"""Claude Code プロバイダ（`claude -p` の subprocess ラッパ）。

Max 契約の枠内で動くため追加課金が発生しない。ニュース選定の精度を
最優先するため、パイプラインで最も判断が難しい2工程
（精密評価・原稿生成）をここに割り当てる。

実測でわかった制約:

  1回あたり約25,000トークンの固定オーバーヘッド
      Claude Code 自身のシステムプロンプトとツール定義が毎回読み込まれる
      （cache_read 18,950 + cache_create 約6,300）。
      短いプロンプトでも必ずこの分がかかるので、**呼び出し回数を増やさず
      1回に詰め込む**のが正しい使い方。パイプラインは1日2回に設計している。

  スキーマを強制できない
      `--output-format json` は封筒（メタデータ + result）の形式を JSON に
      するだけで、result の中身は拘束されない。そこで
      providers.schema_instruction() でスキーマ自体をプロンプトに埋め込み、
      受け取ってから Pydantic で検証し、違反したらエラー内容を添えて
      書き直させる。それでも駄目なら呼び出し側が Ollama に退避する。

  プロンプトは stdin から渡す
      引数で渡すと記事本文を積んだときに長さの上限に当たる。
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..config import load_settings
from . import ProviderError, ProviderUnavailable, schema_instruction

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# 失敗メッセージに含まれていたら「使えない」と判断する語。
# これが出た場合は書き直させても無駄なので即フォールバックする。
_UNAVAILABLE_HINTS = (
    "usage limit",
    "rate limit",
    "not logged in",
    "please run /login",
    "authentication",
    "failed to authenticate",
    # OAuth セッションは自動更新されるが、更新に失敗すると期限切れになる。
    # 実運用で2日間これが起き、気づかないまま品質の低いローカルLLMで
    # 生成され続けた。再ログインが必要なので利用者に知らせる必要がある。
    "oauth session expired",
    "session expired",
    "invalid api key",
    "credit balance",
    "quota",
)

# 退避理由のうち、利用者が手を打たないと直らないもの。
# 単なる一時障害と区別して、通知で明示的に対処を促す。
_NEEDS_ACTION = ("oauth session expired", "session expired", "not logged in", "/login")


def needs_relogin(reason: str) -> bool:
    """退避理由が再ログインを要するものか。"""
    lowered = reason.lower()
    return any(hint in lowered for hint in _NEEDS_ACTION)


class ClaudeCodeProvider:
    """`claude -p` を呼んで構造化出力を得る。"""

    name = "claude_code"

    def __init__(
        self,
        model: str | None = None,
        timeout: float | None = None,
        schema_retry: int | None = None,
    ) -> None:
        cfg = load_settings().llm.get("claude_code", {})
        self.model = model if model is not None else cfg.get("model", "")
        self.timeout = timeout or float(cfg.get("timeout_seconds", 600))
        self.schema_retry = (
            schema_retry if schema_retry is not None else int(cfg.get("schema_retry", 1))
        )
        # ツールを無効にしているので本来1ターンで返るはずだが、1に固定すると
        # 長い出力のときに error_max_turns で落ちることがある（実測）。
        # 暴走はツール無効で防げているので、少しだけ余裕を持たせる。
        self.max_turns = int(cfg.get("max_turns", 3))
        self.binary = shutil.which("claude")
        # 実行のたびに消費トークンを積算し、Max枠の減り方を可視化する
        self.calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    # ── 内部 ──────────────────────────────────────────────────────────

    def _run(self, prompt: str, *, system: str) -> str:
        if not self.binary:
            raise ProviderUnavailable(
                "`claude` コマンドが見つかりません。"
                "Claude Code をインストールし PATH を通してください"
            )

        command = [
            self.binary,
            "-p",
            "--output-format",
            "json",
            "--max-turns",
            str(self.max_turns),
            # ツールは一切使わせない。ファイル探索などを始めると
            # 余計なターンとトークンを消費する
            "--allowed-tools",
            "",
        ]
        if self.model:
            command += ["--model", self.model]
        if system:
            command += ["--append-system-prompt", system]

        try:
            completed = subprocess.run(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise ProviderError(
                f"claude -p がタイムアウトしました（{self.timeout:.0f}秒）"
            ) from exc
        except OSError as exc:
            raise ProviderUnavailable(f"claude を起動できません: {exc}") from exc

        # 異常終了でも本文は JSON の封筒で返ることが多い。生の文字列を
        # そのまま投げると 200 字で切られて肝心の理由が読めなくなるため、
        # まず封筒として解釈し、result / subtype から理由を取り出す。
        envelope: dict[str, Any] | None = None
        if completed.stdout.strip():
            try:
                envelope = json.loads(completed.stdout)
            except json.JSONDecodeError:
                envelope = None

        if completed.returncode != 0 or (envelope and envelope.get("is_error")):
            raise self._classify(
                _failure_reason(envelope, completed.stderr, completed.stdout),
                completed.returncode,
            )

        if envelope is None:
            raise ProviderError(
                f"claude -p の出力を JSON として読めません: {completed.stdout[:300]}"
            )

        self._record_usage(envelope)

        result = envelope.get("result")
        if not isinstance(result, str) or not result.strip():
            raise ProviderError("claude -p が空の応答を返しました")
        return result

    @staticmethod
    def _classify(message: str, returncode: int) -> ProviderError:
        """エラーメッセージから、退避すべきか再試行すべきかを判断する。"""
        lowered = message.lower()
        if any(hint in lowered for hint in _UNAVAILABLE_HINTS):
            return ProviderUnavailable(f"Claude Code が利用できません: {message[:400]}")
        return ProviderError(f"claude -p が失敗しました (exit={returncode}): {message[:400]}")

    def _record_usage(self, envelope: dict[str, Any]) -> None:
        usage = envelope.get("usage") or {}
        self.calls += 1
        # cache_read も Max枠の消費に含まれるので合算して見る
        call_input = (
            int(usage.get("input_tokens", 0))
            + int(usage.get("cache_creation_input_tokens", 0))
            + int(usage.get("cache_read_input_tokens", 0))
        )
        call_output = int(usage.get("output_tokens", 0))
        self.total_input_tokens += call_input
        self.total_output_tokens += call_output
        log.info(
            "claude -p %d回目: 今回 in=%s out=%s / 累計 in=%s out=%s (%.1f秒)",
            self.calls,
            f"{call_input:,}",
            f"{call_output:,}",
            f"{self.total_input_tokens:,}",
            f"{self.total_output_tokens:,}",
            envelope.get("duration_ms", 0) / 1000,
        )

    # ── 公開 API ──────────────────────────────────────────────────────

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        effort: str = "high",
        max_tokens: int | None = None,
        count_hint: int | None = None,
    ) -> T:
        """スキーマ指示を埋め込んで呼び、返答を検証する。

        スキーマ違反なら、何が悪かったかを伝えて書き直させる。
        """
        instruction = schema_instruction(schema, count_hint=count_hint)
        prompt = f"{user}\n{instruction}"
        last_error: Exception | None = None

        for attempt in range(self.schema_retry + 1):
            raw = self._run(prompt, system=system)
            try:
                return schema.model_validate_json(_strip_fence(raw))
            except ValidationError as exc:
                last_error = exc
                log.warning(
                    "Claude Code の出力がスキーマに合いません [%d/%d]: %s",
                    attempt + 1,
                    self.schema_retry + 1,
                    str(exc)[:200],
                )
                if attempt < self.schema_retry:
                    prompt = (
                        f"{user}\n{instruction}\n\n"
                        "# 直前の出力は不正でした\n\n"
                        f"あなたの直前の出力:\n```\n{raw[:1500]}\n```\n\n"
                        f"検証エラー:\n```\n{str(exc)[:800]}\n```\n\n"
                        "エラーを修正し、スキーマに厳密に従った JSON のみを"
                        "出力し直してください。"
                    )

        raise ProviderError(
            f"{schema.__name__} として解釈できませんでした: {str(last_error)[:200]}"
        )

    def text(
        self,
        *,
        system: str,
        user: str,
        effort: str = "medium",
        max_tokens: int | None = None,
    ) -> str:
        return self._run(user, system=system).strip()

    # ── 稼働確認 ──────────────────────────────────────────────────────

    def available(self) -> bool:
        return self.binary is not None

    def usage_summary(self) -> str:
        if not self.calls:
            return "  Claude Code: 呼び出しなし"
        return (
            f"  Claude Code: {self.calls}回  "
            f"入力 {self.total_input_tokens:,} / 出力 {self.total_output_tokens:,} トークン"
        )


def _failure_reason(
    envelope: dict[str, Any] | None, stderr: str, stdout: str
) -> str:
    """失敗の理由を人が読める1行にする。

    claude -p は異常時も JSON の封筒を返す。生の JSON をそのまま
    エラーメッセージにすると、先頭のメタデータだけで文字数を使い切って
    肝心の理由が読めない。result / subtype を優先して取り出す。
    """
    if envelope:
        parts = [
            str(envelope.get(key))
            for key in ("result", "subtype", "api_error_status", "stop_reason")
            if envelope.get(key)
        ]
        if parts:
            return " / ".join(parts)
    return (stderr or stdout or "理由不明").strip()


def _strip_fence(text: str) -> str:
    """```json ... ``` で包まれていた場合に中身を取り出す。"""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 2:
        body = lines[1:]
        if body and body[-1].strip().startswith("```"):
            body = body[:-1]
        return "\n".join(body).strip()
    return stripped


def _selftest() -> int:
    """`python -m ainews.providers.claude_code --selftest` 用。"""
    import time

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    class Item(BaseModel):
        id: str
        headline_ja: str
        interest: int

    class Batch(BaseModel):
        items: list[Item]

    provider = ClaudeCodeProvider()
    print(f"バイナリ: {provider.binary}")
    print(f"モデル: {provider.model or '(Claude Code の既定)'}")
    if not provider.available():
        print("✗ claude コマンドが見つかりません")
        return 1

    payload = [
        {"id": "a0", "title": "OpenAI releases GPT-6 with 3x faster inference"},
        {"id": "a1", "title": "Researchers cut LLM training cost by 60%"},
        {"id": "a2", "title": "EU finalizes AI Act enforcement timeline"},
    ]
    started = time.monotonic()
    try:
        result = provider.structured(
            system="あなたは日本語のAIニュース編集者です。",
            user=(
                "各記事について headline_ja（日本語28字以内の見出し）と "
                "interest（0-100の興味度）を返してください。\n"
                + json.dumps(payload, ensure_ascii=False)
            ),
            schema=Batch,
            count_hint=len(payload),
        )
    except ProviderError as exc:
        print(f"✗ {exc}")
        return 1
    elapsed = time.monotonic() - started

    print(f"✓ stdin入力・JSON再パース・スキーマ検証: 成功（{elapsed:.1f}秒）")
    print(f"  返却件数: {len(result.items)} / {len(payload)}")
    for item in result.items:
        print(f"    {item.id}: [{item.interest}] {item.headline_ja}")
    print(provider.usage_summary())
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_selftest() if "--selftest" in sys.argv else 0)
