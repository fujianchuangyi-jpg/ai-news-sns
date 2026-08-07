"""Ollama プロバイダ（ローカル実行・無料・無制限）。

役割は2つ:
  1. 一次選抜（prefilter）— 60件の候補を20件に絞る。量が多いので
     ここを課金バックエンドに投げると無駄が大きい
  2. フォールバック — Claude Code が使えない日でも投稿を止めない

強み: `format` に JSON Schema を渡すと構造が保証される。
      スキーマ強制ができない Claude Code の最終的な安全網になる。

弱み（実測）:
  - 11 tok/s 程度。60記事の一次選抜で約3分かかる
  - 件数を明示しないと入力より少ない件数を返す
    → プロンプトで件数を強調し、返却件数を検証して不足分を再問い合わせする
"""

from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from ..config import load_settings
from ..llm import sanitize_schema
from . import ProviderError, ProviderUnavailable

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OllamaProvider:
    """ローカルの Ollama サーバを叩く。"""

    name = "ollama"

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        timeout: float | None = None,
    ) -> None:
        cfg = load_settings().llm.get("ollama", {})
        self.model = model or cfg.get("model", "gemma4-ja:latest")
        self.host = (host or cfg.get("host", "http://localhost:11434")).rstrip("/")
        # 60記事の一括処理で数分かかるため、既定のタイムアウトでは足りない
        self.timeout = timeout or float(cfg.get("timeout_seconds", 900))
        self.num_ctx = int(cfg.get("num_ctx", 16384))
        self.temperature = float(cfg.get("temperature", 0.2))

    # ── 内部 ──────────────────────────────────────────────────────────

    def _generate(self, prompt: str, *, system: str, fmt: Any | None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        if fmt is not None:
            payload["format"] = fmt

        try:
            response = httpx.post(
                f"{self.host}/api/generate", json=payload, timeout=self.timeout
            )
        except httpx.ConnectError as exc:
            raise ProviderUnavailable(
                f"Ollama に接続できません（{self.host}）。"
                "`ollama serve` が起動しているか確認してください"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderError(f"Ollama がタイムアウトしました（{self.timeout}秒）") from exc

        if response.status_code == 404:
            raise ProviderUnavailable(
                f"モデル '{self.model}' が見つかりません。"
                f"`ollama pull {self.model}` で取得してください"
            )
        response.raise_for_status()

        body = response.json()
        text = body.get("response", "")
        if not text.strip():
            raise ProviderError("Ollama が空の応答を返しました")

        log.debug(
            "ollama: %s tokens in %.1fs",
            body.get("eval_count"),
            body.get("total_duration", 0) / 1e9,
        )
        return text

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
        """JSON Schema を強制して構造化出力を得る。

        `format` にスキーマを渡すので構文的には必ず通るが、
        配列の件数までは保証されない。件数の検証は呼び出し側で行う。
        """
        prompt = user
        if count_hint is not None:
            prompt += (
                f"\n\n**重要: 入力は {count_hint} 件です。"
                f"{count_hint} 件すべてを漏れなく返してください。**"
            )

        text = self._generate(
            prompt, system=system, fmt=sanitize_schema(schema.model_json_schema())
        )
        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            # format 指定があっても稀に壊れることがあるので、緩く拾い直す
            try:
                return schema.model_validate(json.loads(_extract_json(text)))
            except Exception:
                log.error("Ollama の出力を検証できません: %s", text[:400])
                raise ProviderError(
                    f"{schema.__name__} として解釈できませんでした"
                ) from exc

    def text(
        self,
        *,
        system: str,
        user: str,
        effort: str = "medium",
        max_tokens: int | None = None,
    ) -> str:
        return self._generate(user, system=system, fmt=None).strip()

    # ── 稼働確認 ──────────────────────────────────────────────────────

    def available(self) -> bool:
        """サーバが動いていて、指定モデルが存在するか。"""
        try:
            response = httpx.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
        except Exception:
            return False
        names = {m.get("name", "") for m in response.json().get("models", [])}
        # "gemma4-ja" と "gemma4-ja:latest" のどちらの書き方でも通す
        base = self.model.split(":")[0]
        return any(n == self.model or n.split(":")[0] == base for n in names)


def _extract_json(text: str) -> str:
    """前後に説明文が付いてしまった場合に JSON 部分だけを取り出す。"""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("JSON オブジェクトが見つかりません")
    return text[start : end + 1]


def _selftest() -> int:
    """`python -m ainews.providers.ollama --selftest` 用。

    確認するのは2点:
      1. JSON Schema の強制が効くか
      2. 件数欠落が起きないか（起きるなら指示の効きが足りない）
    """
    import time

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    class Item(BaseModel):
        id: str
        is_ai: bool
        interest: int

    class Batch(BaseModel):
        items: list[Item]

    provider = OllamaProvider()
    print(f"モデル: {provider.model}  ホスト: {provider.host}")
    if not provider.available():
        print("✗ Ollama が利用できません")
        return 1
    print("✓ サーバ稼働・モデル存在を確認")

    titles = [
        "OpenAI releases GPT-6 with 3x faster inference",
        "Apple announces new iPhone case colors",
        "Anthropic publishes position on open-weights models",
        "Google DeepMind unveils Gemini Robotics ER 2",
        "Meta open-sources new speech recognition model",
        "Startup raises $40M for AI chip design",
        "Researchers cut LLM training cost by 60%",
        "Local bakery opens second branch",
    ]
    payload = [{"id": f"a{i}", "title": t} for i, t in enumerate(titles)]

    started = time.monotonic()
    result = provider.structured(
        system="あなたはAIニュースの選別担当です。",
        user=(
            "各記事について is_ai（AI関連か）と interest（0-100の興味度）を判定してください。\n"
            + json.dumps(payload, ensure_ascii=False)
        ),
        schema=Batch,
        count_hint=len(payload),
    )
    elapsed = time.monotonic() - started

    print(f"✓ スキーマ強制: 成功（{elapsed:.1f}秒）")
    print(f"  返却件数: {len(result.items)} / {len(payload)}")
    if len(result.items) != len(payload):
        print("  ⚠ 件数が欠落しています。呼び出し側の再問い合わせが必要です")

    for item in result.items[:4]:
        print(f"    {item.id}: is_ai={item.is_ai} interest={item.interest}")

    ai_flags = {i.id: i.is_ai for i in result.items}
    if ai_flags.get("a0") is False or ai_flags.get("a7") is True:
        print("  ⚠ AI判定が期待とずれています（a0=True, a7=False が期待値）")
    else:
        print("✓ AI判定: 妥当")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_selftest() if "--selftest" in sys.argv else 0)
