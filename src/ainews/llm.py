"""Claude API ラッパ。

方針:
  - モデルは claude-opus-5。thinking は adaptive（Opus 5 は既定で有効）、
    深さは output_config.effort で制御する。
  - 構造化出力は output_config.format に JSON Schema を渡し、
    受け取った JSON を Pydantic で検証する。
  - system プロンプトは固定文字列にして cache_control を付ける
    （同一実行内の連続呼び出しでプロンプトキャッシュが効く）。

Batch API について:
  Batch は50%安いが「最大24時間」の SLA があり、毎朝の定時公開には
  レイテンシのリスクが大きすぎる。日次コストが $0.5 未満のため、
  信頼性を優先して同期呼び出しにしている。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel

from .config import load_settings

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# 構造化出力の JSON Schema がサポートしないキーワード。
# Pydantic は Field(ge=..., max_length=...) からこれらを生成するので、
# 送信前に除去し、検証はクライアント側（Pydantic）で行う。
UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
        "default",
        "examples",
    }
)

SUPPORTED_STRING_FORMATS = frozenset(
    {
        "date-time",
        "time",
        "date",
        "duration",
        "email",
        "hostname",
        "uri",
        "ipv4",
        "ipv6",
        "uuid",
    }
)


class LLMError(RuntimeError):
    """LLM 呼び出しが使える結果を返さなかった。"""


def sanitize_schema(node: Any) -> Any:
    """Pydantic の JSON Schema を構造化出力が受け付ける形に整える。

    - サポート外のキーワードを除去
    - object には additionalProperties: false を付与
    - object の全プロパティを required に入れる（構造化出力の要件）
    """
    if isinstance(node, list):
        return [sanitize_schema(item) for item in node]
    if not isinstance(node, dict):
        return node

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in UNSUPPORTED_SCHEMA_KEYS:
            continue
        if key == "format" and value not in SUPPORTED_STRING_FORMATS:
            continue
        out[key] = sanitize_schema(value)

    if out.get("type") == "object" or "properties" in out:
        out["additionalProperties"] = False
        props = out.get("properties")
        if isinstance(props, dict) and props:
            out["required"] = list(props.keys())
    return out


class LLM:
    """Claude API の薄いラッパ。"""

    def __init__(self, model: str | None = None, max_tokens: int | None = None) -> None:
        settings = load_settings()
        self.model = model or settings.llm["model"]
        self.max_tokens = max_tokens or settings.llm["max_tokens"]
        # API キーは環境変数から解決される（明示的に渡さない）
        self._client = anthropic.Anthropic()

    # ── 内部 ──────────────────────────────────────────────────────────

    def _system_blocks(self, system: str) -> list[dict[str, Any]]:
        """system をキャッシュ対象のブロックにする。

        最小キャッシュ長（Opus 5 で 512 トークン）に満たない場合は
        cache_control を付けても効かないだけで害はない。
        """
        return [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _extract_text(self, response: Any) -> str:
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMError(f"モデルが応答を拒否しました (category={category})")
        if response.stop_reason == "max_tokens":
            log.warning("max_tokens に到達しました。出力が切れている可能性があります")
        parts = [b.text for b in response.content if b.type == "text"]
        if not parts:
            raise LLMError("テキストブロックが返りませんでした")
        return "".join(parts)

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
        """JSON Schema で出力を拘束し、Pydantic モデルとして返す。

        count_hint は他バックエンドとの互換のために受け取るが、ここでは
        使わない。Anthropic API は output_config.format でスキーマを
        強制できるため、件数を言い聞かせる必要がない。
        """
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            system=self._system_blocks(system),
            messages=[{"role": "user", "content": user}],
            output_config={
                "effort": effort,
                "format": {
                    "type": "json_schema",
                    "schema": sanitize_schema(schema.model_json_schema()),
                },
            },
        )
        text = self._extract_text(response)
        try:
            return schema.model_validate_json(text)
        except Exception as exc:
            log.error("構造化出力の検証に失敗: %s", text[:500])
            raise LLMError(f"{schema.__name__} として解釈できませんでした") from exc

    def text(
        self,
        *,
        system: str,
        user: str,
        effort: str = "medium",
        max_tokens: int | None = None,
    ) -> str:
        """自由記述のテキストを返す（週次レポート等）。"""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            system=self._system_blocks(system),
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort},
        )
        return self._extract_text(response)


def json_dump(obj: Any) -> str:
    """プロンプトに埋め込むための JSON 整形（日本語をエスケープしない）。"""
    return json.dumps(obj, ensure_ascii=False, indent=2)


def api_key_available() -> bool:
    """ANTHROPIC_API_KEY が設定されているか（dry-run 判定用）。"""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


class FallbackLLM:
    """主バックエンドが失敗したら退避先に切り替えるラッパ。

    毎朝の投稿を止めないための機構。Claude Code が使えない日
    （利用上限に達した、未ログイン、オフライン）でも、ローカルの
    Ollama が引き継いで下書きを完成させる。

    一度退避したら、その実行中は退避先を使い続ける。主バックエンドが
    使えない状態は数分で直らないことが多く、呼び出しのたびに試すのは
    時間の無駄になるため。
    """

    def __init__(self, primary: Any, fallback: Any) -> None:
        self.primary = primary
        self.fallback = fallback
        self.active = primary
        self.fallback_reason: str = ""

    @property
    def name(self) -> str:
        return getattr(self.active, "name", "unknown")

    @property
    def switched(self) -> bool:
        return self.active is self.fallback

    def _switch(self, exc: Exception) -> None:
        if self.switched:
            return
        self.fallback_reason = f"{getattr(self.primary, 'name', '?')}: {exc}"
        log.warning(
            "%s が使えないため %s に切り替えます: %s",
            getattr(self.primary, "name", "?"),
            getattr(self.fallback, "name", "?"),
            exc,
        )
        self.active = self.fallback

    def _call(self, method: str, **kwargs: Any) -> Any:
        from .providers import ProviderError

        try:
            return getattr(self.active, method)(**kwargs)
        except ProviderError as exc:
            if self.switched:
                raise
            self._switch(exc)
            return getattr(self.active, method)(**kwargs)

    def structured(self, **kwargs: Any) -> Any:
        return self._call("structured", **kwargs)

    def text(self, **kwargs: Any) -> Any:
        return self._call("text", **kwargs)


def _build(backend: str) -> Any:
    """バックエンド名から素のプロバイダを作る。"""
    if backend == "ollama":
        from .providers.ollama import OllamaProvider

        return OllamaProvider()
    if backend == "claude_code":
        from .providers.claude_code import ClaudeCodeProvider

        return ClaudeCodeProvider()
    if backend == "anthropic":
        return LLM()
    if backend == "fake":
        from .fakes import FakeLLM

        return FakeLLM()
    raise ValueError(f"未知のバックエンド: {backend}")


def make_llm(backend: str | None = None, *, with_fallback: bool = True) -> Any:
    """設定に従って LLM バックエンドを組み立てる。

    Args:
        backend: 明示指定。None なら settings.llm.backend を使う。
        with_fallback: 退避先を付けるか。バックエンド比較のように
            素の挙動を見たいときは False にする。
    """
    settings = load_settings()
    backend = backend or settings.llm.get("backend", "anthropic")
    primary = _build(backend)

    if not with_fallback:
        return primary

    fallback_name = settings.llm.get("fallback_backend")
    if not fallback_name or fallback_name == backend:
        return primary

    try:
        fallback = _build(fallback_name)
    except Exception as exc:
        log.warning("退避先 %s を用意できません: %s", fallback_name, exc)
        return primary

    return FallbackLLM(primary, fallback)
