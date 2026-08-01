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
    ) -> T:
        """JSON Schema で出力を拘束し、Pydantic モデルとして返す。"""
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
