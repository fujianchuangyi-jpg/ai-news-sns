"""LLM プロバイダの共通インタフェース。

パイプライン側（select.py / compose.py / analytics.py）は
`structured()` と `text()` の2つだけを見る。どのバックエンドを使うかは
`llm.make_llm()` が決める。

用意しているバックエンド:
    ollama       … ローカル実行。無料・無制限。JSON Schema を強制できる
    claude_code  … `claude -p` を叩く。Max契約の枠内で動く。スキーマは強制できない
    anthropic    … Anthropic API 直（従量課金）。品質最優先のときの保険
    fake         … API を叩かないスタブ（fakes.py）
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class Provider(Protocol):
    """LLM バックエンドが満たすべき最小のインタフェース。"""

    name: str

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        effort: str = "high",
        max_tokens: int | None = None,
    ) -> T:
        """スキーマに沿った構造化出力を返す。"""
        ...

    def text(
        self,
        *,
        system: str,
        user: str,
        effort: str = "medium",
        max_tokens: int | None = None,
    ) -> str:
        """自由記述のテキストを返す。"""
        ...


class ProviderError(RuntimeError):
    """バックエンドが使える結果を返さなかった。

    これを投げるとフォールバック機構が別のバックエンドに切り替える。
    """


class ProviderUnavailable(ProviderError):
    """バックエンド自体が使えない（未インストール・未ログイン・未起動）。

    ProviderError と分けているのは、これが出たら以降の呼び出しも
    確実に失敗するため、即座にフォールバックへ切り替えたいから。
    """


def schema_instruction(schema: type[BaseModel], *, count_hint: int | None = None) -> str:
    """スキーマを強制できないバックエンド向けの指示文を作る。

    Claude Code は `--output-format json` を付けても中身の構造は拘束されない
    （封筒の形式が JSON になるだけ）。そこでスキーマ自体をプロンプトに
    埋め込んで従わせ、受け取った側で Pydantic 検証する。
    """
    import json

    from ..llm import sanitize_schema

    body = json.dumps(
        sanitize_schema(schema.model_json_schema()), ensure_ascii=False, indent=2
    )
    lines = [
        "",
        "# 出力形式（厳守）",
        "",
        "以下の JSON Schema に厳密に従った **JSON オブジェクトのみ** を出力してください。",
        "説明文・前置き・```json などのコードフェンスは一切付けないでください。",
        "最初の文字は `{` で、最後の文字は `}` です。",
        "",
        "```",
        body,
        "```",
    ]
    if count_hint is not None:
        lines += [
            "",
            f"**入力は {count_hint} 件です。{count_hint} 件すべてを漏れなく返してください。**",
            "件数が足りない出力は不正解として扱われます。",
        ]
    return "\n".join(lines)
