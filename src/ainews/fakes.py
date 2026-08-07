"""API キー無しでパイプライン全体を通すためのフェイク LLM。

用途は2つ:
  1. 開発時に画像生成・プレビュー・分析まで一気に確認する
  2. CI で LLM 以外の回帰を検出する

出力はスキーマ的に正しく、入力（記事タイトル）から機械的に組み立てた
それらしい日本語になる。品質の評価には使えないが、配線の検証には十分。
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel

from .models import (
    Assessment,
    AssessmentBatch,
    Category,
    IGCaption,
    XPost,
)

T = TypeVar("T", bound=BaseModel)

_CATEGORIES: list[Category] = [
    "新モデル",
    "製品",
    "資金調達",
    "研究",
    "規制",
    "事件",
    "ツール",
    "業界動向",
]

_HOOK_TYPES = ["数字型", "疑問型", "対比型", "断言型", "物語型"]


def _seed(value: str) -> int:
    """文字列から決定的な整数を作る（実行のたびに同じ結果にする）。"""
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)


def _extract_items(user: str) -> list[dict[str, Any]]:
    """プロンプトに埋め込んだ JSON 配列を取り出す。"""
    match = re.search(r"\[\s*\{.*}\s*]", user, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


class FakeLLM:
    """LLM と同じインタフェースを持つスタブ。"""

    def __init__(self, *_: Any, **__: Any) -> None:
        self.model = "fake"
        self.calls: list[str] = []

    def structured(self, *, system: str, user: str, schema: type[T], **_: Any) -> T:
        self.calls.append(schema.__name__)
        items = _extract_items(user)

        if schema is AssessmentBatch:
            return AssessmentBatch(  # type: ignore[return-value]
                assessments=[self._assessment(i) for i in items]
            )
        if schema.__name__ == "DraftCopy":
            return schema(  # type: ignore[call-arg]
                posts=[self._x_post(i) for i in items],
                caption=self._ig_caption(items),
            )
        if schema.__name__ == "XPostList":
            return schema(posts=[self._x_post(i) for i in items])  # type: ignore[call-arg]
        if schema is XPost:
            # 字数超過時の書き直し要求。短縮版を返す。
            return self._shortened(user)  # type: ignore[return-value]
        if schema is IGCaption:
            return self._ig_caption(items)  # type: ignore[return-value]
        raise NotImplementedError(f"FakeLLM は {schema.__name__} に未対応です")

    def text(self, *, system: str, user: str, **_: Any) -> str:
        self.calls.append("text")
        return "（フェイクLLMによる仮のテキスト出力）"

    # ── 各スキーマの生成 ──────────────────────────────────────────────

    def _assessment(self, item: dict[str, Any]) -> Assessment:
        seed = _seed(str(item.get("id", "")))
        outlets = int(item.get("reported_by_outlets", 1))
        # 報じた媒体数を fame に反映させ、選定ロジックの分岐を実際に動かす
        fame = min(100, 25 + (seed % 45) + outlets * 12)
        return Assessment(
            article_id=str(item.get("id", "")),
            fame=fame,
            interest=20 + (seed // 7) % 75,
            category=_CATEGORIES[seed % len(_CATEGORIES)],
            # 実運用では LLM が日本語に書き換える。ここでは日本語カードの
            # レイアウト（折り返し・字詰め）を確認できる長さの仮文字列を返す。
            headline_ja=f"仮の日本語見出しです{'、詳細は本文を参照' if seed % 2 else ''}",
            hook=f"{str(item.get('title', ''))[:20]}（仮フック）",
            why_matters="フェイクLLMが生成した仮の重要性説明です。",
            risk_flags=[],
        )

    def _x_post(self, item: dict[str, Any]) -> XPost:
        seed = _seed(str(item.get("id", "")))
        title = str(item.get("title", ""))[:40]
        source = item.get("source", "不明")
        body = (
            f"{title}\n\n"
            f"これはフェイクLLMが生成した仮の本文です。実際の原稿は"
            f"記事本文にもとづいて書かれます。\n\n"
            f"#AI #生成AI（出典: {source}）"
        )
        return XPost(
            article_id=str(item.get("id", "")),
            body=body,
            hook_type=_HOOK_TYPES[seed % len(_HOOK_TYPES)],  # type: ignore[arg-type]
        )

    def _shortened(self, user: str) -> XPost:
        match = re.search(r"--- 現在の原稿 ---\n(.*?)\n---", user, re.DOTALL)
        original = match.group(1) if match else ""
        source = ""
        if source_match := re.search(r"（出典: ([^）]+)）", original):
            source = source_match.group(1)
        return XPost(
            article_id="",
            body=f"短縮した仮の本文です。#AI（出典: {source or '不明'}）",
            hook_type="断言型",
        )

    def _ig_caption(self, items: list[dict[str, Any]]) -> IGCaption:
        return IGCaption(
            opening="これはフェイクLLMが生成した仮の冒頭フックです。",
            items=[
                f"{str(i.get('title', ''))[:40]} — 仮の要約テキスト。"
                f"({i.get('source', '不明')})"
                for i in items
            ],
            closing="仮の締めコメントです。毎朝4本、AIニュースをまとめています。",
            hashtags=[
                "AI", "生成AI", "人工知能", "テクノロジー", "AIニュース",
                "ChatGPT", "Claude", "Gemini", "機械学習", "ディープラーニング",
                "LLM", "AI活用", "DX", "IT", "ガジェット",
                "artificialintelligence", "machinelearning", "technews",
                "tech", "startup", "innovation", "programming",
            ],
        )
