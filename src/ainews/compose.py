"""③ 原稿生成。

X は1ニュース=1投稿、Instagram は4ニュース=1カルーセル投稿。

X の文字数だけは投稿可否に直結するので、生成後に必ず検証し、
超過していたら字数を明示して書き直させる。
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from .config import load_settings, prompt
from .llm import LLM, LLMError, json_dump
from .models import IGCaption, ScoredArticle, XPost

log = logging.getLogger(__name__)


class XPostList(BaseModel):
    """複数の X 投稿をまとめて返すためのラッパ。"""

    posts: list[XPost]


def weighted_length(text: str) -> int:
    """X の重み付き文字数を数える。

    X は文字ごとに重み1または2を割り当て、上限280で判定する。
    ラテン文字・記号は1、日本語などそれ以外は2。つまり日本語だけなら
    実質140文字が上限になる。

    重み1の範囲（X の仕様）:
        U+0000-U+10FF, U+2000-U+200D, U+2010-U+201F, U+2032-U+2037
    """
    total = 0
    for char in text:
        code = ord(char)
        light = (
            code <= 0x10FF
            or 0x2000 <= code <= 0x200D
            or 0x2010 <= code <= 0x201F
            or 0x2032 <= code <= 0x2037
        )
        total += 1 if light else 2
    return total


def visible_length(text: str) -> int:
    """人間向けの見た目の文字数（改行を除く）。"""
    return len(text.replace("\n", ""))


class Composer:
    def __init__(self, llm: LLM | None = None) -> None:
        self.settings = load_settings()
        self.cfg = self.settings.compose
        self.llm = llm or LLM()
        self.effort = self.settings.llm["compose_effort"]

    # ── X ─────────────────────────────────────────────────────────────

    def _x_payload(self, item: ScoredArticle) -> dict:
        return {
            "id": item.article.id,
            "title": item.article.title,
            "headline_ja": item.assessment.headline_ja,
            "source": item.article.source_name,
            "hook_seed": item.assessment.hook,
            "why_matters": item.assessment.why_matters,
            "category": item.assessment.category,
            "body": item.article.text_for_llm,
        }

    def compose_x(self, items: list[ScoredArticle]) -> list[XPost]:
        """各ニュースの X 原稿を作り、字数超過は書き直させる。"""
        system = prompt("compose_x")
        limit = self.cfg["x_max_weighted"]

        user = (
            f"以下の {len(items)} 件のニュースそれぞれについて、X の投稿原稿を"
            f"書いてください。\n\n{json_dump([self._x_payload(i) for i in items])}"
        )
        result = self.llm.structured(
            system=system, user=user, schema=XPostList, effort=self.effort
        )
        posts = {p.article_id: p for p in result.posts}

        # 足りない・多い場合に備えて入力順に整える
        ordered = [posts[i.article.id] for i in items if i.article.id in posts]
        missing = [i for i in items if i.article.id not in posts]
        if missing:
            log.warning("X 原稿が生成されなかった記事: %d 件", len(missing))

        return [self._enforce_length(p, items, system, limit) for p in ordered]

    def _enforce_length(
        self,
        post: XPost,
        items: list[ScoredArticle],
        system: str,
        limit: int,
    ) -> XPost:
        """字数超過なら、超過量を伝えて書き直させる。"""
        retries = self.cfg["x_retry_limit"]
        item = next((i for i in items if i.article.id == post.article_id), None)
        if item is None:
            return post

        for attempt in range(retries):
            length = weighted_length(post.body)
            if length <= limit:
                return post
            over = length - limit
            log.info(
                "X原稿が %d 超過（%d/%d）。書き直します [%d/%d]",
                over,
                length,
                limit,
                attempt + 1,
                retries,
            )
            user = (
                f"次の原稿は X の上限を超えています。\n"
                f"現在: 重み付き {length} / 上限 {limit}（{over} 超過。"
                f"日本語なら約 {-(-over // 2)} 文字削る必要があります）\n\n"
                f"--- 現在の原稿 ---\n{post.body}\n---\n\n"
                f"意味と数字を保ったまま短くしてください。"
                f"ハッシュタグは1個に減らしても構いません。"
                f"出典表記は必ず残してください。\n\n"
                f"元のニュース:\n{json_dump(self._x_payload(item))}"
            )
            try:
                rewritten = self.llm.structured(
                    system=system, user=user, schema=XPost, effort=self.effort
                )
                rewritten.article_id = post.article_id
                post = rewritten
            except LLMError as exc:
                log.warning("書き直しに失敗: %s", exc)
                break

        if weighted_length(post.body) > limit:
            log.error(
                "X原稿が上限を超えたままです（%d/%d）: %s",
                weighted_length(post.body),
                limit,
                post.article_id,
            )
        return post

    # ── Instagram ─────────────────────────────────────────────────────

    def compose_ig(self, items: list[ScoredArticle]) -> IGCaption:
        payload = [
            {
                "order": n,
                "title": i.article.title,
                "headline_ja": i.assessment.headline_ja,
                "source": i.article.source_name,
                "hook_seed": i.assessment.hook,
                "why_matters": i.assessment.why_matters,
                "category": i.assessment.category,
                "body": i.article.text_for_llm,
            }
            for n, i in enumerate(items, start=1)
        ]
        count = self.cfg["ig_hashtag_count"]
        user = (
            f"以下の {len(payload)} 件を1つのカルーセル投稿にまとめる"
            f"キャプションを書いてください。ハッシュタグは {count} 個前後。\n\n"
            f"{json_dump(payload)}"
        )
        return self.llm.structured(
            system=prompt("compose_ig"),
            user=user,
            schema=IGCaption,
            effort=self.effort,
        )

    def run(self, items: list[ScoredArticle]) -> tuple[list[XPost], IGCaption]:
        return self.compose_x(items), self.compose_ig(items)


def format_posts(posts: list[XPost], limit: int = 280) -> str:
    """X 原稿を字数付きで表示する（--dry-run 用）。"""
    lines = []
    for post in posts:
        weight = weighted_length(post.body)
        status = "OK " if weight <= limit else "超過"
        lines.append(
            f"  ── [{status}] 重み {weight}/{limit}  見た目 "
            f"{visible_length(post.body)}字  フック型: {post.hook_type}"
        )
        lines.extend(f"     {line}" for line in post.body.split("\n"))
        lines.append("")
    return "\n".join(lines)
