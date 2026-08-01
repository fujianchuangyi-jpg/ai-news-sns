"""本文抽出と OGP 画像の取得。

RSS の summary だけでは LLM が事実関係を判断できないので、記事ページを
取得して本文を抜く。同時にカード画像に使う OGP 画像URLも拾っておく。

全件やると時間もマナーも悪いので、呼び出し側が絞った候補にだけ適用する
（settings.collect.max_fulltext_fetch）。
"""

from __future__ import annotations

import asyncio
import logging

import trafilatura

from .config import load_settings
from .models import Article
from .net import build_client, extract_og_image, get_with_retry

log = logging.getLogger(__name__)

# 画像を取得しないポリシー。text_only は最初から文字カードにする。
NO_IMAGE_POLICIES = frozenset({"text_only"})


class Extractor:
    """記事ページから本文と OGP 画像を取り出す。"""

    def __init__(self) -> None:
        cfg = load_settings().collect
        self.timeout = cfg["timeout_seconds"]
        self.user_agent = cfg["user_agent"]
        self._semaphore = asyncio.Semaphore(cfg["max_concurrency"])

    async def _enrich(self, client, article: Article) -> Article:
        async with self._semaphore:
            try:
                response = await get_with_retry(client, article.url, attempts=2)
            except Exception as exc:
                # 本文が取れなくても summary で先に進める。落とさない。
                log.debug("[%s] 本文取得に失敗: %s", article.id, exc)
                return article

            html = response.text
            body = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
            )
            if body:
                article.fulltext = body.strip()

            if article.image_policy not in NO_IMAGE_POLICIES:
                article.og_image_url = extract_og_image(html, str(response.url))
            return article

    async def enrich_async(self, articles: list[Article]) -> list[Article]:
        async with build_client(
            timeout=self.timeout, user_agent=self.user_agent
        ) as client:
            return list(
                await asyncio.gather(*(self._enrich(client, a) for a in articles))
            )

    def enrich(self, articles: list[Article]) -> list[Article]:
        if not articles:
            return []
        return asyncio.run(self.enrich_async(articles))


def enrichment_summary(articles: list[Article]) -> str:
    with_text = sum(1 for a in articles if a.fulltext)
    with_image = sum(1 for a in articles if a.og_image_url)
    return (
        f"  本文あり {with_text}/{len(articles)}  "
        f"OGP画像あり {with_image}/{len(articles)}"
    )
