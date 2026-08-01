"""① ニュース抽出。

sources.yaml に定義された各ソースを並列に取得し、Article のリストにして返す。

設計方針:
  - 1ソースの失敗で全体を止めない。結果は SourceResult に集約し、
    どのソースが何件取れて何が落ちたかを必ずログに出す。
  - 取得段階では絞り込みを最小限にする（AIキーワードと期間のみ）。
    品質判断は select.py の LLM 評価に任せる。
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import feedparser
import httpx

from .config import Source, load_settings, load_sources
from .models import Article, article_id
from .net import build_client, extract_links, get_with_retry, parse_meta

log = logging.getLogger(__name__)


@dataclass
class SourceResult:
    """1ソースの取得結果。実行サマリの表示に使う。"""

    source_id: str
    fetched: int = 0
    kept: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def _compile_keywords(keywords: list[str]) -> list[re.Pattern[str]]:
    """AIキーワードを正規表現にする。

    ASCII の短い語（AI, LLM 等）は部分一致だと誤爆する（"said" が "ai" に当たる）
    ため単語境界を付ける。日本語は単語境界が効かないので部分一致にする。
    """
    patterns = []
    for kw in keywords:
        if kw.isascii():
            patterns.append(re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE))
        else:
            patterns.append(re.compile(re.escape(kw)))
    return patterns


def _to_utc(value: Any) -> datetime | None:
    """struct_time / epoch 秒 / datetime を aware な UTC datetime に変換する。"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, (int, float)):  # HN の created_at_i など
        try:
            return datetime.fromtimestamp(value, tz=UTC)
        except (OSError, OverflowError, ValueError):
            return None
    try:
        # feedparser の *_parsed は UTC の struct_time
        import calendar

        return datetime.fromtimestamp(calendar.timegm(value), tz=UTC)
    except (TypeError, ValueError):
        return None


def _parse_iso(value: str | None) -> datetime | None:
    """ISO8601 文字列を aware な UTC datetime にする。'Z' 終端にも対応。"""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# og:title は "記事名 | サイト名" 形式が多い。カード画像と原稿にそのまま
# 載るので、末尾の媒体名を落としておく。
_TITLE_SUFFIX = re.compile(r"\s*[|｜]\s*[^|｜]{1,30}$")


def clean_title(title: str) -> str:
    cleaned = _TITLE_SUFFIX.sub("", title.strip())
    # 全部消えてしまう見出し（"| Mistral" だけ 等）は元を返す
    return cleaned if len(cleaned) >= 8 else title.strip()


class Collector:
    """全ソースを並列取得する。

    Args:
        first_seen: 記事ID → 初めて観測した日時。公開日メタを持たない
            HTML ソースで、既知の記事に元の日付を割り当てるために使う。
            通常は store から渡す。
    """

    def __init__(self, first_seen: dict[str, datetime] | None = None) -> None:
        self.settings = load_settings()
        self.sources = load_sources()
        self.first_seen = first_seen or {}
        cfg = self.settings.collect
        self.lookback = timedelta(hours=cfg["lookback_hours"])
        self.max_per_source = cfg["max_items_per_source"]
        self.timeout = cfg["timeout_seconds"]
        self.user_agent = cfg["user_agent"]
        self._ai_patterns = _compile_keywords(self.sources.ai_keywords)
        self._exclude = [k.lower() for k in self.sources.exclude_keywords]
        self._semaphore = asyncio.Semaphore(cfg["max_concurrency"])

    # ── フィルタ ──────────────────────────────────────────────────────

    def _is_ai_related(self, text: str) -> bool:
        return any(p.search(text) for p in self._ai_patterns)

    def _is_excluded(self, text: str) -> bool:
        lowered = text.lower()
        return any(k in lowered for k in self._exclude)

    def _keep(self, source: Source, title: str, summary: str, published: datetime) -> bool:
        if datetime.now(UTC) - published > self.lookback:
            return False
        haystack = f"{title} {summary}"
        if self._is_excluded(haystack):
            return False
        if source.needs_ai_filter and not self._is_ai_related(haystack):
            return False
        return True

    def _build(
        self,
        source: Source,
        *,
        title: str,
        url: str,
        summary: str,
        published: datetime,
        points: int = 0,
    ) -> Article:
        return Article(
            id=article_id(url),
            source_id=source.id,
            source_name=source.name,
            tier=source.tier,
            lang=source.lang,
            image_policy=source.image_policy,
            title=clean_title(title),
            url=url,
            summary=summary[:1500],
            published_at=published,
            fetched_at=datetime.now(UTC),
            points=points,
        )

    # ── 各ソース種別の取得 ────────────────────────────────────────────

    async def _fetch_rss(
        self, client: httpx.AsyncClient, source: Source, result: SourceResult
    ) -> list[Article]:
        response = await get_with_retry(client, source.url)
        feed = feedparser.parse(response.content)
        # bozo は「整形式でないが読めた」場合も立つので、エントリが取れていれば続行
        if feed.bozo and not feed.entries:
            raise ValueError(f"フィードを解釈できません: {feed.bozo_exception}")

        articles: list[Article] = []
        for entry in feed.entries[: self.max_per_source]:
            result.fetched += 1
            url = entry.get("link") or ""
            title = entry.get("title") or ""
            if not url or not title:
                continue
            published = (
                _to_utc(entry.get("published_parsed"))
                or _to_utc(entry.get("updated_parsed"))
                # 日付が無いフィードは「今取れた＝新着」とみなす
                or datetime.now(UTC)
            )
            summary = _strip_html(entry.get("summary", ""))
            if not self._keep(source, title, summary, published):
                continue
            articles.append(
                self._build(
                    source, title=title, url=url, summary=summary, published=published
                )
            )
        return articles

    async def _fetch_hn(
        self, client: httpx.AsyncClient, source: Source, result: SourceResult
    ) -> list[Article]:
        opts = source.options or {}
        min_points = opts.get("min_points", 80)
        queries = opts.get("queries", ["AI"])
        since = int((datetime.now(UTC) - self.lookback).timestamp())

        seen: dict[str, Article] = {}
        for query in queries:
            response = await get_with_retry(
                client,
                source.url,
                params={
                    "query": query,
                    "tags": "story",
                    "numericFilters": f"points>={min_points},created_at_i>{since}",
                    "hitsPerPage": self.max_per_source,
                },
            )
            for hit in response.json().get("hits", []):
                result.fetched += 1
                title = hit.get("title") or ""
                # Ask HN など外部URLが無い投稿は HN のスレッドを指す
                url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit['objectID']}"
                published = _to_utc(hit.get("created_at_i")) or datetime.now(UTC)
                if not title or not self._keep(source, title, "", published):
                    continue
                article = self._build(
                    source,
                    title=title,
                    url=url,
                    summary="",
                    published=published,
                    points=int(hit.get("points", 0)),
                )
                # 複数クエリで重複したら points が高い方を残す
                if article.id not in seen or seen[article.id].points < article.points:
                    seen[article.id] = article
        return list(seen.values())

    async def _fetch_reddit(
        self, client: httpx.AsyncClient, source: Source, result: SourceResult
    ) -> list[Article]:
        # Reddit の .rss は通常の RSS なので同じ経路で読める
        return await self._fetch_rss(client, source, result)

    async def _fetch_hf_papers(
        self, client: httpx.AsyncClient, source: Source, result: SourceResult
    ) -> list[Article]:
        response = await get_with_retry(client, source.url)
        articles: list[Article] = []
        for item in response.json()[: self.max_per_source]:
            result.fetched += 1
            paper = item.get("paper") or {}
            paper_id = paper.get("id")
            title = paper.get("title") or ""
            if not paper_id or not title:
                continue
            # 論文の初出日と日次リスト掲載日の2つがある。話題になったのは
            # 遅い方のタイミングなので、新しい方を採る。
            dates = [
                d
                for d in (
                    _parse_iso(item.get("publishedAt")),
                    _parse_iso(paper.get("publishedAt")),
                )
                if d is not None
            ]
            published = max(dates) if dates else datetime.now(UTC)
            summary = _strip_html(paper.get("summary", ""))
            if not self._keep(source, title, summary, published):
                continue
            articles.append(
                self._build(
                    source,
                    title=title,
                    url=f"https://huggingface.co/papers/{paper_id}",
                    summary=summary,
                    published=published,
                    points=int(paper.get("upvotes", 0)),
                )
            )
        return articles

    async def _fetch_html(
        self, client: httpx.AsyncClient, source: Source, result: SourceResult
    ) -> list[Article]:
        """RSS を持たない公式ブログを一覧ページから拾う（Anthropic, Mistral 等）。

        これらの記事ページは公開日のメタタグを持たないため、
        「初めて見つけた時刻＝公開日」として扱う。過去に取得済みの URL は
        DB に記録した最初の日付を再利用するので、古い記事が毎日
        新着として蘇ることはない。

        一覧の上位だけを見る（options.max_items）。下まで辿ると初回実行時に
        過去記事が一気に流れ込むため。
        """
        opts = source.options or {}
        link_contains = opts.get("link_contains", "/news/")
        limit = int(opts.get("max_items", 8))

        index = await get_with_retry(client, source.url)
        index_url = str(index.url).rstrip("/")
        links = [
            (url, text)
            for url, text in extract_links(
                index.text, str(index.url), contains=link_contains
            )
            # 一覧ページ自身やカテゴリ一覧へのリンクは記事ではない
            if url.rstrip("/") != index_url and url.rstrip("/") != source.url.rstrip("/")
        ]
        # 一覧ページの上の方ほど新しい、という前提で先頭から見る
        links = links[:limit]

        async def load(url: str, anchor_text: str) -> Article | None:
            result.fetched += 1
            try:
                page = await get_with_retry(client, url, attempts=2)
            except Exception:
                return None
            meta = parse_meta(page.text)
            title = clean_title(meta.get("og:title") or anchor_text)
            published = (
                _parse_iso(meta.get("article:published_time"))
                or _parse_iso(meta.get("article:modified_time"))
                or self.first_seen.get(article_id(url))
                or datetime.now(UTC)
            )
            summary = meta.get("og:description") or meta.get("description") or ""
            if not title or not self._keep(source, title, summary, published):
                return None
            article = self._build(
                source, title=title, url=url, summary=summary, published=published
            )
            # 一覧に載っている＝一次情報として即座に使える。OGP も取得済み。
            article.og_image_url = meta.get("og:image", "")
            return article

        gathered = await asyncio.gather(*(load(u, t) for u, t in links))
        return [a for a in gathered if a is not None]

    # ── オーケストレーション ──────────────────────────────────────────

    async def _fetch_one(
        self, client: httpx.AsyncClient, source: Source
    ) -> tuple[list[Article], SourceResult]:
        result = SourceResult(source_id=source.id)
        handlers = {
            "rss": self._fetch_rss,
            "hn": self._fetch_hn,
            "reddit": self._fetch_reddit,
            "hf_papers": self._fetch_hf_papers,
            "html": self._fetch_html,
        }
        handler = handlers.get(source.type)
        if handler is None:
            result.error = f"未知のソース種別: {source.type}"
            return [], result

        async with self._semaphore:
            try:
                articles = await handler(client, source, result)
                result.kept = len(articles)
                return articles, result
            except Exception as exc:
                # 1ソースの失敗で日次実行を止めない
                result.error = f"{type(exc).__name__}: {exc}"
                log.warning("[%s] 取得失敗: %s", source.id, result.error)
                return [], result

    async def collect_async(self) -> tuple[list[Article], list[SourceResult]]:
        async with build_client(
            timeout=self.timeout, user_agent=self.user_agent
        ) as client:
            outcomes = await asyncio.gather(
                *(self._fetch_one(client, s) for s in self.sources.enabled())
            )

        articles: dict[str, Article] = {}
        results: list[SourceResult] = []
        for items, result in outcomes:
            results.append(result)
            for article in items:
                # 同じURLが複数ソースに出たら tier の高い方を残す
                existing = articles.get(article.id)
                if existing is None or _tier_rank(article.tier) > _tier_rank(existing.tier):
                    articles[article.id] = article
        return list(articles.values()), results

    def collect(self) -> tuple[list[Article], list[SourceResult]]:
        return asyncio.run(self.collect_async())


_TIER_ORDER = {"official": 4, "major": 3, "research": 2, "niche": 2, "community": 1}


def _tier_rank(tier: str) -> int:
    return _TIER_ORDER.get(tier, 0)


def format_summary(results: list[SourceResult]) -> str:
    """実行サマリを人が読める形にする。"""
    lines = []
    ok = [r for r in results if r.ok]
    ng = [r for r in results if not r.ok]
    for r in sorted(ok, key=lambda r: -r.kept):
        lines.append(f"  {r.source_id:<22} 取得 {r.fetched:>3} → 採用 {r.kept:>3}")
    for r in ng:
        lines.append(f"  {r.source_id:<22} ✗ {r.error}")
    lines.append(f"  ── 成功 {len(ok)} / 失敗 {len(ng)} ソース")
    return "\n".join(lines)
