"""HTTP 共通処理。

collect.py（一覧取得）と extract.py（本文・OGP取得）が共有する:
  - リトライ付き GET
  - HTML の meta / OGP 解析

リトライが要るのは、コネクション再利用中のサーバ切断
（RemoteProtocolError）が一定の頻度で起きるため。単発リクエストでは
成功するサイトでも並列プールでは落ちることがあり、日次実行では
そこで1ソース欠けるのが惜しい。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import urljoin

import httpx
from selectolax.parser import HTMLParser

log = logging.getLogger(__name__)

DEFAULT_UA = "ainews-bot/0.1 (+https://github.com/)"

# 再試行する価値がある一時的な失敗
TRANSIENT_ERRORS = (
    httpx.RemoteProtocolError,
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
    httpx.ConnectError,
    httpx.PoolTimeout,
)


def build_client(
    *, timeout: float = 20.0, user_agent: str = DEFAULT_UA
) -> httpx.AsyncClient:
    """収集・抽出で共有する AsyncClient。

    ヘッダは実測で決めている:
      - Accept を "*/*" にすると一部 CDN(Cloudflare/Fastly) が
        レスポンスを返さず切断する。ブラウザ相当の Accept を送る。
      - User-Agent はブラウザを詐称せず素直に bot 名を送る。詐称すると
        TLS フィンガープリントとの不一致で逆に弾かれるサイトがある。
    """
    return httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers={
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )


async def get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    attempts: int = 3,
    backoff: float = 0.8,
    **kwargs: Any,
) -> httpx.Response:
    """一時的なネットワーク失敗をリトライする GET。

    2回目以降は Connection: close を付けて、切断された keep-alive
    コネクションを掴み直さないようにする。
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            headers = dict(kwargs.pop("headers", {}) or {})
            if attempt > 0:
                headers["Connection"] = "close"
            response = await client.get(url, headers=headers or None, **kwargs)
            response.raise_for_status()
            return response
        except TRANSIENT_ERRORS as exc:
            last = exc
            if attempt < attempts - 1:
                await asyncio.sleep(backoff * (2**attempt))
        except httpx.HTTPStatusError as exc:
            # 4xx は再試行しても無駄（5xx だけ粘る）
            if exc.response.status_code < 500 or attempt == attempts - 1:
                raise
            last = exc
            await asyncio.sleep(backoff * (2**attempt))
    assert last is not None
    raise last


def parse_meta(html: str) -> dict[str, str]:
    """<meta> の property/name → content を辞書にする。

    og:image, article:published_time などを取り出すのに使う。
    """
    tree = HTMLParser(html)
    meta: dict[str, str] = {}
    for node in tree.css("meta"):
        key = node.attributes.get("property") or node.attributes.get("name")
        content = node.attributes.get("content")
        if key and content:
            meta.setdefault(key.lower(), content)
    return meta


def extract_og_image(html: str, base_url: str) -> str:
    """OGP 画像の絶対URLを返す。無ければ空文字。"""
    meta = parse_meta(html)
    for key in ("og:image:secure_url", "og:image", "twitter:image", "twitter:image:src"):
        if value := meta.get(key):
            return urljoin(base_url, value.strip())
    return ""


def extract_links(html: str, base_url: str, *, contains: str) -> list[tuple[str, str]]:
    """アンカーから (絶対URL, テキスト) を集める。

    RSS を持たない公式ブログ（Anthropic, Mistral 等）の一覧ページ用。
    contains にパスの一部（例 "/news/"）を渡して絞り込む。
    """
    tree = HTMLParser(html)
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for node in tree.css("a"):
        href = node.attributes.get("href") or ""
        if contains not in href:
            continue
        url = urljoin(base_url, href).split("#")[0].rstrip("/")
        if url in seen:
            continue
        text = " ".join(node.text().split())
        seen.add(url)
        out.append((url, text))
    return out
