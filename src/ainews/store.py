"""SQLite 永続化層。

ORM は使わず素の sqlite3 で書く。スキーマは小さく、リポジトリに
data/ainews.db をコミットして GitHub Actions 間で状態を引き継ぐ。

役割:
  - 収集した記事の保管（再実行時の重複取得を避ける）
  - 過去に下書き化した記事の履歴（既出チェックの土台）
  - 日次下書きの保管（プレビュー再生成・分析の入力）
  - 実投稿と実績メトリクスの保管（Phase 6）
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .config import data_dir
from .models import Article, Draft

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id            TEXT PRIMARY KEY,
    source_id     TEXT NOT NULL,
    title         TEXT NOT NULL,
    url           TEXT NOT NULL,
    published_at  TEXT NOT NULL,
    fetched_at    TEXT NOT NULL,
    cluster_id    TEXT DEFAULT '',
    cluster_size  INTEGER DEFAULT 1,
    payload       TEXT NOT NULL          -- Article 全体の JSON
);
CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at);
CREATE INDEX IF NOT EXISTS idx_articles_cluster   ON articles(cluster_id);

-- 下書きに採用された記事の履歴。既出チェックはこのテーブルを見る。
CREATE TABLE IF NOT EXISTS drafted_articles (
    article_id    TEXT NOT NULL,
    draft_date    TEXT NOT NULL,
    title         TEXT NOT NULL,
    url           TEXT NOT NULL,
    cluster_id    TEXT DEFAULT '',
    PRIMARY KEY (article_id, draft_date)
);
CREATE INDEX IF NOT EXISTS idx_drafted_date ON drafted_articles(draft_date);

-- 1日分の下書き（Draft モデルの JSON）
CREATE TABLE IF NOT EXISTS drafts (
    date          TEXT PRIMARY KEY,
    generated_at  TEXT NOT NULL,
    payload       TEXT NOT NULL
);

-- 実際に公開された投稿。analytics.py が API から取得して下書きと突き合わせる。
CREATE TABLE IF NOT EXISTS posts (
    platform      TEXT NOT NULL,         -- instagram | x
    post_id       TEXT NOT NULL,
    posted_at     TEXT NOT NULL,
    body_prefix   TEXT NOT NULL DEFAULT '',
    draft_date    TEXT DEFAULT '',       -- 突き合わせできた下書きの日付
    article_id    TEXT DEFAULT '',       -- X は投稿1件=記事1件。IG は空
    permalink     TEXT DEFAULT '',
    PRIMARY KEY (platform, post_id)
);
CREATE INDEX IF NOT EXISTS idx_posts_draft ON posts(draft_date);

-- 投稿ごとの実績。同じ投稿を複数回サンプリングするので時系列で持つ。
CREATE TABLE IF NOT EXISTS metrics (
    platform      TEXT NOT NULL,
    post_id       TEXT NOT NULL,
    collected_at  TEXT NOT NULL,
    payload       TEXT NOT NULL,         -- {impressions, likes, saves, ...}
    PRIMARY KEY (platform, post_id, collected_at)
);

-- ニュースが薄い日に使うエバーグリーンのストック
CREATE TABLE IF NOT EXISTS stock (
    id            TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    used_at       TEXT DEFAULT '',
    payload       TEXT NOT NULL
);
"""


def db_path() -> Path:
    return data_dir() / "ainews.db"


@contextmanager
def connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """コネクションを開いてスキーマを保証し、正常終了時にコミットする。"""
    conn = sqlite3.connect(path or db_path())
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── 記事 ──────────────────────────────────────────────────────────────


def upsert_articles(conn: sqlite3.Connection, articles: Iterable[Article]) -> int:
    """記事を保存。既存IDは payload ごと上書きする（本文抽出で内容が増えるため）。"""
    rows = [
        (
            a.id,
            a.source_id,
            a.title,
            a.url,
            a.published_at.isoformat(),
            a.fetched_at.isoformat(),
            a.cluster_id,
            a.cluster_size,
            a.model_dump_json(),
        )
        for a in articles
    ]
    conn.executemany(
        """
        INSERT INTO articles
            (id, source_id, title, url, published_at, fetched_at,
             cluster_id, cluster_size, payload)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            title        = excluded.title,
            fetched_at   = excluded.fetched_at,
            cluster_id   = excluded.cluster_id,
            cluster_size = excluded.cluster_size,
            payload      = excluded.payload
        """,
        rows,
    )
    return len(rows)


def known_article_ids(conn: sqlite3.Connection, ids: Iterable[str]) -> set[str]:
    """与えたIDのうち、すでに DB にあるものを返す。"""
    id_list = list(ids)
    if not id_list:
        return set()
    found: set[str] = set()
    # SQLite のプレースホルダ上限（既定 999）を避けて分割する
    for i in range(0, len(id_list), 500):
        chunk = id_list[i : i + 500]
        placeholders = ",".join("?" * len(chunk))
        cur = conn.execute(
            f"SELECT id FROM articles WHERE id IN ({placeholders})", chunk
        )
        found.update(r["id"] for r in cur)
    return found


def first_seen_dates(conn: sqlite3.Connection, days: int = 120) -> dict[str, datetime]:
    """記事ID → 最初に記録した公開日時。

    公開日メタを持たない HTML ソース（Anthropic, Mistral 等）で、
    既知の記事が毎日「新着」として蘇るのを防ぐために使う。
    """
    since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    cur = conn.execute(
        "SELECT id, published_at FROM articles WHERE published_at >= ?", (since,)
    )
    out: dict[str, datetime] = {}
    for row in cur:
        try:
            out[row["id"]] = datetime.fromisoformat(row["published_at"])
        except ValueError:
            continue
    return out


def recent_articles(conn: sqlite3.Connection, hours: int) -> list[Article]:
    """直近 N 時間に公開された記事を新しい順で返す。"""
    since = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
    cur = conn.execute(
        "SELECT payload FROM articles WHERE published_at >= ? ORDER BY published_at DESC",
        (since,),
    )
    return [Article.model_validate_json(r["payload"]) for r in cur]


# ── 既出チェック ───────────────────────────────────────────────────────


def record_drafted(conn: sqlite3.Connection, draft_date: str, articles: Iterable[Article]) -> None:
    conn.executemany(
        """
        INSERT INTO drafted_articles (article_id, draft_date, title, url, cluster_id)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(article_id, draft_date) DO NOTHING
        """,
        [(a.id, draft_date, a.title, a.url, a.cluster_id) for a in articles],
    )


def drafted_history(conn: sqlite3.Connection, days: int) -> list[dict[str, Any]]:
    """過去 N 日に下書き化した記事（既出判定に使う）。"""
    since = (datetime.now(UTC) - timedelta(days=days)).date().isoformat()
    cur = conn.execute(
        "SELECT article_id, title, url, cluster_id, draft_date "
        "FROM drafted_articles WHERE draft_date >= ?",
        (since,),
    )
    return [dict(r) for r in cur]


# ── 下書き ────────────────────────────────────────────────────────────


def save_draft(conn: sqlite3.Connection, draft: Draft) -> None:
    conn.execute(
        """
        INSERT INTO drafts (date, generated_at, payload) VALUES (?, ?, ?)
        ON CONFLICT(date) DO UPDATE SET
            generated_at = excluded.generated_at,
            payload      = excluded.payload
        """,
        (draft.date, draft.generated_at.isoformat(), draft.model_dump_json()),
    )


def load_draft(conn: sqlite3.Connection, date: str) -> Draft | None:
    cur = conn.execute("SELECT payload FROM drafts WHERE date = ?", (date,))
    row = cur.fetchone()
    return Draft.model_validate_json(row["payload"]) if row else None


def load_drafts_since(conn: sqlite3.Connection, days: int) -> list[Draft]:
    since = (datetime.now(UTC) - timedelta(days=days)).date().isoformat()
    cur = conn.execute(
        "SELECT payload FROM drafts WHERE date >= ? ORDER BY date DESC", (since,)
    )
    return [Draft.model_validate_json(r["payload"]) for r in cur]


def latest_draft_date(conn: sqlite3.Connection) -> str | None:
    cur = conn.execute("SELECT date FROM drafts ORDER BY date DESC LIMIT 1")
    row = cur.fetchone()
    return row["date"] if row else None


# ── 実投稿と実績（Phase 6） ────────────────────────────────────────────


def upsert_post(
    conn: sqlite3.Connection,
    *,
    platform: str,
    post_id: str,
    posted_at: str,
    body_prefix: str = "",
    draft_date: str = "",
    article_id: str = "",
    permalink: str = "",
) -> None:
    conn.execute(
        """
        INSERT INTO posts
            (platform, post_id, posted_at, body_prefix, draft_date, article_id, permalink)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(platform, post_id) DO UPDATE SET
            body_prefix = excluded.body_prefix,
            -- 突き合わせ結果は一度決まったら消さない（空で上書きしない）
            draft_date  = CASE WHEN excluded.draft_date != '' THEN excluded.draft_date
                               ELSE posts.draft_date END,
            article_id  = CASE WHEN excluded.article_id != '' THEN excluded.article_id
                               ELSE posts.article_id END,
            permalink   = excluded.permalink
        """,
        (platform, post_id, posted_at, body_prefix, draft_date, article_id, permalink),
    )


def unlinked_posts(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """まだ下書きと突き合わせできていない投稿。"""
    cur = conn.execute("SELECT * FROM posts WHERE draft_date = ''")
    return [dict(r) for r in cur]


def record_metrics(
    conn: sqlite3.Connection,
    *,
    platform: str,
    post_id: str,
    collected_at: str,
    payload: dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO metrics (platform, post_id, collected_at, payload)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(platform, post_id, collected_at) DO UPDATE SET
            payload = excluded.payload
        """,
        (platform, post_id, collected_at, json.dumps(payload, ensure_ascii=False)),
    )


def latest_metrics(conn: sqlite3.Connection, days: int) -> list[dict[str, Any]]:
    """投稿ごとの最新メトリクスを、下書き情報と結合して返す。"""
    since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    cur = conn.execute(
        """
        SELECT m.platform, m.post_id, m.collected_at, m.payload,
               p.draft_date, p.article_id, p.posted_at, p.permalink
        FROM metrics m
        JOIN posts p ON p.platform = m.platform AND p.post_id = m.post_id
        JOIN (
            SELECT platform, post_id, MAX(collected_at) AS latest
            FROM metrics GROUP BY platform, post_id
        ) t ON t.platform = m.platform
           AND t.post_id  = m.post_id
           AND t.latest   = m.collected_at
        WHERE p.posted_at >= ?
        """,
        (since,),
    )
    out = []
    for r in cur:
        row = dict(r)
        row["payload"] = json.loads(row["payload"])
        out.append(row)
    return out
