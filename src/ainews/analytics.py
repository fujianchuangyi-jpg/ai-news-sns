"""⑥ 実績収集とトピック分析。

重要な点: **手動投稿でも完全に自動化できる**。
Instagram も X も「自分の投稿一覧」を API で取得できるので、
投稿ボタンを人が押していても、翌日には実績を機械的に回収できる。

投稿と下書きの紐付けは本文の先頭一致で行う。人が微修正して投稿しても
先頭数十文字はたいてい残るため、実運用で十分機能する。

X API はクレジット購入が必要なので、代替として analytics.x.com から
書き出した CSV の取り込みにも対応する。
"""

from __future__ import annotations

import csv
import logging
import os
import re
import sqlite3
import unicodedata
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import httpx

from . import store
from .config import load_settings
from .llm import LLM
from .models import Draft

log = logging.getLogger(__name__)

GRAPH_API = "https://graph.facebook.com/v21.0"
X_API = "https://api.x.com/2"

# 突き合わせに使う先頭文字数。長すぎると微修正で外れ、短すぎると誤マッチする。
MATCH_PREFIX = 24

IG_METRICS = "reach,likes,comments,saved,shares,total_interactions"


def _normalize(text: str) -> str:
    """比較用に正規化する。空白・記号・絵文字の差を吸収する。"""
    normalized = unicodedata.normalize("NFKC", text)
    return re.sub(r"[\s\W_]+", "", normalized).lower()


def _prefix(text: str) -> str:
    return _normalize(text)[:MATCH_PREFIX]


# ── 取得 ──────────────────────────────────────────────────────────────


def fetch_instagram_posts() -> list[dict[str, Any]]:
    """自分の Instagram 投稿とインサイトを取得する。

    必要な環境変数: IG_USER_ID, META_ACCESS_TOKEN
    権限は読み取りのみで足りる（投稿権限は不要）。
    """
    user_id = os.environ.get("IG_USER_ID", "").strip()
    token = os.environ.get("META_ACCESS_TOKEN", "").strip()
    if not (user_id and token):
        log.info("Instagram の認証情報が未設定のためスキップします")
        return []

    days = load_settings().analytics["lookback_days"]
    out: list[dict[str, Any]] = []
    try:
        media = httpx.get(
            f"{GRAPH_API}/{user_id}/media",
            params={
                "fields": "id,caption,timestamp,permalink,media_type",
                "limit": 50,
                "access_token": token,
            },
            timeout=30,
        )
        media.raise_for_status()
        for item in media.json().get("data", []):
            posted_at = item.get("timestamp", "")
            if posted_at and _age_days(posted_at) > days:
                continue
            metrics = _fetch_ig_insights(item["id"], token)
            out.append(
                {
                    "platform": "instagram",
                    "post_id": item["id"],
                    "posted_at": posted_at,
                    "body": item.get("caption", ""),
                    "permalink": item.get("permalink", ""),
                    "metrics": metrics,
                }
            )
    except Exception as exc:
        log.error("Instagram の取得に失敗しました: %s", exc)
    return out


def _fetch_ig_insights(media_id: str, token: str) -> dict[str, int]:
    try:
        response = httpx.get(
            f"{GRAPH_API}/{media_id}/insights",
            params={"metric": IG_METRICS, "access_token": token},
            timeout=30,
        )
        response.raise_for_status()
        return {
            row["name"]: row["values"][0]["value"]
            for row in response.json().get("data", [])
            if row.get("values")
        }
    except Exception as exc:
        log.debug("インサイト取得に失敗 (%s): %s", media_id, exc)
        return {}


def fetch_x_posts() -> list[dict[str, Any]]:
    """自分の X 投稿と公開指標を取得する。

    必要な環境変数: X_BEARER_TOKEN, X_USER_ID
    自分の投稿の読み取りは $0.001/件。月120件で約 $0.12。
    """
    token = os.environ.get("X_BEARER_TOKEN", "").strip()
    user_id = os.environ.get("X_USER_ID", "").strip()
    if not token:
        log.info("X の認証情報が未設定のためスキップします")
        return []

    headers = {"Authorization": f"Bearer {token}"}
    out: list[dict[str, Any]] = []
    try:
        if not user_id:
            me = httpx.get(f"{X_API}/users/me", headers=headers, timeout=30)
            me.raise_for_status()
            user_id = me.json()["data"]["id"]

        response = httpx.get(
            f"{X_API}/users/{user_id}/tweets",
            headers=headers,
            params={
                "max_results": 100,
                "tweet.fields": "public_metrics,created_at,text",
                "exclude": "retweets,replies",
            },
            timeout=30,
        )
        response.raise_for_status()
        for tweet in response.json().get("data", []):
            metrics = tweet.get("public_metrics", {})
            out.append(
                {
                    "platform": "x",
                    "post_id": tweet["id"],
                    "posted_at": tweet.get("created_at", ""),
                    "body": tweet.get("text", ""),
                    "permalink": f"https://x.com/i/status/{tweet['id']}",
                    "metrics": metrics,
                }
            )
    except Exception as exc:
        log.error("X の取得に失敗しました: %s", exc)
    return out


def load_x_csv(path: Path) -> list[dict[str, Any]]:
    """analytics.x.com が書き出す CSV を取り込む。

    API のクレジットを使わずに実績を回収するための代替経路。
    列名は書き出し時期で変わるため、それらしい列を柔軟に拾う。
    """
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            lowered = {k.strip().lower(): v for k, v in row.items() if k}

            def pick(*names: str) -> str:
                for name in names:
                    if value := lowered.get(name):
                        return value
                return ""

            post_id = pick("tweet id", "post id", "id")
            body = pick("tweet text", "post text", "text")
            if not post_id or not body:
                continue
            out.append(
                {
                    "platform": "x",
                    "post_id": post_id,
                    "posted_at": pick("time", "date", "created at"),
                    "body": body,
                    "permalink": pick("permalink", "tweet permalink"),
                    "metrics": {
                        "impression_count": _to_int(pick("impressions", "impression")),
                        "like_count": _to_int(pick("likes", "like")),
                        "reply_count": _to_int(pick("replies", "reply")),
                        "retweet_count": _to_int(pick("retweets", "retweet", "reposts")),
                        "bookmark_count": _to_int(pick("bookmarks", "bookmark")),
                    },
                }
            )
    return out


def _to_int(value: str) -> int:
    try:
        return int(float(str(value).replace(",", "") or 0))
    except ValueError:
        return 0


def _age_days(iso: str) -> float:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    return (datetime.now(UTC) - dt).total_seconds() / 86400


# ── 下書きとの突き合わせ ───────────────────────────────────────────────


def build_match_index(drafts: list[Draft]) -> dict[str, tuple[str, str]]:
    """本文先頭 → (下書き日, article_id) の索引を作る。

    X は投稿1件=記事1件なので article_id が入る。
    Instagram はカルーセル1件=記事4件なので article_id は空にする。
    """
    index: dict[str, tuple[str, str]] = {}
    for draft in drafts:
        for post in draft.x_posts:
            index[_prefix(post.body)] = (draft.date, post.article_id)
        index[_prefix(draft.ig_caption.render())] = (draft.date, "")
    return index


def match_post(body: str, index: dict[str, tuple[str, str]]) -> tuple[str, str]:
    """投稿本文から下書きを引き当てる。見つからなければ ('', '')。"""
    prefix = _prefix(body)
    if hit := index.get(prefix):
        return hit
    # 人が冒頭を微修正した場合に備え、前方一致でも探す
    for key, value in index.items():
        if key and (prefix.startswith(key[:16]) or key.startswith(prefix[:16])):
            return value
    return ("", "")


def collect_metrics(conn: sqlite3.Connection, *, csv_path: str | None = None) -> str:
    """API（と任意の CSV）から実績を集め、下書きと紐付けて保存する。"""
    drafts = store.load_drafts_since(conn, days=load_settings().analytics["lookback_days"])
    index = build_match_index(drafts)

    posts: list[dict[str, Any]] = []
    posts += fetch_instagram_posts()
    posts += fetch_x_posts()
    if csv_path:
        posts += load_x_csv(Path(csv_path))

    now = datetime.now(UTC).isoformat()
    matched = 0
    for post in posts:
        draft_date, article_id = match_post(post["body"], index)
        matched += bool(draft_date)
        store.upsert_post(
            conn,
            platform=post["platform"],
            post_id=post["post_id"],
            posted_at=post["posted_at"] or now,
            body_prefix=post["body"][:120],
            draft_date=draft_date,
            article_id=article_id,
            permalink=post.get("permalink", ""),
        )
        if post["metrics"]:
            store.record_metrics(
                conn,
                platform=post["platform"],
                post_id=post["post_id"],
                collected_at=now,
                payload=post["metrics"],
            )

    if not posts:
        return (
            "── 実績収集\n"
            "  取得できた投稿がありません。\n"
            "  IG_USER_ID / META_ACCESS_TOKEN、X_BEARER_TOKEN のいずれかを設定するか、\n"
            "  --import-csv で analytics.x.com の書き出しを渡してください。"
        )
    return (
        f"── 実績収集\n  投稿 {len(posts)} 件を取得、"
        f"うち {matched} 件を下書きと紐付けました"
    )


# ── 分析 ──────────────────────────────────────────────────────────────

# 指標名は媒体で異なるので、代表値として使う優先順を決めておく
REACH_KEYS = ("reach", "impression_count", "impressions")
ENGAGEMENT_KEYS = (
    "total_interactions", "like_count", "likes", "saved", "bookmark_count"
)


def _reach(payload: dict[str, Any]) -> int:
    for key in REACH_KEYS:
        if key in payload:
            return int(payload[key] or 0)
    return 0


def _engagement(payload: dict[str, Any]) -> int:
    return sum(int(payload.get(key, 0) or 0) for key in ENGAGEMENT_KEYS)


def analyze(conn: sqlite3.Connection) -> dict[str, Any]:
    """下書きのメタデータと実績を結合して集計する。

    下書き段階でカテゴリ・バケット・フック型を記録してあるので、
    後から任意の軸で切れる。
    """
    days = load_settings().analytics["lookback_days"]
    rows = store.latest_metrics(conn, days)
    drafts = {d.date: d for d in store.load_drafts_since(conn, days)}

    by_category: dict[str, list[int]] = defaultdict(list)
    by_bucket: dict[str, list[int]] = defaultdict(list)
    by_hook: dict[str, list[int]] = defaultdict(list)
    by_source: dict[str, list[int]] = defaultdict(list)
    items: list[dict[str, Any]] = []

    for row in rows:
        draft = drafts.get(row["draft_date"])
        if draft is None:
            continue
        reach = _reach(row["payload"])
        engagement = _engagement(row["payload"])
        # リーチが取れない媒体もあるのでエンゲージメントを主指標にする
        score = engagement if engagement else reach

        scored = draft.scored_by_id(row["article_id"]) if row["article_id"] else None
        if scored is not None:
            by_category[scored.assessment.category].append(score)
            by_bucket[scored.bucket].append(score)
            by_source[scored.article.source_name].append(score)
            hook = next(
                (p.hook_type for p in draft.x_posts if p.article_id == row["article_id"]),
                "",
            )
            if hook:
                by_hook[hook].append(score)
            items.append(
                {
                    "title": scored.article.title,
                    "category": scored.assessment.category,
                    "bucket": scored.bucket,
                    "score": score,
                    "reach": reach,
                    "platform": row["platform"],
                    "permalink": row["permalink"],
                }
            )

    def averages(data: dict[str, list[int]]) -> list[tuple[str, float, int]]:
        return sorted(
            ((key, mean(values), len(values)) for key, values in data.items() if values),
            key=lambda t: t[1],
            reverse=True,
        )

    return {
        "sample_size": len(items),
        "by_category": averages(by_category),
        "by_bucket": averages(by_bucket),
        "by_hook": averages(by_hook),
        "by_source": averages(by_source),
        "top": sorted(items, key=lambda i: i["score"], reverse=True)[:5],
        "bottom": sorted(items, key=lambda i: i["score"])[:3],
    }


def weekly_report(conn: sqlite3.Connection, *, llm: LLM | None = None) -> str:
    """週次レポートを Markdown で作る。"""
    result = analyze(conn)
    if result["sample_size"] == 0:
        return (
            "── 週次レポート\n"
            "  分析できる実績がまだありません。\n"
            "  投稿してから analytics を実行し、実績を蓄積してください。"
        )

    lines = [
        f"# 週次レポート（対象 {result['sample_size']} 投稿）",
        "",
        "## 有名 vs ニッチ",
    ]
    for bucket, avg, count in result["by_bucket"]:
        label = "有名ニュース" if bucket == "famous" else "ニッチニュース"
        lines.append(f"- {label}: 平均 {avg:.0f}（{count}件）")

    lines += ["", "## カテゴリ別"]
    for name, avg, count in result["by_category"]:
        lines.append(f"- {name}: 平均 {avg:.0f}（{count}件）")

    if result["by_hook"]:
        lines += ["", "## フック型別"]
        for name, avg, count in result["by_hook"]:
            lines.append(f"- {name}: 平均 {avg:.0f}（{count}件）")

    lines += ["", "## 伸びた投稿"]
    for item in result["top"]:
        lines.append(f"- [{item['bucket']}] {item['title'][:50]} — {item['score']}")

    lines += ["", "## 伸びなかった投稿"]
    for item in result["bottom"]:
        lines.append(f"- [{item['bucket']}] {item['title'][:50]} — {item['score']}")

    report = "\n".join(lines)

    if llm is not None:
        try:
            summary = llm.text(
                system=(
                    "あなたはSNS運用のアナリストです。渡された実績データから、"
                    "次週の投稿方針として実行できる示唆を3点だけ、日本語の箇条書きで出してください。"
                    "データから読み取れないことは書かないでください。"
                ),
                user=report,
                effort="medium",
            )
            report += f"\n\n## 次週への示唆\n{summary}"
        except Exception as exc:
            log.warning("レポートの要約に失敗しました: %s", exc)

    return report
