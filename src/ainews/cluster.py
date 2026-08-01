"""同一ニュースの束ね（クラスタリング）と既出チェック。

なぜ必要か:
  1. 大きなニュースは10社が同時に報じる。そのまま選定に流すと4枠が
     同じ話題で埋まる。束ねて代表1本にする。
  2. 束ねたサイズ（何社が報じたか）は「有名度」の最も素直な機械シグナル
     になるので、select.py がこれを使う。
  3. 昨日投稿した話題を今日も出さないよう、過去の下書き履歴と照合する。

手法は軽量な語彙ベースの類似度。埋め込みAPIを使わないのは、記事タイトルの
一致判定にはこれで十分実用的で、コストとレイテンシがゼロで済むため。
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

from .config import load_settings
from .models import Article

log = logging.getLogger(__name__)

# タイトル比較の前に落とすノイズ（媒体名の接尾辞や煽り表現）。
# 区切り記号は前後に空白を要求する。要求しないと "GPT-6を発表…" の
# ハイフンを媒体名の区切りと誤認して見出し本体を丸ごと削ってしまう。
_NOISE_PATTERNS = [
    re.compile(r"\s+[|｜]\s*[^|｜]{1,25}$"),  # 末尾の「 | ITmedia」など
    re.compile(r"\s+[-–—]\s+\S[^-–—]{0,24}$"),  # 末尾の「 - The Verge」など
    re.compile(r"^\s*【[^】]{1,12}】"),  # 先頭の【速報】など
    re.compile(r"\[[^\]]{1,12}\]\s*$"),
]


def normalize_title(title: str) -> str:
    """全角半角・記号・媒体名を落として比較用の文字列にする。"""
    text = unicodedata.normalize("NFKC", title).strip()
    for pattern in _NOISE_PATTERNS:
        text = pattern.sub("", text)
    text = text.lower()
    # "gpt-6" → "gpt6"。製品名のバージョン番号は同一ニュース判定で
    # 最も効く手がかりなので、ハイフンで分断させない。
    text = re.sub(r"(?<=[a-z])[-_.](?=[0-9])", "", text)
    text = re.sub(r"[^\w\s぀-ヿ一-鿿]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# 情報量が乏しく、どの見出しにも出るため比較の役に立たない語
_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "from", "into", "this", "that", "its",
        "new", "now", "how", "why", "what", "you", "your", "are", "was",
        "will", "can", "has", "have", "not", "but", "out", "all", "more",
        "says", "said", "about", "over", "after", "than", "when",
    }
)


def title_tokens(title: str) -> set[str]:
    """比較用トークン集合。

    英語は単語、日本語は文字bigramにする。日本語は分かち書きが無いと
    単語分割できないが、bigram なら形態素解析器なしで十分な精度が出る。
    """
    text = normalize_title(title)
    tokens: set[str] = set()

    for word in re.findall(r"[a-z0-9]+", text):
        # 1文字は雑音。純粋な数字は年号などで誤マッチしやすいので落とす。
        if len(word) < 2 or word.isdigit() or word in _STOPWORDS:
            continue
        tokens.add(word)

    # CJK は連続部分ごとに文字bigram
    for run in re.findall(r"[぀-ヿ㐀-䶿一-鿿]+", text):
        if len(run) == 1:
            tokens.add(run)
        else:
            tokens.update(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def similarity(a: set[str], b: set[str]) -> float:
    """同一ニュース判定用の類似度（重なり係数）。

    Jaccard ではなく overlap coefficient を使う。同じ話題でも媒体によって
    見出しの長さが倍近く違うことがあり、Jaccard は和集合が膨らむぶん
    長さの差だけでスコアが落ちてしまう。重なり係数は短い方を基準に
    見るのでその影響を受けない。
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


_TIER_ORDER = {"official": 4, "major": 3, "research": 2, "niche": 2, "community": 1}


def _representative_rank(article: Article) -> tuple[int, int, float]:
    """クラスタ代表を選ぶ優先順位。

    一次情報 > 大手 の順。同点なら本文が長い方（＝情報量が多い方）。
    """
    return (
        _TIER_ORDER.get(article.tier, 0),
        article.points,
        len(article.fulltext or article.summary),
    )


def assign_clusters(articles: list[Article]) -> list[Article]:
    """類似タイトルをまとめ、cluster_id と cluster_size を埋める。

    単純な逐次クラスタリング（各記事を既存クラスタの代表と比較）。
    1日100〜300件の規模なら O(n*k) で十分速い。
    """
    threshold = load_settings().dedup["title_similarity_threshold"]

    # 新しい順に見て、先に来た記事をクラスタの起点にする
    ordered = sorted(articles, key=lambda a: a.published_at, reverse=True)
    clusters: list[dict[str, Any]] = []

    for article in ordered:
        tokens = title_tokens(article.title)
        for cluster in clusters:
            if similarity(tokens, cluster["tokens"]) >= threshold:
                cluster["members"].append(article)
                # クラスタの語彙を広げすぎると無関係な記事まで吸うので広げない
                break
        else:
            clusters.append(
                {"id": article.id, "tokens": tokens, "members": [article]}
            )

    for cluster in clusters:
        members: list[Article] = cluster["members"]
        size = len(members)
        for member in members:
            member.cluster_id = cluster["id"]
            member.cluster_size = size

    log.info("%d 記事 → %d クラスタ", len(articles), len(clusters))
    return ordered


def representatives(articles: list[Article]) -> list[Article]:
    """各クラスタから代表1本だけを残す。"""
    best: dict[str, Article] = {}
    for article in articles:
        key = article.cluster_id or article.id
        current = best.get(key)
        if current is None or _representative_rank(article) > _representative_rank(current):
            best[key] = article
    return sorted(best.values(), key=lambda a: a.published_at, reverse=True)


def filter_already_drafted(
    articles: list[Article], history: list[dict[str, Any]]
) -> tuple[list[Article], list[Article]]:
    """過去に下書き化した記事と重複するものを除く。

    URL 完全一致だけでなくタイトル類似も見る。同じ話題を別媒体で
    拾い直したケースを弾くため。

    Returns:
        (残った記事, 除外された記事)
    """
    if not history:
        return articles, []

    threshold = load_settings().dedup["title_similarity_threshold"]
    seen_urls = {h["url"] for h in history}
    seen_ids = {h["article_id"] for h in history}
    seen_tokens = [title_tokens(h["title"]) for h in history]

    kept: list[Article] = []
    dropped: list[Article] = []
    for article in articles:
        if article.id in seen_ids or article.url in seen_urls:
            dropped.append(article)
            continue
        tokens = title_tokens(article.title)
        if any(similarity(tokens, prev) >= threshold for prev in seen_tokens):
            dropped.append(article)
            continue
        kept.append(article)

    if dropped:
        log.info("既出として %d 件を除外", len(dropped))
    return kept, dropped


def cluster_summary(articles: list[Article], top: int = 5) -> str:
    """大きいクラスタ（＝多く報じられた話題）を表示する。"""
    sizes: dict[str, list[Article]] = {}
    for a in articles:
        sizes.setdefault(a.cluster_id or a.id, []).append(a)
    big = sorted(sizes.values(), key=len, reverse=True)[:top]
    lines = []
    for members in big:
        if len(members) < 2:
            continue
        head = max(members, key=_representative_rank)
        names = ", ".join(sorted({m.source_name for m in members}))
        lines.append(f"  ×{len(members)} {head.title[:50]}  ({names})")
    return "\n".join(lines) if lines else "  （複数社が報じた話題はなし）"
