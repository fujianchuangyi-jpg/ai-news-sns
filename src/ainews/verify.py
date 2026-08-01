"""生成原稿のファクト照合。

ニュースアカウントは誤情報1回で信頼を失う。LLM が本文にない数字や
固有名詞を書いてしまう事故を、下書きの段階で機械的に潰す。

やること: 原稿から「検証できる要素」だけを抜き、元記事本文に
その根拠があるか照合する。無ければ VerificationIssue として報告し、
プレビュー画面に警告を出す（自動で書き換えはしない。判断は人に残す）。

誤検知を避けるため、確度の高いものだけを見る:
  - 単位付きの数値（3倍, 40%, 1200万ドル など）と2桁以上の数値
  - カタカナ4文字以上／英大文字始まり4文字以上の固有名詞
ハッシュタグ・出典表記・定型句の中の要素は対象外にする。
"""

from __future__ import annotations

import logging
import re
import unicodedata

from .models import VerificationIssue

log = logging.getLogger(__name__)

# 数値＋単位。単位が付くものは記事本文の主張そのものなので必ず検証する。
_NUMBER_WITH_UNIT = re.compile(
    r"(\d[\d,.]*)\s*"
    r"(%|％|倍|割|人|件|社|億|万|兆|円|ドル|年|月|日|時間|分|秒|"
    r"GB|TB|MB|KB|B|W|台|個|種類|カ国|か国|ヶ月|カ月|回)"
)

# 単位なしでも2桁以上なら固有の主張であることが多い
_BARE_NUMBER = re.compile(r"(?<![\d/.\-])(\d{2,}(?:[,.]\d+)*)(?![\d/%.\-])")

# カタカナ語（4文字以上）
_KATAKANA = re.compile(r"[ァ-ヴー]{4,}")

# 英語の固有名詞らしき語（大文字始まり4文字以上、または全大文字3文字以上）
_ASCII_PROPER = re.compile(r"\b([A-Z][A-Za-z0-9]{3,}|[A-Z]{3,})\b")

# 照合対象から外す領域。
# ハッシュタグは \S+ にすると "#GPT6（出典: X）" のように後続の括弧まで
# 飲み込んでしまい、出典表記の除去が効かなくなる。区切り記号で止める。
_HASHTAG = re.compile(r"#[^\s#（）()、。，．]+")
_SOURCE_NOTE = re.compile(r"[（(]\s*出典[:：][^）)]*[）)]")

# よく出るが検証不要な一般語（誤検知の主因）
_COMMON_WORDS = frozenset(
    {
        "AI", "API", "OpenAI", "ChatGPT", "Claude", "Gemini", "Google", "Meta",
        "Microsoft", "Apple", "Amazon", "NVIDIA", "LLM", "GPT", "IT", "PC",
        "アップデート", "テクノロジー", "サービス", "ユーザー", "エンジニア",
        "モデル", "データ", "システム", "ツール", "プラットフォーム",
        "リリース", "アプリケーション", "パフォーマンス", "インターネット",
        "ニュース", "コメント", "ポイント", "スタート", "チェック",
    }
)


def _normalize(text: str) -> str:
    """全角/半角とカンマを吸収して比較しやすくする。"""
    normalized = unicodedata.normalize("NFKC", text)
    return normalized.replace(",", "").lower()


def _strip_uncheckable(text: str) -> str:
    """ハッシュタグと出典表記を落とす（本文に無くて当然の部分）。"""
    text = _HASHTAG.sub(" ", text)
    return _SOURCE_NOTE.sub(" ", text)


def _number_in_source(value: str, source: str) -> bool:
    """数値が本文に現れるか。

    単純な部分一致だと "5ドル" が本文の "15ドル" にマッチしてしまい、
    捏造した数値を見逃す。前後に数字が続かないことを条件にする。
    表記ゆれ（1,200 と 1200、3.0 と 3）は吸収する。
    """
    candidates = [value.replace(",", "")]
    if "." in candidates[0]:
        trimmed = candidates[0].rstrip("0").rstrip(".")
        if trimmed:
            candidates.append(trimmed)
    return any(
        re.search(rf"(?<![\d.]){re.escape(c)}(?![\d])", source) for c in candidates
    )


def verify_text(draft_text: str, source_text: str) -> list[VerificationIssue]:
    """原稿1本を元記事本文と照合する。

    Args:
        draft_text: 生成された投稿原稿
        source_text: 元記事の本文（fulltext or summary）

    Returns:
        根拠が見つからなかった要素のリスト。空なら問題なし。
    """
    if not source_text.strip():
        # 本文が取れていない記事は照合しようがない。黙って通すと
        # 検証済みと誤解されるので、その旨を1件返す。
        return [
            VerificationIssue(
                kind="数値",
                value="-",
                note="元記事の本文を取得できていないため照合できません",
            )
        ]

    target = _strip_uncheckable(draft_text)
    haystack = _normalize(source_text)
    issues: list[VerificationIssue] = []
    reported: set[str] = set()

    def add(kind: str, value: str, note: str) -> None:
        key = f"{kind}:{value}"
        if key not in reported:
            reported.add(key)
            issues.append(VerificationIssue(kind=kind, value=value, note=note))  # type: ignore[arg-type]

    normalized_target = _normalize(target)
    checked_numbers: set[str] = set()

    # 数値（単位付き）
    for number, unit in _NUMBER_WITH_UNIT.findall(normalized_target):
        checked_numbers.add(number)
        if not _number_in_source(number, haystack):
            add("数値", f"{number}{unit}", "この数値の根拠が本文に見つかりません")

    # 数値（2桁以上、単位なし）。単位付きで報告済みの値は重複させない。
    for number in _BARE_NUMBER.findall(normalized_target):
        if number in checked_numbers:
            continue
        if not _number_in_source(number, haystack):
            add("数値", number, "この数値の根拠が本文に見つかりません")

    # 固有名詞（カタカナ）
    for word in _KATAKANA.findall(target):
        if word in _COMMON_WORDS:
            continue
        if _normalize(word) not in haystack:
            add("固有名詞", word, "この語が本文に見つかりません")

    # 固有名詞（英語）
    for word in _ASCII_PROPER.findall(target):
        if word in _COMMON_WORDS:
            continue
        if _normalize(word) not in haystack:
            add("固有名詞", word, "この語が本文に見つかりません")

    return issues


def verify_draft(
    x_bodies: dict[str, str],
    ig_items: dict[str, str],
    sources: dict[str, str],
) -> dict[str, list[VerificationIssue]]:
    """下書き全体を照合する。

    Args:
        x_bodies: article_id → X 原稿
        ig_items: article_id → IG キャプションの該当項目
        sources:  article_id → 元記事本文

    Returns:
        article_id → 問題リスト（問題があった記事だけを含む）
    """
    out: dict[str, list[VerificationIssue]] = {}
    for article_id, source_text in sources.items():
        combined = "\n".join(
            part
            for part in (x_bodies.get(article_id), ig_items.get(article_id))
            if part
        )
        if not combined:
            continue
        if issues := verify_text(combined, source_text):
            out[article_id] = issues
    return out


def format_issues(issues: dict[str, list[VerificationIssue]], titles: dict[str, str]) -> str:
    if not issues:
        return "  ファクト照合: 問題なし"
    lines = ["  ファクト照合で要確認:"]
    for article_id, found in issues.items():
        lines.append(f"    ▸ {titles.get(article_id, article_id)[:50]}")
        for issue in found:
            lines.append(f"        [{issue.kind}] {issue.value} — {issue.note}")
    return "\n".join(lines)
