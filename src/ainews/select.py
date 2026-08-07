"""② スコアリングと選定。

「有名2本 : ニッチ2本」を成立させるのがこのモジュールの仕事。

なぜ機械シグナルと LLM を混ぜるか:
  LLM 単体に「有名か」を聞くと、その日の記事の並びに引きずられて基準が
  揺れる。何社が報じたか（cluster_size）と媒体の格（tier）は揺れない
  客観指標なので、これを混ぜて日ごとのブレを抑える。

  逆に「面白いか」は機械では測れないので LLM に任せる。
"""

from __future__ import annotations

import logging
import math
from collections import Counter

from .config import load_settings, prompt
from .llm import LLM, LLMError, json_dump
from .models import (
    Article,
    Assessment,
    AssessmentBatch,
    ScoredArticle,
)

log = logging.getLogger(__name__)

# 媒体の格。一次情報と大手ほど「広く知られる」方に寄与する。
TIER_SCORE = {
    "official": 100.0,
    "major": 85.0,
    "niche": 45.0,
    "research": 30.0,
    "community": 35.0,
}

# 1回の呼び出しで評価する記事数の上限。
#
# 以前は12件ずつ刻んでいたが、Claude Code は1回あたり約25,000トークンの
# 固定オーバーヘッド（Claude Code 自身のシステムプロンプト）を伴うため、
# 刻むほど無駄が増える。一次選抜で20件程度まで絞ったうえで1回に
# まとめて渡すのが最も効率がよい。
#
# 上限を残してあるのは、一次選抜が無効な場合や候補が異常に多い日の保険。
ASSESS_CHUNK = 24


def signal_score(article: Article) -> float:
    """機械的に測れる「広く報じられている度合い」を 0-100 で返す。

    内訳:
      50点 何社が報じたか（対数。1社→0点, 3社→約32点, 8社→50点）
      30点 媒体の格
      20点 コミュニティでの反応（HN/Reddit のスコア。対数）
    """
    coverage = min(1.0, math.log(article.cluster_size + 1, 2) / 3.0) * 50.0
    tier = TIER_SCORE.get(article.tier, 40.0) / 100.0 * 30.0
    buzz = min(1.0, math.log(article.points + 1, 10) / 3.0) * 20.0
    return coverage + tier + buzz


class Selector:
    def __init__(self, llm: LLM | None = None) -> None:
        self.settings = load_settings()
        self.cfg = self.settings.select
        self.llm = llm or LLM()
        self._system = prompt("assess")

    # ── LLM 評価 ──────────────────────────────────────────────────────

    def _assess_chunk(self, articles: list[Article]) -> list[Assessment]:
        payload = [
            {
                "id": a.id,
                "title": a.title,
                "source": a.source_name,
                "published": a.published_at.strftime("%Y-%m-%d %H:%M UTC"),
                "reported_by_outlets": a.cluster_size,
                "body": a.text_for_llm,
            }
            for a in articles
        ]
        user = (
            f"以下の {len(payload)} 件の記事を評価してください。\n\n"
            f"{json_dump(payload)}"
        )
        batch = self.llm.structured(
            system=self._system,
            user=user,
            schema=AssessmentBatch,
            effort=self.settings.llm["select_effort"],
            # スキーマを強制できないバックエンド（Claude Code）と、
            # 件数を落としがちなバックエンド（Ollama）の両方に効く
            count_hint=len(payload),
        )
        return batch.assessments

    def assess(self, articles: list[Article]) -> dict[str, Assessment]:
        """記事を評価して article_id → Assessment の辞書にする。

        チャンクが1つ失敗しても、残りの評価だけで選定は続行できる。
        """
        results: dict[str, Assessment] = {}
        chunks = [
            articles[i : i + ASSESS_CHUNK]
            for i in range(0, len(articles), ASSESS_CHUNK)
        ]
        for index, chunk in enumerate(chunks, start=1):
            try:
                for assessment in self._assess_chunk(chunk):
                    results[assessment.article_id] = assessment
                log.info("評価 %d/%d チャンク完了", index, len(chunks))
            except LLMError as exc:
                log.warning("評価チャンク %d が失敗（スキップ）: %s", index, exc)
        return results

    # ── 選定 ──────────────────────────────────────────────────────────

    def score(
        self, articles: list[Article], assessments: dict[str, Assessment]
    ) -> list[ScoredArticle]:
        """評価とシグナルを合成して fame_final を出し、バケットに振り分ける。"""
        w_llm = self.cfg["llm_weight"]
        w_sig = self.cfg["signal_weight"]
        threshold = self.cfg["fame_threshold"]
        blocking = set(self.cfg["blocking_risk_flags"])

        scored: list[ScoredArticle] = []
        for article in articles:
            assessment = assessments.get(article.id)
            if assessment is None:
                continue
            if blocking & set(assessment.risk_flags):
                log.info(
                    "リスクフラグで除外 [%s] %s",
                    ",".join(assessment.risk_flags),
                    article.title[:50],
                )
                continue
            signal = signal_score(article)
            fame_final = w_llm * assessment.fame + w_sig * signal
            scored.append(
                ScoredArticle(
                    article=article,
                    assessment=assessment,
                    signal_score=round(signal, 1),
                    fame_final=round(fame_final, 1),
                    bucket="famous" if fame_final >= threshold else "niche",
                )
            )
        return scored

    def pick(self, scored: list[ScoredArticle]) -> list[ScoredArticle]:
        """有名バケットとニッチバケットから設定比率で選ぶ。

        同じカテゴリで埋まらないよう max_per_category で頭打ちにする。
        片方のバケットが足りない場合は、もう片方から補って総数を満たす。
        """
        want_famous = self.cfg["famous"]
        want_niche = self.cfg["niche"]
        total = self.cfg["total"]
        max_per_category = self.cfg["max_per_category"]

        famous = sorted(
            (s for s in scored if s.bucket == "famous"),
            key=lambda s: s.assessment.interest,
            reverse=True,
        )
        niche = sorted(
            (s for s in scored if s.bucket == "niche"),
            key=lambda s: s.assessment.interest,
            reverse=True,
        )

        picked: list[ScoredArticle] = []
        categories: Counter[str] = Counter()

        def take(pool: list[ScoredArticle], limit: int) -> None:
            for candidate in pool:
                if limit <= 0:
                    return
                category = candidate.assessment.category
                if categories[category] >= max_per_category:
                    continue
                picked.append(candidate)
                categories[category] += 1
                limit -= 1

        take(famous, want_famous)
        take(niche, want_niche)

        # 片方が枯れた日は、残りをもう片方から interest 順に埋める
        if len(picked) < total:
            chosen = {s.article.id for s in picked}
            rest = sorted(
                (s for s in scored if s.article.id not in chosen),
                key=lambda s: s.assessment.interest,
                reverse=True,
            )
            take(rest, total - len(picked))

        # 有名 → ニッチ の順に並べる（投稿の掴みを強くする）
        picked.sort(key=lambda s: (s.bucket != "famous", -s.assessment.interest))
        return picked[:total]

    def run(self, articles: list[Article]) -> tuple[list[ScoredArticle], list[ScoredArticle]]:
        """候補の評価から選定までを通す。

        Returns:
            (選ばれた記事, 全スコア付き候補)  ※後者は検証・デバッグ用
        """
        candidates = sorted(articles, key=signal_score, reverse=True)[
            : self.cfg["max_candidates"]
        ]
        log.info("候補 %d 件を LLM 評価にかけます", len(candidates))
        assessments = self.assess(candidates)
        scored = self.score(candidates, assessments)
        return self.pick(scored), scored


def format_scores(scored: list[ScoredArticle], picked: list[ScoredArticle]) -> str:
    """スコアの内訳を人が読める表にする（--explain 用）。"""
    chosen = {s.article.id for s in picked}
    rows = sorted(scored, key=lambda s: s.fame_final, reverse=True)
    lines = [
        f"  {'':2} {'fame':>5} {'興味':>4} {'信号':>5} {'束':>3}  "
        f"{'バケット':<8} {'カテゴリ':<8} 見出し",
    ]
    for s in rows:
        mark = "★" if s.article.id in chosen else " "
        lines.append(
            f"  {mark:2} {s.fame_final:>5.1f} {s.assessment.interest:>4} "
            f"{s.signal_score:>5.1f} {s.article.cluster_size:>3}  "
            f"{s.bucket:<8} {s.assessment.category:<8} {s.display_title[:44]}"
        )
    return "\n".join(lines)
