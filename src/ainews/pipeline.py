"""日次パイプラインのオーケストレーション。

収集 → クラスタリング → 既出除外 → 本文抽出 → 選定 → 原稿生成 → 照合
までを1本に繋ぐ。画像生成とプレビュー出力は別コマンドに分けてある
（原稿だけ作り直したい、画像だけ作り直したい、が頻繁に起きるため）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from . import store
from .cluster import (
    assign_clusters,
    cluster_summary,
    filter_already_drafted,
    representatives,
)
from .collect import Collector, SourceResult, format_summary
from .compose import Composer
from .config import load_settings
from .extract import Extractor, enrichment_summary
from .llm import LLM, make_llm
from .prefilter import Prefilter, PrefilterResult
from .models import Article, Draft, ScoredArticle
from .select import Selector, format_scores, signal_score
from .verify import audit_draft, format_audit, format_issues, verify_draft

log = logging.getLogger(__name__)

JST = ZoneInfo("Asia/Tokyo")


def today_jst() -> str:
    """投稿日は日本時間で決める（読者が日本時間で生活しているため）。"""
    return datetime.now(JST).date().isoformat()


@dataclass
class PipelineReport:
    """実行内容の記録。CLI の表示と Actions のログに使う。"""

    date: str
    source_results: list[SourceResult] = field(default_factory=list)
    collected: int = 0
    clusters: int = 0
    after_dedup: int = 0
    dropped_as_seen: int = 0
    candidates: int = 0
    draft: Draft | None = None
    scored: list[ScoredArticle] = field(default_factory=list)
    articles: list[Article] = field(default_factory=list)
    prefilter: PrefilterResult | None = None

    def render(self, *, explain: bool = False) -> str:
        lines = [
            f"── 収集（{self.date}）",
            format_summary(self.source_results),
            "",
            f"  記事 {self.collected} 件 → {self.clusters} クラスタ",
            f"  既出除外 {self.dropped_as_seen} 件 → 候補 {self.after_dedup} 件",
            "",
        ]
        if self.prefilter is not None:
            lines += ["── 一次選択（ローカル）", self.prefilter.render(explain=explain), ""]
        if self.draft is not None and self.draft.fallback_reason:
            lines += [
                f"  ⚠ 主バックエンドが使えず {self.draft.llm_backend} で生成しました",
                f"    理由: {self.draft.fallback_reason}",
                "",
            ]
        if self.draft is None:
            return "\n".join(lines)

        if explain and self.scored:
            lines += ["── スコア内訳（★=採用）", format_scores(self.scored, self.draft.selected), ""]

        lines.append("── 選定")
        for item in self.draft.selected:
            mark = "有名" if item.bucket == "famous" else "ニッチ"
            lines.append(
                f"  [{mark}] fame {item.fame_final:>5.1f} / 興味 "
                f"{item.assessment.interest:>3} / {item.assessment.category}"
            )
            lines.append(f"         {item.display_title[:60]}")
            lines.append(f"         {item.article.source_name} — {item.article.url}")
        lines.append("")

        titles = {s.article.id: s.display_title for s in self.draft.selected}
        lines.append(format_issues(self.draft.verification_issues, titles))
        lines.append("")
        from .verify import AuditIssue

        lines.append(
            format_audit(
                [AuditIssue.model_validate(i) for i in self.draft.audit_issues], titles
            )
        )
        return "\n".join(lines)


def collect_stage(conn) -> tuple[list[Article], list[SourceResult]]:
    """収集してクラスタリングし、DB に保存する。"""
    first_seen = store.first_seen_dates(conn)
    articles, results = Collector(first_seen=first_seen).collect()
    articles = assign_clusters(articles)
    store.upsert_articles(conn, articles)
    return articles, results


def prepare_candidates(
    conn, articles: list[Article], *, screen: bool = True, explain: bool = False
) -> tuple[list[Article], int, PrefilterResult | None]:
    """代表記事に絞り、既出を除き、一次選抜を通し、本文を取得する。

    本文取得（HTTPアクセス）は一次選抜の**後**に行う。先に取ると、
    落とす記事の本文まで取りに行くことになり、時間も相手サーバへの
    負荷も無駄になる。
    """
    settings = load_settings()
    reps = representatives(articles)
    history = store.drafted_history(conn, settings.dedup["history_days"])
    kept, dropped = filter_already_drafted(reps, history)

    candidates = sorted(kept, key=signal_score, reverse=True)[
        : settings.collect["max_fulltext_fetch"]
    ]

    result: PrefilterResult | None = None
    if screen:
        result = Prefilter().run(candidates, explain=explain)
        candidates = result.kept

    candidates = Extractor().enrich(candidates)
    store.upsert_articles(conn, candidates)
    return candidates, len(dropped), result


def run_daily(
    *,
    date: str | None = None,
    llm: LLM | None = None,
    explain: bool = False,
) -> PipelineReport:
    """収集から原稿生成までを実行し、下書きを保存する。"""
    date = date or today_jst()
    report = PipelineReport(date=date)

    with store.connect() as conn:
        articles, results = collect_stage(conn)
        report.source_results = results
        report.collected = len(articles)
        report.clusters = len({a.cluster_id or a.id for a in articles})
        report.articles = articles
        log.info("複数社が報じた話題:\n%s", cluster_summary(articles))

        shared_llm = llm if llm is not None else make_llm()

        # 一次選抜はローカルの Ollama が担う。主バックエンドが
        # 既に Ollama の場合は二重に走らせても意味がないので省く。
        screen = getattr(shared_llm, "name", "") != "ollama"
        candidates, dropped, screened = prepare_candidates(
            conn, articles, screen=screen, explain=explain
        )
        report.dropped_as_seen = dropped
        report.after_dedup = len(candidates)
        report.candidates = len(candidates)
        report.prefilter = screened
        log.info("本文抽出:\n%s", enrichment_summary(candidates))

        if not candidates:
            log.error("候補が0件です。収集かフィルタ設定を確認してください")
            return report
        picked, scored = Selector(llm=shared_llm).run(candidates)
        report.scored = scored
        if not picked:
            log.error("選定結果が0件です")
            return report

        x_posts, ig_caption = Composer(llm=shared_llm).run(picked)

        # ファクト照合: X 原稿と IG の該当項目を、元記事本文と突き合わせる
        x_bodies = {p.article_id: p.body for p in x_posts}
        ig_items = {
            item.article.id: ig_caption.items[i]
            for i, item in enumerate(picked)
            if i < len(ig_caption.items)
        }
        sources = {i.article.id: (i.article.fulltext or i.article.summary) for i in picked}
        issues = verify_draft(x_bodies, ig_items, sources)

        # 機械照合は「本文に無い数値・固有名詞」しか見ない。語彙は正しいのに
        # 意味がずれているケース（価格の話→供給の話 など）は素通りするので、
        # LLM に原稿と元記事を並べて読ませ、記事にない主張を検出させる。
        titles_for_audit = {i.article.id: i.display_title for i in picked}
        audit = audit_draft(shared_llm, x_bodies, ig_items, sources, titles_for_audit)

        draft = Draft(
            date=date,
            generated_at=datetime.now(UTC),
            selected=picked,
            x_posts=x_posts,
            ig_caption=ig_caption,
            verification_issues=issues,
            llm_backend=getattr(shared_llm, "name", ""),
            fallback_reason=getattr(shared_llm, "fallback_reason", ""),
            audit_issues=[i.model_dump() for i in audit],
        )
        store.save_draft(conn, draft)
        store.record_drafted(conn, date, [i.article for i in picked])
        report.draft = draft

    return report
