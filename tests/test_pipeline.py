"""パイプラインの回帰テスト。

LLM を呼ばずに検証できる部分（字数計算・クラスタリング・ファクト照合・
選定ロジック・スキーマ整形）を押さえる。ここが壊れると投稿できない
原稿が出たり、同じニュースを二度出したりする。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ainews.cluster import (
    assign_clusters,
    filter_already_drafted,
    representatives,
    similarity,
    title_tokens,
)
from ainews.compose import visible_length, weighted_length
from ainews.llm import sanitize_schema
from ainews.models import Article, Assessment, IGCaption, article_id
from ainews.select import Selector, signal_score
from ainews.verify import verify_text


def make_article(
    title: str,
    *,
    source_id: str = "test",
    tier: str = "major",
    url: str | None = None,
    hours_ago: int = 1,
    points: int = 0,
) -> Article:
    url = url or f"https://example.com/{abs(hash(title))}"
    now = datetime.now(UTC)
    return Article(
        id=article_id(url),
        source_id=source_id,
        source_name=source_id,
        tier=tier,
        lang="ja",
        image_policy="ogp_ok",
        title=title,
        url=url,
        summary="",
        published_at=now - timedelta(hours=hours_ago),
        fetched_at=now,
        points=points,
    )


# ── X の文字数 ────────────────────────────────────────────────────────


class TestWeightedLength:
    """X は日本語1文字を2として数える。ここを誤ると投稿できない原稿が出る。"""

    def test_japanese_140_chars_is_exactly_the_limit(self):
        assert weighted_length("あ" * 140) == 280

    def test_one_char_over_exceeds(self):
        assert weighted_length("あ" * 141) == 282

    def test_ascii_counts_as_one(self):
        assert weighted_length("a" * 280) == 280

    def test_mixed(self):
        # 半角3文字 + 全角2文字 = 3 + 4
        assert weighted_length("abcあい") == 7

    def test_visible_length_ignores_newlines(self):
        assert visible_length("あい\nうえ") == 4


# ── クラスタリング ────────────────────────────────────────────────────


class TestClustering:
    def test_same_story_different_outlets_are_similar(self):
        a = title_tokens("OpenAI、GPT-6を発表　推論性能が大幅向上")
        b = title_tokens("OpenAIがGPT-6を発表、推論性能を大幅に向上 | ITmedia")
        assert similarity(a, b) >= 0.62

    def test_different_stories_are_not_similar(self):
        a = title_tokens("OpenAI、GPT-6を発表")
        b = title_tokens("Google、Gemini 4を公開　マルチモーダル強化")
        assert similarity(a, b) < 0.62

    def test_version_number_survives_hyphen(self):
        """"GPT-6" が媒体名除去で削られないこと（過去に踏んだバグ）。"""
        assert "gpt6" in title_tokens("OpenAI、GPT-6を発表")

    def test_media_suffix_is_stripped(self):
        with_suffix = title_tokens("Gemini 4 launches today - The Verge")
        without = title_tokens("Gemini 4 launches today")
        assert similarity(with_suffix, without) == pytest.approx(1.0)

    def test_cluster_size_counts_outlets(self):
        articles = [
            make_article("OpenAI、GPT-6を発表　推論性能が向上", source_id="a"),
            make_article("OpenAIがGPT-6を発表、推論性能が向上", source_id="b"),
            make_article("Google、Gemini 4を公開", source_id="c"),
        ]
        clustered = assign_clusters(articles)
        sizes = {a.title: a.cluster_size for a in clustered}
        assert sizes["OpenAI、GPT-6を発表　推論性能が向上"] == 2
        assert sizes["Google、Gemini 4を公開"] == 1

    def test_representative_prefers_primary_source(self):
        articles = [
            make_article("OpenAI、GPT-6を発表　推論性能が向上", source_id="media", tier="major"),
            make_article("OpenAIがGPT-6を発表、推論性能が向上", source_id="openai", tier="official"),
        ]
        reps = representatives(assign_clusters(articles))
        assert len(reps) == 1
        assert reps[0].tier == "official"


# ── 既出チェック ──────────────────────────────────────────────────────


class TestDedup:
    def test_same_url_is_dropped(self):
        article = make_article("OpenAI、GPT-6を発表", url="https://example.com/a")
        history = [
            {
                "article_id": article.id,
                "title": article.title,
                "url": article.url,
                "cluster_id": "",
            }
        ]
        kept, dropped = filter_already_drafted([article], history)
        assert kept == [] and len(dropped) == 1

    def test_same_story_from_another_outlet_is_dropped(self):
        """昨日 A 社で出した話題を、今日 B 社の記事で再掲しないこと。"""
        today = make_article(
            "OpenAIがGPT-6を発表、推論性能を大幅に向上", url="https://b.example/1"
        )
        history = [
            {
                "article_id": "yesterday",
                "title": "OpenAI、GPT-6を発表　推論性能が大幅向上",
                "url": "https://a.example/1",
                "cluster_id": "",
            }
        ]
        kept, dropped = filter_already_drafted([today], history)
        assert kept == [] and len(dropped) == 1

    def test_unrelated_story_survives(self):
        today = make_article("Google、Gemini 4を公開", url="https://b.example/2")
        history = [
            {
                "article_id": "yesterday",
                "title": "OpenAI、GPT-6を発表",
                "url": "https://a.example/1",
                "cluster_id": "",
            }
        ]
        kept, _ = filter_already_drafted([today], history)
        assert kept == [today]


# ── ファクト照合 ──────────────────────────────────────────────────────


class TestVerify:
    SOURCE = (
        "OpenAIは新モデルGPT-6を発表した。推論性能は従来比で3倍に向上し、"
        "APIの価格は1,200万トークンあたり15ドルとなる。学習には約90日を要した。"
    )

    def test_accurate_draft_has_no_issues(self):
        draft = "GPT-6が登場。推論性能は3倍、価格は15ドル。\n\n#AI #GPT6（出典: TechCrunch）"
        assert verify_text(draft, self.SOURCE) == []

    def test_fabricated_number_is_caught(self):
        draft = "推論性能は10倍に。\n\n#AI（出典: TechCrunch）"
        values = [i.value for i in verify_text(draft, self.SOURCE)]
        assert "10倍" in values

    def test_substring_number_is_not_a_false_negative(self):
        """"5ドル" が本文の "15ドル" に部分一致して見逃されないこと。"""
        draft = "価格は5ドル。\n\n#AI（出典: TechCrunch）"
        values = [i.value for i in verify_text(draft, self.SOURCE)]
        assert "5ドル" in values

    def test_fabricated_entity_is_caught(self):
        draft = "ソフトバンクが出資した。\n\n#AI（出典: TechCrunch）"
        values = [i.value for i in verify_text(draft, self.SOURCE)]
        assert "ソフトバンク" in values

    def test_number_format_variants_are_tolerated(self):
        draft = "1200万トークンあたり15ドル。90日で学習。\n\n#AI（出典: TechCrunch）"
        assert verify_text(draft, self.SOURCE) == []

    def test_missing_source_text_is_reported(self):
        issues = verify_text("なんらかの原稿", "")
        assert len(issues) == 1 and "照合できません" in issues[0].note


# ── 選定 ──────────────────────────────────────────────────────────────


def make_scored(selector, title, *, fame, interest, category, tier="major", cluster=1):
    article = make_article(title, tier=tier)
    article.cluster_size = cluster
    assessment = Assessment(
        article_id=article.id,
        fame=fame,
        interest=interest,
        category=category,
        headline_ja=title,
        hook="フック",
        why_matters="重要",
        risk_flags=[],
    )
    return article, assessment


class TestSelection:
    def _selector(self):
        return Selector(llm=object())  # LLM は使わないので何でもよい

    def test_signal_score_rises_with_coverage(self):
        one = make_article("A")
        many = make_article("B")
        many.cluster_size = 8
        assert signal_score(many) > signal_score(one)

    def test_picks_two_famous_and_two_niche(self):
        selector = self._selector()
        specs = [
            ("有名1", 95, 80, "新モデル"),
            ("有名2", 90, 70, "製品"),
            ("有名3", 88, 60, "資金調達"),
            ("ニッチ1", 20, 95, "研究"),
            ("ニッチ2", 15, 90, "ツール"),
            ("ニッチ3", 10, 85, "業界動向"),
        ]
        articles, assessments = [], {}
        for title, fame, interest, category in specs:
            article, assessment = make_scored(
                selector, title, fame=fame, interest=interest, category=category
            )
            articles.append(article)
            assessments[article.id] = assessment

        scored = selector.score(articles, assessments)
        picked = selector.pick(scored)

        assert len(picked) == 4
        assert sum(1 for p in picked if p.bucket == "famous") == 2
        assert sum(1 for p in picked if p.bucket == "niche") == 2

    def test_category_cap_prevents_monoculture(self):
        """4本すべてが同じカテゴリにならないこと。"""
        selector = self._selector()
        articles, assessments = [], {}
        for index in range(6):
            article, assessment = make_scored(
                selector,
                f"新モデル{index}",
                fame=90 - index,
                interest=90 - index,
                category="新モデル",
            )
            articles.append(article)
            assessments[article.id] = assessment

        picked = selector.pick(selector.score(articles, assessments))
        categories = [p.assessment.category for p in picked]
        assert categories.count("新モデル") <= 2

    def test_risky_articles_are_excluded(self):
        selector = self._selector()
        article, assessment = make_scored(
            selector, "噂話", fame=80, interest=90, category="事件"
        )
        assessment.risk_flags = ["未確認情報"]
        assert selector.score([article], {article.id: assessment}) == []


# ── 構造化出力のスキーマ ──────────────────────────────────────────────


class TestSchemaSanitize:
    def test_unsupported_keywords_are_removed(self):
        schema = sanitize_schema(Assessment.model_json_schema())
        fame = schema["properties"]["fame"]
        # Field(ge=0, le=100) 由来の minimum/maximum は構造化出力が受け付けない
        assert "minimum" not in fame and "maximum" not in fame

    def test_objects_are_closed_and_fully_required(self):
        schema = sanitize_schema(Assessment.model_json_schema())
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


# ── キャプション整形 ──────────────────────────────────────────────────


class TestIGCaption:
    def test_render_numbers_items_and_prefixes_hashtags(self):
        caption = IGCaption(
            opening="冒頭",
            items=["1件目", "2件目"],
            closing="締め",
            hashtags=["AI", "#生成AI"],
        )
        rendered = caption.render()
        assert "1. 1件目" in rendered
        assert "2. 2件目" in rendered
        # 入力に # が付いていても二重にしない
        assert "#AI #生成AI" in rendered
        assert "##" not in rendered


class TestVerifySourceNote:
    """出典表記の媒体名を「本文に無い固有名詞」として誤検出しないこと。

    括弧付き・括弧なしの両方が実際に生成される（ローカルLLMへ退避した日は
    書式が崩れやすい）。ここを取りこぼすと毎回ノイズの警告が出て、
    本物の捏造が埋もれる。
    """

    SOURCE = "OpenAIは新モデルを発表した。推論性能は3倍に向上した。"

    def test_parenthesized_source_is_ignored(self):
        draft = "推論性能が3倍に。\n\n#AI（出典: ITmedia AI+）"
        assert verify_text(draft, self.SOURCE) == []

    def test_bare_source_is_ignored(self):
        draft = "推論性能が3倍に。\n\n#AI\n出典: ITmedia AI+"
        assert verify_text(draft, self.SOURCE) == []

    def test_halfwidth_parens_are_ignored(self):
        draft = "推論性能が3倍に。\n\n#AI (出典: TechCrunch)"
        assert verify_text(draft, self.SOURCE) == []

    def test_real_fabrication_still_caught(self):
        """出典を無視するようにしても、本文の捏造は検出できること。"""
        draft = "推論性能が10倍に。ソフトバンクが出資。\n\n#AI（出典: ITmedia）"
        values = [i.value for i in verify_text(draft, self.SOURCE)]
        assert "10倍" in values and "ソフトバンク" in values


class TestRunDateConsistency:
    """通し実行が日付をまたいでも、全ステップが同じ日を見ること。

    ローカルLLMへ退避した日は生成が数時間かかることがあり、実際に
    daily が保存した下書きを images が見つけられず失敗した。
    """

    def test_run_fixes_the_date_once(self, monkeypatch):
        import argparse

        from ainews import cli

        seen: list[str | None] = []

        def fake_step(args):
            seen.append(args.date)
            return 0

        monkeypatch.setattr(cli, "cmd_daily", fake_step)
        monkeypatch.setattr(cli, "cmd_images", fake_step)
        monkeypatch.setattr(cli, "cmd_render", fake_step)
        monkeypatch.setattr(cli, "cmd_notify", fake_step)

        # 各ステップの間で日付が変わる状況を作る
        dates = iter(["2026-08-10", "2026-08-11", "2026-08-11", "2026-08-11"])
        monkeypatch.setattr(
            "ainews.pipeline.today_jst", lambda: next(dates, "2026-08-11")
        )

        args = argparse.Namespace(date=None)
        assert cli.cmd_run(args) == 0
        assert seen == ["2026-08-10"] * 4, "全ステップが同じ日付を見ること"
