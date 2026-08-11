"""プロバイダ層と一次選抜の回帰テスト。

外部（Ollama サーバ・claude コマンド）は呼ばない。壊れると
「毎朝の投稿が止まる」or「静かに品質が落ちる」経路だけを押さえる。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import BaseModel

from ainews.llm import FallbackLLM
from ainews.models import Article, article_id
from ainews.prefilter import Prefilter, Screening, ScreeningBatch
from ainews.providers import (
    ProviderError,
    ProviderUnavailable,
    schema_instruction,
)
from ainews.providers.claude_code import ClaudeCodeProvider, _strip_fence


class Sample(BaseModel):
    name: str
    count: int


def make_article(title: str, *, tier: str = "major", cluster: int = 1) -> Article:
    url = f"https://example.com/{abs(hash(title))}"
    now = datetime.now(UTC)
    article = Article(
        id=article_id(url),
        source_id="test",
        source_name="test",
        tier=tier,
        lang="ja",
        image_policy="ogp_ok",
        title=title,
        url=url,
        summary="",
        published_at=now - timedelta(hours=1),
        fetched_at=now,
    )
    article.cluster_size = cluster
    return article


# ── スキーマ指示文 ────────────────────────────────────────────────────


class TestSchemaInstruction:
    """Claude Code はスキーマを強制できないので、指示文が要になる。"""

    def test_embeds_the_schema(self):
        text = schema_instruction(Sample)
        assert "count" in text and "name" in text

    def test_forbids_prose_and_fences(self):
        text = schema_instruction(Sample)
        assert "```json" in text or "コードフェンス" in text
        assert "JSON オブジェクトのみ" in text

    def test_count_hint_is_stated_when_given(self):
        """件数を言わないと Ollama も Claude Code も件数を落とす。"""
        assert "7 件" in schema_instruction(Sample, count_hint=7)

    def test_no_count_hint_when_absent(self):
        assert "件すべて" not in schema_instruction(Sample)

    def test_unsupported_keywords_are_stripped(self):
        """Field(ge=...) 由来の minimum などは構造化出力が受け付けない。"""

        from pydantic import Field

        class Bounded(BaseModel):
            score: int = Field(ge=0, le=100)

        assert "minimum" not in schema_instruction(Bounded)


# ── Claude Code の出力整形 ────────────────────────────────────────────


class TestStripFence:
    def test_plain_json_passes_through(self):
        assert _strip_fence('{"a": 1}') == '{"a": 1}'

    def test_json_fence_is_removed(self):
        assert _strip_fence('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_bare_fence_is_removed(self):
        assert _strip_fence('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_surrounding_whitespace_is_trimmed(self):
        assert _strip_fence('\n\n{"a": 1}\n') == '{"a": 1}'


class TestClaudeCodeErrorClassification:
    """利用上限に当たったら、書き直させても無駄なので即退避したい。"""

    @pytest.mark.parametrize(
        "message",
        [
            "Usage limit reached. Try again later.",
            "You are not logged in. Please run /login",
            "Credit balance is too low",
        ],
    )
    def test_unavailable_messages_trigger_fallback(self, message):
        assert isinstance(
            ClaudeCodeProvider._classify(message, 1), ProviderUnavailable
        )

    def test_other_failures_are_retryable(self):
        error = ClaudeCodeProvider._classify("unexpected token at line 3", 1)
        assert isinstance(error, ProviderError)
        assert not isinstance(error, ProviderUnavailable)


# ── フォールバック ────────────────────────────────────────────────────


class _Boom:
    """常に失敗するプロバイダ。"""

    name = "boom"

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def structured(self, **_):
        self.calls += 1
        raise self.error

    def text(self, **_):
        self.calls += 1
        raise self.error


class _Fine:
    """常に成功するプロバイダ。"""

    name = "fine"

    def __init__(self) -> None:
        self.calls = 0

    def structured(self, **_):
        self.calls += 1
        return Sample(name="ok", count=1)

    def text(self, **_):
        self.calls += 1
        return "ok"


class TestFallback:
    """毎朝の投稿を止めないための機構。ここが壊れると障害日に何も出ない。"""

    def test_switches_on_failure(self):
        primary, backup = _Boom(ProviderUnavailable("上限")), _Fine()
        llm = FallbackLLM(primary, backup)
        assert llm.structured(system="", user="", schema=Sample).name == "ok"
        assert llm.switched
        assert backup.calls == 1

    def test_records_reason_for_the_preview_banner(self):
        llm = FallbackLLM(_Boom(ProviderUnavailable("利用上限")), _Fine())
        llm.structured(system="", user="", schema=Sample)
        assert "利用上限" in llm.fallback_reason
        assert llm.name == "fine"

    def test_does_not_retry_primary_after_switching(self):
        """一度使えなくなった相手を毎回試すのは時間の無駄。"""
        primary, backup = _Boom(ProviderUnavailable("上限")), _Fine()
        llm = FallbackLLM(primary, backup)
        for _ in range(3):
            llm.structured(system="", user="", schema=Sample)
        assert primary.calls == 1
        assert backup.calls == 3

    def test_raises_when_both_fail(self):
        llm = FallbackLLM(_Boom(ProviderError("A")), _Boom(ProviderError("B")))
        with pytest.raises(ProviderError):
            llm.structured(system="", user="", schema=Sample)

    def test_no_fallback_when_primary_succeeds(self):
        primary, backup = _Fine(), _Fine()
        llm = FallbackLLM(primary, backup)
        llm.structured(system="", user="", schema=Sample)
        assert not llm.switched and backup.calls == 0


# ── 一次選抜 ──────────────────────────────────────────────────────────


class _StubScreener:
    """判定結果を差し込めるスタブ。"""

    name = "stub"

    def __init__(self, verdicts: dict[str, tuple[bool, int]], *, fail: bool = False):
        self.verdicts = verdicts
        self.fail = fail

    def structured(self, *, user: str, **_):
        if self.fail:
            raise ProviderError("模擬失敗")
        ids = [item["id"] for item in json.loads(user[user.index("[") :])]
        return ScreeningBatch(
            items=[
                Screening(id=i, is_ai=self.verdicts[i][0], interest=self.verdicts[i][1])
                for i in ids
                if i in self.verdicts
            ]
        )


class TestPrefilter:
    def _articles(self, n: int) -> list[Article]:
        return [make_article(f"記事{i}について") for i in range(n)]

    @staticmethod
    def _prefilter(provider, *, keep: int = 20) -> Prefilter:
        """枠割れ救済が働かない十分な件数で検証するための生成ヘルパ。"""
        prefilter = Prefilter(provider=provider)
        prefilter.keep = keep
        return prefilter

    def test_drops_non_ai_and_low_interest(self):
        # 救済が働かないよう、通過する記事を十分に用意する
        articles = self._articles(12)
        verdicts = {a.id: (True, 90) for a in articles}
        verdicts[articles[1].id] = (False, 90)  # AI無関係
        verdicts[articles[2].id] = (True, 5)  # 興味度不足

        result = self._prefilter(_StubScreener(verdicts)).run(articles)
        kept_ids = {a.id for a in result.kept}
        assert articles[1].id not in kept_ids
        assert articles[2].id not in kept_ids
        assert result.dropped_not_ai == [articles[1]]
        assert result.dropped_low_interest == [articles[2]]

    def test_rescues_dropped_articles_when_too_few_survive(self):
        """ニュースが薄い日に絞り込みすぎて投稿本数を満たせなくなるのを防ぐ。"""
        articles = self._articles(6)
        verdicts = {a.id: (True, 5) for a in articles}  # 全部が興味度不足
        verdicts[articles[0].id] = (True, 90)

        result = self._prefilter(_StubScreener(verdicts)).run(articles)
        assert len(result.kept) == 6
        assert result.dropped_low_interest == []

    def test_unjudged_articles_are_kept(self):
        """判定漏れを落とすと取りこぼしになる。通す側に倒すのが正解。"""
        articles = self._articles(2)
        # 1件だけ判定を返さない
        provider = _StubScreener({articles[0].id: (True, 80)})
        result = self._prefilter(provider).run(articles)
        assert articles[1] in result.kept
        assert articles[1] in result.unjudged

    def test_provider_failure_keeps_everything(self):
        """Ollama が落ちた日も、候補を全部通して後段に判断させる。"""
        articles = self._articles(3)
        result = self._prefilter(_StubScreener({}, fail=True)).run(articles)
        assert len(result.kept) == 3
        assert len(result.unjudged) == 3

    def test_ranking_blends_interest_and_signal(self):
        """興味度だけで切ると、多数の媒体が報じた大きな話題を取り逃がす。"""
        plain = make_article("普通の記事です", tier="niche", cluster=1)
        big = make_article("多数が報じた話題です", tier="official", cluster=8)
        verdicts = {plain.id: (True, 80), big.id: (True, 74)}

        prefilter = Prefilter(provider=_StubScreener(verdicts))
        prefilter.keep = 1
        result = prefilter.run([plain, big])
        # 興味度は plain が上だが、報道の広がりで big が勝つ
        assert result.kept == [big]

    def test_disabled_prefilter_passes_everything_through(self):
        articles = self._articles(3)
        prefilter = self._prefilter(_StubScreener({}, fail=True))
        prefilter.cfg = {**prefilter.cfg, "enabled": False}
        result = prefilter.run(articles)
        assert result.kept == articles
        assert result.elapsed == 0.0


# ── カード画像の折り返し ──────────────────────────────────────────────


class TestWrap:
    """和文と欧文が混ざった見出しで単語が壊れないこと。

    製品名は見出しの核なので、"Claude Opus 5" が "ClaudeOpus 5" に
    なると投稿できる品質でなくなる。
    """

    def _renderer(self):
        from ainews.imagegen import CardRenderer

        return CardRenderer()

    def test_space_between_ascii_and_cjk_survives(self):
        renderer = self._renderer()
        font = renderer.font(52, bold=True)
        assert renderer.wrap("Anthropic、Claude Opus 5を公開", font, 900) == [
            "Anthropic、Claude Opus 5を公開"
        ]

    def test_version_number_is_not_split(self):
        renderer = self._renderer()
        font = renderer.font(52, bold=True)
        assert renderer.wrap("MythosとGPT-5.6 Solが暴走", font, 900) == [
            "MythosとGPT-5.6 Solが暴走"
        ]

    def test_wraps_without_losing_words(self):
        renderer = self._renderer()
        font = renderer.font(52, bold=True)
        text = "Google DeepMind、Gemini Robotics ER 2を発表"
        lines = renderer.wrap(text, font, 900)
        assert len(lines) > 1
        # 折り返しても、空白を詰めれば元の文字列に戻る
        assert "".join(lines).replace(" ", "") == text.replace(" ", "")

    def test_no_leading_space_on_wrapped_lines(self):
        renderer = self._renderer()
        font = renderer.font(52, bold=True)
        for line in renderer.wrap("Google DeepMind、Gemini Robotics ER 2を発表", font, 900):
            assert line == line.strip()


# ── Discord 配信 ──────────────────────────────────────────────────────


class TestDiscordPackage:
    """配信の要件は「長押しコピーで原稿がそのまま取れる」こと。

    Discord の「テキストをコピー」は content だけを拾い embed は含めない。
    そのため content に余計な見出しが混ざると、貼り付けのたびに手直しが
    必要になり、この仕組みの価値が失われる。
    """

    def _capture(self, monkeypatch, tmp_path):
        import ainews.notify as notify

        sent: list[tuple[dict, list | None]] = []
        monkeypatch.setattr(
            notify, "_post", lambda w, p, a=None: (sent.append((p, a)), True)[1]
        )
        monkeypatch.setattr(notify, "_webhook", lambda: "https://discord.test/x")
        monkeypatch.setattr(notify, "SEND_INTERVAL", 0)
        return notify, sent

    def _draft(self):
        from ainews.models import Assessment, Draft, IGCaption, ScoredArticle, XPost
        from datetime import UTC, datetime

        items, posts = [], []
        for i in range(2):
            article = make_article(f"見出し{i}です")
            items.append(
                ScoredArticle(
                    article=article,
                    signal_score=50,
                    fame_final=70,
                    bucket="famous",
                    assessment=Assessment(
                        article_id=article.id,
                        fame=70,
                        interest=80,
                        category="新モデル",
                        headline_ja=f"見出し{i}です",
                        hook="フック",
                        why_matters="重要",
                        risk_flags=[],
                    ),
                )
            )
            posts.append(
                XPost(
                    article_id=article.id,
                    body=f"本文{i}です。\n\n#AI（出典: test）",
                    hook_type="断言型",
                )
            )
        return Draft(
            date="2026-08-07",
            generated_at=datetime.now(UTC),
            selected=items,
            x_posts=posts,
            ig_caption=IGCaption(
                opening="冒頭", items=["1件目", "2件目"], closing="締め", hashtags=["AI"]
            ),
        )

    def test_content_is_the_post_body_verbatim(self, monkeypatch, tmp_path):
        notify, sent = self._capture(monkeypatch, tmp_path)
        draft = self._draft()
        notify.send_draft_package(draft, tmp_path)

        contents = [p.get("content") for p, _ in sent if p.get("content")]
        bodies = {p.body for p in draft.x_posts} | {draft.ig_caption.render()}
        assert set(contents) == bodies

    def test_metadata_never_leaks_into_content(self, monkeypatch, tmp_path):
        """番号や字数が content に混ざると、コピーした本文が汚れる。"""
        notify, sent = self._capture(monkeypatch, tmp_path)
        notify.send_draft_package(self._draft(), tmp_path)

        for payload, _ in sent:
            content = payload.get("content", "")
            assert "字数" not in content
            assert "長押し" not in content
            assert not content.startswith("X ")

    def test_summary_message_carries_no_content(self, monkeypatch, tmp_path):
        """ヘッダは embed だけにする。content があると誤コピーの元になる。"""
        notify, sent = self._capture(monkeypatch, tmp_path)
        notify.send_draft_package(self._draft(), tmp_path)
        assert not sent[0][0].get("content")
        assert "の下書き" in sent[0][0]["embeds"][0]["title"]

    def test_skips_without_webhook(self, monkeypatch, tmp_path):
        import ainews.notify as notify

        monkeypatch.setattr(notify, "_webhook", lambda: "")
        assert notify.send_draft_package(self._draft(), tmp_path) is False


class TestSplit:
    def test_short_text_stays_whole(self):
        from ainews.notify import _split

        assert _split("abc", 100) == ["abc"]

    def test_splits_on_newline_when_possible(self):
        from ainews.notify import _split

        parts = _split("a" * 40 + "\n" + "b" * 40, 50)
        assert parts == ["a" * 40, "b" * 40]

    def test_splits_by_length_without_newline(self):
        from ainews.notify import _split

        parts = _split("a" * 120, 50)
        assert all(len(p) <= 50 for p in parts)
        assert "".join(parts) == "a" * 120


class TestFailureReason:
    """claude -p は異常時も JSON の封筒を返す。生の JSON を投げると
    先頭のメタデータで文字数を使い切り、肝心の理由が読めなくなる。"""

    def test_prefers_result_over_raw_json(self):
        from ainews.providers.claude_code import _failure_reason

        envelope = {"is_error": True, "result": "Usage limit reached", "subtype": "error"}
        assert "Usage limit reached" in _failure_reason(envelope, "", "{...}")

    def test_falls_back_to_subtype(self):
        from ainews.providers.claude_code import _failure_reason

        assert "error_max_turns" in _failure_reason(
            {"is_error": True, "subtype": "error_max_turns"}, "", ""
        )

    def test_uses_stderr_without_envelope(self):
        from ainews.providers.claude_code import _failure_reason

        assert _failure_reason(None, "command not found", "") == "command not found"

    def test_usage_limit_in_envelope_triggers_fallback(self):
        """封筒の中に理由が埋もれていても、退避すべきと判定できること。"""
        from ainews.providers.claude_code import ClaudeCodeProvider, _failure_reason

        envelope = {"is_error": True, "result": "Claude Code usage limit reached"}
        error = ClaudeCodeProvider._classify(_failure_reason(envelope, "", ""), 1)
        assert isinstance(error, ProviderUnavailable)


class TestRelogin:
    """認証切れは放置すると毎日품質が落ちたまま気づけない。
    一時障害と区別して、通知で対処を促せること。"""

    def test_oauth_expiry_is_treated_as_unavailable(self):
        from ainews.providers.claude_code import ClaudeCodeProvider

        error = ClaudeCodeProvider._classify(
            "Failed to authenticate: OAuth session expired and could not be refreshed", 1
        )
        assert isinstance(error, ProviderUnavailable)

    def test_oauth_expiry_needs_relogin(self):
        from ainews.providers.claude_code import needs_relogin

        assert needs_relogin("OAuth session expired and could not be refreshed")
        assert needs_relogin("You are not logged in. Please run /login")

    def test_transient_failure_does_not_need_relogin(self):
        from ainews.providers.claude_code import needs_relogin

        assert not needs_relogin("usage limit reached")
        assert not needs_relogin("connection reset by peer")
