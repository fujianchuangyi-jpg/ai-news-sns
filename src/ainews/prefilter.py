"""① ローカルでの一次選抜。

なぜ必要か:
  `claude -p` は1回あたり約25,000トークンの固定オーバーヘッドがかかる
  （Claude Code 自身のシステムプロンプト）。候補60件を12件ずつ5回に
  分けて評価すると、それだけで12万トークンが overhead に消える。

  そこで「明らかに不要な記事を落とす」という判断力をあまり要さない仕事を
  ローカルの Ollama に任せ、Claude Code には絞り込んだ20件を1回で
  渡す。ユーザーの優先事項である選定精度を、限られた呼び出し回数に
  集中投下するための工夫。

方針:
  取りこぼしを最も嫌う。ここで落ちた記事は二度と検討されないので、
  判定に失敗した記事は**落とさず通す**（fail-open）。通しすぎた分は
  後段の精密評価が正しく落とす。
"""

from __future__ import annotations

import logging
import time

from pydantic import BaseModel, Field

from .config import load_settings, prompt
from .models import Article
from .providers import ProviderError
from .select import signal_score

log = logging.getLogger(__name__)


class Screening(BaseModel):
    """記事1件の一次判定。"""

    id: str = Field(description="入力の id をそのまま返す")
    is_ai: bool = Field(description="AI関連のニュースか")
    interest: int = Field(description="興味度 0-100")


class ScreeningBatch(BaseModel):
    items: list[Screening]


class PrefilterResult:
    """一次選抜の結果と、その過程の記録。"""

    def __init__(self) -> None:
        self.kept: list[Article] = []
        self.dropped_not_ai: list[Article] = []
        self.dropped_low_interest: list[Article] = []
        self.dropped_overflow: list[Article] = []
        self.unjudged: list[Article] = []
        self.elapsed: float = 0.0
        self.scores: dict[str, int] = {}

    @property
    def total_dropped(self) -> int:
        return (
            len(self.dropped_not_ai)
            + len(self.dropped_low_interest)
            + len(self.dropped_overflow)
        )

    def render(self, *, explain: bool = False) -> str:
        lines = [
            f"  {len(self.kept)} 件に絞り込み"
            f"（AI無関係 {len(self.dropped_not_ai)} / "
            f"興味度不足 {len(self.dropped_low_interest)} / "
            f"枠あふれ {len(self.dropped_overflow)} を除外、"
            f"{self.elapsed:.0f}秒）"
        ]
        if self.unjudged:
            lines.append(
                f"  ⚠ {len(self.unjudged)} 件は判定できなかったため"
                f"落とさず通しました（取りこぼし防止）"
            )
        if explain:
            lines.append("  ── 通過")
            for article in self.kept:
                score = self.scores.get(article.id, -1)
                lines.append(f"    [{score:>3}] {article.title[:60]}")
            lines.append("  ── 除外（AI無関係）")
            for article in self.dropped_not_ai[:10]:
                lines.append(f"          {article.title[:60]}")
            lines.append("  ── 除外（興味度不足）")
            for article in self.dropped_low_interest[:10]:
                score = self.scores.get(article.id, -1)
                lines.append(f"    [{score:>3}] {article.title[:60]}")
        return "\n".join(lines)


class Prefilter:
    """Ollama を使って候補を絞り込む。"""

    def __init__(self, provider=None) -> None:
        settings = load_settings()
        self.cfg = settings.prefilter
        self.keep = int(self.cfg["keep"])
        self.chunk = int(self.cfg["chunk"])
        self.min_interest = int(self.cfg["min_interest"])
        self._system = prompt("prefilter")
        if provider is None:
            from .providers.ollama import OllamaProvider

            provider = OllamaProvider()
        self.provider = provider

    def _screen_chunk(self, articles: list[Article]) -> dict[str, Screening]:
        payload = [
            {
                "id": a.id,
                "title": a.title,
                "source": a.source_name,
                # 本文はまだ取得していない段階なので概要だけ渡す。
                # 一次選抜には見出しと概要で十分。
                "summary": a.summary[:300],
            }
            for a in articles
        ]
        import json

        batch = self.provider.structured(
            system=self._system,
            user=(
                f"以下の {len(payload)} 件を判定してください。\n\n"
                + json.dumps(payload, ensure_ascii=False, indent=2)
            ),
            schema=ScreeningBatch,
            count_hint=len(payload),
        )
        return {item.id: item for item in batch.items}

    def run(self, articles: list[Article], *, explain: bool = False) -> PrefilterResult:
        """候補を絞り込む。

        候補が少ない日でも実行する。件数を減らす必要が無くても、
        非AI記事や中身の薄い記事を落とす価値があるため。
        （候補が少なければチャンク数も減るので所要時間も短い）
        """
        result = PrefilterResult()
        if not self.cfg.get("enabled", True) or not articles:
            result.kept = articles
            return result

        started = time.monotonic()
        judged: dict[str, Screening] = {}

        chunks = [
            articles[i : i + self.chunk] for i in range(0, len(articles), self.chunk)
        ]
        for index, chunk in enumerate(chunks, start=1):
            try:
                judged.update(self._screen_chunk(chunk))
                log.info("一次選抜 %d/%d チャンク完了", index, len(chunks))
            except ProviderError as exc:
                # チャンクごと失敗しても、その記事は落とさず通す
                log.warning("一次選抜チャンク %d が失敗（通過扱い）: %s", index, exc)

        # 件数欠落した記事だけをもう一度まとめて問い合わせる
        missing = [a for a in articles if a.id not in judged]
        if missing:
            log.info("判定漏れ %d 件を再問い合わせ", len(missing))
            for i in range(0, len(missing), self.chunk):
                try:
                    judged.update(self._screen_chunk(missing[i : i + self.chunk]))
                except ProviderError as exc:
                    log.warning("再問い合わせが失敗（通過扱い）: %s", exc)

        result.elapsed = time.monotonic() - started

        survivors: list[Article] = []
        for article in articles:
            screening = judged.get(article.id)
            if screening is None:
                # 判定できなかった記事は落とさない。ここでの取りこぼしは
                # 後段で取り返せないため。
                result.unjudged.append(article)
                survivors.append(article)
                continue
            result.scores[article.id] = screening.interest
            if not screening.is_ai:
                result.dropped_not_ai.append(article)
            elif screening.interest < self.min_interest:
                result.dropped_low_interest.append(article)
            else:
                survivors.append(article)

        # 残った中から、興味度と機械シグナルを合わせた順に keep 件を取る。
        # 興味度だけで切ると、複数社が報じた大きなニュースが
        # 「見出しが地味」という理由で落ちることがある。
        def rank(article: Article) -> float:
            interest = result.scores.get(article.id, self.min_interest)
            return interest * 0.7 + signal_score(article) * 0.3

        survivors.sort(key=rank, reverse=True)

        # ニュースが少ない日に絞り込みすぎると、その日の投稿本数を
        # 満たせなくなる。足りない場合は落とした記事を点数順に戻す。
        minimum = int(load_settings().select["total"]) * 2
        if len(survivors) < minimum:
            rescued = sorted(
                result.dropped_low_interest + result.dropped_not_ai,
                key=lambda a: result.scores.get(a.id, 0),
                reverse=True,
            )[: minimum - len(survivors)]
            if rescued:
                log.info(
                    "候補が %d 件しか残らないため %d 件を戻します",
                    len(survivors),
                    len(rescued),
                )
                for article in rescued:
                    if article in result.dropped_low_interest:
                        result.dropped_low_interest.remove(article)
                    else:
                        result.dropped_not_ai.remove(article)
                survivors += rescued

        result.kept = survivors[: self.keep]
        result.dropped_overflow = survivors[self.keep :]
        return result
