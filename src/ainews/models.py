"""パイプライン全体で共有するデータモデル。

LLM の構造化出力に使うものは Pydantic、DB との受け渡しに使うものも Pydantic で
統一する（SQLite への保存は store.py が JSON 化して行う）。
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Category = Literal[
    "新モデル",
    "製品",
    "資金調達",
    "研究",
    "規制",
    "事件",
    "ツール",
    "業界動向",
]

HookType = Literal["数字型", "疑問型", "対比型", "断言型", "物語型"]

Bucket = Literal["famous", "niche"]


def article_id(url: str) -> str:
    """URL から安定した記事IDを作る。同じ記事の再取得で同じIDになる。"""
    normalized = re.sub(r"[?#].*$", "", url.strip().lower()).rstrip("/")
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


class Article(BaseModel):
    """収集した記事1件。"""

    id: str
    source_id: str
    source_name: str
    tier: str
    lang: str
    image_policy: str
    title: str
    url: str
    summary: str = ""
    published_at: datetime
    fetched_at: datetime
    # extract.py が埋める
    fulltext: str = ""
    og_image_url: str = ""
    # cluster.py が埋める
    cluster_id: str = ""
    cluster_size: int = 1
    # コミュニティソース由来のシグナル
    points: int = 0

    @property
    def text_for_llm(self) -> str:
        """LLM に渡す本文。長すぎると無駄なので先頭を切る。"""
        body = self.fulltext or self.summary
        return body[:4000]


class Assessment(BaseModel):
    """LLM による記事1件の評価（構造化出力）。"""

    article_id: str = Field(description="評価対象の記事ID（入力の id をそのまま返す）")
    fame: int = Field(ge=0, le=100, description="どれだけ広く報じられ知られているか")
    interest: int = Field(
        ge=0, le=100, description="一般のAI関心層が『へえ』と思う度合い"
    )
    category: Category
    headline_ja: str = Field(
        description="カード画像に載せる日本語の見出し（28字以内）。原題が英語でも必ず日本語にする"
    )
    hook: str = Field(description="日本語の一言フック（30字以内）")
    why_matters: str = Field(description="なぜ重要か 1〜2文の日本語")
    risk_flags: list[str] = Field(
        default_factory=list,
        description="該当するものだけ: 未確認情報 / 誤情報の疑い / センシティブ / 宣伝色 / 古い話題",
    )


class AssessmentBatch(BaseModel):
    """複数記事の評価をまとめて返すためのラッパ。"""

    assessments: list[Assessment]


class ScoredArticle(BaseModel):
    """評価とスコアが付いた記事。select.py の中間表現。"""

    article: Article
    assessment: Assessment
    signal_score: float = Field(description="機械シグナルの正規化スコア 0-100")
    fame_final: float = Field(description="LLM評価と機械シグナルの加重平均")
    bucket: Bucket

    @property
    def display_title(self) -> str:
        """カード画像とプレビューに出す見出し。

        読者は日本語話者なので、英語記事でも日本語見出しを優先する。
        LLM が生成に失敗した場合だけ原題にフォールバックする。
        """
        return self.assessment.headline_ja.strip() or self.article.title


class XPost(BaseModel):
    """X 用の投稿原稿（構造化出力）。"""

    article_id: str
    body: str = Field(
        description="X投稿の本文。URLは含めない。ハッシュタグと出典表記を含む完成形"
    )
    hook_type: HookType = Field(description="冒頭フックの型（後の分析用）")


class IGCaption(BaseModel):
    """Instagram 用のキャプション（構造化出力）。"""

    opening: str = Field(description="冒頭フック 1〜2行")
    items: list[str] = Field(description="ニュース4件それぞれの要約。各2〜3行")
    closing: str = Field(description="締めの一言とCTA")
    hashtags: list[str] = Field(description="# を含まないタグ名のリスト")

    def render(self) -> str:
        """投稿にそのまま貼れる形に整形する。"""
        parts = [self.opening, ""]
        for i, item in enumerate(self.items, start=1):
            parts.append(f"{i}. {item}")
            parts.append("")
        parts.append(self.closing)
        parts.append("")
        parts.append(" ".join(f"#{t.lstrip('#')}" for t in self.hashtags))
        return "\n".join(parts).strip()


class VerificationIssue(BaseModel):
    """ファクト照合で見つかった問題。"""

    kind: Literal["数値", "固有名詞", "日付"]
    value: str
    note: str


class ImageCard(BaseModel):
    """生成した画像1枚のメタデータ。"""

    path: str
    platform: Literal["instagram", "x"]
    role: Literal["cover", "news", "cta"]
    article_id: str = ""
    used_og_image: bool = False
    og_image_url: str = ""


class Draft(BaseModel):
    """1日分の下書き。これがプレビューサイトと分析の入力になる。"""

    date: str  # YYYY-MM-DD
    generated_at: datetime
    selected: list[ScoredArticle]
    x_posts: list[XPost]
    ig_caption: IGCaption
    images: list[ImageCard] = Field(default_factory=list)
    verification_issues: dict[str, list[VerificationIssue]] = Field(
        default_factory=dict, description="article_id → 検出された問題"
    )

    def article_by_id(self, aid: str) -> Article | None:
        return next((s.article for s in self.selected if s.article.id == aid), None)

    def scored_by_id(self, aid: str) -> ScoredArticle | None:
        return next((s for s in self.selected if s.article.id == aid), None)
