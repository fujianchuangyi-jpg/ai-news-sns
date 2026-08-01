"""設定ファイルの読み込み。

config/settings.yaml と config/sources.yaml を読み、属性アクセスできる形で返す。
プロジェクトルートは環境変数 AINEWS_ROOT で上書きできる（テスト用）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """プロジェクトルート（pyproject.toml がある階層）を返す。"""
    if env := os.environ.get("AINEWS_ROOT"):
        return Path(env).resolve()
    # src/ainews/config.py → src/ainews → src → root
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Source:
    """ニュースソース1件。sources.yaml の1エントリに対応。"""

    id: str
    name: str
    type: str
    url: str
    tier: str
    lang: str
    image_policy: str = "ogp_ok"
    enabled: bool = True
    # None = 自動判定。公式ブログと研究フィードは元々AI専門なのでフィルタ不要、
    # 総合ニュース系はAIキーワードで絞り込む。yaml で明示指定すれば上書きできる。
    filter_ai: bool | None = None
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Source:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})

    @property
    def needs_ai_filter(self) -> bool:
        if self.filter_ai is not None:
            return self.filter_ai
        return self.tier not in ("official", "research")


@dataclass(frozen=True)
class Sources:
    """sources.yaml 全体。"""

    sources: list[Source]
    ai_keywords: list[str]
    exclude_keywords: list[str]

    def enabled(self) -> list[Source]:
        return [s for s in self.sources if s.enabled]

    def by_id(self, source_id: str) -> Source | None:
        return next((s for s in self.sources if s.id == source_id), None)


class Settings:
    """settings.yaml へのドット/ブラケット両対応アクセサ。

    settings.collect["lookback_hours"] のように使う。ネストは dict のまま返す。
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as exc:  # pragma: no cover - 設定ミスの早期検出用
            raise AttributeError(f"settings.yaml に '{name}' がありません") from exc

    def get(self, path: str, default: Any = None) -> Any:
        """'image.colors.accent' のようなドットパスで引く。"""
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def raw(self) -> dict[str, Any]:
        return self._data


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    path = project_root() / "config" / "settings.yaml"
    with path.open(encoding="utf-8") as fh:
        return Settings(yaml.safe_load(fh))


@lru_cache(maxsize=1)
def load_sources() -> Sources:
    path = project_root() / "config" / "sources.yaml"
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Sources(
        sources=[Source.from_dict(s) for s in raw.get("sources", [])],
        ai_keywords=raw.get("ai_keywords", []),
        exclude_keywords=raw.get("exclude_keywords", []),
    )


def data_dir() -> Path:
    d = project_root() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def docs_dir() -> Path:
    d = project_root() / "docs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def prompt(name: str) -> str:
    """config/prompts/<name>.md を読む。"""
    return (project_root() / "config" / "prompts" / f"{name}.md").read_text(
        encoding="utf-8"
    )
