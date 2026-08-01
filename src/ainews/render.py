"""⑤ プレビューサイトの生成。

これが「下書きのみ運用」の要。スマホで開いて、コピーボタンを押して、
画像を保存して、投稿画面を開く——までを1画面で完結させる。

出力先は docs/ 配下（GitHub Pages）。ここに置いた画像は公開URLを持つので、
将来 Instagram Graph API での自動投稿に進むときも、画像ホスティングを
別途用意する必要がない。
"""

from __future__ import annotations

import logging
import sqlite3
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import store
from .compose import visible_length, weighted_length
from .config import load_settings
from .models import Draft

log = logging.getLogger(__name__)

JST = ZoneInfo("Asia/Tokyo")

ROLE_LABELS = {"cover": "表紙", "news": "ニュース", "cta": "CTA"}


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _build_zip(draft: Draft, output_dir: Path) -> str | None:
    """Instagram のカルーセル画像をまとめた zip を作る。

    スマホで6枚を1枚ずつ保存するのは手間なので、まとめて落とせるようにする。
    """
    ig_images = [c for c in draft.images if c.platform == "instagram"]
    if not ig_images:
        return None
    name = f"instagram_{draft.date}.zip"
    path = output_dir / name
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, card in enumerate(ig_images, start=1):
            source = output_dir / card.path
            if source.exists():
                archive.write(source, arcname=f"{index:02d}_{card.path}")
    return name


def render_preview(draft: Draft, output_dir: Path) -> Path:
    """1日分のプレビューHTMLを書き出してパスを返す。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = load_settings()
    x_limit = settings.compose["x_max_weighted"]

    # テンプレートで使う派生値を先に計算しておく
    for post in draft.x_posts:
        post._weight = weighted_length(post.body)  # type: ignore[attr-defined]
        post._visible = visible_length(post.body)  # type: ignore[attr-defined]
        post._intent = quote(post.body)  # type: ignore[attr-defined]

    x_images = {
        card.article_id: card.path
        for card in draft.images
        if card.platform == "x" and card.article_id
    }
    ig_images = [
        {"path": card.path, "label": ROLE_LABELS.get(card.role, card.role)}
        for card in draft.images
        if card.platform == "instagram"
    ]

    html = _environment().get_template("preview.html").render(
        draft=draft,
        generated_jst=draft.generated_at.astimezone(JST).strftime("%m/%d %H:%M"),
        x_images=x_images,
        ig_images=ig_images,
        ig_caption=draft.ig_caption.render(),
        x_limit=x_limit,
        zip_name=_build_zip(draft, output_dir),
    )
    path = output_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


INDEX_TEMPLATE = """<!doctype html>
<html lang="ja"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>AIニュース 下書き一覧</title>
<style>
 body{{margin:0;background:#0e1116;color:#f5f7fa;
      font-family:-apple-system,BlinkMacSystemFont,"Hiragino Sans",sans-serif;line-height:1.8}}
 .wrap{{max-width:640px;margin:0 auto;padding:1.5rem}}
 h1{{font-size:1.3rem}} a{{color:#3ddc97;text-decoration:none}}
 li{{border-bottom:1px solid #262d38;padding:.7rem 0;list-style:none}}
 .t{{color:#8b95a5;font-size:.85rem}}
 ul{{padding:0}}
</style></head><body><div class="wrap">
<h1>AIニュース 下書き一覧</h1>
<ul>
{items}
</ul>
</div></body></html>
"""


def build_index(conn: sqlite3.Connection, docs_root: Path) -> Path:
    """日付一覧のトップページを作る。過去の下書きに戻れるようにする。"""
    drafts = store.load_drafts_since(conn, days=90)
    items = []
    for draft in drafts:
        titles = "、".join(s.display_title[:22] for s in draft.selected[:2])
        items.append(
            f'<li><a href="d/{draft.date}/">{draft.date}</a>'
            f'<div class="t">{titles}…</div></li>'
        )
    path = docs_root / "index.html"
    path.write_text(
        INDEX_TEMPLATE.format(items="\n".join(items) or "<li>まだ下書きがありません</li>"),
        encoding="utf-8",
    )
    # GitHub Pages が _ 始まりのパスを無視しないようにする
    (docs_root / ".nojekyll").touch()
    return path


def preview_url(date: str) -> str:
    """公開URL（settings.account.site_base_url が設定されている場合）。"""
    base = load_settings().account.get("site_base_url", "").rstrip("/")
    return f"{base}/d/{date}/" if base else ""
