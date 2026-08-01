"""④ カード画像の生成。

方針は「元記事のOGP画像 ＋ ブランドカード合成」。報道写真をそのまま
転載するのではなく、見出し・出典・日付を焼き込んだ引用の体裁にする。
著作権リスクを下げつつ、実写の情報量は残す。

生成物:
  Instagram … 1080×1350 を6枚（表紙 + ニュース4 + CTA）。
               カルーセルは全枚が1枚目の比率に切られるので比率を揃える。
  X          … 1600×900 を1ニュースにつき1枚。

OGP画像が取れない、またはソースの image_policy が text_only の場合は
自動でタイポグラフィのみのカードに切り替える。画像が無いせいで
その日の投稿が落ちることはない。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date as date_cls
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .config import load_settings, project_root
from .models import Draft, ImageCard, ScoredArticle
from .net import DEFAULT_UA

log = logging.getLogger(__name__)

# カテゴリごとのアクセント色。1日4枚が同じ色にならないようにする。
CATEGORY_COLORS = {
    "新モデル": "#3DDC97",
    "製品": "#4EA8DE",
    "資金調達": "#F4B860",
    "研究": "#B98AE0",
    "規制": "#E86A6A",
    "事件": "#E8735A",
    "ツール": "#5FD0C4",
    "業界動向": "#8FA0B5",
}


@dataclass
class Palette:
    background: str
    surface: str
    text: str
    muted: str
    accent: str


class Canvas:
    """カード1枚の描画コンテキスト。"""

    def __init__(self, width: int, height: int, palette: Palette) -> None:
        self.width = width
        self.height = height
        self.palette = palette
        self.image = Image.new("RGB", (width, height), palette.background)
        self.draw = ImageDraw.Draw(self.image)


class CardRenderer:
    """設定に従ってカード画像を描く。"""

    def __init__(self) -> None:
        settings = load_settings()
        self.cfg = settings.image
        self.account = settings.account
        colors = self.cfg["colors"]
        self.palette = Palette(**colors)
        root = project_root()
        self._bold_path = root / self.cfg["font"]["bold"]
        self._regular_path = root / self.cfg["font"]["regular"]
        self._font_cache: dict[tuple[bool, int], ImageFont.FreeTypeFont] = {}

    # ── フォント・テキスト ────────────────────────────────────────────

    def font(self, size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
        key = (bold, size)
        if key not in self._font_cache:
            path = self._bold_path if bold else self._regular_path
            self._font_cache[key] = ImageFont.truetype(str(path), size)
        return self._font_cache[key]

    def wrap(
        self, text: str, font: ImageFont.FreeTypeFont, max_width: int
    ) -> list[str]:
        """幅に収まるよう折り返す。

        日本語は単語境界が無いので1文字ずつ積む。英単語は途中で切ると
        読めなくなるので、半角スペース区切りの語は塊のまま扱う。
        """
        lines: list[str] = []
        current = ""
        # 空白で切りつつ、日本語部分は1文字ずつ足せるよう細かい単位にする
        chunks: list[str] = []
        for word in text.split(" "):
            if word.isascii():
                chunks.append(word + " ")
            else:
                chunks.extend(list(word))
        for chunk in chunks:
            trial = current + chunk
            if self.draw_width(trial.rstrip(), font) <= max_width:
                current = trial
            else:
                if current.strip():
                    lines.append(current.rstrip())
                current = chunk
        if current.strip():
            lines.append(current.rstrip())
        return lines

    @staticmethod
    def draw_width(text: str, font: ImageFont.FreeTypeFont) -> float:
        return font.getlength(text)

    def fit_text(
        self,
        text: str,
        *,
        max_width: int,
        max_height: int,
        max_size: int,
        min_size: int,
        bold: bool = True,
        line_spacing: float = 1.28,
    ) -> tuple[list[str], ImageFont.FreeTypeFont, int]:
        """指定領域に収まる最大のフォントサイズを二分探索的に選ぶ。

        Returns:
            (折り返し済みの行, フォント, 行の高さ)
        """
        best: tuple[list[str], ImageFont.FreeTypeFont, int] | None = None
        for size in range(max_size, min_size - 1, -2):
            font = self.font(size, bold=bold)
            lines = self.wrap(text, font, max_width)
            line_height = int(size * line_spacing)
            if len(lines) * line_height <= max_height:
                return lines, font, line_height
            best = (lines, font, line_height)
        # 最小サイズでも入らなければ、入る行数で切って省略記号を付ける
        lines, font, line_height = best or ([text], self.font(min_size, bold=bold), min_size)
        allowed = max(1, max_height // line_height)
        if len(lines) > allowed:
            lines = lines[:allowed]
            lines[-1] = lines[-1][:-1] + "…"
        return lines, font, line_height

    def text_block(
        self,
        canvas: Canvas,
        lines: list[str],
        *,
        x: int,
        y: int,
        font: ImageFont.FreeTypeFont,
        line_height: int,
        fill: str,
    ) -> int:
        for index, line in enumerate(lines):
            canvas.draw.text((x, y + index * line_height), line, font=font, fill=fill)
        return y + len(lines) * line_height

    # ── パーツ ────────────────────────────────────────────────────────

    def badge(
        self, canvas: Canvas, label: str, *, x: int, y: int, color: str
    ) -> None:
        font = self.font(max(20, canvas.width // 38), bold=True)
        pad_x, pad_y = 22, 12
        width = int(self.draw_width(label, font)) + pad_x * 2
        height = font.size + pad_y * 2
        canvas.draw.rounded_rectangle(
            [x, y, x + width, y + height], radius=height // 2, fill=color
        )
        canvas.draw.text(
            (x + pad_x, y + pad_y - 2), label, font=font, fill=canvas.palette.background
        )

    def photo_panel(
        self, canvas: Canvas, image: Image.Image, *, height: int
    ) -> None:
        """上部に写真を敷き、下端をグラデーションで背景に溶かす。

        写真の下端で色が急に変わると帯が見えてしまうので、
        フェードは写真の高さの6割を使い、最下段は完全に背景色にする。
        写真全体にも薄いスクリムをかけて、見出しの可読性を確保する。
        """
        panel = _cover_crop(image, canvas.width, height)

        # 全面スクリム（暗くして文字を乗せられる明度に落とす）
        scrim = Image.new("RGB", (canvas.width, height), canvas.palette.background)
        panel = Image.blend(panel, scrim, 0.22)
        canvas.image.paste(panel, (0, 0))

        fade_height = max(1, int(height * 0.60))
        gradient = Image.new("L", (1, fade_height))
        for row in range(fade_height):
            ratio = row / (fade_height - 1) if fade_height > 1 else 1.0
            # 指数を上げるほど「上は透明・下で一気に不透明」になり、
            # 写真の情報を残しつつ境界だけを消せる
            gradient.putpixel((0, row), int(255 * ratio**2.6))
        mask = gradient.resize((canvas.width, fade_height))
        overlay = Image.new("RGB", (canvas.width, fade_height), canvas.palette.background)
        canvas.image.paste(overlay, (0, height - fade_height), mask)

        # 最下段は完全に背景色にして継ぎ目を消す
        seam = max(2, int(height * 0.02))
        canvas.draw.rectangle(
            [0, height - seam, canvas.width, height], fill=canvas.palette.background
        )

    def footer(
        self, canvas: Canvas, *, source: str, date_label: str, y: int
    ) -> None:
        font = self.font(max(20, canvas.width // 42))
        canvas.draw.text(
            (self.margin(canvas), y),
            f"出典: {source}",
            font=font,
            fill=canvas.palette.muted,
        )
        width = self.draw_width(date_label, font)
        canvas.draw.text(
            (canvas.width - self.margin(canvas) - width, y),
            date_label,
            font=font,
            fill=canvas.palette.muted,
        )

    @staticmethod
    def margin(canvas: Canvas) -> int:
        return int(canvas.width * 0.065)

    def accent_bar(self, canvas: Canvas, *, y: int, color: str) -> None:
        m = self.margin(canvas)
        canvas.draw.rounded_rectangle(
            [m, y, m + int(canvas.width * 0.11), y + 8], radius=4, fill=color
        )

    # ── カード ────────────────────────────────────────────────────────

    def news_card(
        self,
        item: ScoredArticle,
        *,
        size: tuple[int, int],
        photo: Image.Image | None,
        date_label: str,
        index: int | None = None,
    ) -> Image.Image:
        width, height = size
        canvas = Canvas(width, height, self.palette)
        m = self.margin(canvas)
        accent = CATEGORY_COLORS.get(item.assessment.category, self.palette.accent)

        content_top = int(height * 0.30)
        footer_y = height - m - int(width * 0.030)
        text_width = width - m * 2
        available = footer_y - content_top - int(height * 0.05)
        gap = int(height * 0.022)

        # 見出しは全体の6割まで。残りをフックに使う。
        lines, font, line_height = self.fit_text(
            item.display_title,
            max_width=text_width,
            max_height=int(available * 0.6),
            max_size=int(width * 0.062),
            min_size=int(width * 0.030),
        )
        headline_height = len(lines) * line_height

        hook = item.assessment.hook.strip()
        hook_lines: list[str] = []
        hook_font = font
        hook_lh = 0
        if hook:
            hook_lines, hook_font, hook_lh = self.fit_text(
                hook,
                max_width=text_width,
                max_height=max(1, available - headline_height - gap),
                max_size=int(width * 0.033),
                min_size=int(width * 0.022),
                bold=False,
            )

        # テキストは下寄せにする。上寄せだと見出しが短い日に
        # カード下半分が丸ごと空いて間延びする。
        total = headline_height + (len(hook_lines) * hook_lh + gap if hook_lines else 0)
        start = max(content_top, footer_y - int(height * 0.045) - total)

        # 写真はテキストの直前まで伸ばす。固定比率にすると、テキストが
        # 短い日に写真とテキストの間へ帯状の空白ができる。
        if photo is not None:
            photo_height = min(
                int(height * 0.78),
                max(int(height * 0.42), start + int(height * 0.015)),
            )
            self.photo_panel(canvas, photo, height=photo_height)

        self.badge(
            canvas, item.assessment.category, x=m, y=int(height * 0.045), color=accent
        )
        if index is not None:
            num_font = self.font(int(height * 0.075), bold=True)
            canvas.draw.text(
                (width - m - self.draw_width(f"{index}", num_font), int(height * 0.04)),
                f"{index}",
                font=num_font,
                fill=accent,
            )

        y = self.text_block(
            canvas, lines, x=m, y=start, font=font, line_height=line_height,
            fill=self.palette.text,
        )
        if hook_lines:
            self.text_block(
                canvas, hook_lines, x=m, y=y + gap,
                font=hook_font, line_height=hook_lh, fill=self.palette.muted,
            )

        self.footer(
            canvas, source=item.article.source_name, date_label=date_label, y=footer_y
        )
        return canvas.image

    def cover_card(
        self, items: list[ScoredArticle], *, size: tuple[int, int], date_label: str
    ) -> Image.Image:
        width, height = size
        canvas = Canvas(width, height, self.palette)
        m = self.margin(canvas)

        self.accent_bar(canvas, y=int(height * 0.085), color=self.palette.accent)

        # タイトルは実測の行高で積む。割合で決め打ちすると "4選" と
        # 日付が詰まって重なる。
        title_size = int(width * 0.085)
        title_font = self.font(title_size, bold=True)
        title_line = int(title_size * 1.18)
        y = int(height * 0.115)
        canvas.draw.text((m, y), "今日のAIニュース", font=title_font, fill=self.palette.text)
        y += title_line
        canvas.draw.text(
            (m, y), f"{len(items)}選", font=title_font, fill=self.palette.accent
        )
        y += title_line + int(height * 0.012)

        date_font = self.font(int(width * 0.036))
        canvas.draw.text((m, y), date_label, font=date_font, fill=self.palette.muted)

        # 4本の見出しを箇条書きにする（表紙で中身が分かるようにする）。
        # 見出しの長さは日によって変わるので、まず各項目の高さを測り、
        # 余った縦方向を項目間に均等配分する。決め打ちの行送りだと
        # 見出しが短い日に下半分が空いて間延びする。
        handle_y = height - m - int(width * 0.030)
        list_top = int(height * 0.40)
        list_bottom = handle_y - int(height * 0.045)
        indent = int(width * 0.085)
        num_font = self.font(int(width * 0.032), bold=True)

        blocks = [
            self.fit_text(
                item.display_title,
                max_width=width - m * 2 - indent,
                max_height=int(height * 0.10),
                max_size=int(width * 0.034),
                min_size=int(width * 0.022),
                bold=False,
            )
            for item in items
        ]
        content_height = sum(len(lines) * lh for lines, _, lh in blocks)
        gaps = max(1, len(items) - 1)
        slack = max(0, (list_bottom - list_top) - content_height)
        # 行送りが開きすぎるとリストに見えなくなるので上限を設ける。
        # 上限で余った分はブロックごと中央に寄せて、下だけが空くのを避ける。
        spacing = min(int(height * 0.070), slack // gaps)
        y = list_top + max(0, (slack - spacing * gaps) // 2)
        for index, (item, (lines, font, line_height)) in enumerate(
            zip(items, blocks), start=1
        ):
            accent = CATEGORY_COLORS.get(item.assessment.category, self.palette.accent)
            canvas.draw.text((m, y + 4), f"{index:02d}", font=num_font, fill=accent)
            self.text_block(
                canvas, lines, x=m + indent, y=y, font=font,
                line_height=line_height, fill=self.palette.text,
            )
            y += len(lines) * line_height + spacing

        handle_font = self.font(int(width * 0.030))
        canvas.draw.text(
            (m, handle_y), self.account["handle"],
            font=handle_font, fill=self.palette.muted,
        )
        return canvas.image

    def cta_card(self, *, size: tuple[int, int]) -> Image.Image:
        width, height = size
        canvas = Canvas(width, height, self.palette)
        m = self.margin(canvas)

        self.accent_bar(canvas, y=int(height * 0.30), color=self.palette.accent)
        title_font = self.font(int(width * 0.062), bold=True)
        canvas.draw.text(
            (m, int(height * 0.35)), "毎朝、AIニュースを\n4本だけ。",
            font=title_font, fill=self.palette.text, spacing=int(width * 0.030),
        )
        body_font = self.font(int(width * 0.034))
        canvas.draw.text(
            (m, int(height * 0.56)),
            "有名なニュースとニッチな話題を\n半分ずつ。5分で今日のAIが分かります。",
            font=body_font, fill=self.palette.muted, spacing=int(width * 0.020),
        )
        handle_font = self.font(int(width * 0.044), bold=True)
        canvas.draw.text(
            (m, int(height * 0.74)), self.account["handle"],
            font=handle_font, fill=self.palette.accent,
        )
        return canvas.image


def _cover_crop(image: Image.Image, width: int, height: int) -> Image.Image:
    """アスペクト比を保ったまま中央を切り出して指定サイズに合わせる。"""
    image = image.convert("RGB")
    src_ratio = image.width / image.height
    dst_ratio = width / height
    if src_ratio > dst_ratio:
        new_width = int(image.height * dst_ratio)
        left = (image.width - new_width) // 2
        image = image.crop((left, 0, left + new_width, image.height))
    else:
        new_height = int(image.width / dst_ratio)
        top = (image.height - new_height) // 3  # 中央より少し上を残す
        image = image.crop((0, top, image.width, top + new_height))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def fetch_image(url: str, *, timeout: float = 15.0) -> Image.Image | None:
    """OGP画像を取得する。失敗しても None を返すだけで落とさない。"""
    if not url:
        return None
    try:
        response = httpx.get(
            url,
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": DEFAULT_UA},
        )
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        image.load()
        # ロゴだけの小さい画像は写真として使えない
        if image.width < 400 or image.height < 220:
            log.info("OGP画像が小さすぎるため文字カードにします: %s", url)
            return None
        return image
    except Exception as exc:
        log.info("OGP画像の取得に失敗（文字カードにします）: %s (%s)", url, exc)
        return None


class ImageBuilder:
    """1日分のカード画像をまとめて生成する。"""

    def __init__(self, output_dir: Path) -> None:
        self.renderer = CardRenderer()
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.renderer.cfg
        self.ig_size = (cfg["instagram"]["width"], cfg["instagram"]["height"])
        self.x_size = (cfg["x"]["width"], cfg["x"]["height"])

    def build(self, draft: Draft) -> list[ImageCard]:
        date_label = _format_date(draft.date)
        items = draft.selected
        cards: list[ImageCard] = []

        # 記事ごとに1回だけ画像を取りにいき、IG と X で使い回す
        photos: dict[str, Image.Image | None] = {}
        for item in items:
            photos[item.article.id] = fetch_image(item.article.og_image_url)

        # Instagram: 表紙 → ニュース4 → CTA
        cover_path = self.output_dir / "ig_01_cover.jpg"
        self.renderer.cover_card(items, size=self.ig_size, date_label=date_label).save(
            cover_path, quality=92
        )
        cards.append(ImageCard(path=cover_path.name, platform="instagram", role="cover"))

        for index, item in enumerate(items, start=1):
            photo = photos[item.article.id]
            path = self.output_dir / f"ig_{index + 1:02d}_news.jpg"
            self.renderer.news_card(
                item, size=self.ig_size, photo=photo, date_label=date_label, index=index
            ).save(path, quality=92)
            cards.append(
                ImageCard(
                    path=path.name,
                    platform="instagram",
                    role="news",
                    article_id=item.article.id,
                    used_og_image=photo is not None,
                    og_image_url=item.article.og_image_url,
                )
            )

        cta_path = self.output_dir / f"ig_{len(items) + 2:02d}_cta.jpg"
        self.renderer.cta_card(size=self.ig_size).save(cta_path, quality=92)
        cards.append(ImageCard(path=cta_path.name, platform="instagram", role="cta"))

        # X: 1ニュース1枚
        for index, item in enumerate(items, start=1):
            photo = photos[item.article.id]
            path = self.output_dir / f"x_{index:02d}.jpg"
            self.renderer.news_card(
                item, size=self.x_size, photo=photo, date_label=date_label
            ).save(path, quality=92)
            cards.append(
                ImageCard(
                    path=path.name,
                    platform="x",
                    role="news",
                    article_id=item.article.id,
                    used_og_image=photo is not None,
                    og_image_url=item.article.og_image_url,
                )
            )

        return cards


def _format_date(iso_date: str) -> str:
    d = date_cls.fromisoformat(iso_date)
    return f"{d.year}.{d.month:02d}.{d.day:02d}"


def image_summary(cards: list[ImageCard]) -> str:
    ig = [c for c in cards if c.platform == "instagram"]
    x = [c for c in cards if c.platform == "x"]
    with_photo = sum(1 for c in cards if c.used_og_image)
    news = sum(1 for c in cards if c.role == "news")
    return (
        f"  Instagram {len(ig)}枚 / X {len(x)}枚  "
        f"（うち実写あり {with_photo}/{news}、残りは文字カード）"
    )
