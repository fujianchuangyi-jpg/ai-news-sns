"""Discord への配信。

毎朝、下書きを**そのまま投稿できる形**でスマホに届ける。
Discord アプリはすでに手元にあり、長押しでテキストをコピーでき、
画像もその場で保存できるので、実質これが投稿画面になる。

コピーしやすさのための設計:
    Discord の「テキストをコピー」はメッセージ本文（content）だけを
    コピーし、埋め込み（embed）は含めない。そこで **content には原稿を
    一字一句そのまま入れ**、番号・カテゴリ・字数・警告はすべて embed に
    逃がしている。長押し1回で貼り付けられる状態にするため。

Webhook URL が未設定なら黙ってスキップする（ローカル実行の邪魔をしない）。
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import httpx

from .compose import weighted_length
from .config import load_settings
from .models import Draft
from .render import preview_url

log = logging.getLogger(__name__)

# Discord embed の説明欄の上限
DESCRIPTION_LIMIT = 4000
# メッセージ本文の上限
CONTENT_LIMIT = 2000
# 1メッセージに添付できるファイル数
ATTACHMENT_LIMIT = 10

# 配信の色分け
COLOR_OK = 0x3DDC97
COLOR_WARN = 0xE86A6A
COLOR_INFO = 0x4EA8DE

# 連投でレート制限に当たらないよう、メッセージ間に置く間隔（秒）
SEND_INTERVAL = 0.4


def _build_payload(draft: Draft) -> dict:
    url = preview_url(draft.date)
    limit = load_settings().compose["x_max_weighted"]

    lines = []
    for index, item in enumerate(draft.selected, start=1):
        mark = "🔥" if item.bucket == "famous" else "💎"
        lines.append(
            f"{mark} **{index}. {item.article.title[:60]}**\n"
            f"　{item.article.source_name}・{item.assessment.category}"
            f"・興味度 {item.assessment.interest}"
        )

    warnings = []
    if draft.fallback_reason:
        from .providers.claude_code import needs_relogin

        warnings.append(
            f"⚠️ 予備のLLM（{draft.llm_backend}）で生成。品質を念入りに確認してください"
        )
        if needs_relogin(draft.fallback_reason):
            # 放置すると毎日この状態が続き、品質が落ちたまま気づけない。
            # 具体的な対処をそのまま書く。
            warnings.append(
                "🔑 **Claude Code の再ログインが必要です。**"
                "ターミナルで `claude` を起動し `/login` を実行してください"
            )
    over = [p for p in draft.x_posts if weighted_length(p.body) > limit]
    if over:
        warnings.append(f"⚠️ X原稿 {len(over)}件が字数超過（手直しが必要）")
    if draft.verification_issues:
        warnings.append(
            f"⚠️ ファクト照合で {len(draft.verification_issues)}件が要確認"
        )

    description = "\n\n".join(lines)
    if warnings:
        description += "\n\n" + "\n".join(warnings)

    embed: dict = {
        "title": f"📰 {draft.date} の下書きができました",
        "description": description[:DESCRIPTION_LIMIT],
        "color": 0xE86A6A if warnings else 0x3DDC97,
        "footer": {
            "text": (
                f"X {len(draft.x_posts)}投稿 / Instagram カルーセル1件"
                + (f" / {draft.llm_backend}" if draft.llm_backend else "")
            )
        },
    }
    if url:
        embed["url"] = url

    content = f"今日の下書き → {url}" if url else "今日の下書きができました"
    return {"content": content, "embeds": [embed]}


def send_draft_notification(draft: Draft) -> bool:
    """下書き完成を通知する。送ったら True、未設定なら False。"""
    webhook = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook:
        log.info("DISCORD_WEBHOOK_URL が未設定のため通知をスキップします")
        return False
    try:
        response = httpx.post(webhook, json=_build_payload(draft), timeout=15)
        response.raise_for_status()
        return True
    except Exception as exc:
        # 通知の失敗で日次実行を落とさない。下書き自体は完成している。
        log.error("Discord への通知に失敗しました: %s", exc)
        return False


def _webhook() -> str:
    return os.environ.get("DISCORD_WEBHOOK_URL", "").strip()


def _post(
    webhook: str,
    payload: dict,
    attachments: list[Path] | None = None,
) -> bool:
    """Webhook に1メッセージ送る。画像があれば multipart で添付する。"""
    try:
        if attachments:
            files = {}
            described = []
            for index, path in enumerate(attachments[:ATTACHMENT_LIMIT]):
                files[f"files[{index}]"] = (
                    path.name,
                    path.read_bytes(),
                    "image/jpeg",
                )
                described.append({"id": index, "filename": path.name})
            payload = {**payload, "attachments": described}
            response = httpx.post(
                webhook,
                data={"payload_json": json.dumps(payload, ensure_ascii=False)},
                files=files,
                timeout=60,
            )
        else:
            response = httpx.post(webhook, json=payload, timeout=30)
        response.raise_for_status()
        return True
    except Exception as exc:
        detail = getattr(getattr(exc, "response", None), "text", "")
        log.error("Discord への送信に失敗しました: %s %s", exc, detail[:200])
        return False


def send_draft_package(draft: Draft, draft_dir: Path) -> bool:
    """下書き一式を Discord に流し込む。

    投稿1件につき1メッセージにする。スマホで長押し →「テキストをコピー」
    したときに、余計な見出しが混ざらず原稿だけが取れるようにするため。

    Args:
        draft: 対象の下書き
        draft_dir: 画像が置かれているディレクトリ（docs/d/<日付>）
    """
    webhook = _webhook()
    if not webhook:
        log.info("DISCORD_WEBHOOK_URL が未設定のため配信をスキップします")
        return False

    limit = load_settings().compose["x_max_weighted"]
    images = {card.article_id: draft_dir / card.path for card in draft.images}
    sent = 0

    def deliver(payload: dict, files: list[Path] | None = None) -> None:
        nonlocal sent
        if _post(webhook, payload, files):
            sent += 1
        time.sleep(SEND_INTERVAL)

    # ── 1. ヘッダ（その日の全体像と警告） ───────────────────────────
    deliver({"embeds": [_summary_embed(draft, limit)]})

    # ── 2. X 原稿（1本 = 1メッセージ） ──────────────────────────────
    for index, post in enumerate(draft.x_posts, start=1):
        item = draft.scored_by_id(post.article_id)
        weight = weighted_length(post.body)
        over = weight > limit
        issues = draft.verification_issues.get(post.article_id, [])

        fields = [
            {
                "name": "字数",
                "value": f"{weight}/{limit}" + ("  ⚠️超過" if over else "  ✓"),
                "inline": True,
            },
            {"name": "フック", "value": post.hook_type, "inline": True},
        ]
        if item is not None:
            fields.insert(
                0,
                {
                    "name": "区分",
                    "value": f"{'有名' if item.bucket == 'famous' else 'ニッチ'}"
                    f"・{item.assessment.category}",
                    "inline": True,
                },
            )
        if issues:
            fields.append(
                {
                    "name": "⚠️ 要確認（本文に根拠なし）",
                    "value": "\n".join(
                        f"[{i.kind}] {i.value}" for i in issues[:5]
                    )[:1000],
                    "inline": False,
                }
            )

        source = item.article.source_name if item else ""
        url = item.article.url if item else ""
        embed = {
            "title": f"X {index}/{len(draft.x_posts)}　↑この上の本文を長押ししてコピー",
            "description": f"[元記事を開く（{source}）]({url})" if url else "",
            "color": COLOR_WARN if (over or issues) else COLOR_OK,
            "fields": fields,
        }

        attachment = images.get(post.article_id)
        card = [attachment] if attachment and attachment.exists() else None
        # content に原稿だけを入れる。ここが「コピーして貼るだけ」の要。
        deliver({"content": post.body[:CONTENT_LIMIT], "embeds": [embed]}, card)

    # ── 3. Instagram のカルーセル画像 ───────────────────────────────
    carousel = [
        draft_dir / card.path
        for card in draft.images
        if card.platform == "instagram" and (draft_dir / card.path).exists()
    ]
    if carousel:
        deliver(
            {
                "embeds": [
                    {
                        "title": f"Instagram カルーセル（{len(carousel)}枚）",
                        "description": (
                            "画像を長押しして保存 → この順番でカルーセル投稿します。\n"
                            "キャプションは次のメッセージにあります。"
                        ),
                        "color": COLOR_INFO,
                    }
                ]
            },
            carousel,
        )

    # ── 4. Instagram のキャプション ─────────────────────────────────
    caption = draft.ig_caption.render()
    for part_index, chunk in enumerate(_split(caption, CONTENT_LIMIT), start=1):
        total_parts = -(-len(caption) // CONTENT_LIMIT)
        title = "Instagram キャプション　↑この上を長押ししてコピー"
        if total_parts > 1:
            title = f"Instagram キャプション ({part_index}/{total_parts})"
        deliver(
            {
                "content": chunk,
                "embeds": [{"title": title, "color": COLOR_INFO}],
            }
        )

    log.info("Discord に %d メッセージを送信しました", sent)
    return sent > 0


def _summary_embed(draft: Draft, limit: int) -> dict:
    """その日の全体像。ここだけ見れば何が来たか分かるようにする。"""
    lines = []
    for index, item in enumerate(draft.selected, start=1):
        mark = "🔥" if item.bucket == "famous" else "💎"
        lines.append(
            f"{mark} **{index}. {item.display_title[:56]}**\n"
            f"　{item.article.source_name}・{item.assessment.category}"
            f"・興味度 {item.assessment.interest}"
        )

    warnings = []
    if draft.fallback_reason:
        from .providers.claude_code import needs_relogin

        warnings.append(
            f"⚠️ 予備のLLM（{draft.llm_backend}）で生成。品質を念入りに確認してください"
        )
        if needs_relogin(draft.fallback_reason):
            # 放置すると毎日この状態が続き、品質が落ちたまま気づけない。
            # 具体的な対処をそのまま書く。
            warnings.append(
                "🔑 **Claude Code の再ログインが必要です。**"
                "ターミナルで `claude` を起動し `/login` を実行してください"
            )
    if over := [p for p in draft.x_posts if weighted_length(p.body) > limit]:
        warnings.append(f"⚠️ X原稿 {len(over)}件が字数超過（手直しが必要）")
    if draft.verification_issues:
        warnings.append(
            f"⚠️ ファクト照合で {len(draft.verification_issues)}件が要確認"
        )

    description = "\n\n".join(lines)
    if warnings:
        description += "\n\n" + "\n".join(warnings)
    if url := preview_url(draft.date):
        description += f"\n\n[プレビューページを開く]({url})"

    return {
        "title": f"📰 {draft.date} の下書き",
        "description": description[:DESCRIPTION_LIMIT],
        "color": COLOR_WARN if warnings else COLOR_OK,
        "footer": {
            "text": (
                f"X {len(draft.x_posts)}投稿 / Instagram カルーセル1件"
                + (f" / {draft.llm_backend}" if draft.llm_backend else "")
            )
        },
    }


def _split(text: str, size: int) -> list[str]:
    """長い本文を、なるべく改行位置で分割する。"""
    if len(text) <= size:
        return [text]
    parts: list[str] = []
    rest = text
    while len(rest) > size:
        cut = rest.rfind("\n", 0, size)
        if cut < size // 2:  # 適当な改行が無ければ文字数で切る
            cut = size
        parts.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip("\n")
    if rest:
        parts.append(rest)
    return parts


def send_text(message: str) -> bool:
    """任意のテキストを通知する（週次レポート等）。"""
    webhook = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook:
        return False
    try:
        httpx.post(webhook, json={"content": message[:1900]}, timeout=15).raise_for_status()
        return True
    except Exception as exc:
        log.error("Discord への通知に失敗しました: %s", exc)
        return False


def _main() -> int:
    """`python -m ainews.notify --file report.md` でファイルを通知する。

    ワークフローから使う。YAML の run ブロックに Python を直接埋め込むと
    インデントで壊れやすいので、モジュールとして呼べるようにしてある。
    """
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Discord にテキストを送る")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="送信するテキストファイル")
    group.add_argument("--message", help="送信する文字列")
    args = parser.parse_args()

    text = (
        Path(args.file).read_text(encoding="utf-8") if args.file else args.message
    )
    if send_text(text):
        print("送信しました")
        return 0
    print("送信をスキップしました（DISCORD_WEBHOOK_URL 未設定または失敗）")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
