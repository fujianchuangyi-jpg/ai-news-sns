"""Discord への完成通知。

毎朝、下書きができたことをスマホにプッシュする。ここを押せば
プレビューが開き、そのままコピペ投稿に入れる、という導線を作る。

Webhook URL が未設定なら黙ってスキップする（ローカル実行の邪魔をしない）。
"""

from __future__ import annotations

import logging
import os

import httpx

from .compose import weighted_length
from .config import load_settings
from .models import Draft
from .render import preview_url

log = logging.getLogger(__name__)

# Discord embed の説明欄の上限
DESCRIPTION_LIMIT = 4000


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
        warnings.append(
            f"⚠️ 予備のLLM（{draft.llm_backend}）で生成。品質を念入りに確認してください"
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
