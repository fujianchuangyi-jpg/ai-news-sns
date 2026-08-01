"""コマンドラインインタフェース。

各工程を単体で回せるようにしてある。原稿だけ作り直す、画像だけ作り直す、
プレビューだけ再生成する、が開発中も運用中も頻繁に必要になるため。

    ainews collect  --dry-run     収集だけ試す（LLM を呼ばない）
    ainews daily    --explain     収集〜原稿生成。スコア内訳も表示
    ainews images   --date ...    保存済み下書きから画像を生成
    ainews render                 プレビューサイトを生成
    ainews notify                 Discord に通知
    ainews run                    上記を通しで実行（GitHub Actions 用）
    ainews analytics              実績を収集して分析

LLM を呼ぶコマンドは --fake-llm でスタブに差し替えられる。
API キーが無い状態でも配線とレイアウトを確認できる。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import store
from .config import docs_dir, load_settings
from .llm import LLM, api_key_available


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname).1s %(message)s",
    )
    # httpx のリクエストログは量が多いので抑える
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _make_llm(args: argparse.Namespace) -> LLM:
    if getattr(args, "fake_llm", False):
        from .fakes import FakeLLM

        print("※ フェイクLLMを使用中（原稿の品質は評価できません）\n")
        return FakeLLM()  # type: ignore[return-value]
    if not api_key_available():
        print(
            "ANTHROPIC_API_KEY が設定されていません。\n"
            "  実キーで動かす: export ANTHROPIC_API_KEY=sk-ant-...\n"
            "  配線だけ確認する: --fake-llm を付けて再実行",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return LLM()


def _draft_dir(date: str) -> Path:
    """その日の生成物を置くディレクトリ（GitHub Pages 配下）。"""
    return docs_dir() / "d" / date


# ── 各コマンド ────────────────────────────────────────────────────────


def cmd_collect(args: argparse.Namespace) -> int:
    from .cluster import cluster_summary
    from .collect import format_summary
    from .pipeline import collect_stage, prepare_candidates, today_jst

    with store.connect() as conn:
        articles, results = collect_stage(conn)
        print(f"── 収集（{args.date or today_jst()}）")
        print(format_summary(results))
        print(f"\n  記事 {len(articles)} 件")
        print("\n── 複数社が報じた話題")
        print(cluster_summary(articles))

        if not args.dry_run:
            candidates, dropped = prepare_candidates(conn, articles)
            from .extract import enrichment_summary

            print(f"\n── 候補（既出 {dropped} 件を除外）")
            print(f"  {len(candidates)} 件")
            print(enrichment_summary(candidates))
            for article in candidates[:15]:
                print(f"  [{article.source_name}] {article.title[:60]}")
    return 0


def cmd_daily(args: argparse.Namespace) -> int:
    from .pipeline import run_daily

    report = run_daily(date=args.date, llm=_make_llm(args), explain=args.explain)
    print(report.render(explain=args.explain))
    return 0 if report.draft else 1


def cmd_images(args: argparse.Namespace) -> int:
    from .imagegen import ImageBuilder, image_summary
    from .pipeline import today_jst

    date = args.date or today_jst()
    with store.connect() as conn:
        draft = store.load_draft(conn, date)
        if draft is None:
            print(f"{date} の下書きがありません。先に daily を実行してください", file=sys.stderr)
            return 1
        cards = ImageBuilder(_draft_dir(date)).build(draft)
        draft.images = cards
        store.save_draft(conn, draft)

    print(f"── 画像生成（{date}）")
    print(image_summary(cards))
    print(f"  出力先: {_draft_dir(date)}")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    from .pipeline import today_jst
    from .render import build_index, render_preview

    date = args.date or today_jst()
    with store.connect() as conn:
        draft = store.load_draft(conn, date)
        if draft is None:
            print(f"{date} の下書きがありません", file=sys.stderr)
            return 1
        path = render_preview(draft, _draft_dir(date))
        build_index(conn, docs_dir())

    print(f"── プレビュー生成\n  {path}")
    if url := load_settings().account.get("site_base_url"):
        print(f"  公開URL: {url.rstrip('/')}/d/{date}/")
    return 0


def cmd_notify(args: argparse.Namespace) -> int:
    from .notify import send_draft_notification
    from .pipeline import today_jst

    date = args.date or today_jst()
    with store.connect() as conn:
        draft = store.load_draft(conn, date)
    if draft is None:
        print(f"{date} の下書きがありません", file=sys.stderr)
        return 1
    ok = send_draft_notification(draft)
    print("── 通知を送信しました" if ok else "── 通知はスキップされました（Webhook 未設定）")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """日次の通し実行。GitHub Actions から呼ぶ。"""
    for step in (cmd_daily, cmd_images, cmd_render, cmd_notify):
        code = step(args)
        if code != 0:
            return code
        print()
    return 0


def cmd_analytics(args: argparse.Namespace) -> int:
    from .analytics import collect_metrics, weekly_report

    if args.report:
        with store.connect() as conn:
            print(weekly_report(conn, llm=_make_llm(args) if args.summarize else None))
        return 0

    with store.connect() as conn:
        summary = collect_metrics(conn, csv_path=args.import_csv)
    print(summary)
    return 0


# ── パーサ ────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ainews", description="AIニュースSNS 下書き生成パイプライン"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="詳細ログ")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_text: str, func) -> argparse.ArgumentParser:
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--date", help="対象日 (YYYY-MM-DD)。既定は今日(JST)")
        sub.set_defaults(func=func)
        return sub

    collect = add("collect", "ニュースを収集する", cmd_collect)
    collect.add_argument(
        "--dry-run", action="store_true", help="本文抽出をせず収集結果だけ見る"
    )

    daily = add("daily", "収集から原稿生成まで実行する", cmd_daily)
    daily.add_argument("--fake-llm", action="store_true", help="LLMをスタブに差し替える")
    daily.add_argument("--explain", action="store_true", help="スコア内訳を表示する")

    add("images", "下書きからカード画像を生成する", cmd_images)
    add("render", "プレビューサイトを生成する", cmd_render)
    add("notify", "Discord に下書き完成を通知する", cmd_notify)

    run = add("run", "daily→images→render→notify を通しで実行する", cmd_run)
    run.add_argument("--fake-llm", action="store_true", help="LLMをスタブに差し替える")
    run.add_argument("--explain", action="store_true", help="スコア内訳を表示する")

    analytics = add("analytics", "実績を収集・分析する", cmd_analytics)
    analytics.add_argument("--report", action="store_true", help="週次レポートを出力する")
    analytics.add_argument("--summarize", action="store_true", help="レポートをLLMで要約する")
    analytics.add_argument("--fake-llm", action="store_true", help="LLMをスタブに差し替える")
    analytics.add_argument("--import-csv", help="X analytics の CSV を取り込む")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
