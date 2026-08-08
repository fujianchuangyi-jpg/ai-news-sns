#!/bin/bash
# 日次実行のエントリポイント。launchd から呼ばれる。
#
# GitHub Actions ではなく Mac 上で動かしている理由:
#   Claude Code の認証情報を Mac のローカルに置いたまま、公式CLIを
#   そのまま呼ぶため。認証をクラウドCIに持ち出さないことが、この構成の
#   要点になっている。Ollama もローカルにしか無い。
#
# 成果物は GitHub に push し、GitHub Pages がプレビューを配信する
# （スマホから開けるようにするため）。

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

LOG_DIR="$PROJECT_DIR/data"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/daily.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# launchd の PATH は極端に狭い（/usr/bin:/bin:/usr/sbin:/sbin のみ）。
# uv / claude / ollama はいずれもここに入っていないので明示的に足す。
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

# 個人設定（Discord Webhook など）があれば読む。認証情報を
# リポジトリに置かずに済ませるための逃げ道。
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$PROJECT_DIR/.env"
    set +a
fi

# --check: 環境だけ確認して終わる（LLM を呼ばない）。
# launchd から起動したときに PATH や認証が正しく解決できるかを、
# 本番と同じ経路で確かめるための入口。launchd の PATH は極端に狭く、
# ここが原因で「毎朝静かに失敗し続ける」のが最もありがちな事故なので、
# 登録直後に必ずこれを通す。
CHECK_ONLY=false
[ "${1:-}" = "--check" ] && CHECK_ONLY=true

if $CHECK_ONLY; then
    log "───────── 環境チェック（launchd 経由）"
else
    log "───────── 日次実行を開始"
fi

log "  PATH=$PATH"

for cmd in uv claude ollama; do
    if command -v "$cmd" >/dev/null 2>&1; then
        log "  ✓ $cmd: $(command -v "$cmd")"
    else
        # claude が無ければ Ollama にフォールバックして続行できる。
        # uv が無いと何もできないのでここで止める。
        log "  ⚠ $cmd が見つかりません"
        [ "$cmd" = "uv" ] && { log "uv が必須です。中止します"; exit 1; }
    fi
done

if [ -n "${DISCORD_WEBHOOK_URL:-}" ]; then
    log "  ✓ DISCORD_WEBHOOK_URL: 設定済み"
else
    log "  ⚠ DISCORD_WEBHOOK_URL: 未設定（配信されません）"
fi

# Ollama が寝ていると一次選抜もフォールバックも動かないので起こす
if ! curl -s -m 3 http://localhost:11434/api/version >/dev/null 2>&1; then
    log "  Ollama が停止中。起動します"
    ollama serve >>"$LOG" 2>&1 &
    for _ in $(seq 1 30); do
        sleep 1
        curl -s -m 2 http://localhost:11434/api/version >/dev/null 2>&1 && break
    done
fi
if curl -s -m 3 http://localhost:11434/api/version >/dev/null 2>&1; then
    log "  ✓ Ollama: 応答あり"
else
    log "  ⚠ Ollama: 応答なし（一次選抜とフォールバックが使えません）"
fi

if $CHECK_ONLY; then
    log "───────── チェック完了（本番実行はしていません）"
    exit 0
fi

log "下書きを生成中（収集 → 一次選抜 → 選定 → 原稿 → 画像 → プレビュー）"
if uv run ainews run >>"$LOG" 2>&1; then
    log "✓ 下書きの生成が完了"
else
    STATUS=$?
    log "✗ 下書きの生成に失敗 (exit=$STATUS)"
    if [ -n "${DISCORD_WEBHOOK_URL:-}" ]; then
        curl -sS -H "Content-Type: application/json" \
            -d "{\"content\":\"⚠️ 日次下書きの生成に失敗しました。ログ: $LOG\"}" \
            "$DISCORD_WEBHOOK_URL" >/dev/null 2>&1
    fi
    exit "$STATUS"
fi

# 成果物を GitHub へ。push できなくてもローカルには残るので致命傷ではない。
if git rev-parse --git-dir >/dev/null 2>&1; then
    git add docs data/ainews.db 2>/dev/null
    if git diff --staged --quiet; then
        log "変更なし（コミットをスキップ）"
    else
        git -c user.name="ainews-bot" -c user.email="ainews-bot@local" \
            commit -q -m "下書き $(date +%Y-%m-%d)" && log "✓ コミット"
        if git push -q 2>>"$LOG"; then
            log "✓ push 完了（GitHub Pages に反映されます）"
        else
            log "⚠ push に失敗（ローカルには保存済み。手動で push してください）"
        fi
    fi
fi

log "───────── 完了"
