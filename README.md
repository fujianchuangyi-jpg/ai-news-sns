# AIニュース SNS 下書き生成システム

毎朝AIニュースを4本選び、Instagram と X の**投稿原稿とカード画像を自動生成**します。
人がやるのは「確認して投稿ボタンを押す」ことだけです。

- ニュースの取捨選択は **有名2本 : ニッチ2本**（設定で変更可）
- カード画像は **元記事のOGP画像＋見出し合成**（出典・日付を焼き込む引用の体裁）
- X は **URLを含めない**（X APIはURL付き投稿が13倍高いため。出典は媒体名で表記）
- 投稿は手動でも、**実績の収集と分析は完全自動**
- **運用コストは $0**（Claude Code の契約枠 ＋ ローカルLLM ＋ GitHub Pages）

---

## セットアップ

### 1. 依存のインストール

```bash
cd ~/dev/ai-news-sns
uv sync
```

### 2. LLM バックエンドを用意

既定は **Claude Code（契約枠内）＋ Ollama（ローカル）** の組み合わせで、
追加のAPI課金は発生しません。

```bash
# Claude Code にログイン済みであること
claude --version

# 一次選抜とフォールバックに使うローカルモデル
ollama pull gemma4-ja
ollama serve            # 起動していなければ

export DISCORD_WEBHOOK_URL=https://...     # 任意（完成通知）
```

従量課金の Anthropic API を使いたい場合のみ:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv run ainews daily --backend anthropic
```

何も用意せず配線だけ確認するなら `--fake-llm` を付けます。

### 3. アカウント情報を設定

`config/settings.yaml` の `account` を編集します。

```yaml
account:
  name: "AI NEWS DAILY"
  handle: "@your_handle"
  site_base_url: "https://<ユーザー名>.github.io/ai-news-sns"
```

---

## 使い方

`~/.local/bin/ainews` にリンクしてあるので、どこからでも叩けます。

```bash
ainews                 # 今日のプレビューをブラウザで開く（引数なしの既定）
ainews run             # 収集から Discord 配信まで通しで実行
ainews notify          # 下書きを Discord に配信し直す
```

プロジェクト内で細かく回す場合:

```bash
# 収集だけ試す（LLMを呼ばない）
uv run ainews collect --dry-run

# 一次選抜の結果を確認（60件→20件の足切り内訳）
uv run ainews collect --explain

# 収集〜原稿生成。--explain で全候補のスコア内訳を表示
uv run ainews daily --explain

# カード画像を生成
uv run ainews images

# プレビューサイトを生成して開く
uv run ainews render
open docs/d/$(date +%F)/index.html

# 上記を通しで実行（GitHub Actions が毎朝叩くのはこれ）
uv run ainews run

# バックエンドを切り替えて比較する
uv run ainews daily --backend ollama      --explain   # 全ローカル
uv run ainews daily --backend claude_code --explain   # 本番構成
uv run ainews daily --backend claude_code --no-fallback  # 退避なしで素の挙動を見る

# LLM無しで全工程を確認
uv run ainews run --fake-llm
```

### 毎朝の運用（Discord で完結）

1. 6:00 JST に Mac の launchd が実行し、**Discord に下書き一式が届く**
2. スマホの Discord を開く。届くのは7メッセージ:

   | | 内容 |
   |---|---|
   | 1 | その日の全体像（4本の見出し・警告） |
   | 2〜5 | X原稿1本ずつ（**本文のみ**＋カード画像1枚） |
   | 6 | Instagram カルーセル画像6枚 |
   | 7 | Instagram キャプション（**本文のみ**） |

3. **X**: メッセージを長押し →「テキストをコピー」→ 画像を長押しして保存 →
   X アプリに貼り付けて投稿
4. **Instagram**: 画像6枚を保存 → キャプションを長押しコピー → カルーセル投稿
5. 翌日、`analytics` が実績を自動収集して下書きと突き合わせます

> 原稿のメッセージには**本文しか入れていません**。Discord の「テキストをコピー」は
> メッセージ本文だけを拾い、番号や字数などの情報（埋め込み部分）は含まれません。
> つまり長押し1回で、そのまま貼れる状態のテキストが手に入ります。

プレビューページ（`ainews` で開く）も引き続き使えます。PC で作業するとき向けです。

---

## 仕組み

```
collect    ニュース収集（RSS / HTML / Hacker News / Reddit / HF Papers）
   ↓
cluster    同一ニュースの束ね ＋ 過去30日の既出除外
   ↓
prefilter  ローカルLLM(Ollama)で一次選抜 60件→20件   ← 無料・無制限
   ↓
extract    本文抽出（trafilatura）と OGP 画像URL取得
   ↓
select     機械シグナル ＋ LLM評価 → 有名2 : ニッチ2   ← Claude Code 1回目
   ↓
compose    X原稿4本 ＋ IGキャプション（同一呼び出し）  ← Claude Code 2回目
   ↓
verify   ファクト照合（本文にない数値・固有名詞を検出）
   ↓
imagegen カード画像生成（IG 1080×1350 ×6 / X 1600×900 ×4）
   ↓
render   プレビューサイト（docs/）
   ↓
notify   Discord 通知
```

### なぜ2段構えなのか

`claude -p` は**1回あたり約25,000トークンの固定オーバーヘッド**を伴います
（Claude Code 自身のシステムプロンプトが毎回読み込まれるため）。
候補60件を刻んで評価すると、それだけで12万トークンが消えます。

そこで「明らかに不要な記事を落とす」という、判断力をあまり要さない仕事を
ローカルの Ollama に任せ、**Claude Code の呼び出しは1日2回**に抑えています。
選定精度が最も効く工程に、限られた呼び出しを集中させる設計です。

X原稿とIGキャプションを1回にまとめているのも同じ理由です。分けて呼ぶと
同じ記事本文（約8,000トークン）を2回送ることになります。

### フォールバック

Claude Code が使えない日（利用上限・未ログイン・オフライン）は、
Ollama が自動的に引き継いで下書きを完成させます。投稿は止まりません。
その場合はプレビューと Discord に警告が出るので、原稿を念入りに確認してください。

### 選定ロジック

「有名度」は LLM 単体だと日によって基準が揺れるため、客観指標と混ぜています。

```
fame_final = 0.6 × LLM評価 + 0.4 × 機械シグナル

機械シグナル = 50点 何社が報じたか（対数）
             + 30点 媒体の格（公式 > 大手 > ニッチ）
             + 20点 HN/Reddit のスコア（対数）
```

`fame_final >= 60` を有名バケット、それ未満をニッチバケットとし、
各バケットから「興味度」上位を取ります。同じカテゴリは1日2本までです。

### 画像の権利ハンドリング

`config/sources.yaml` の `image_policy` でソースごとに制御します。

| 値 | 挙動 |
|---|---|
| `ogp_ok` | OGP画像を取得して合成（既定）。出典名・日付を必ず焼き込む |
| `official_only` | ベンダー公式・CC系のみ |
| `text_only` | 画像を使わず文字カードのみ（通信社・UGC向け） |

OGP画像の取得に失敗した場合や、画像が小さすぎる（ロゴのみ等）場合も
自動で文字カードにフォールバックします。画像が無くて投稿が落ちることはありません。

---

## 設定

すべて `config/` 配下で完結します。コード変更は不要です。

| ファイル | 内容 |
|---|---|
| `config/settings.yaml` | 選定比率、字数上限、画像サイズ、配色、モデル設定 |
| `config/sources.yaml` | ニュースソース、AIキーワード、除外キーワード |
| `config/prompts/*.md` | LLMプロンプト（評価・X原稿・IGキャプション） |

ソースを1つ止めたいときは `enabled: false` を付けます。

---

## 定時実行（launchd）

本番の定時実行は **Mac の launchd** です。GitHub Actions ではありません。
Claude Code の認証情報をクラウドCIに持ち出さず、ローカルで公式CLIを
そのまま呼ぶためです（Ollama もローカルにしかありません）。

```bash
cp scripts/com.ainews.daily.plist ~/Library/LaunchAgents/
sed -i '' "s|__PROJECT_DIR__|$HOME/dev/ai-news-sns|g" \
    ~/Library/LaunchAgents/com.ainews.daily.plist
launchctl load ~/Library/LaunchAgents/com.ainews.daily.plist

# 即時テスト
launchctl start com.ainews.daily
tail -f data/daily.log
```

Mac がスリープしていた場合、launchd は復帰後に遅れて実行します。
数時間ずれても下書きは残るので運用は破綻しません。

Discord の Webhook などは `.env`（gitignore 済み）に置くと読み込まれます。

```bash
echo 'DISCORD_WEBHOOK_URL=https://...' > .env
```

### GitHub Pages

リポジトリを public にし、Settings → Pages で `main` ブランチの `/docs` を
公開すると、プレビューがスマホから開けるようになります。
`scripts/run_daily.sh` が生成物を自動で push します。

### GitHub Actions（バックアップ経路）

| ワークフロー | 実行 | 内容 |
|---|---|---|
| `daily.yml` | 手動のみ | Mac が長期間使えないときの逃げ道。`ANTHROPIC_API_KEY`（従量課金）が必要 |
| `analytics.yml` | 毎晩 23:00 JST | 実績収集（日曜のみ週次レポート）。LLM をほぼ使わないのでクラウドで回す |

Actions 用の Secrets:

| Secret | 用途 |
|---|---|
| `DISCORD_WEBHOOK_URL` | 下書き配信・失敗通知 |
| `IG_USER_ID` / `META_ACCESS_TOKEN` | Instagram 実績収集 |
| `X_BEARER_TOKEN` / `X_USER_ID` | X 実績収集（CSV取り込みで代替可） |
| `ANTHROPIC_API_KEY` | バックアップ経路を使うときだけ |

---

## 運用コスト

| 項目 | 月額 |
|---|---|
| LLM（Claude Code の契約枠 ＋ ローカル Ollama） | **$0** |
| GitHub Actions + Pages（publicリポジトリ） | $0 |
| Instagram Graph API | $0 |
| X 実績取得（CSV取り込みを使う場合） | $0 |
| **合計** | **$0** |

X API で実績を自動取得する場合のみ約 $0.12/月。
`--import-csv` で analytics.x.com の書き出しを取り込めば $0 のままです。

### Max枠の消費について

1日2回 × 約25,000トークンの固定費に、記事本文が上乗せされます。
運用開始後しばらくは Claude Code の `/status` で消費を観察してください。
上限に近づくようなら `config/settings.yaml` の `llm.claude_code.model` を
`sonnet` に下げると消費を抑えられます。

---

## テスト

```bash
uv run pytest -q
```

字数計算・クラスタリング・既出除外・ファクト照合・選定ロジックに加えて、
プロバイダ層（スキーマ指示・エラー分類・フォールバック）と一次選抜を検証します。
外部（Ollama サーバ・claude コマンド）は呼びません。

各バックエンドの単体確認:

```bash
uv run python -m ainews.providers.ollama --selftest       # スキーマ強制と件数欠落
uv run python -m ainews.providers.claude_code --selftest  # stdin入力とJSON再パース
```

---

## 実績分析

投稿は手動でも、実績収集は自動です。Instagram も X も「自分の投稿一覧」を
API で取得でき、本文の先頭一致で下書きと突き合わせます。

```bash
uv run ainews analytics                    # 実績を収集
uv run ainews analytics --report           # 週次レポート
uv run ainews analytics --import-csv x.csv # X の CSV 書き出しを取り込む
```

下書き段階でカテゴリ・バケット・フック型を記録してあるので、
「ニッチとメジャーどちらが伸びるか」「どのフックが効くか」を後から分析できます。

---

## 将来の自動投稿

`src/ainews/publish/` に投稿モジュールを追加すると、承認後の自動投稿に移行できます。
Instagram Graph API は無料で、`docs/` に置いた画像がそのまま公開URLとして使えるため、
インフラの追加は不要です。
