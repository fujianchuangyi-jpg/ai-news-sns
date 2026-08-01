# AIニュース SNS 下書き生成システム

毎朝AIニュースを4本選び、Instagram と X の**投稿原稿とカード画像を自動生成**します。
人がやるのは「確認して投稿ボタンを押す」ことだけです。

- ニュースの取捨選択は **有名2本 : ニッチ2本**（設定で変更可）
- カード画像は **元記事のOGP画像＋見出し合成**（出典・日付を焼き込む引用の体裁）
- X は **URLを含めない**（X APIはURL付き投稿が13倍高いため。出典は媒体名で表記）
- 投稿は手動でも、**実績の収集と分析は完全自動**

---

## セットアップ

### 1. 依存のインストール

```bash
cd ~/dev/ai-news-sns
uv sync
```

### 2. APIキーを設定

```bash
export ANTHROPIC_API_KEY=sk-ant-...        # 必須（原稿生成）
export DISCORD_WEBHOOK_URL=https://...     # 任意（完成通知）
```

キーが無くても `--fake-llm` を付ければ配線とレイアウトの確認はできます。

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

```bash
# 収集だけ試す（LLMを呼ばない・課金なし）
uv run ainews collect --dry-run

# 収集〜原稿生成。--explain で全候補のスコア内訳を表示
uv run ainews daily --explain

# カード画像を生成
uv run ainews images

# プレビューサイトを生成して開く
uv run ainews render
open docs/d/$(date +%F)/index.html

# 上記を通しで実行（GitHub Actions が毎朝叩くのはこれ）
uv run ainews run

# APIキー無しで全工程を確認
uv run ainews run --fake-llm
```

### 毎朝の運用

1. 6:00 JST に GitHub Actions が実行され、Discord に通知が届く
2. 通知のリンクからプレビューを開く（スマホ推奨）
3. X: 「本文をコピー」→「Xの投稿画面を開く」（本文入力済みで開きます）→ 画像を添付して投稿
4. Instagram: 「6枚まとめてダウンロード」→ キャプションをコピー → アプリでカルーセル投稿
5. 翌日、`analytics` が実績を自動収集して下書きと突き合わせます

---

## 仕組み

```
collect  ニュース収集（RSS / HTML / Hacker News / Reddit / HF Papers）
   ↓
cluster  同一ニュースの束ね ＋ 過去30日の既出除外
   ↓
extract  本文抽出（trafilatura）と OGP 画像URL取得
   ↓
select   機械シグナル ＋ LLM評価 → 有名2 : ニッチ2 を選定
   ↓
compose  X原稿4本 ＋ IGキャプション生成（字数超過は自動で書き直し）
   ↓
verify   ファクト照合（本文にない数値・固有名詞を検出）
   ↓
imagegen カード画像生成（IG 1080×1350 ×6 / X 1600×900 ×4）
   ↓
render   プレビューサイト（docs/）
   ↓
notify   Discord 通知
```

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

## GitHub Actions

| ワークフロー | 実行時刻 | 内容 |
|---|---|---|
| `daily.yml` | 毎朝 6:00 JST | 下書き生成 → コミット → Discord通知 |
| `analytics.yml` | 毎晩 23:00 JST | 実績収集（日曜のみ週次レポート） |

必要な Secrets:

| Secret | 用途 | 必須 |
|---|---|---|
| `ANTHROPIC_API_KEY` | 原稿生成 | ✅ |
| `DISCORD_WEBHOOK_URL` | 完成通知・失敗通知 | 推奨 |
| `IG_USER_ID` / `META_ACCESS_TOKEN` | Instagram 実績収集 | 分析時 |
| `X_BEARER_TOKEN` / `X_USER_ID` | X 実績収集 | 分析時 |

リポジトリを public にし、Settings → Pages で `main` ブランチの `/docs` を
公開すると、プレビューがスマホから開けるようになります。

---

## 運用コスト

| 項目 | 月額 |
|---|---|
| Claude API（`claude-opus-5`、プロンプトキャッシュ適用） | 約 $10 |
| GitHub Actions + Pages（publicリポジトリ） | $0 |
| Instagram Graph API | $0 |
| X API（実績の読み取りのみ、月120件） | 約 $0.12 |

X の自動投稿に進む場合は +約 $2.4/月（URLなし方針のため）。

---

## テスト

```bash
uv run pytest -q
```

字数計算・クラスタリング・既出除外・ファクト照合・選定ロジックを検証します。
LLM は呼ばないので課金されません。

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
