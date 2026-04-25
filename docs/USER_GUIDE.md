# ユーザーガイド

このアプリの**操作方法**を順を追ってまとめたドキュメント。
設計の詳細は [`DESIGN.md`](DESIGN.md)、アーキテクチャ概要は
[`ARCHITECTURE.md`](ARCHITECTURE.md) を参照。

---

## 1. アプリの全体像

このアプリは大きく **3つのレイヤー** で動きます。

| レイヤー | 何をする |
|----------|---------|
| **CLI** (`python -m src.fx.cli ...`) | バックテスト、analyze、evaluate、postmortem、sentiment収集、trade(発注) |
| **Webダッシュボード** (`http://localhost:5000`) | 判断プロセスの可視化、履歴閲覧、教訓の確認 |
| **LINE Bot** (オプション) | 新シグナル / 評価結果 / 教訓を Push、コマンドで対話 |

データは全部 `data/fx.db` (SQLite) と `data/sentiment.json` に貯まります。

---

## 2. 最初の起動 (5分)

### 必要なもの

- Python 3.10 以上
- `pip` が動くこと
- (推奨) `ANTHROPIC_API_KEY` — Anthropic Console で取得
- (オプション) LINE Messaging API — 通知が欲しいなら

### インストール

```bash
git clone <this-repo>
cd test
pip install -r requirements.txt
```

### 環境変数 (`.env` でも、shell rc でも、cron で `Environment=` でも)

```bash
# 必須(LLMを使うなら)
export ANTHROPIC_API_KEY=sk-ant-...

# 任意の上書き
export FX_MODEL=claude-opus-4-7      # デフォルト
export FX_EFFORT=medium               # low | medium | high | xhigh | max
export FX_DB_PATH=data/fx.db

# LINE Bot を使うなら
export LINE_CHANNEL_ACCESS_TOKEN=...
export LINE_CHANNEL_SECRET=...

# 通知バックエンドを明示したい場合(自動判定で十分なことが多い)
export NOTIFY_BACKEND=line   # "line" | "log" | "null"
```

### 起動 (ダブルクリック)

| OS | ファイル | 初回だけ必要な設定 |
|----|---------|--------------------|
| macOS | `start.command` | `chmod +x start.command` (1回) |
| Linux | `start.sh` | `chmod +x start.sh` (1回) |
| Windows | `start.bat` | なし |

ダブルクリックすると:
1. 必要パッケージを自動インストール
2. 空きポート (5000-5010) を選んで Flask 起動
3. ブラウザを自動オープン

`ANTHROPIC_API_KEY` が無くてもテクニカル分析だけは動きます。

### 起動 (CLIから)

```bash
python launch.py        # ダブルクリックと同じ動作
python sample.py        # Flask debug モード
gunicorn sample:app     # 本番デプロイ向け
```

---

## 3. 5分で全部試す

```bash
# 1. バックテストで戦略の素性を即座に確認(API キー不要、ネットだけあればOK)
python -m src.fx.cli backtest --symbol USDJPY=X --period 1y

# 2. ダッシュボード起動して /analyze を1回押す(Claudeあり)
python launch.py
#   → http://localhost:5000/analyze で Run pipeline
#   → 11ステップが順番に光るのが見える

# 3. 群衆センチメントを集める
python -m src.fx.cli sentiment-refresh \
    --symbols USDJPY=X EURUSD=X BTC-USD

# 4. /sentiment ページでスコア確認
#   ブラウザの上ナビから Sentiment

# 5. (時間が経ってから) 期日経過した予測を採点
python -m src.fx.cli evaluate

# 6. WRONG 判定にClaude が原因分析
python -m src.fx.cli postmortem

# 7. 蓄積した教訓を確認
python -m src.fx.cli lessons --symbol USDJPY=X
#   /lessons ページでもグラフで見える
```

---

## 4. CLIコマンド一覧

### 分析・予測

| コマンド | 用途 | 主なオプション |
|---------|-----|----------------|
| `analyze` | 1回の分析 + 反証可能予測を保存 | `--symbol`, `--interval`, `--period`, `--no-llm`, `--no-news`, `--no-correlation`, `--no-events`, `--no-lessons`, `--no-sentiment` |
| `watch` | analyzeを定期実行 | 上記 + `--every <秒>` |
| `backtest` | 過去データで戦略を再生 | `--symbol`, `--period`, `--start`, `--end`, `--warmup` |

### 学習ループ

| コマンド | 用途 |
|---------|-----|
| `evaluate` | 期日経過した PENDING 予測を実価格と照合 → CORRECT/PARTIAL/WRONG/INCONCLUSIVE |
| `postmortem` | WRONG だった予測に Claude を通して原因分析 |
| `lessons` | 蓄積した教訓のサマリ + シンボル別最近のレッスン |
| `review` | 過去トレード全体を Claude にレビューさせる(週次) |

### 取引

| コマンド | 用途 |
|---------|-----|
| `trade` | analyze → リスクプラン → (任意で) 発注 |
| | `--broker paper` (デフォルト) または `--broker oanda --confirm-demo` (デモ口座のみ) |
| | `--dry-run` で発注せず計画だけ表示 |

### 補助

| コマンド | 用途 |
|---------|-----|
| `calendar-seed` | `data/events.json` のひな形を作る |
| `sentiment-refresh` | 全コレクター実行 → スコアリング → JSON保存 |
| `sentiment-show` | キャッシュ済みスナップショットを表示 |

各コマンドに `--help` を付けると詳細オプションが出ます。

---

## 5. Webダッシュボードの使い方

| URL | 役割 |
|-----|------|
| `/` | 全体ステータス: 累計analyses数、予測の正解率、根本原因の分布 |
| `/analyze` | **判断プロセス実況中継** (SSE で11ステップ順次表示) |
| `/analysis/<id>` | 個別分析: スナップショット全項目 / Claudeの理由 / 反証可能予測 / 評価結果 |
| `/predictions` | 予測一覧 (Status/Symbolフィルタ) |
| `/sentiment` | 群衆センチメント: 各シンボルのスコア・ソース別内訳・注目投稿 |
| `/lessons` | 蓄積した教訓 (根本原因の棒グラフ + 改善ルール本文) |

**`/analyze` の見方**: フォームで Symbol/Interval/Period を選んで「Run pipeline」を押すと、11個のステップカードが順次:

```
○ pending → ● ok (緑)  | ○ skip (灰) | ✕ error (赤)
```

各ステップに **何秒かかったか** と **何を取れたか** が出るので、
ボトルネックが目で見えます (例: ヘッドライン取得が4秒、Claudeが7秒、など)。

---

## 6. LINE Bot 通知 (オプション)

### LINE側のセットアップ

1. [LINE Developers Console](https://developers.line.biz/console/) で
   "Messaging API" のチャネルを作成
2. **チャネルアクセストークン**(長いやつ)と**チャネルシークレット**を控える
3. Webhook URLを設定: `https://<your-public-host>/webhook/line`
   - ローカル開発時は ngrok などで https を晒す
   - 本番なら Heroku/Render/Railway などにデプロイ
4. "Auto-reply messages" は OFF に(自前で reply するため)
5. ボットを友だち追加

### サーバー側

```bash
export LINE_CHANNEL_ACCESS_TOKEN="..."
export LINE_CHANNEL_SECRET="..."
python launch.py    # Webhook を待ち受ける
```

### Bot との対話

Bot に話しかけられるコマンド一覧:

| メッセージ | 動作 |
|-----------|-----|
| `subscribe` | 通知を受け取り始める |
| `unsubscribe` | 通知を止める |
| `status` | 累計analyses / 予測内訳 / 勝率 |
| `lessons [SYMBOL]` | 最近の教訓 |
| `predictions [STATUS]` | 最近の予測 (デフォルト PENDING) |
| `sentiment SYMBOL` | キャッシュ済みセンチメント |
| `analyze SYMBOL` | 分析を依頼 (バックグラウンド処理 → 完了時 Push) |
| `help` | コマンド一覧 |

スラッシュ(`/help` 等)もOK。

### 自動 Push される通知

| イベント | いつ | 形式 |
|---------|------|------|
| **新シグナル** | `analyze` で BUY/SELL かつ confidence ≥ 各購読者の閾値 | 🟢 USDJPY=X BUY (conf 0.75) ... |
| **判定結果** | `evaluate` で CORRECT/PARTIAL/WRONG が確定 | ✅ USDJPY=X BUY → CORRECT ... |
| **新しい教訓** | `postmortem` が WRONG を分析した時 | 🧠 New lesson [TREND_MISREAD] ... |
| **サマリ** | `evaluate` 実行後、何か変化があれば | 📊 Summary - evaluate predictions: ... |

各購読者ごとに `min_confidence` を設定可能(将来の機能;現状は 0.6 デフォルト)。

### LINE 無しでテストしたい時

```bash
export NOTIFY_BACKEND=log
python -m src.fx.cli analyze --symbol USDJPY=X
# → stderr に通知内容が出力される
```

---

## 7. 自動運用 (cron)

```cron
# 15分ごとに analyze (3シンボル)
*/15 * * * * cd /path/to/test && python -m src.fx.cli analyze --symbol USDJPY=X
*/15 * * * * cd /path/to/test && python -m src.fx.cli analyze --symbol EURUSD=X
*/15 * * * * cd /path/to/test && python -m src.fx.cli analyze --symbol BTC-USD

# 1時間ごとにセンチメント refresh
0 * * * * cd /path/to/test && python -m src.fx.cli sentiment-refresh

# 1日1回 evaluate と postmortem
0 1 * * * cd /path/to/test && python -m src.fx.cli evaluate
30 1 * * * cd /path/to/test && python -m src.fx.cli postmortem
```

> **コスト目安**: Claude Opus 4.7 で 15分 × 3シンボル × 24時間 = 288回/日 → 月 $5〜10。プロンプトキャッシュが効くのでもっと安い。Sentiment は Haiku なので無視できるレベル。

---

## 8. トラブルシューティング

| 症状 | 原因 / 対処 |
|------|------------|
| `ModuleNotFoundError: anthropic` | `pip install -r requirements.txt` |
| `ANTHROPIC_API_KEY not set` | shell で `export` するか `.env` に書く。テクニカルだけなら `--no-llm` |
| analyze が `No data returned for ...` | yfinance に一時的にレート制限されている。数分待つ |
| Webhook が `403 invalid signature` | `LINE_CHANNEL_SECRET` が違う、または webhook body が改変されている |
| Webhook が呼ばれない | LINE Console で Webhook URL を Verify、Webhook=ON、Auto-reply=OFF を確認 |
| センチメントが空 | `sentiment-refresh` を1回実行する。各ソースが全部失敗してる場合 `errors_by_source` を確認 |
| `OANDABroker requires confirm_demo=True` | 仕様。3重ロック解除のため `--confirm-demo` フラグ + `OANDA_ENV=practice` + `OANDA_ALLOW_LIVE_ORDERS=yes` をすべて満たす |

---

## 9. Decision Engine と Risk Gate

このアプリの最終 BUY/SELL/HOLD は **AI ではなく固定ルールの Decision Engine** が決めます。Claude は助言者であり、Risk Gate は AI が上書きできない強制 HOLD 条件の集合体です。詳しくは [`DESIGN.md`](DESIGN.md) §3 と [`AUDIT.md`](AUDIT.md)。

判断順序:

1. **Risk Gate** (AI 不可侵) — データ品質 / 経済イベント / スプレッド / 日次損失 / 連敗 / ルール検証
2. **波形・パターン確認** — 三山やダブルトップ等は **ネックライン割れ後のみ** SELL 候補
3. **上位足整合** — 上位足と逆行する setup は HOLD
4. **Risk-Reward フロア** — RR < 1.5 は HOLD
5. **AI 助言者チェック** — confidence < 0.6 は HOLD、テクニカルと不一致も HOLD
6. **発注承認**

ダッシュボードの `/analyze` ページで 1 ステップずつ可視化されます。

## 10. 安全に運用するためのチェックリスト

- [ ] バックテストで PF > 1.3 / 最大 DD < 20% を確認した
- [ ] ペーパートレード (`--broker paper`) で 1ヶ月以上動かした
- [ ] `lessons` の集計を見て、システムプロンプトを 1 回以上チューニングした
- [ ] OANDA demo の鍵を `.env` に入れて gitignore してある
- [ ] `git status` で `.env` が「Untracked」「Ignored」のどちらかになっている
- [ ] cron 実行時のログを `>> /var/log/fx.log 2>&1` に流して**過去ログを残している**

これらをクリアしてから初めて、デモ口座での発注を検討する流れを推奨します。
本番(リアルマネー)は **自己責任** で。
