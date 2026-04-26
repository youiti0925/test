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

### バックテスト二系統

| コマンド | 用途 |
|---------|-----|
| `backtest` | 過去データを **テクニカル素** で再生 (ベースライン比較用) |
| `backtest-engine` | 過去データを **本番Decision Engine + Risk Gate** で再生 (spec §4) |

`backtest-engine` の出力には `hold_reasons` が含まれ、ゲートが何回・
何で止めたかが集計されます。普段の判断はこちらを基準に。

### 研究 (Phase B〜D)

| コマンド | 用途 |
|---------|-----|
| `build-timeline` | OHLCV + テクニカル + パターン + 上位足 + イベント + マクロ + decision を **point-in-time** で1表に (spec §5) |
| `event-overlay` | FOMC/BOJ/CPI/NFP 前後の値動き集計 (spec §6) |
| `waveform-build-library` | 過去波形をスライディングで JSONL に蓄積 (spec §7.2) |
| `waveform-match` | 現在波形 → 類似過去波形 → 方向バイアス (spec §7.3) |
| `attribution-report` | 大きい move の **原因候補** を closed taxonomy でランキング (spec §8) |
| `calibrate` | パラメータ走査 → **安定性スコアで** ランキング (spec §10) |
| `compare-external` | 外部ベンダー CSV と engine 予測を突き合わせ (spec §9) |

### 学習ループ

| コマンド | 用途 |
|---------|-----|
| `evaluate` | 期日経過した PENDING 予測を実価格と照合 → CORRECT/PARTIAL/WRONG/INCONCLUSIVE |
| `postmortem` | WRONG だった予測に Claude を通して原因分析 (`blocked_by`/`rule_chain`/`pattern`/`event_risk`/`spread`/`risk_reward` を `context_json` に保存) |
| `lessons` | 蓄積した教訓のサマリ + シンボル別最近のレッスン |
| `review` | 過去トレード全体を Claude にレビューさせる(週次) |

### 取引

| コマンド | 用途 |
|---------|-----|
| `trade` | analyze → Decision Engine → リスクプラン → (任意で) 発注 |
| | `--broker paper` (デフォルト) または `--broker oanda --confirm-demo` (デモ口座のみ) |
| | `--dry-run` で発注せず計画だけ表示 |

OANDA 経路 (`--broker oanda`) では:
* 起動時に `data/events.json` の鮮度をチェックし、stale/欠落なら HOLD
* 発注前に Bid/Ask を取得 (取れなければ発注拒否)
* 約定後、実 fill 価格・スリッページ・レイテンシ・broker_order_id を
  `trades` テーブルに自動保存

### 補助

| コマンド | 用途 |
|---------|-----|
| `calendar-seed` | `data/events.json` のひな形を作る |
| `sentiment-refresh` | 全コレクター実行 → スコアリング → JSON保存 |
| `sentiment-show` | キャッシュ済みスナップショットを表示 |

各コマンドに `--help` を付けると詳細オプションが出ます。

### 典型的な研究ワークフロー

```bash
# 1. 経済カレンダーを seed (もしくは自前 JSON を data/events.json に置く)
python -m src.fx.cli calendar-seed

# 2. 本番ルールでバックテスト
python -m src.fx.cli backtest-engine --symbol USDJPY=X --interval 1h --period 180d

# 3. 同じ期間で hold_reasons を見て、何が止めてるか把握
#    (上の出力に metrics.hold_reasons がある)

# 4. パラメータをグリッド走査して安定性スコアで比較
python -m src.fx.cli calibrate --symbol USDJPY=X \
    --stop-atr 1.5 2.0 2.5 --tp-atr 2.5 3.0 3.5

# 5. 大きい move の原因候補ランキング
python -m src.fx.cli attribution-report --symbol USDJPY=X --top-n 10

# 6. CPI 前後の値動きアグリゲート
python -m src.fx.cli event-overlay --symbol USDJPY=X --event CPI --period 2y

# 7. 波形ライブラリを構築 → 現在波形を照合
python -m src.fx.cli waveform-build-library \
    --symbol USDJPY=X --period 2y --output data/waveforms/lib.jsonl
python -m src.fx.cli waveform-match \
    --symbol USDJPY=X --library data/waveforms/lib.jsonl --horizon 24
```

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

- [ ] **`backtest-engine` で** PF > 1.3 / 最大 DD < 20% を確認した
      (`backtest` 単独ではなく Decision Engine 経由で)
- [ ] `metrics.hold_reasons` を見て、ゲートが何を止めているか把握した
- [ ] `calibrate` で **安定性スコア** (`stability_score`) ベースに
      パラメータを選んだ (収益最大ではなく)
- [ ] ペーパートレード (`--broker paper`) で 1ヶ月以上動かした
- [ ] `lessons` の集計と `postmortem` の `context_json` を見て、
      システムプロンプトを 1 回以上チューニングした
- [ ] **`data/events.json` を毎日更新する仕組み** (cron + 自前
      フィード) を整えた (live broker は stale で発注しない)
- [ ] OANDA demo の鍵を `.env` に入れて gitignore してある
- [ ] `git status` で `.env` が「Untracked」「Ignored」のどちらかになっている
- [ ] cron 実行時のログを `>> /var/log/fx.log 2>&1` に流して**過去ログを残している**
- [ ] `trade --broker oanda --dry-run` で **fill オーディット**
      (Bid/Ask, slippage, latency) が想定通り出ることを確認した
- [ ] `compare-external` で外部ベンダーとの一致率を一度測った
      (盲目的に追従しないため)

これらをクリアしてから初めて、デモ口座での発注を検討する流れを推奨します。
本番(リアルマネー)は **自己責任** で。
