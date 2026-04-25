# FX Auto-Trading Analysis Toolkit

FXと暗号資産の自動売買研究用のツールキット。Claude APIで市場分析と週次レビューを行う。**Phase 1-3 (分析・バックテスト・レビュー) のみ実装。実発注は未実装。**

詳細: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## セットアップ

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# .env に ANTHROPIC_API_KEY を設定
```

## 使い方

```bash
# 1回だけ分析 (Claudeが呼ばれる、ヘッドラインも自動取得)
python -m src.fx.cli analyze --symbol USDJPY=X --interval 1h

# Claudeなしで分析 (テクニカルのみ)
python -m src.fx.cli analyze --symbol USDJPY=X --no-llm

# ニュースだけ無効化
python -m src.fx.cli analyze --symbol USDJPY=X --no-news

# バックテスト (テクニカルのみ、180日分)
python -m src.fx.cli backtest --symbol USDJPY=X --interval 1h --period 180d

# 暗号資産も可
python -m src.fx.cli backtest --symbol BTC-USD --interval 1d --period 1y

# 定期分析ループ (15分ごと)
python -m src.fx.cli watch --symbol USDJPY=X --interval 15m --every 900

# 週次セルフレビュー (過去のトレードをClaudeに分析させる)
python -m src.fx.cli review --limit 50

# 経済カレンダーのひな形を作る (data/events.json)
python -m src.fx.cli calendar-seed

# 分析 + リスクプラン + PaperBrokerに発注 (dry-run)
python -m src.fx.cli trade --symbol USDJPY=X --dry-run

# PaperBrokerに実発注 (メモリ内)
python -m src.fx.cli trade --symbol USDJPY=X --capital 10000 --risk-pct 0.01

# OANDA demo発注 (3重の安全装置をクリアする必要あり、本体コード参照)
export OANDA_API_KEY=... OANDA_ACCOUNT_ID=... OANDA_ENV=practice OANDA_ALLOW_LIVE_ORDERS=yes
python -m src.fx.cli trade --broker oanda --confirm-demo --symbol USDJPY=X
```

## 学習ループの回し方

```bash
# 1. 普段のanalyzeでClaudeに反証可能な予測を保存(自動)
python -m src.fx.cli analyze --symbol USDJPY=X

# 2. 数時間/数日後、期日経過した予測を実価格と照合
python -m src.fx.cli evaluate

# 3. WRONGになった予測にClaudeを通して根本原因分析 + 改善ルール提案
python -m src.fx.cli postmortem

# 4. 蓄積した教訓を確認(カテゴリ別集計 + 該当シンボルの最近のレッスン)
python -m src.fx.cli lessons --symbol USDJPY=X

# 次回 analyze からは過去のWRONGケースが自動的にプロンプトに注入される
# → Claudeが「以前同じ状況でXが原因で間違えた」を読んで判断に反映
```

cronで `evaluate` と `postmortem` を1日1回回せば、放置していても学習データが蓄積されていく。

## Webダッシュボード

ブラウザでBotの状態と判断プロセスを見える化:

### 一番簡単な起動方法 (ダブルクリック)

| OS | ファイル | 初回だけ必要な手順 |
|----|---------|--------------------|
| macOS | `start.command` | ターミナルで `chmod +x start.command` 一度だけ |
| Linux | `start.sh` | `chmod +x start.sh` 一度だけ |
| Windows | `start.bat` | なし |

ダブルクリックで:
1. 必要パッケージを自動インストール (初回だけ時間かかる)
2. 空きポート (5000〜5010 のうち最初に空いてるもの) で起動
3. ブラウザを自動オープン

`ANTHROPIC_API_KEY` が無くてもテクニカル分析だけは動きます。

### コマンドラインから

```bash
python launch.py     # 上と同じ動作
python sample.py     # Flask標準のdebugモード
gunicorn sample:app  # 本番デプロイ向け
```

ページ:
- `/` — Dashboard: 累計分析数、予測の正解率、根本原因の分布、最近の判断履歴
- `/analyze` — Analyze: フォームから実行ボタンを押すと、SSE で**パイプラインを1ステップずつ実況**:
  データ取得 → テクニカル → ATR → ニュース → 相関 → イベント → 過去レッスン → Claude → 合意ルール → リスクプラン → 保存
- `/analysis/<id>` — 各分析の詳細(スナップショット、Claudeの理由、反証可能予測、評価結果)
- `/predictions` — 予測一覧、ステータスでフィルタ
- `/lessons` — 蓄積した教訓(根本原因の棒グラフ + 改善ルール)

Webサーバから叩く analyze は `predictions` テーブルにも保存されるので、CLIの `fx evaluate` / `fx postmortem` の対象になります。

## テスト

```bash
pytest tests/ -v
```

## 構成

| モジュール | 役割 |
|-----------|------|
| `src/fx/data.py` | OHLCV取得 (yfinance) |
| `src/fx/indicators.py` | テクニカル指標 (SMA/EMA/RSI/MACD/BB) |
| `src/fx/news.py` | ヘッドライン取得 (yfinance、FX用は差し替え推奨) |
| `src/fx/correlation.py` | 関連通貨ペア/指数との相関分析 |
| `src/fx/calendar.py` | 経済指標カレンダー (data/events.json) |
| `src/fx/risk.py` | ATR/Kelly/ポジションサイジング |
| `src/fx/analyst.py` | Claude API分析 (Opus 4.7 + adaptive thinking + prompt caching、テクニカル+ファンダメンタル+相関+イベント+過去レッスン注入) |
| `src/fx/strategy.py` | テクニカル×LLMの合意ルール |
| `src/fx/prediction.py` | 予測の評価ロジック(CORRECT/PARTIAL/WRONG/INCONCLUSIVE) |
| `src/fx/postmortem.py` | WRONG予測の根本原因分析 + 改善ルール提案(Claude) |
| `src/fx/broker.py` | ブローカー抽象 + PaperBroker (メモリ内発注) |
| `src/fx/oanda.py` | OANDA demo口座スキャフォールド (3重の安全装置付き) |
| `src/fx/backtest.py` | イベント駆動型バックテスト |
| `src/fx/storage.py` | SQLite履歴管理 (analyses/trades/predictions/postmortems/backtest_runs) |
| `src/fx/cli.py` | CLIエントリポイント |
| `src/fx/web/` | Flaskダッシュボード(SSEで判断プロセスを実況) |
| `sample.py` | gunicornエントリ(`gunicorn sample:app`) |

## リスクに関する注意

- 本リポジトリは**研究・バックテスト用途**。実資金の取引には使用しないこと。
- Phase 4 (デモ口座発注) に進む前に、6ヶ月以上のバックテストで PF > 1.3、最大DD < 20% を確認すること。
- FX自動売買は国ごとに法規制あり。日本ではレバレッジ上限や自主規制あり。
