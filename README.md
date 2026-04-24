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
```

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
| `src/fx/analyst.py` | Claude API分析 (Opus 4.7 + adaptive thinking + prompt caching、テクニカル+ファンダメンタル) |
| `src/fx/strategy.py` | テクニカル×LLMの合意ルール |
| `src/fx/backtest.py` | イベント駆動型バックテスト |
| `src/fx/storage.py` | SQLite履歴管理 |
| `src/fx/cli.py` | CLIエントリポイント |

## リスクに関する注意

- 本リポジトリは**研究・バックテスト用途**。実資金の取引には使用しないこと。
- Phase 4 (デモ口座発注) に進む前に、6ヶ月以上のバックテストで PF > 1.3、最大DD < 20% を確認すること。
- FX自動売買は国ごとに法規制あり。日本ではレバレッジ上限や自主規制あり。
