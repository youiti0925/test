# FX Auto-Trading App — Architecture

## 目的

FX(および暗号資産)の自動売買を、Claude APIで市場分析・戦略レビューを行いながら実行するシステム。

このフェーズではまず **分析とバックテスト** に限定し、実発注は実装しない(安全のため)。

## システム構成

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           CLI (src/fx/cli.py)                            │
│  analyze │ trade │ backtest │ review │ watch │ calendar-seed             │
└──────┬──────┬────────┬─────────┬────────┬─────────┬──────────────────────┘
       │      │        │         │        │         │
       ▼      ▼        ▼         ▼        ▼         ▼
                        _gather_inputs()
                               │
    ┌──────────┬───────────────┼───────────────┬──────────┐
    ▼          ▼               ▼               ▼          ▼
┌────────┐ ┌──────────┐ ┌────────────┐ ┌──────────┐ ┌──────────┐
│ Data   │ │Indicators│ │Correlation │ │ Calendar │ │   News   │
│yfinance│ │SMA/RSI/+ │ │ multi-pair │ │ events   │ │headlines │
└────────┘ └────┬─────┘ └─────┬──────┘ └────┬─────┘ └────┬─────┘
                └────────┬────┴─────────────┴────────────┘
                         ▼
                  ┌──────────────┐
                  │   Analyst    │  Opus 4.7 + adaptive thinking
                  │  (Claude)    │  + prompt caching
                  └──────┬───────┘
                         │ TradeSignal (action, confidence, reason)
                         ▼
                 ┌───────────────┐
                 │   Strategy    │  Consensus rule
                 │   combine()   │
                 └──────┬────────┘
                        ▼
                 ┌──────────────┐
                 │     Risk     │  ATR stops, Kelly sizing, RiskPlan
                 │  plan_trade  │
                 └──────┬───────┘
                        ▼
              ┌─────────────────────┐
              │       Broker        │
              ├─────────────────────┤
              │  PaperBroker (test) │
              │  OANDABroker (demo) │
              └──────────┬──────────┘
                         ▼
                 ┌──────────────┐
                 │   Storage    │  SQLite
                 │              │  analyses / trades / backtests
                 └──────────────┘
```

## コンポーネント

### 1. Data層 (`src/fx/data.py`)
- **責務**: 為替/暗号資産のOHLCV取得
- **現在の実装**: `yfinance` (無料、USDJPY=X 等が使える)
- **差し替え先**: OANDA API(本格運用時)、bitFlyer/Binance API(暗号資産)

### 2. Indicators層 (`src/fx/indicators.py`)
- **責務**: 純粋な数値計算(テクニカル指標)
- SMA / EMA / RSI / MACD / Bollinger Bands
- LLMに渡す前の「数値の要約」を作る

### 3. Analyst層 (`src/fx/analyst.py`)
- **責務**: Claude API でマーケットを分析し、トレード判断を生成
- **モデル**: `claude-opus-4-7` + adaptive thinking
- **プロンプトキャッシュ**: 固定のシステムプロンプトをキャッシュし、変動する市場データだけを差分送信
- **構造化出力**: `output_config.format` で `{action, confidence, reason}` をJSON強制
- **分離**: テクニカル計算とLLM推論を完全に切り離している(テストしやすさ)

### 4. Strategy層 (`src/fx/strategy.py`)
- **責務**: テクニカルシグナル + LLM判断を組み合わせて最終シグナルを出す
- 現在は **合意ベース**(両方がBUYならBUY、不一致ならHOLD)
- ポジションサイジング、ストップロス、テイクプロフィットもここ

### 5. Storage層 (`src/fx/storage.py`)
- **責務**: SQLiteでトレード履歴・分析履歴・バックテスト結果を保存
- テーブル: `trades`, `analyses`, `backtest_runs`

### 6. Backtest層 (`src/fx/backtest.py`)
- **責務**: 過去データでストラテジーを再生し、パフォーマンスを計測
- イベント駆動型(1バーずつ前進、未来情報を見ない)
- 出力: 勝率、PF(プロフィットファクター)、最大ドローダウン、累積PnL

### 7. News層 (`src/fx/news.py`)
- **責務**: ヘッドライン取得(ファンダメンタル情報)
- yfinanceの新旧両方のレスポンス形式に対応
- 取得失敗時は空リスト(「ニュースなし」は正常状態)
- 本格運用時はReuters/Bloomberg/中央銀行RSSに差し替え可能

### 8. Correlation層 (`src/fx/correlation.py`)
- **責務**: 関連通貨ペア・指数との相関を計算
- USDJPY=X なら EURJPY=X / GBPJPY=X / DXYと比較
- 相関が高い他ペアとの24h変動方向が**確認 or 乖離**かLLMに伝える
- 乖離は「このペアの動きは他と違う → 確度を下げる」判断材料

### 9. Calendar層 (`src/fx/calendar.py`)
- **責務**: 経済指標スケジュール(FOMC/日銀/ECB等)
- `data/events.json` にJSONで保持(自分でseed/scrape)
- `upcoming_for_symbol()` でシンボルに関係する通貨の今後N時間のイベントを抽出
- LLMに「あと4時間でFOMC → HOLD推奨」のような判断材料として渡す

### 10. Risk層 (`src/fx/risk.py`)
- **責務**: 純粋な数学(外部依存なし、テスト容易)
- **ATR (Average True Range)** — Wilder's EWM版
- **ATRベースのストップロス/テイクプロフィット** — エントリーから `N*ATR` 離したところ
- **Kelly基準** — 勝率と損益比から最適ベットサイズを計算
- **Fractional Kelly** — フル Kellyはボラが高すぎるので 1/4 Kelly + 20% cap
- **Position sizing** — リスク許容額(資金の1%等)をストップ距離で割ってユニット数を決める
- `plan_trade()` が全部合わせて `RiskPlan` を返す

### 11. Broker層 (`src/fx/broker.py`, `src/fx/oanda.py`)
- **責務**: 発注の抽象化
- `Broker` ABC — place_order / close_position / balance / mark_to_market
- `PaperBroker` — メモリ内シミュレータ。ストップ/TPを自動判定して close
- `OANDABroker` — OANDA v20 APIラッパー、**practice口座限定**
  - `OANDA_ENV=practice` 強制(liveは弾く)
  - `OANDA_ALLOW_LIVE_ORDERS=yes` が必要(セッションごとのopt-in)
  - `confirm_demo=True` が必要(明示的コンストラクタ引数)
  - 3条件全部満たさないと `LiveTradingBlocked` 例外
  - `oandapyV20` は**遅延import**(未インストールでもテストは通る)

### 12. Prediction層 (`src/fx/prediction.py`)
- **責務**: 「Claudeの予測 vs 実際の価格動向」の照合
- 各analyze時、Claudeに`expected_direction`, `expected_magnitude_pct`, `horizon_bars`, `invalidation_price`を強制(反証可能性)
- `evaluate_prediction()`が後続バーを見て5カテゴリに分類
- `max_favorable_pct` / `max_adverse_pct` も記録 → 「方向は当たったが我慢が足りなかった」も検出可能
- 自動的にinvalidation_priceにヒットしたかも判定

### 13. Postmortem層 (`src/fx/postmortem.py`)
- **責務**: WRONG判定だけClaudeに失敗の根本原因分析させる
- 11カテゴリの閉じたタクソノミ(集計のため)
- 改善ルール提案も構造化出力
- 結果は`postmortems`テーブルへ
- 次回`analyze`で同一シンボルの過去失敗を取り出してプロンプトに自動注入(`Storage.relevant_postmortems()`)

### 14. Sentiment 収集サブシステム (`src/sentiment/`)
- **責務**: Web群衆の声を集めて数値化、JSONで保存
- **完全に独立した別モジュール** — `src/fx/` には依存しない
- 連携は `data/sentiment.json` 経由 (`src/fx/sentiment.py` が薄いリーダー)
- ソース:
  | ソース | 認証 | 安定性 | 備考 |
  |--------|-----|--------|------|
  | Reddit | なし | ◎ | `.json` エンドポイント |
  | Stocktwits | なし | ◎ | 公式REST、無料 |
  | TradingView ideas | なし | △ | HTMLスクレイプ |
  | Twitter/X (curated) | option | △ | snscrape→Nitter→API v2 の3段フォールバック |
  | RSS | なし | ◎ | キーワードフィルタ |
- スコアリング: Claude **Haiku 4.5** で1記事ずつ `bullish/bearish/neutral` + 信頼度
- 集計: 24h集計 + 24h-48hとの差分でvelocity算出
- 翻訳: 非英語ソース混入時は Haiku で英訳してからスコア
- リフレッシュは `fx sentiment-refresh` CLI で(cronで1時間ごと推奨)

### 15. Web層 (`src/fx/web/`)
- **責務**: Botの「**何を見て、どう考え、どう判断したか**」をブラウザで可視化
- Flask + Jinja2 + Tailwind CDN(ビルド不要)
- `/analyze` は **Server-Sent Events** でパイプラインを1ステップずつ実況
  (データ取得 → テクニカル → ATR → ニュース → 相関 → イベント → 過去レッスン → Claude → 合意 → リスクプラン → 保存)
- `pipeline_stream.run_pipeline()` がジェネレータで各ステップ完了時にイベントをyield
- 既存の `sample.py` を `gunicorn sample:app` の入口として保持し、新Flaskアプリへ繋ぐ

### 15. CLI層 (`src/fx/cli.py`)
- `fx analyze USDJPY=X` — 1回ライブ分析(過去レッスン自動注入)
- `fx trade USDJPY=X --dry-run` — 分析→リスクプラン作成→(任意で)発注
- `fx backtest USDJPY=X --period 180d` — バックテスト実行
- `fx evaluate` — 期日経過した予測を全部実価格と照合
- `fx postmortem` — WRONG予測にClaudeを通して原因分析
- `fx lessons --symbol USDJPY=X` — 蓄積した教訓を確認
- `fx review` — 過去トレードをClaudeにレビューさせる(週次まとめ)
- `fx watch USDJPY=X --interval 15m` — 定期分析ループ
- `fx calendar-seed` — `data/events.json` のひな形を作る

## Claude API の使い分け

| 用途 | モデル | 呼び出し頻度 | キャッシュ |
|-----|-------|------------|-----------|
| リアルタイム市場分析 | Opus 4.7 + adaptive thinking | 5分〜1時間毎 | system promptキャッシュ |
| バックテスト中のシグナル | Opus 4.7 | バーごと(高頻度) | 必須(コスト削減) |
| 週次セルフレビュー | Opus 4.7 + effort="high" | 週1 | 不要 |

## 学習ループ(Claudeとの連携)

「予測 → 検証 → 失敗の根本原因分析 → 次回プロンプトに注入」の閉ループ:

```
  ┌──────────────────────────────────────────┐
  │ 1. analyze: Claudeが反証可能な予測を生成   │
  │    expected_direction / magnitude /       │
  │    horizon_bars / invalidation_price      │
  │    → predictions テーブルに PENDING       │
  └─────────────────┬────────────────────────┘
                    ▼
  ┌──────────────────────────────────────────┐
  │ 2. evaluate: horizon経過後に実価格と照合   │
  │    → CORRECT / PARTIAL / WRONG /          │
  │       INCONCLUSIVE / INSUFFICIENT_DATA    │
  └─────────────────┬────────────────────────┘
                    ▼
  ┌──────────────────────────────────────────┐
  │ 3. postmortem: WRONGだけClaudeが原因分析   │
  │    11カテゴリの根本原因 + 改善ルール提案  │
  │    → postmortems テーブル                 │
  └─────────────────┬────────────────────────┘
                    ▼
  ┌──────────────────────────────────────────┐
  │ 4. 次回 analyze: 同じシンボルの過去失敗を │
  │    プロンプトに注入。Claudeが「以前同じ   │
  │    パターンで X が原因で間違えた」を読み、 │
  │    今回もそれが当てはまるか検証する       │
  └──────────────────────────────────────────┘
```

### 根本原因の閉じたタクソノミ

集計と検索のため、Claudeは以下11カテゴリのいずれかに必ず分類:

| カテゴリ | 意味 |
|---------|------|
| `TREND_MISREAD` | トレンド継続を反転と読んだ(逆も) |
| `NEWS_SHOCK` | フィードに無いニュースで動いた |
| `LIQUIDITY_WHIPSAW` | 薄商いのノイズで損切りされた |
| `CORRELATION_BREAKDOWN` | 関連ペアが想定外に乖離した |
| `INDICATOR_LAG` | テクニカルが動きの後追いだった |
| `FALSE_SIGNAL` | 指標が空売りで点灯した |
| `EVENT_VOLATILITY` | スケジュールイベントで吹き飛んだ |
| `REGIME_CHANGE` | レンジ→トレンド等の地合い変化 |
| `OVER_CONFIDENCE` | シグナル弱いのに張った |
| `UNDER_HORIZON` | 方向は正しいが時間軸違い |
| `OTHER` | 上記に当てはまらない |

`fx lessons` でカテゴリ別集計を確認 → 多発カテゴリがプロンプト改善の優先順位になる。

## 段階的ロードマップ

| Phase | 内容 | 実装状況 |
|-------|-----|---------|
| 1 | データ取得・テクニカル・分析ログ(発注なし) | ✅ |
| 2 | バックテスト基盤 | ✅ |
| 3 | セルフレビュー(週次学習) | ✅ |
| 3.5 | ニュース / 相関 / 経済カレンダー / リスク管理 | ✅ |
| 4 | PaperBroker + OANDA demoスキャフォールド | ✅ (安全装置付き) |
| 4.5 | OANDA demo口座で実発注 | 手動実行 |
| 5 | リアルマネー(小額から) | 未(自己責任) |

**Phase 4以降に進む前の検証基準(目安):**
- 6ヶ月以上のバックテストで PF > 1.3
- 最大DD < 20%
- パラメータのロバストネス確認(別期間でも性能維持)
- 実時間ペーパートレードで1ヶ月以上検証

## セキュリティ・リスク

- **APIキー**: `.env` に保存(コミット禁止)
- **実資金の扱い**: Phase 4まで完全に無効化
- **Claudeの判断ミス**: 必ず人間が最終確認 or ハードストップロスで損失上限を設定
- **法規制**: FX自動売買は国ごとに規制あり。日本ではレバレッジ上限、自動売買ツールの自主規制あり
