# Detailed Design Specification

このドキュメントは、各モジュールの**責務・契約・依存関係**を1か所に集めた詳細設計仕様。
日々の操作方法は [`USER_GUIDE.md`](USER_GUIDE.md)、システム全体の俯瞰は
[`ARCHITECTURE.md`](ARCHITECTURE.md)、最終判断のロジックは仕様書 §10/§17 と本ドキュメント §3 を参照。

## 目次

1. システム概要
2. モジュール依存図
3. Decision Engine 仕様
4. Risk Gate 仕様
5. データスキーマ (SQLite + JSON)
6. 設定と環境変数
7. 拡張ポイント (新ソース・新ブローカー追加)
8. セキュリティ
9. 既知の制限と非目標

---

## §1 システム概要

### 1.1 製品の位置づけ

このアプリは **FX/暗号資産の研究プロトタイプ**。実発注は無効化されており、
3 つの主要ユースケースを持つ:

| ユースケース | 入力 | 出力 |
|------------|------|------|
| **バックテスト** | OHLCV 履歴 | 勝率 / PF / 最大DD / トレード一覧 |
| **リアルタイム分析** | 現在の市場データ | 反証可能予測 + リスクプラン (発注はしない) |
| **学習ループ** | 過去予測の評価結果 | 根本原因タグ + 改善ルール提案 |

### 1.2 設計の最重要原則

仕様書 §0 を本実装の最上位制約として固定:

1. **BUY/SELL/HOLD の最終判断は AI ではなく固定ルール `decision_engine` が下す。**
2. **Risk Gate は AI が上書きできない。**
3. AI (Claude) は「相場解釈・根拠整理・失敗分析・改善案作成」担当。
4. ファイルが存在することと、最終判断に効いていることは別問題。
   コミット時のテストで **接続性** を毎回検証する。
5. 実発注は OFF。OANDA demo 接続も 3 重ロック必須。

### 1.3 システム全体の流れ

```
価格取得 (yfinance)
    ↓
データ品質チェック [Risk Gate]
    ↓
経済イベント窓口チェック [Risk Gate]
    ↓
スプレッド・流動性 [Risk Gate / 現状 yfinance では UNKNOWN]
    ↓
波形・構造認識 (patterns.py)
    ↓
テクニカル指標 (indicators.py)
    ↓
SNS センチメント取得 (src/sentiment/)
    ↓
AI 助言 (analyst.py — Claude Opus 4.7)
    ↓
固定ルール Decision Engine (decision_engine.py)
    ↓
BUY / SELL / HOLD + 反証可能予測
    ↓
Storage (SQLite) + 通知 (LINE/log)
    ↓
horizon 経過 → evaluate
    ↓
WRONG → postmortem (Claude Haiku/Opus) → lesson library
    ↓
次回 analyze 時に lesson が自動的にプロンプトへ注入
```

各ステップは Web `/analyze/stream` で SSE として実況される。

---

## §2 モジュール依存図

### 2.1 パッケージ階層

```
test/                              ← repo root
├── launch.py / start.{command,sh,bat}    ダブルクリック起動
├── sample.py                              gunicorn / Flask エントリ
├── data/
│   ├── fx.db                              SQLite (analyses, predictions, …)
│   ├── events.json                        経済カレンダー
│   └── sentiment.json                     最新センチメントスナップショット
│
├── src/fx/                                FX 本体
│   ├── data.py            yfinance ラッパー (OHLCV)
│   ├── indicators.py      SMA/EMA/RSI/MACD/BB + 数値スナップショット
│   ├── patterns.py        波形・構造・パターン検出 (Future-leak ナシ)
│   ├── higher_timeframe.py 上位足の trend_state
│   ├── correlation.py     関連通貨ペア相関
│   ├── calendar.py        経済イベント (events.json)
│   ├── news.py            ヘッドライン
│   ├── sentiment.py       SNS スナップショットの薄いリーダー
│   ├── risk.py            ATR / Kelly / position sizing
│   ├── risk_gate.py       強制 HOLD ゲート集合体
│   ├── context.py         MarketContext (拡張 Snapshot)
│   ├── analyst.py         Claude API (助言者)
│   ├── decision_engine.py 固定ルール最終判定
│   ├── strategy.py        互換 shim → decision_engine
│   ├── prediction.py      予測の事後評価
│   ├── postmortem.py      失敗の根本原因分析
│   ├── backtest.py        イベント駆動バックテスト
│   ├── storage.py         SQLite + マイグレーション
│   ├── broker.py          PaperBroker (in-memory)
│   ├── oanda.py           OANDA demo (3重ロック)
│   ├── cli.py             コマンドライン
│   └── web/               Flask ダッシュボード
│       ├── app.py
│       ├── routes.py      / /analyze /predictions /sentiment /lessons /webhook/line
│       ├── pipeline_stream.py SSE イベント生成
│       ├── webhook.py     LINE コマンドパーサー
│       └── templates/     Jinja2 テンプレ
│
├── src/sentiment/                         独立アプリ (FX に依存しない)
│   ├── base.py            Post / Source ABC
│   ├── config.py          シンボル別キーワード, インフルエンサー
│   ├── collectors/        reddit / stocktwits / tradingview / twitter / rss
│   ├── scoring.py         Claude Haiku で各投稿を採点
│   ├── aggregator.py      24h 集計 + velocity
│   ├── snapshot.py        data/sentiment.json への atomic write
│   └── refresh.py         end-to-end オーケストレーター
│
├── src/notify/                            通知抽象
│   ├── base.py
│   ├── line.py            LINE Messaging API (push/reply/署名検証)
│   ├── log.py / null.py
│   └── formatters.py      signal / verdict / lesson / summary 整形
│
├── tests/
│   ├── test_indicators.py (pure math)
│   ├── test_patterns / test_decision_engine (safety guarantees)
│   ├── test_risk_gate / test_risk
│   ├── test_correlation / test_calendar / test_news
│   ├── test_sentiment* / test_storage*
│   ├── test_notify / test_web
│   └── test_launch
│
└── docs/
    ├── ARCHITECTURE.md  (俯瞰)
    ├── USER_GUIDE.md    (操作)
    ├── DESIGN.md        (本ファイル — 詳細仕様)
    └── AUDIT.md         (現状監査レポート)
```

### 2.2 重要な依存方向 (許される向き)

```
                ┌──────────────┐
                │  cli.py      │
                │  web/        │
                └─┬──────┬─────┘
                  │      │
        ┌─────────┴──┐  ┌┴───────────────┐
        ▼            ▼  ▼                ▼
  decision_engine ◀── analyst       prediction
        │       \
        │        ▶ risk_gate ◀── calendar
        │                      ◀── (sentiment)
        ▼
   patterns ─── indicators ─── data (yfinance)
                                │
                                ▼
                              storage (SQLite)
```

ルール:
- `decision_engine` は `analyst` の出力を **受け取って参照** する
  だけで、`analyst` は `decision_engine` を一切 import しない。
  AI の助言が決定ロジックを汚染しない構造。
- `risk_gate` は外部 I/O をしない (引数で受け取った `RiskState`
  だけで判断する)。テストが容易、決定論的。
- `patterns` / `indicators` / `risk` は純粋関数のみ。
- `src/sentiment/` は `src/fx/` に依存しない。連携は
  `data/sentiment.json` ファイル経由のみ。

### 2.3 禁止される依存方向

| してはいけない | 理由 |
|--------------|------|
| `risk_gate` から `analyst` を呼ぶ | AI が Risk Gate を間接的に触れる経路ができる |
| `analyst` から `decision_engine` を import | プロンプトが結論を知ってしまう |
| `patterns` / `indicators` から I/O | ピュア性が崩れる、バックテストでもネットを叩く |
| `cli` から直接 SQLite を SQL で叩く | `storage` の API を経由する |

### 2.4 主要データ構造のフロー

```
Snapshot (numeric)            ◀─ build_snapshot(df)
   │
   ▼
PatternResult                 ◀─ analyse(df)
   │
   ▼
MarketContext  ───── to_dict() ─────▶  Claude Analyst の入力 JSON
   │                                          │
   ▼                                          ▼
RiskState     ───── evaluate() ──▶ GateResult (allow_trade?)
   │                                          │
   ▼                                          ▼
            decide(...)  ────────────▶ Decision (action/confidence/reason
                                                /blocked_by/rule_chain/advisory)
                                                │
                                                ▼
                                       Storage.save_analysis +
                                       Storage.save_prediction
                                                │
                                                ▼
                                       Notification (LINE/log)
```

すべてのデータクラスは frozen=True で、`to_dict()` を持つ
(JSON シリアライズ可能、ログ・通知・LLM 入力に再利用可能)。

---

## §3 Decision Engine 仕様

### 3.1 目的と契約

`src/fx/decision_engine.py` の `decide(...)` は、このリポジトリで
**唯一**実エントリの BUY/SELL を生成できる関数。仕様書 §10 に
完全準拠する。

入力契約:

| 引数 | 型 | 必須 | 用途 |
|------|---|-----|------|
| `technical_signal` | `"BUY" \| "SELL" \| "HOLD"` | 必須 | ルールベースの方向 |
| `pattern` | `PatternResult \| None` | 任意 | 波形・ネックライン情報 |
| `higher_timeframe_trend` | `str \| None` | 任意 | 上位足の trend_state |
| `risk_reward` | `float \| None` | 任意 | RR(なければスキップ) |
| `risk_state` | `RiskState` | **必須** | Risk Gate に渡す状態 |
| `llm_signal` | `TradeSignal \| None` | 任意 | Claude の助言 |
| `min_confidence` | `float = 0.6` | デフォルト | LLM 最低信頼度 |
| `min_risk_reward` | `float = 1.5` | デフォルト | RR フロア |

出力契約: `Decision(action, confidence, reason, blocked_by, rule_chain, advisory)`

* `action` ∈ {`BUY`, `SELL`, `HOLD`}
* `blocked_by` は `BlockReason.code` のタプル
* `rule_chain` は通過した順序付き名前タプル
* `advisory` は LLM の出力(action / confidence / reason / key_risks)を保持

`reason` は人間が読める文。後の Postmortem や通知メッセージで再利用。

### 3.2 判定アルゴリズム

```
1. risk_gate                Risk Gate を回す
                            block あり → HOLD (advisory に LLM の声を残す)
2. technical_directionality technical_signal が HOLD なら HOLD
                            (SNS だけでは絶対トレードしない保証)
3. pattern_check            top-pattern × SELL かつ neckline_broken=False → HOLD
                            bottom-pattern × BUY かつ neckline_broken=False → HOLD
4. higher_tf_alignment      BUY × 上位足 DOWNTREND → HOLD
                            SELL × 上位足 UPTREND → HOLD
5. risk_reward_floor        RR < 1.5 → HOLD
6. llm_advisory             LLM confidence < 0.6 → HOLD
                            LLM action ≠ technical_signal → HOLD
7. approve                  ここに来たら technical_signal で承認、
                            confidence は LLM があれば LLM の値、
                            なければ 0.55 (LLM 不在時のデフォルト)
```

各ステップは `chain.append(name)` で履歴を残す。HOLD 時は
`reason` で「どこで落ちたか」が分かる。Postmortem の検索キーとして
そのまま使える。

### 3.3 AI が**できないこと**

| AI のしたいこと | 結果 |
|---------------|------|
| Risk Gate NG なのに BUY と言う | gate のブロックが優先 → HOLD |
| confidence 0.4 で BUY を主張 | min_confidence に達せず HOLD |
| technical=HOLD なのに BUY と言う | step 2 で弾かれる |
| technical=BUY なのに SELL を主張 | step 6 で不一致として HOLD |
| ネックライン未割れの三山で SELL を主張 | step 3 で弾かれる |

`tests/test_decision_engine.py::test_ai_cannot_override_risk_gate`
で監視。

### 3.4 AI が**できること**

| AI のしたいこと | 結果 |
|---------------|------|
| ルールが許す BUY を補強する説明を書く | `reason` に乗る、confidence は LLM の数値 |
| confidence を低く返して HOLD に倒す | step 6 で HOLD、`reason` に LLM の理由が出る |
| 過去の同種失敗(`past_postmortems`)を参照して警告する | プロンプトで自動注入される |
| 反証可能予測(direction/magnitude/horizon/invalidation)を出す | `predictions` テーブルに保存され、後で評価 |

### 3.5 反証可能予測

すべての LLM 出力は構造化スキーマで以下を必須にする
(`analyst.SIGNAL_SCHEMA`):

* `expected_direction` ∈ {`UP`, `DOWN`, `FLAT`}
* `expected_magnitude_pct`(0.5 等の絶対値%)
* `horizon_bars`(整数;何バー後の評価か)
* `invalidation_price`(数値 or null)

これにより `evaluate` が後で実価格と機械的に照合できる
(`prediction.py:evaluate_prediction`)。

---

## §4 Risk Gate 仕様

### 4.1 設計思想

Risk Gate は **HOLD のみ生成できる**強制ゲートの集合。AI
を信頼しない。仕様書 §6 の優先順位を厳格に守る。

```
1. data_quality       データが無い・短い・NaN混入・価格0以下
2. event_high         FOMC/BOJ/CPI/PCE/NFP の窓口内
3. spread_abnormal    スプレッドが閾値超え
4. daily_loss_cap     当日累計損失が上限超え
5. consecutive_losses 連敗回数が上限超え
6. rule_unverified    ルール変更から検証期間が経っていない
7. sentiment_spike    投稿数 + velocity + 危機ワード(派生ゲート)
```

各チェックは独立した小さな関数で、Maybe[BlockReason] を返す。
`evaluate(state)` がこの順序で **最初に block を返したものを採用**
する(`blocked_codes` には他にも該当したものを残す = 診断用)。

### 4.2 BlockReason の構造

```python
@dataclass(frozen=True)
class BlockReason:
    code: str        # "data_quality" | "event_high" | ...
    message: str     # 人間向け短文
    detail: dict     # 機械可読 (event名, window_h, mention数等)
```

* `code` はテストとダッシュボード集計で使う識別子。**変更禁止に近い**。
* `message` はログ・通知・LLM プロンプトに引用される。
* `detail` は Postmortem 時の証拠保全。

### 4.3 各ゲートの仕様

#### 4.3.1 `check_data_quality(df, min_bars=50)`

* `df` が空 → block
* バー数 < `min_bars` → block (デフォルト 50; 指標計算に必要)
* `last_close` が NaN/inf/0以下 → block
* 直近 5 バーの OHLC に NaN がある → block (フィードが壊れた合図)

#### 4.3.2 `check_high_impact_event(events, now=None)`

* `events` の中で `impact == "high"` のものだけ対象
* タイトルから窓口時間を決定:

  | キーワード(タイトル含む、大小区別なし) | 窓口 |
  |--------------------------------------|-----|
  | `FOMC` | ±6h |
  | `BOJ` | ±6h |
  | `ECB` | ±4h |
  | `CPI` / `PCE` / `NFP` / `GDP` | ±2h |
  | `RATE_DECISION` | ±6h |
  | `INTERVENTION` | ±12h |
  | (上記なし、impact=high) | ±4h |

* `|now − event.when| ≤ 窓口` → block
* キーワードが取れない / events.json が無い場合は **保守的に
  扱う**(イベントなしと見なすが、上位の運用で「スケジュール
  情報なし」自体を `rule_unverified` で扱える)

#### 4.3.3 `check_spread(spread_pct, max_pct=0.05)`

* `spread_pct` が None → スキップ(yfinance は bid/ask を出さない)
* `spread_pct > max_pct` (0.05% ≈ 5bp) → block
* NaN/inf → block

#### 4.3.4 `check_daily_loss_cap(pnl_today, cap)`

* `pnl_today ≤ -|cap|` → block

#### 4.3.5 `check_consecutive_losses(streak, cap=3)`

* `streak ≥ cap` → block

#### 4.3.6 `check_rule_unverified(age_h, min_age_hours=24.0)`

* ルール更新からの経過時間が `min_age_hours` 未満 → block
* デモ/バックテストでルール変更を検証する強制期間

#### 4.3.7 `check_sentiment_spike(sentiment, mention=200, velocity=0.6)`

* mention_count_24h ≥ 200 かつ |sentiment_velocity| ≥ 0.6
* かつ `notable_posts` に `fomc / boj / 日銀 / cpi / pce / nfp /
  rate decision / intervention / 介入 / panic / crash / breaking`
  のいずれかが含まれる場合 → block
* SNS 「だけ」を理由に**取引する**ことは禁止だが、SNS が
  「警戒材料」として強く出た時は **HOLD する** のはOK。
  ここで実装。

### 4.4 Risk Gate を回避する正規ルート

ない。回避は仕様違反。
`tests/test_decision_engine.py::test_ai_cannot_override_risk_gate`
で防止。

---

## §5 データスキーマ

### 5.1 SQLite (`data/fx.db`)

`src/fx/storage.py` で管理。`CREATE TABLE IF NOT EXISTS` + 軽量
ALTER TABLE マイグレーションで安全に進化させる。

#### 5.1.1 `analyses`

各 `analyze` 実行のスナップショットと最終 action。

```sql
CREATE TABLE analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,                    -- ISO 8601 UTC
    symbol TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,         -- MarketContext.to_dict()
    technical_signal TEXT NOT NULL,      -- BUY|SELL|HOLD (rule)
    llm_action TEXT,                     -- BUY|SELL|HOLD or NULL
    llm_confidence REAL,
    llm_reason TEXT,
    final_action TEXT NOT NULL           -- decision_engine.Decision.action
);
CREATE INDEX idx_analyses_symbol_ts ON analyses(symbol, ts);
```

#### 5.1.2 `predictions`

反証可能予測 + 評価結果 + Postmortem 用の証拠フィールド。

```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    entry_price REAL NOT NULL,

    -- LLM 由来の反証可能予測
    action TEXT NOT NULL,                -- BUY|SELL|HOLD
    confidence REAL,
    reason TEXT,
    expected_direction TEXT NOT NULL,    -- UP|DOWN|FLAT
    expected_magnitude_pct REAL NOT NULL,
    horizon_bars INTEGER NOT NULL,
    invalidation_price REAL,

    -- evaluate の結果
    status TEXT NOT NULL DEFAULT 'PENDING',  -- PENDING/CORRECT/PARTIAL/WRONG/INCONCLUSIVE
    actual_direction TEXT,
    actual_magnitude_pct REAL,
    max_favorable_pct REAL,
    max_adverse_pct REAL,
    invalidation_hit INTEGER,
    evaluated_at TEXT,
    evaluation_note TEXT,

    -- 仕様書 §13 Postmortem-friendly fields (back-fill OK)
    spread_at_entry REAL,
    spread_at_exit REAL,
    slippage REAL,
    rule_version TEXT,                   -- e.g. "2025-04-25-r3"
    detected_pattern TEXT,
    trend_state TEXT,
    higher_timeframe_trend TEXT,
    event_risk_level TEXT,               -- LOW|MEDIUM|HIGH
    economic_events_nearby TEXT,         -- JSON array
    sentiment_score REAL,
    sentiment_volume_spike INTEGER,
    blocked_by TEXT,                     -- カンマ区切り Risk Gate codes
    final_reason TEXT,                   -- decision_engine.reason

    FOREIGN KEY (analysis_id) REFERENCES analyses(id)
);
CREATE INDEX idx_predictions_status ON predictions(status);
CREATE INDEX idx_predictions_symbol_ts ON predictions(symbol, ts);
```

#### 5.1.3 `postmortems`

WRONG 判定された予測に対する根本原因タグ + 改善案。

```sql
CREATE TABLE postmortems (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL UNIQUE,
    ts TEXT NOT NULL,
    root_cause TEXT NOT NULL,            -- TREND_MISREAD / NEWS_SHOCK / ...
    narrative TEXT NOT NULL,
    proposed_rule TEXT,
    tags TEXT,                           -- カンマ区切り

    -- 仕様書 §12 損失分類
    loss_category TEXT,                  -- A|B|C|D|E|F
    is_system_accident INTEGER NOT NULL DEFAULT 0,
    context_json TEXT,                   -- 任意の構造化データ
    rule_applied INTEGER NOT NULL DEFAULT 0,

    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
);
CREATE INDEX idx_postmortems_root_cause ON postmortems(root_cause);
CREATE INDEX idx_postmortems_loss_category ON postmortems(loss_category);
```

`root_cause` の閉じたタクソノミ(`postmortem.py:ROOT_CAUSES`):

| code | 意味 |
|------|------|
| `TREND_MISREAD` | トレンド継続を反転と読んだ(逆も) |
| `NEWS_SHOCK` | フィードに無いニュースで動いた |
| `LIQUIDITY_WHIPSAW` | 薄商いノイズで損切り |
| `CORRELATION_BREAKDOWN` | 関連ペアが想定外に乖離 |
| `INDICATOR_LAG` | 指標が動きの後追い |
| `FALSE_SIGNAL` | 指標が空打ち |
| `EVENT_VOLATILITY` | スケジュールイベントで吹き飛ぶ |
| `REGIME_CHANGE` | レンジ→トレンド等の地合い変化 |
| `OVER_CONFIDENCE` | 弱いシグナルなのに張った |
| `UNDER_HORIZON` | 方向は正しいが時間軸違い |
| `OTHER` | 上記に当てはまらない |

`loss_category` の閉じたタクソノミ(仕様書 §12):

| code | 意味 | 対策方針 |
|------|------|---------|
| `A` | 正しい負け | 基本対策しない |
| `B` | 入ってはいけない負け(イベント前等) | Risk Gate 強化 |
| `C` | 利確/損切り設計ミス | ATR/RR/トレーリング見直し |
| `D` | 波形認識ミス | patterns.py 条件修正 |
| `E` | 指標依存ミス | 構造優先に変更 |
| `F` | 実行系問題(スリッページ等) | broker/risk/order 修正 |

#### 5.1.4 `backtest_runs`

```sql
CREATE TABLE backtest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    strategy TEXT NOT NULL,
    metrics_json TEXT NOT NULL           -- {win_rate, PF, max_dd, n_trades, ...}
);
```

#### 5.1.5 `subscribers`

LINE / Slack 等の通知受信者。

```sql
CREATE TABLE subscribers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backend TEXT NOT NULL,               -- "line" | "slack" | ...
    user_id TEXT NOT NULL,
    display_name TEXT,
    subscribed_at TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    notify_signal INTEGER NOT NULL DEFAULT 1,
    notify_verdict INTEGER NOT NULL DEFAULT 1,
    notify_lesson INTEGER NOT NULL DEFAULT 1,
    notify_summary INTEGER NOT NULL DEFAULT 1,
    min_confidence REAL NOT NULL DEFAULT 0.6,
    UNIQUE(backend, user_id)
);
```

`notify_*` で種別ごとの ON/OFF。`min_confidence` は将来の per-user
閾値(現状はデフォルト 0.6)。

### 5.2 JSON ファイル

#### 5.2.1 `data/events.json`

経済イベント一覧。`fx calendar-seed` でひな形生成、本運用では
ForexFactory 等から日次で再生成する想定。

```json
[
  {
    "when": "2025-05-01T18:00:00+00:00",
    "currency": "USD",
    "title": "FOMC Statement & Rate Decision",
    "impact": "high",                   // low | medium | high
    "forecast": "5.25%",
    "previous": "5.25%"
  }
]
```

`impact` と `title` は Risk Gate の窓口判定に直接効く。
タイトルに `FOMC` / `BOJ` / `CPI` 等のキーワードを含めること。

#### 5.2.2 `data/sentiment.json`

```json
{
  "as_of": "2025-04-25T10:00:00+00:00",
  "symbols": {
    "USDJPY=X": {
      "symbol": "USDJPY=X",
      "as_of": "2025-04-25T10:00:00+00:00",
      "mention_count_24h": 312,
      "sentiment_score": 0.45,
      "sentiment_velocity": 0.18,
      "by_source": {
        "reddit": {"count": 180, "score": 0.52},
        "stocktwits": {"count": 95, "score": 0.31}
      },
      "notable_posts": [
        {"source": "reddit", "author": "u/x", "score": 0.8, "text": "..."}
      ]
    }
  }
}
```

`SentimentSnapshot.from_dict / to_dict` で対称的に往復する。
書き込みは `path.with_suffix('.tmp').replace(path)` で atomic。

### 5.3 マイグレーションポリシー

* スキーマ変更時は `Storage._migrate(conn)` の `targets` dict
  に新カラムを追記。`PRAGMA table_info(table)` で existence を
  チェックし、未存在なら `ALTER TABLE ADD COLUMN` する。
* 既存カラムの **削除・型変更は禁止**。代替カラムを追加して
  並行期間を設ける。
* 必須の意味変化(例: status 値の追加)はテストで pin する。

---

## §6 設定と環境変数

### 6.1 環境変数の一覧

| 変数 | デフォルト | 用途 |
|------|-----------|------|
| `ANTHROPIC_API_KEY` | (なし) | Claude API。なければ analyst は無効 |
| `FX_MODEL` | `claude-opus-4-7` | analyst モデル |
| `FX_EFFORT` | `medium` | adaptive thinking effort |
| `FX_DB_PATH` | `data/fx.db` | SQLite パス |
| `FX_SYMBOL` | `USDJPY=X` | デフォルトシンボル |
| `FX_INTERVAL` | `1h` | デフォルト足 |
| `LINE_CHANNEL_ACCESS_TOKEN` | (なし) | LINE push/reply |
| `LINE_CHANNEL_SECRET` | (なし) | webhook 署名検証 |
| `NOTIFY_BACKEND` | (auto) | `line` \| `log` \| `null` を強制 |
| `OANDA_API_KEY` | (なし) | OANDA demo (broker=oanda 時のみ) |
| `OANDA_ACCOUNT_ID` | (なし) | OANDA account |
| `OANDA_ENV` | `practice` | **必ず practice**。live は LiveTradingBlocked |
| `OANDA_ALLOW_LIVE_ORDERS` | (なし) | `yes` で OANDA arm。3 重ロックの 1 つ |
| `TWITTER_BEARER_TOKEN` | (なし) | Twitter API v2 (3rd fallback only) |

`.env` 推奨。コミット禁止 (`.gitignore`済み)。

### 6.2 コード内設定

| 場所 | 内容 |
|------|------|
| `src/fx/risk_gate.py:HIGH_IMPACT_WINDOWS_HOURS` | イベント別 HOLD 窓口 |
| `src/fx/risk_gate.py:SENTIMENT_KEYWORD_TRIGGERS` | センチメント警戒ワード |
| `src/fx/decision_engine.py:MIN_CONFIDENCE` | デフォルト 0.6 |
| `src/fx/decision_engine.py:MIN_RISK_REWARD` | デフォルト 1.5 |
| `src/fx/correlation.py:RELATED_SYMBOLS` | 関連通貨ペア |
| `src/fx/calendar.py:SYMBOL_CURRENCIES` | シンボル→通貨マップ |
| `src/sentiment/config.py:SYMBOL_PROFILES` | 各ソースの検索キーワード |
| `src/sentiment/config.py:TWITTER_INFLUENCERS` | curated アカウント |
| `src/sentiment/config.py:RSS_FEEDS` | RSS 一覧 |

これらは**ファイル編集して再起動**で反映。CLI 引数では変えない
(変えると全シンボル / 全ユーザに反映できないため)。

### 6.3 ルールのバージョニング

仕様書 §6 の `rule_unverified` ゲートのために、ルールに
`rule_version` 文字列を持たせる(現状は環境変数 / コミット
SHA を流用する想定)。`predictions.rule_version` カラムに記録
することで、後から「どのバージョンの結果か」を集計できる。

将来の実装案:

```bash
export FX_RULE_VERSION="$(git rev-parse --short HEAD)"
```

ルール変更時は新コミット = 新 SHA → 自動的に `rule_version` が
変わる。`rule_unverified` ゲートには「FX_RULE_VERSION_UPDATED_AT」
を別途持たせて経過時間で判定する。

---

## §7 拡張ポイント

### 7.1 新しいセンチメントソースを足す

1. `src/sentiment/collectors/` に新ファイル `mysite.py`
2. `Source` ABC を継承し、`fetch(query, limit) -> list[Post]` を実装
3. 失敗時は **例外を上げず `[]` を返す**(全コレクターで統一)
4. `src/sentiment/refresh.py:refresh()` の `factories` に追加
5. `src/sentiment/config.py:SYMBOL_PROFILES` に検索キーワードを追加
6. ネット I/O はテストでは `http_get` を注入してモック

### 7.2 新しいパターン検出を足す

1. `src/fx/patterns.py` に `detect_xxx(highs, lows, atr_value) ->
   tuple[bool, neckline | None, confidence]` を追加
2. `analyse(df)` の優先順位リストに挿入(より specific を先に)
3. **Decision Engine の対応分岐** を追加:
   - top パターンか bottom パターンかを判定する分類に追加
   - tests/test_decision_engine.py にネックライン未割れで HOLD する
     pin テストを追加
4. AI プロンプトの「Waveform / structure guidelines」節に新パターンの
   ルールを書く

### 7.3 新しい Risk Gate を足す

1. `src/fx/risk_gate.py` に `check_xxx(state) -> BlockReason | None`
2. `RiskState` にゲートが必要な情報を追加(任意フィールド)
3. `evaluate()` の `checks` リストに spec 順序で挿入
4. `tests/test_decision_engine.py` に「これでブロックされる」pin テスト
5. ダッシュボードの `/analyze` パイプライン表示で見えるように、
   `pipeline_stream.py` に該当ステップを yield するコードを追加

### 7.4 新しいブローカーを足す (例: bitFlyer Lightning)

1. `src/fx/broker.py` の `Broker` ABC を継承
2. `place_order` / `close_position` / `balance` / `mark_to_market` /
   `open_positions` を実装
3. **発注の安全装置を多重化する** — 環境変数 + 明示的引数の両方で
   gate (`oanda.py` を参考)
4. 遅延 import で依存ライブラリは `Broker` 構築時にだけロード
5. `cli.py:_build_broker` に `--broker bitflyer` 分岐を追加

### 7.5 新しい通知バックエンドを足す (例: Slack)

1. `src/notify/slack.py` に `Notifier` 継承クラス
2. `src/notify/factory.py:build_notifier` に `NOTIFY_BACKEND=slack`
   分岐を追加
3. `src/fx/storage.py` の `subscribers.backend` に `"slack"` を許容
4. `src/fx/web/routes.py` に `/webhook/slack` を追加(Slack の
   署名検証は `X-Slack-Signature`、HMAC-SHA256)

### 7.6 新しいテクニカル指標を足す

1. `src/fx/indicators.py` にピュア関数を追加
2. `Snapshot` フィールドを増やす(後方互換のため `to_dict` に
   キーを増やすだけ;既存キー名は変えない)
3. `technical_signal(snap)` の投票ロジックに組み込む

---

## §8 セキュリティ

### 8.1 シークレットの取り扱い

* `.env` は `.gitignore` 済み。**絶対にコミットしない**。
* `LINE_CHANNEL_SECRET` を含むあらゆるキーは環境変数のみ。
* OANDA は practice 限定 + 3 重ロック (`oanda.py`):
  1. `OANDA_ENV == "practice"` 強制
  2. `OANDA_ALLOW_LIVE_ORDERS == "yes"` セッション毎 opt-in
  3. `OANDABroker(..., confirm_demo=True)` 明示引数

### 8.2 Webhook 署名検証

* LINE webhook は `verify_signature(secret, body, X-Line-Signature)`
  を `hmac.compare_digest` で検証 (タイミング攻撃耐性)
* `LINE_CHANNEL_SECRET` 未設定時は dev モードとして検証スキップ
  するが、本番デプロイでは必ず設定すること

### 8.3 SQLite SQL インジェクション

* `src/fx/storage.py` は **すべての可変値をパラメータバインド**
  (`?` プレースホルダ)。文字列結合での SQL 構築禁止。
* 唯一の例外は ALTER TABLE (列名は内部固定値のみ)

### 8.4 LLM プロンプトインジェクション

* AI が出力した文字列(`reason`, `narrative`)はログ・通知に出る
  だけで、**コード実行・SQL クエリ・ファイル操作には使わない**。
* AI の `action` フィールドは Decision Engine が独立に検証する
  ため、不正な値が来ても影響しない。
* `analyze` コマンドで AI が返した文字列がそのまま LINE に
  reply される経路があるが、LINE 側がプレーンテキストとして
  レンダリングするため XSS 等は発生しない。

### 8.5 yfinance / Reddit / Stocktwits の利用規約

* **公開エンドポイントへの低頻度アクセス**(15分1回程度)
* User-Agent を明示 (`fx-sentiment-bot/0.1` 等)
* TradingView は Terms of Service で禁止される行為に注意。本実装は
  公開 ideas ページの軽量スクレイピングに留め、**ログイン回避や
  認証突破は行わない**
* Twitter は API v2 を使う場合のみ TOS 準拠。snscrape / Nitter は
  Best-effort で、失敗しても他のソースで運用が成立する設計

### 8.6 個人情報・取引履歴

* `data/fx.db` には自分のトレード判断履歴のみ。個人情報なし。
* LINE `user_id` が `subscribers` に保存される。本番デプロイ時は
  バックアップから漏洩しないよう取り扱い注意。

---

## §9 既知の制限と非目標

### 9.1 設計上の非目標 (やらないこと)

| 項目 | 理由 |
|------|------|
| AI 単独での発注 | 仕様書 §0; 構造的に不可能 |
| 高頻度 (HFT) 取引 | yfinance のレート制限と、Claude API のレイテンシ |
| マルチユーザ SaaS | 個人/小チーム向け前提 |
| バックアップ・冗長化 | 単一マシンで運用、定期 `cp data/fx.db backup/` で十分 |
| 多言語 UI | ダッシュボードは英語/カタカナ混在 |

### 9.2 既知の制限

| 制限 | 内容 | 影響 |
|------|------|------|
| spread 不可視 | yfinance は bid/ask を返さない。`spread_pct=None` 固定 | `spread_abnormal` ゲートが事実上スキップされる。OANDA broker からは取得できる |
| events.json は手動更新 | 自動取得スクレイパー未実装 | cron で `fx calendar-seed` するだけだとプレースホルダになる。本運用は ForexFactory 等から日次取得が必要 |
| TradingView パーサ | HTML 構造変化で `0` 件になる可能性 | 他ソースで運用継続可能 |
| Twitter | snscrape 不安定、Nitter インスタンス頻繁にダウン | curated アカウントのみ取れない事もある |
| 上位足取得が yfinance | 同じ API のレート制限を共有 | 連続 `analyze` で間隔短いと制限に当たる |
| LLM コスト | Opus 4.7 で 1 回 ~$0.05;cron で頻繁に回すと月 $10〜30 | プロンプトキャッシュ + Haiku 使い分けで軽減 |
| 同時実行 | SQLite は 1 writer のみ; 複数 cron が同時 `analyze` するとロック発生 | プロセスを直列化 (cron で `flock` 等) |

### 9.3 仕様書の完了条件 (§17) と本実装の対応

| 仕様書 §17 の条件 | 本実装 |
|------------------|------|
| Risk Gate NG 時に BUY/SELL が絶対に出ない | `test_ai_cannot_override_risk_gate` 含む 3 テスト |
| FOMC・日銀・CPI・PCE・雇用統計の前後で HOLD | `test_fomc_forces_hold` / `test_boj_forces_hold` / `test_high_impact_event_forces_hold` |
| 三山候補だけでは SELL せず、ネックライン割れ後のみ | `test_triple_top_requires_neckline_break` |
| SNS データだけでは BUY/SELL にならない | `test_sentiment_alone_never_trades` |
| AI が BUY/SELL しても、固定ルール違反なら HOLD | Decision Engine 全体の構造 + `test_ai_cannot_override_risk_gate` |
| すべての判断理由がログに残る | `Decision.reason` + `blocked_by` + `rule_chain` + analyses テーブル |
| バックテストで未来参照がないことをテストで示す | `test_patterns_no_future_leak` + `test_backtest_never_peeks_at_future` |
| 実発注は OFF のまま、デモ検証に進める状態 | `OANDABroker` 3 重ロック維持 |

### 9.4 残課題 (将来作業)

* events.json の自動再生成 (ForexFactory or Investing.com スクレイプ)
* 上位足分析を独立にキャッシュする (重複 fetch 削減)
* ダッシュボードに **Risk Gate 状態のリアルタイム可視化**
  (`blocked_by` を色付きバッジで表示)
* `loss_category` の自動分類 (現状 Postmortem 後に手で付与)
* 1日損失上限 / 連敗上限の自動計算 (現状 `RiskState` に手で渡す)
* デモ口座の往復テスト (PaperBroker で実装済みだが、OANDA 実 demo
  との突き合わせは未)

---

このドキュメントは仕様書 (FX自動売買システム Claude Code 引き継ぎ・
実装指示書) §1-§19 と一対一で対応している。仕様書が更新されたら
本ファイルも更新する。
