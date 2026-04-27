# 引継ぎ資料 (Handover)

**スナップショット時点: 2026-04-27 (セクション切り替え)**

このドキュメントは「FX/暗号資産研究プロトタイプ」プロジェクトを
**次のセクション** (= 次の作業者 / 次のフェーズ) に引き継ぐためのもの。
PR #2 (`claude/forex-trading-app-E122S`) と PR #3 (`fix/backtest-synthetic-metrics`)
は **共に main にマージ済み**。直近の作業 PR #4 (`claude/decision-trace-logging`,
decision_trace v1) は **open / レビュー待ち** の状態で次セクションに引き継ぐ。

このドキュメント自体は **`claude/create-handover-docs-SpYQB` ブランチ** に置く。
PR #4 (`claude/decision-trace-logging`) とは独立し、レビューやマージの影響を受けない。

**読む順番**:
1. このファイル (HANDOVER.md) で全体像と今やるべきこと
2. `DESIGN.md` で仕様 §1〜§14 / 仕様書 §1〜§19 との対応
3. `USER_GUIDE.md` で CLI / ダッシュボード / cron / トラブルシューティング
4. `ARCHITECTURE.md` でシステム俯瞰
5. `AUDIT.md` でブランチ初期の整備記録 (レガシー)

---

## 1. ひとことでいうと

**FX/暗号資産の研究プロトタイプ**。実発注は構造的にロックされており、
研究 → デモ口座 → 仮説検証 → ルール改善 のループを回すための土台。

中核思想は仕様書 §0 の宣言:

> AIは売買の最終判断者ではなく、分析・説明・反省・改善提案の役割に限定する。
> Risk Gate と Decision Engine を最優先にする。

`decision_engine.decide()` が **唯一** の BUY/SELL/HOLD 判断点。
LLM・センチメント・波形マッチは助言・拒否権のみで、エントリーを「起こす」権限はない。

---

## 2. リポジトリの現状 (2026-04-27 時点)

### ブランチとPRサマリー

| ブランチ | 役割 | 状態 |
|---|---|---|
| `main` | 統合先 (`4532e16`) | PR #2, #3 まで取り込み済み |
| `claude/forex-trading-app-E122S` | PR #2 元ブランチ (`a850c7e`) | **merged 2026-04-26** |
| `fix/backtest-synthetic-metrics` | PR #3 元ブランチ (`827f23a`) | **merged 2026-04-26** |
| `claude/decision-trace-logging` | PR #4 元ブランチ (`ba2a47a`) | **open / レビュー待ち** |
| `claude/create-handover-docs-SpYQB` | この HANDOVER.md 用 (`482fc13`) | docs only。PR は未作成 (任意) |
| `claude/create-handover-docs-fAEL8` | 並行で切られた docs 用 | 内容は SpYQB と同一 (1 本に統一推奨) |

### main の最新ライン (`4532e16`)

```
4532e16  Merge pull request #3 from youiti0925/fix/backtest-synthetic-metrics
827f23a  Test synthetic execution metadata in engine backtest metrics
27f57e5  Mark engine backtest metrics as synthetic execution
b9a4bff  Merge pull request #2 from youiti0925/claude/forex-trading-app-E122S
bd5d5a4  Merge main into branch — resolve 4-file conflict
bb82457  audit: minimal fixes from stabilization pass
e377969  Wire StrategyConfig into the live decision path
105aafe  docs: cover Phases A-D and §16 fill audit in DESIGN/USER_GUIDE
4efc1b8  Phase 5 (spec §16): execution-fill audit on every live order
349d403  Phase D: attribution, calibration, external comparator
3cc6653  Phase C: waveform matching + advisory-only Decision Engine wiring
62beb4b  Phase B: market timeline + event overlay
ff92ebe  Phase A: live-trade safety hardening + Decision Engine backtest
```

### テスト・規模

| 指標 | 値 |
|---|---|
| main 上のテスト | **250 passed / 1 skipped** (`pytest tests/ --ignore=tests/test_notify.py`) |
| PR #4 適用後のテスト | **281 passed / 1 skipped** (decision_trace で +31 件 / 1 skip) |
| Python 推奨 | 3.11+ (ローカルでは 3.11.15 で通過) |
| 主要ディレクトリ規模 | `src/fx/` 30+ モジュール、`tests/` 約 280 件 |

---

## 3. 直近セクションで完了したもの (PR #2 + PR #3、main にマージ済み)

> 詳細フェーズはこの後の Phase A〜D と Phase 5 を参照。  
> ここでは「このセクションで何が確定したか」を一行ずつまとめる。

- **PR #2** (`b9a4bff` で main に merge): Phase A〜D + §16 fill audit + StrategyConfig
  配線 + 監査修正をまとめた大型 PR。仕様書 §0〜§19 のうち実装可能な範囲を全部
  カバーし、テスト 250 件で全不変条件を pin。
- **PR #3** (`4532e16` で main に merge): `backtest_engine.metrics()` に
  `synthetic_execution=True` ほか 6 キーを追加。**バックテスト結果が研究用で
  あること**を metrics 側からも明示的に分かるようにした (`spread_mode` /
  `slippage_mode` / `fill_model` / `bid_ask_mode` / `sentiment_archive`)。
  検証は `pytest tests/test_backtest_engine.py` → 8 passed。
- **副次効果**: `BacktestResult.metrics()` を読むコード (ダッシュボード、CLI、
  外部レポート) は 6 キーを表示できる前提で書ける。

## 4. 進行中のセクション (PR #4 — open、レビュー待ち)

### 概要

| 項目 | 値 |
|---|---|
| PR | `#4` (Add decision_trace v1: per-bar audit log for backtest_engine) |
| ブランチ | `claude/decision-trace-logging` (HEAD `ba2a47a`) |
| ベース | `main` (`4532e16`, PR #3 merge 後) |
| 状態 | **open / unmerged / mergeable=clean** |
| 行数 | +2,500 / −3、5 ファイル |
| テスト | **281 passed / 1 skipped** (回帰なし、+31 件 / 1 skip) |

### 何が入っているか

- `src/fx/decision_trace.py` (新規 620 行): 12 個の dataclass で 1 バー分の
  完全な意思決定オーディットを表現
  (`RunMetadata` / `BarDecisionTrace` / `MarketSlice` / `TechnicalSlice` /
  `WaveformSlice` / `HigherTimeframeSlice` / `FundamentalSlice` /
  `ExecutionAssumptionSlice` / `ExecutionTraceSlice` / `RuleCheck` /
  `DecisionSlice` / `FutureOutcomeSlice`)。
- `src/fx/decision_trace_build.py` (新規 1,130 行): 上記 dataclass を
  `backtest_engine` の内部状態から組み立てる純関数群 + `future_outcome` の
  第二パス + `hypothetical_technical_trade` シミュレータ。
- `src/fx/backtest_engine.py` (+235 / −3): `run_engine_backtest()` に
  `capture_traces` フラグを追加。**売買判断のロジックは無変更**。
- `src/fx/indicators.py` (+41): 既存 `technical_signal()` は無変更のまま、
  reason_codes を返す薄いラッパー `technical_signal_reasons()` を追加
  (line 156-194、line 189 で `technical_signal(snap)` に action を委譲)。
- `tests/test_decision_trace.py` (新規 477 行 / 32 tests): 不変性 3 件を含む。

### 不変性テスト (PR #4 でレビュー側が一番見るべきところ)

| テスト | 何を pin したか |
|---|---|
| `test_decisions_unchanged_with_trace_logging` | `capture_traces=True/False` で trades / hold_reasons / bars_processed が完全一致 |
| `test_metrics_dict_unchanged_with_trace_logging` | `metrics()` 戻り値が完全一致 |
| `test_hold_reasons_unchanged_with_trace_logging` | `hold_reasons` dict が完全一致 |
| `test_future_outcome_does_not_affect_decisions` | 第二パス追加で trades 一致 |
| `test_hypothetical_technical_trade_does_not_affect_decisions` | 仮想取引シミュレータが本体に影響しない |
| `test_trade_id_links_trace_and_engine_trade` | `engine_trade_ids ⊆ trace_trade_ids` |
| `test_missing_evidence_reasons_distinct_from_empty_arrays` | 空配列でも `missing_*_reason` 文字列が必須 |

### PR #4 のマージ前にやるべきこと

1. `pytest tests/ --ignore=tests/test_notify.py` を 281 passed / 1 skipped で再現確認
2. PR 説明の「レビュー重点 6 点」を順に確認 (`indicators.py` 無変更 / 売買判断
   不変 / future_outcome の第二パス / trade_id リンク / `decision_trace_build.py`
   の構造 / missing evidence 明示)
3. **squash merge ではなく merge commit** でマージ (PR #2 と同じ理由: dataclass
   ごとのコミット粒度を残す)

### PR #4 が **scope 外** に切り出している項目 (後続 PR 用)

| 項目 | 接続点 |
|---|---|
| `data_sources.jsonl` / `source_documents.jsonl` / `extracted_facts.jsonl` 連携 | `event_evidence_ids: list[str]` を埋めるだけで接続可。slice 構造は変えない |
| trace を JSONL に書き出す runner / CLI | `result.to_decision_trace_records()` が `list[dict]` を返すので `json.dumps` だけでよい |
| ニュース API 連携 | `news_evidence_ids` / `missing_news_evidence_reason` を埋める |
| OANDA live trace | `decision_trace_build` を import して live 経路でも再利用 (dataclass は I/O 非依存) |
| `_technical_score` の共有関数化 | 本 PR は B 案 (`technical_signal_reasons` が `technical_signal` に委譲) で実装。共有スコアリング関数化はリファクタとして別 PR |

---

## 5. 旧セクション (PR #2) の中身 — フェーズ別記録

> ここから先は PR #2 の作業履歴。**main に統合済み** だが、各モジュールの
> 由来を辿りたいときの参照用。新規開発時に同じものを再実装しないこと。

### Phase A — Live-trade safety hardening (commit `ff92ebe`)
- `calendar_freshness()` + `RiskState.require_calendar_fresh` 追加
  → live broker 経路では events.json 古い/欠損で **強制 HOLD**
- `Quote(bid, ask)` 抽象 + `RiskState.require_spread`
  → OANDA は Bid/Ask 取れないと発注拒否 (`spread_unavailable` block)
- `predictions` テーブルに `final_decision_action / executed_action`
  カラム追加 (LLM助言 vs エンジン決定 vs 実行 を分離)
- `backtest_engine.py` 新設 — Decision Engine 経由のバックテスト

### Phase B — Information integration (commit `62beb4b`)
- `src/fx/macro.py` — yields / DXY / equity / VIX を `asof` で
  point-in-time 整列
- `src/fx/market_timeline.py` — 1バー = 1行で OHLCV + テクニカル +
  パターン + 上位足 + イベント + マクロ + decision を統合
- `src/fx/event_overlay.py` — FOMC/BOJ/CPI/NFP の前後リターン集計

### Phase C — Waveform matching (commit `3cc6653`)
- `src/fx/waveform_matcher.py` — z-score 正規化 + cosine/correlation/DTW
- `src/fx/waveform_library.py` — JSONL でスライディング波形を蓄積
- `src/fx/waveform_backtest.py` — top-K類似 → 順方向リターン → `WaveformBias`
- `decision_engine.decide(waveform_bias=)` — **助言のみ**、不一致なら HOLD veto

### Phase D — Attribution / Calibration / External (commit `349d403`)
- `src/fx/attribution.py` — closed taxonomy で「原因候補」をランキング
  (CPI/FOMC/BOJ/NFP/YIELD/DXY/...) — 断定はしない
- `src/fx/strategy_config.py` + `src/fx/calibration.py` — パラメータ束ね +
  グリッド走査。**安定性スコアでランキング**（収益単独ではない）
- `src/fx/external/` — CSV 取り込みベースの外部ベンダー比較
  (スクレイピング禁止、CSV/Webhook のみ)

### Phase 5 (§16) — Execution-fill audit (commit `4efc1b8`)
- `ExecutionFill` dataclass — slippage / latency / Bid/Ask / broker_order_id
- `OANDABroker.place_order` が v20 fill response から実 fill 価格抽出
- `trades` テーブルに約定オーディット 14 列追加 (idempotent migration)
- `cmd_trade` が自動で fill を `save_trade` に渡す

### Documentation (commit `105aafe`)
- `DESIGN.md` に §10 (backtest 二系統) / §11 (timeline+overlay+macro) /
  §12 (waveform) / §13 (attribution+calibration+external) / §14 (fill audit)
  を追加
- §9.3 の §17/§19 マッピング表を 18 行に拡張
- `USER_GUIDE.md` に「典型的な研究ワークフロー」を追加

### StrategyConfig wiring (commit `e377969`)
- `_gather_inputs` と `cmd_trade` が `--config <yaml/json>` を受け取るように
- `decide()` の `min_risk_reward` / `min_confidence` が config 由来になる

### Audit / stabilization (commit `bb82457`)
- 上位足 map に **30m / 2h / 4h** を追加 (オフライン resample が
  silently UNKNOWN になっていた)
- `EngineTrade.exit_reason` の文字列リストを実装と整合
  (`stop | take_profit | max_holding | end_of_data` のみ。"flip" は未実装と明記)
- `decision_engine.py` の docstring を実挙動に揃える
  (LLM は「補助」ではなく「保守的に止める権限あり」)

### Merge (commit `bd5d5a4`)
- main の整理 (`.venv` / `templates/` / `Procfile` / `runtime.txt` 削除) を取り込み
- `requirements.txt` と `sample.py` はブランチ側を維持
- `sample.py` docstring から削除済み Procfile への参照を除去

---

## 6. ファイル構成 (主要なものだけ)

```
src/fx/
├── decision_engine.py    # 唯一の BUY/SELL/HOLD 判断点
├── risk_gate.py          # 閉じた taxonomy で blocking ルール
├── patterns.py           # swing / pattern / trend 認識 (no future leak)
├── analyst.py            # Claude を使った助言 (反証可能予測必須)
├── indicators.py         # テクニカル指標 + technical_signal
├── higher_timeframe.py   # live 上位足 fetch
├── correlation.py        # 関連通貨ペア相関
├── news.py               # yfinance ヘッドライン
├── calendar.py           # events.json + freshness
├── context.py            # MarketContext 集約
├── backtest.py           # 旧 backtest (テクニカル単独、ベースライン)
├── backtest_engine.py    # 新 backtest (Decision Engine 全段経由) ★
├── market_timeline.py    # point-in-time タイムライン ★
├── event_overlay.py      # イベント前後の値動き集計 ★
├── macro.py              # yields/DXY/VIX/equity の point-in-time fetch ★
├── waveform_matcher.py   # 類似度プリミティブ ★
├── waveform_library.py   # JSONL 波形ライブラリ ★
├── waveform_backtest.py  # 類似 → bias ★
├── attribution.py        # 候補原因ランキング ★
├── calibration.py        # パラメータグリッド走査 ★
├── strategy_config.py    # 設定束ね (YAML/JSON) ★
├── external/             # 外部ベンダー CSV 比較 ★
├── broker.py             # ABC + PaperBroker + ExecutionFill
├── oanda.py              # OANDA practice (3 重ロック + Bid/Ask 必須)
├── storage.py            # SQLite (analyses/predictions/trades/postmortems)
├── postmortem.py         # 失敗分析
├── prediction.py         # PENDING → CORRECT/PARTIAL/WRONG 評価
├── risk.py               # ATR / Kelly / position sizing
├── sentiment.py          # FX 側 sentiment reader
├── cli.py                # argparse エントリ (全サブコマンド)
└── web/                  # Flask ダッシュボード + SSE pipeline

src/sentiment/            # SNS/掲示板/RSS コレクター + Claude スコアリング
src/notify/               # LINE/log/null Notifier

tests/                    # 274 件、すべて pass
docs/
├── DESIGN.md             # §1-§14、仕様書 §1-§19 と一対一
├── USER_GUIDE.md         # CLI/ダッシュボード/cron/トラブルシューティング
├── ARCHITECTURE.md       # システム俯瞰
├── AUDIT.md              # ブランチ初期の整備記録
└── HANDOVER.md           # ★ このファイル
```

★ = PR #2 で追加/再構築されたモジュール (現在は main 上)

PR #4 が追加するファイル (まだ main にはない):

```
src/fx/
├── decision_trace.py         # 12 個の dataclass (オーディット用) ☆
└── decision_trace_build.py   # slice ビルダー + 第二パス + 仮想シミュレータ ☆
tests/
└── test_decision_trace.py    # 32 件 (不変性 7 件含む) ☆
```

☆ = PR #4 で導入予定 (open)。ロジック変更なし、観測のみ。

---

## 7. 重要な不変条件 (テストで pin 済み)

仕様書 §17/§19 と本実装の対応は `DESIGN.md §9.3` の表に集約。代表例:

| 不変条件 | テスト |
|---|---|
| Risk Gate NG なら必ず HOLD | `test_ai_cannot_override_risk_gate` |
| FOMC/BOJ/CPI/NFP の窓内は HOLD | `test_fomc_forces_hold` 他 |
| 三山候補だけでは SELL しない | `test_triple_top_requires_neckline_break` |
| SNS だけでは BUY/SELL しない | `test_sentiment_alone_never_trades` |
| 波形マッチだけでは BUY/SELL しない | `test_waveform_alone_never_trades_when_technical_is_hold` |
| バックテストで未来参照なし | `test_engine_backtest_no_future_leak` 他 |
| events.json 古い/欠落で live HOLD | `test_gate_blocks_when_calendar_required_and_stale` |
| spread_pct=None で live 拒否 | `test_gate_require_spread_blocks_without_quote` |
| Calibration は安定性で best | `test_calibrate_best_is_highest_stability_score` |
| 30m/2h/4h で上位足が UNKNOWN にならない | `test_higher_tf_coverage` |
| `capture_traces=True/False` で売買結果が完全一致 | `test_decisions_unchanged_with_trace_logging` (PR #4) |
| `metrics()` が trace ON/OFF で完全一致 | `test_metrics_dict_unchanged_with_trace_logging` (PR #4) |
| `future_outcome` 第二パスが判断に影響しない | `test_future_outcome_does_not_affect_decisions` (PR #4) |
| `event_evidence_ids = []` のときは `missing_event_evidence_reason` が必須 | `test_missing_evidence_reasons_distinct_from_empty_arrays` (PR #4) |

---

## 8. 既知の制限と研究時の前提

`backtest_engine.py` は **研究用** であり、本番成績ではない:

| 項目 | 状態 | 備考 |
|---|---|---|
| spread | ⚠️ 未反映 | `spread_pct=None` ハードコード |
| Bid/Ask 履歴 | ⚠️ 不可避 | yfinance に履歴なし |
| slippage | ⚠️ 未反映 | エントリー = bar の close |
| actual fill | ⚠️ 未反映 | live専用 |
| sentiment 過去データ | ⚠️ 未反映 | アーカイブ未保存 |
| events.json 鮮度判定 | ⚠️ 未反映 | live のみ |

**PR #3 (`fix/backtest-synthetic-metrics`, merged `4532e16`)** で `metrics()` に
`synthetic_execution: true` / `spread_mode` / `slippage_mode` / `fill_model` /
`bid_ask_mode` / `sentiment_archive` の 6 キーが追加された。この限界が metrics
側からも一目で分かる構造になっている (`_synthetic_execution_metadata()` で集中管理)。

PR #4 で各バーの `ExecutionAssumptionSlice` にも同じ 6 キーが乗るので、metrics
レベル / バーレベルの両方で synthetic 性が辿れるようになる。

---

## 9. ローカル開発のはじめ方

```bash
# 1. クローン + ブランチ選択
git clone <repo>
cd test

# 通常の開発: main から始める
git switch main

# PR #4 をローカル検証する場合
git switch claude/decision-trace-logging

# 2. 依存
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. テスト
pytest tests/ --ignore=tests/test_notify.py
# main:           250 passed / 1 skipped
# decision-trace: 281 passed / 1 skipped

# 4. 環境変数 (任意。LLM・通知・broker を使うなら)
cp .env.example .env
# ANTHROPIC_API_KEY=...     # LLM 助言を有効化したい場合
# OANDA_API_KEY=...         # demo broker 接続テスト
# OANDA_ACCOUNT_ID=...
# OANDA_ENV=practice
# OANDA_ALLOW_LIVE_ORDERS=yes
# LINE_CHANNEL_SECRET=...   # LINE Bot を使う場合

# 5. 動作確認の最短経路
python -m src.fx.cli analyze --symbol USDJPY=X --interval 1h --no-llm
python -m src.fx.cli backtest-engine --symbol USDJPY=X --interval 1h --period 60d
python -m src.fx.cli build-timeline --symbol USDJPY=X --interval 1h --period 30d

# 6. Web ダッシュボード
python sample.py     # debug mode :5000
# あるいは launch.py / start.command でダブルクリック起動
```

---

## 10. PR をマージするときの注意

- **PR #4 のマージ前**: PR 説明の「レビュー重点 6 点」を順番に確認。特に
  `indicators.py` line 121-153 の `technical_signal()` が **無変更** であること、
  `decide_action()` の前後コードが **無変更** であることを diff で確認する。
- マージ方式は **squash merge ではなく merge commit** 推奨
  (PR #2, #3 と同じ理由: dataclass 単位 / フェーズ単位のコミット粒度を
  研究記録として残す)。
- `mergeable_state` は GitHub の MCP 経由で確認 (`pull_request_read get`)。
  PR #4 push 時点 (`ba2a47a`) では mergeable=clean。
- main は既に PR #2 の大量削除 (`.venv/` / `templates/` / `Procfile` /
  `runtime.txt`) を取り込み済み。`requirements.txt` と `sample.py` は
  PR #2 で復活させてあるので追加対応は不要。

---

## 11. 次セクションでやるべき最小タスク (優先順)

これは `DESIGN.md §9.4` の残課題リスト + PR #4 の OUT-OF-SCOPE と整合。

### A. PR #4 を閉じる — 最優先
1. **PR #4 (`decision_trace v1`) のレビュー対応 → merge commit でマージ**
   - 281 passed / 1 skipped を再現確認
   - レビュー指摘があれば dataclass slice の追加フィールドで対応 (frozen=True
     を維持。判断ロジックには絶対に触らない)
   - `mergeable=clean` のままなら `merge_pull_request` で merge commit

### B. PR #4 の延長線 (decision_trace を実用化する小さな PR)
2. **trace を JSONL に書き出す CLI**: `python -m src.fx.cli backtest-engine
   --capture-traces --trace-out runs/<run_id>.jsonl`。`result.to_decision_trace_records()`
   が既に `list[dict]` を返すので `json.dumps` するだけ。1 PR / 〜100 行。
3. **`event_evidence_ids` を `data_sources.jsonl` / `extracted_facts.jsonl` と接続**:
   slice 構造は変更せず、`fact_<id>` 文字列を埋めるだけ。1 PR / 〜200 行。
4. **OANDA live trace の配線**: `decision_trace_build` を import して live 経路でも
   trace を出す。dataclass は I/O 非依存なのでビルダー再利用が可能。

### C. 中規模・low-risk (DESIGN.md §9.4 から継続)
5. `backtest-compare` CLI — 旧/新 backtest を 1 コマンド並走で diff JSON
6. events.json の自動更新 cron スクリプト + 信頼できるソース選定
7. ダッシュボードに `blocked_by` バッジ表示 (decision_trace の `blocked_by`
   list をそのまま使うとシンプル)
8. `loss_category` の半自動分類

### D. 検証 (未通電のもの)
9. OANDA demo に実接続して `--broker oanda --dry-run` で fill audit と
   live trace が想定通り出るか手動確認 (コードはOK、未通電状態)
10. 1日損失上限 / 連敗上限を `RiskState` に渡す自動化

### E. 触らない方が良いところ
- waveform / external comparator / YouTube context は **新規実装しない**
  (前ユーザーから明示の禁止事項)
- live (real-money) アカウント対応は絶対にしない (3 重ロックの spirit)
- `_technical_score` の共有関数化は **scope を切ってから別 PR** で。PR #4 は
  B 案 (`technical_signal_reasons` が `technical_signal` に委譲) でドリフト不可
  にしている。同 PR で合体させない。
- `decision_engine.decide()` のシグネチャや内部判断順序を勝手に変えない。
  仕様書 §0 の「AI は最終判断者ではない」を守るため、判断点は1個に留める。

---

## 12. 引き継ぎチェックリスト (次セクション開始時)

新しい作業者 / 新しいフェーズに入るときは順に確認:

- [ ] `git switch main && git pull origin main` で `4532e16` を確認
- [ ] `git fetch origin claude/decision-trace-logging` で PR #4 の HEAD を確認
- [ ] `pytest tests/ --ignore=tests/test_notify.py` を main で 250 passed を確認
- [ ] PR #4 をローカル checkout して 281 passed を確認 (回帰なしを実体験)
- [ ] `mcp__github__pull_request_read get` で PR #4 の最新コメント / レビューを確認
- [ ] このファイル (HANDOVER.md) 全部 + `DESIGN.md §10〜§14` を読む
- [ ] `USER_GUIDE.md` の「典型的な研究ワークフロー」を一度通しで実行
- [ ] `claude/create-handover-docs-fAEL8` ブランチを `claude/create-handover-docs-SpYQB`
      に統一するか判断 (内容は同一なので片方を削除して良い)

---

## 13. 重要な思想 (仕様書 §20 を再掲)

> このシステムの目的は、いきなり儲けることではない。
>
> 目的は、過去の値動きと情報を重ねて、何が効いて、何がダメで、
> どの条件では入ってはいけないかを見つけること。

数値が良いバックテスト結果を信用しないこと。**Risk Gate が何を止めたか**
(`hold_reasons`) を読み、**安定性スコア** (`stability_score`) を見て、
**仮説 → 検証 → 失敗分類 → ルール修正** のループを回すための土台。

PR #4 の `decision_trace` は **このループを支えるオーディット層**。
バーごとに「なぜそうなったか」が JSON 1 行で全部辿れるので、`hold_reasons`
の集計より一段細かく失敗分類できるようになる。マージ後はこれを `loss_category`
の半自動分類 (タスク #8) に直接つなげるのが筋。

---

このドキュメントの起点コミット: `482fc13` (HANDOVER.md = PR #3 mark merged)。  
反映時点: 2026-04-27 (PR #4 が open になった当日)。  
コードが進んだら、特に PR #4 が merge された / 後続 PR が開始された段階で
このファイルを更新すること (§4 と §11 が真っ先に陳腐化する)。
