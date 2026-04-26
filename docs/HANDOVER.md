# 引継ぎ資料 (Handover)

このブランチ `claude/forex-trading-app-E122S` (PR #2) を次の担当者に引き継ぐためのドキュメント。
**仕様の詳細**は `DESIGN.md`、**操作方法**は `USER_GUIDE.md`、**アーキテクチャ俯瞰**は
`ARCHITECTURE.md`、**レガシー監査記録**は `AUDIT.md` を参照。

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

## 2. ブランチの現状

| 項目 | 値 |
|---|---|
| ブランチ | `claude/forex-trading-app-E122S` |
| PR | `#2` (state=open, **mergeable=clean**) |
| 最新コミット | `bd5d5a4` (Merge main into branch — resolve 4-file conflict) |
| テスト | **274 passed / 0 failed** |
| 行数 | +19,267 / 112 files changed (PRトータル) |
| Python | 3.11+ 推奨 (ローカルテストは 3.11.15 で通過) |

### 直近のコミット 8 本

```
bd5d5a4  Merge main into branch — resolve 4-file conflict
bb82457  audit: minimal fixes from stabilization pass
e377969  Wire StrategyConfig into the live decision path
105aafe  docs: cover Phases A-D and §16 fill audit in DESIGN/USER_GUIDE
4efc1b8  Phase 5 (spec §16): execution-fill audit on every live order
349d403  Phase D: attribution, calibration, external comparator (spec §8/§9/§10)
3cc6653  Phase C: waveform matching + advisory-only Decision Engine wiring
62beb4b  Phase B: market timeline + event overlay (spec §5/§6)
```

---

## 3. このブランチで何が入ったか (フェーズ別)

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

## 4. ファイル構成 (主要なものだけ)

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

★ = このブランチで追加/再構築されたモジュール

---

## 5. 重要な不変条件 (テストで pin 済み)

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

---

## 6. 既知の制限と研究時の前提

`backtest_engine.py` は **研究用** であり、本番成績ではない:

| 項目 | 状態 | 備考 |
|---|---|---|
| spread | ⚠️ 未反映 | `spread_pct=None` ハードコード |
| Bid/Ask 履歴 | ⚠️ 不可避 | yfinance に履歴なし |
| slippage | ⚠️ 未反映 | エントリー = bar の close |
| actual fill | ⚠️ 未反映 | live専用 |
| sentiment 過去データ | ⚠️ 未反映 | アーカイブ未保存 |
| events.json 鮮度判定 | ⚠️ 未反映 | live のみ |

**ユーザーの直近の修正（未push）** で `metrics()` に
`synthetic_execution: true` 等のフラグが追加される予定。これによりこの限界が
metrics 側からも一目で分かる構造になる。

---

## 7. ローカル開発のはじめ方

```bash
# 1. クローン + ブランチ
git clone <repo>
cd test
git switch claude/forex-trading-app-E122S

# 2. 依存
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. テスト
pytest tests/                       # 274 passed が出れば OK

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

## 8. PR をマージするときの注意

- `mergeable_state="clean"` 確認済み (commit `bd5d5a4` 時点)
- main は **大量削除コミット** (`.venv/` / `templates/` / `Procfile` /
  `runtime.txt` / `requirements.txt`) を取り込んだ状態。本ブランチは
  `requirements.txt` と `sample.py` のみ復活させてある
- マージ時は **squash merge ではなく merge commit** 推奨
  (フェーズ別のコミット粒度が研究記録として有用)

---

## 9. 次にやるべき最小タスク (優先順)

これは `DESIGN.md §9.4` の残課題リストとも整合。

### A. 取り込み確認
1. **ユーザーが local で作業した synthetic_execution metrics の push & 確認**
   - `backtest_engine.py` と `tests/test_backtest_engine.py` 2ファイル変更
   - 51 行追加、2 コミット
   - push 後 `pytest tests/test_backtest_engine.py` を実行

### B. 中規模・low-risk
2. `backtest-compare` CLI — 旧/新 backtest を 1 コマンド並走で diff JSON
3. events.json の自動更新 cron スクリプト + 信頼できるソース選定
4. ダッシュボードに `blocked_by` バッジ表示
5. `loss_category` の半自動分類

### C. 検証
6. OANDA demo に実接続して `--broker oanda --dry-run` で fill audit
   が想定通り出るか手動確認 (コードはOK、未通電状態)
7. 1日損失上限 / 連敗上限を `RiskState` に渡す自動化

### D. 触らない方が良いところ
- waveform / external comparator / YouTube context は **新規実装しない**
  (前回ユーザーから明示の禁止事項)
- live (real-money) アカウント対応は絶対にしない (3 重ロックの spirit)

---

## 10. 重要な思想 (仕様書 §20 を再掲)

> このシステムの目的は、いきなり儲けることではない。
>
> 目的は、過去の値動きと情報を重ねて、何が効いて、何がダメで、
> どの条件では入ってはいけないかを見つけること。

数値が良いバックテスト結果を信用しないこと。**Risk Gate が何を止めたか**
（`hold_reasons`）を読み、**安定性スコア**（`stability_score`）を見て、
**仮説 → 検証 → 失敗分類 → ルール修正** のループを回すための土台。

---

このドキュメントは引継ぎ時点 (`bd5d5a4`, 2026-04-26) の状態を反映。
コードが進んだら適宜更新を。
