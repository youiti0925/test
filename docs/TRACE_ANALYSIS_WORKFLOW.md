# Decision Trace 分析ワークフロー

PR #4 から PR #10 までで完成した **検証パイプライン** の使い方、出力の読み方、判断ルール、禁止事項を 1 箇所にまとめたドキュメント。

このドキュメントはコードの状態を **PR #10 マージ時点 (main = `bd1f024`)** に固定して書かれている。新しい集計や live 配線が入ったらこのファイルも更新すること。

---

## 1. 現在の到達点

| PR | 役割 | 主な追加 |
|---|---|---|
| **PR #4** | judge & record | `decision_trace v1` — 12 dataclass / 19 rule_check ID で 1 バー = 1 完全オーディット |
| **PR #5** | save to disk | `backtest-engine --trace-out / --trace-out-default` で `runs/<run_id>/{run_metadata.json, decision_traces.jsonl, summary.json}` を書き出し |
| **PR #6** | read & tally | `trace-stats` CLI — 1 jsonl から 9 セクションの集計 JSON を出力 |
| **PR #7** | cross-section view | `cross_stats.{hold_reason_outcome, gate_effect_by_technical_action, final_action_by_outcome}` を追加 |
| **PR #8** | pooled across runs | `trace-stats-multi` CLI / `aggregate_many()` — per-run + global pooled aggregate |
| **PR #9** | Risk Gate energised | `data/events.json` を seed して `event_high` が初めて発火するように |
| **PR #10** | blocked_by attribution | `cross_stats.blocked_by_outcome` — Risk Gate コード別の効果評価 |

到達点を一行で：**「backtest_engine の各バー判断を ① 完全に記録し、② ディスクに保存し、③ 個別／pool で集計し、④ ルール別に効果を数値化できる」** 状態。

main 上のテスト数: **412 passed / 0 skipped** (PR #3 時点 250 → +162)

---

## 2. 基本コマンド

### 2-1. backtest を走らせて trace を書き出す

```bash
# trace-out-default → runs/<run_id>/ に自動出力
python -m src.fx.cli backtest-engine \
    --symbol USDJPY=X \
    --interval 1h \
    --period 180d \
    --trace-out-default

# 明示的なディレクトリへ
python -m src.fx.cli backtest-engine \
    --symbol USDJPY=X --interval 1h --period 180d \
    --trace-out runs/USDJPY_180d_v1/

# gzip 圧縮
python -m src.fx.cli backtest-engine ... --trace-out runs/X/ --gzip
```

生成物 (`runs/<run_id>/`):

```
run_metadata.json       (843 B 程度 — strategy hash / data hash / commit_sha 等)
decision_traces.jsonl   (約 14 MB / 4000 traces — 1 行 = 1 BarDecisionTrace)
summary.json            (1.3 KB — n_trades / win_rate / metrics 等)
```

### 2-2. 1 run を集計する

```bash
# コンパクト 1 行 JSON (default)
python -m src.fx.cli trace-stats runs/<run_id>/decision_traces.jsonl

# 整形出力 (人間が読む用)
python -m src.fx.cli trace-stats runs/<run_id>/decision_traces.jsonl --pretty

# top hold reasons の N を変更
python -m src.fx.cli trace-stats runs/<run_id>/decision_traces.jsonl \
    --pretty --top-hold-reasons 5

# .gz も透過読み
python -m src.fx.cli trace-stats runs/<run_id>/decision_traces.jsonl.gz --pretty
```

### 2-3. 複数 run を pool して集計する

```bash
# glob で全 run を集計 (per-run + global)
python -m src.fx.cli trace-stats-multi runs/*/decision_traces.jsonl --pretty

# 個別指定 (.jsonl と .gz の混在も可)
python -m src.fx.cli trace-stats-multi \
    runs/USDJPY_180d/decision_traces.jsonl \
    runs/EURUSD_180d/decision_traces.jsonl.gz \
    --pretty
```

### 2-4. ライブラリ呼び出し

```python
from src.fx.decision_trace_io import export_run
from src.fx.decision_trace_stats import aggregate_stats, aggregate_many

# 単一 run
stats = aggregate_stats("runs/X/decision_traces.jsonl")

# 複数 run pool
multi = aggregate_many([
    "runs/USDJPY/decision_traces.jsonl",
    "runs/EURUSD/decision_traces.jsonl",
])
```

### 失敗系の挙動 (CLI)

| 状況 | 挙動 |
|---|---|
| ファイル不在 | stderr + exit 2 |
| 完全に空 / 全行 malformed | `ValueError`、stderr + exit 2 |
| 一部の行が malformed | `consistency_checks.errors` に記録、stdout に集計 JSON、exit 2 |
| `aggregate_many` で 1 run 失敗 + 他 OK | `failed_runs[]` に記録、stdout に集計 JSON、exit 2 |
| 全 run 失敗 | stderr + exit 2 |
| 同じ run_id を複数 jsonl に検出 | `warnings` に記録 (error にしない)、exit 0 |

---

## 3. 見るべき指標

`trace-stats` の出力 JSON の中で、**ルール調整／検証判断のときに見るべき** フィールドを優先順に。

### 3-1. `top_hold_reasons` (上位 N の HOLD 理由)

```json
"top_hold_reasons": [
  {"reason": "Technical signal is HOLD; nothing to confirm", "count": 3768},
  {"reason": "Risk gate blocked: Non-Farm Payrolls (USD) within ±4h ...", "count": 16},
  ...
]
```

- 「**何が一番 HOLD を引き起こしているか**」の概要把握。
- 上位は通常 `"Technical signal is HOLD"` (シグナル空白) が支配的。
- 2 位以下に **Risk gate blocked: ...** や **Counter-trend ... against higher-timeframe ...** などのルール由来理由が出れば調査対象。

### 3-2. `cross_stats.hold_reason_outcome` (PR #7) — HOLD 理由ごとの効果

```json
"hold_reason_outcome": {
  "Counter-trend SELL against higher-timeframe UPTREND": {
    "n": 16,
    "gate_effect": {"COST_OPPORTUNITY": 15, "PROTECTED": 1},
    "outcome_if_technical_action_taken": {"WIN_MISSED": 15, "LOSS_AVOIDED": 1},
    "technical_only_action": {"SELL": 16},
    "return_stats": {
      "n_with_return": 16,
      "avg_pct": 0.304, "sum_pct": 4.858,
      "min_pct": -0.265, "max_pct": 0.390
    }
  }
}
```

- 「**この HOLD 理由は損失回避だったか、機会損失だったか**」を per-reason で集計。
- 主に **Decision Engine の chain 系ルール** (higher_tf_alignment / pattern_check / risk_reward_floor / llm_advisory) の評価に使う。

### 3-3. `cross_stats.blocked_by_outcome` (PR #10) — Risk Gate コードごとの効果

```json
"blocked_by_outcome": {
  "event_high": {
    "n": 798,
    "gate_effect": {"NO_CHANGE": 785, "COST_OPPORTUNITY": 8, "PROTECTED": 5},
    "outcome_if_technical_action_taken": {"N/A": 785, "WIN_MISSED": 8, "LOSS_AVOIDED": 5},
    "technical_only_action": {"HOLD": 785, "BUY": 11, "SELL": 2},
    "return_stats": {
      "n_with_return": 13,
      "avg_pct": 0.136, "sum_pct": 1.769,
      "min_pct": -0.356, "max_pct": 0.521
    }
  },
  "no_block": {...}
}
```

- 「**Risk Gate の各コード (event_high / spread_abnormal / etc.) は実際にどれだけ得／損したか**」を per-code で集計。
- 主に **Risk Gate のチューニング** (event の窓幅 / spread 閾値) の評価に使う。
- `blocked_by` が空のバーは合成キー `no_block` に入る。1 trace に複数コードがある場合は **各キーに +1 ずつ** (additive)。

### 3-4. `cross_stats.gate_effect_by_technical_action` (PR #7)

```json
"gate_effect_by_technical_action": {
  "BUY":  {"NO_CHANGE": 58, "PROTECTED": 10, "COST_OPPORTUNITY": 12},
  "SELL": {"NO_CHANGE": 60, "PROTECTED": 8,  "COST_OPPORTUNITY": 16}
}
```

- 「**Gate は BUY 側／SELL 側のどちらに偏って効いているか**」。両側で偏りがあれば、ルールの非対称性を疑う。

### 3-5. `cross_stats.final_action_by_outcome` (PR #7)

```json
"final_action_by_outcome": {
  "BUY":  {"WIN": 28, "LOSS": 30},
  "SELL": {"WIN": 26, "LOSS": 34},
  "HOLD": {"N/A": 12251, "LOSS_AVOIDED": 18, "WIN_MISSED": 28}
}
```

- 実際に取った action の outcome を確認。 `metrics.win_rate` との sanity check。

### 3-6. `return_stats` 系 — 数値で語るときに見るフィールド

| フィールド | 意味 |
|---|---|
| `n_with_return` | hypothetical_technical_trade_return_pct が non-null のバー数 |
| `sum_pct` | これらの return の合計 (パーセント) |
| `avg_pct` | `sum_pct / n_with_return` (再計算済み、null なら null) |
| `min_pct` / `max_pct` | 全 run / 全 bar 横断の最小／最大 |

`n_with_return` は HOLD かつ technical_only_action が directional な bar のみ正の値を持つ。

---

## 4. 解釈ルール

### 4-1. gate_effect の意味

| ラベル | 意味 |
|---|---|
| `PROTECTED` | engine が止めた → 走らせていたら **負けトレード**だった (損失回避) |
| `COST_OPPORTUNITY` | engine が止めた → 走らせていたら **勝ちトレード**だった (機会損失) |
| `NO_CHANGE` | technical 自体が HOLD だったので engine は何も変えていない (実質影響ゼロ) |

### 4-2. outcome_if_technical_action_taken の意味

| ラベル | 意味 |
|---|---|
| `WIN` | 実際に BUY/SELL を取り、勝った |
| `LOSS` | 実際に BUY/SELL を取り、負けた |
| `LOSS_AVOIDED` | HOLD 後、走らせていたら負けていた (= PROTECTED と対応) |
| `WIN_MISSED` | HOLD 後、走らせていたら勝っていた (= COST_OPPORTUNITY と対応) |
| `N/A` | technical も HOLD で測れない |

### 4-3. ルール調整の判断ガイド

「あるルール (HOLD 理由 / blocked_by コード) は net で得しているか？」を評価するときの読み方：

1. **`n` で対象数を確認**
   - n が小さい (< 10〜15) なら **断定しない**。「強い疑い」の表現にとどめる。
2. **`gate_effect` のうち判断を変えた件数 = `PROTECTED + COST_OPPORTUNITY`** を見る
   - `NO_CHANGE` は実質影響ゼロなので除外して考える。
3. **`return_stats.sum_pct` の符号で方向を判定**
   - **`sum_pct > 0`** → 「止めたことで取り逃し傾向」(機会損失寄り、ルールは net コスト)
   - **`sum_pct < 0`** → 「止めたことで損失回避傾向」(ルールは net 得)
4. **`min_pct` と `max_pct` で外れ値の影響を確認**
   - 1 件の極端な外れ値が `sum_pct` を支配していないか目視。
5. **per-symbol で違うか確認**
   - 1 通貨で出ている偏りが、3 通貨 pool でも残るか。

### 4-4. 判定の言い回しテンプレート

| n | sum_pct 符号 | 推奨表現 |
|---|---|---|
| n < 10 | + | "弱い示唆 — n 不足、参考値" |
| n < 10 | − | "弱い示唆 — n 不足、参考値" |
| 10 ≤ n < 30 | + | "**強い疑い** — net で取り逃し傾向、検討材料が揃った" |
| 10 ≤ n < 30 | − | "ルールは機能している様子、サンプル増で再評価" |
| n ≥ 30 | + | "ルールは net 機会損失、撤廃／緩和を検討すべき"（ただし統計検定は別途） |
| n ≥ 30 | − | "ルールは net で損失回避、維持" |

---

## 5. 現時点の実データからの示唆 (PR #10 マージ時点)

USDJPY=X / EURUSD=X / GBPUSD=X を 1h × 180d で `trace-stats-multi` 集計した結果に基づく。

### 5-1. `higher_tf_alignment` (Counter-trend 系) — **強い疑い**

```
"hold_reason_outcome" (3 通貨 pool):
  Counter-trend SELL against higher-timeframe UPTREND:
    n=16, gate_effect={COST_OPPORTUNITY:15, PROTECTED:1}, sum_pct=+4.86%
  Counter-trend BUY against higher-timeframe DOWNTREND:
    n=18, gate_effect={PROTECTED:8, COST_OPPORTUNITY:10}, sum_pct=+2.12%
```

- 3 通貨 pool で n=16〜18、両側で `sum_pct > 0` (取り逃し傾向)。
- とくに Counter-trend SELL は **15/16 が WIN_MISSED** で偏りが顕著。
- **判断**: 強い疑いだが n はまだ小さい (10 ≤ n < 30 帯)。**この PR ではルール変更しない**。
  - 次の評価窓: symbol 拡大 (AUD/CAD/CHF) または期間延長 (365d) で n を倍に増やしてから再評価。

### 5-2. `DOUBLE_TOP / DOUBLE_BOTTOM neckline` — **機能している様子**

```
"hold_reason_outcome" (3 通貨 pool):
  DOUBLE_TOP_CANDIDATE: neckline not yet broken on close:
    n=8, gate_effect={PROTECTED:7, COST_OPPORTUNITY:1}, sum_pct=-1.73%
  DOUBLE_BOTTOM_CANDIDATE: neckline not yet broken on close:
    n=4, gate_effect={PROTECTED:2, COST_OPPORTUNITY:2}, sum_pct=+0.17%
```

- DOUBLE_TOP は **7/8 が PROTECTED** で `sum_pct < 0` (損失回避方向)。期待どおり機能。
- DOUBLE_BOTTOM は n=4 で判断保留 (n < 10)。
- **判断**: DOUBLE_TOP は維持。DOUBLE_BOTTOM はサンプル増で再評価。

### 5-3. `event_high` (Risk Gate) — **n 不足、現時点で機会損失寄りに見える**

```
"blocked_by_outcome" (3 通貨 pool):
  event_high:
    n=798,
    gate_effect={NO_CHANGE:785, COST_OPPORTUNITY:8, PROTECTED:5},
    technical_only_action={HOLD:785, BUY:11, SELL:2},
    return_stats: {n_with_return:13, sum_pct:+1.769, avg_pct:+0.136}
```

- 接触したバー数は n=798 と多いが、**実際に判断を変えたのは PROTECTED 5 + COST_OPPORTUNITY 8 = 13 件のみ**。
  - 残り 785 はそもそも technical も HOLD だったので Gate の有無に関わらず HOLD。
- 13 件の `sum_pct = +1.77%` (機会損失寄り)、`WIN_MISSED:8 vs LOSS_AVOIDED:5`。
- **判断**: n=13 はまだ小さく断定しない。「**event_high は現時点で機会損失寄りに見える、ただし n 不足**」までが正しい言い回し。

### 5-4. 通貨ペア別の傾向

| symbol | 180d metrics |
|---|---|
| USDJPY=X | n_trades:19, win_rate:0.263, profit_factor:0.58, total_return:-1.59% |
| EURUSD=X | n_trades:11, win_rate:0.455, profit_factor:1.38, total_return:+0.55% |
| GBPUSD=X | n_trades:23, win_rate:0.609, profit_factor:2.37, total_return:+2.90% |

- USDJPY は赤字、GBPUSD は健全。**戦略パラメータが GBPUSD に最適化されすぎている可能性** がある (本 PR では仮説止まり)。

---

## 6. 推奨ワークフロー

「ルール調整を考える」ときの基本サイクル:

1. `runs/` を消す → backtest を流す → 集計を取る
   ```bash
   rm -rf runs/
   for s in USDJPY=X EURUSD=X GBPUSD=X; do
     python -m src.fx.cli backtest-engine --symbol "$s" --interval 1h \
        --period 180d --trace-out-default
   done
   python -m src.fx.cli trace-stats-multi runs/*/decision_traces.jsonl --pretty \
      > /tmp/multi.json
   ```
2. **`global.cross_stats.hold_reason_outcome` を見る** — chain 系ルールの効果を確認
3. **`global.cross_stats.blocked_by_outcome` を見る** — Risk Gate コードの効果を確認
4. **`global.top_hold_reasons` を見る** — 上位の HOLD 理由が偏っていないか
5. n と sum_pct の符号で「§4-4 言い回しテンプレート」に当てはめる
6. **n が小さい場合は判断保留** → サンプル増 (symbol 追加 / 期間延長 / 自動更新 cron) を先にやる
7. n が十分なら次の PR でルール調整 (このドキュメント PR ではしない)

---

## 7. やってはいけないこと (本 PR / ドキュメント整備)

- **このドキュメント PR では売買ロジックを変更しない**
  - `decision_engine.py` / `backtest_engine.py` の entry/exit ロジックは無変更
  - `risk_gate.py` の event_high / spread_abnormal などのロジックは無変更
- **`decision_trace.py` の schema を変更しない**
  - 12 個の dataclass / 19 個の rule_check ID は無変更
- **`data/events.json` の追加修正をしない**
  - PR #9 で公式照合済み、本 PR ではデータも触らない
- **live trace をまだ実装しない**
  - PR #11 (本 PR) はドキュメント、live 配線は次以降の PR
- **OANDA live / AI Review / dashboard / CSV / HTML / cron をいま入れない**
  - スコープ外、別 PR

### 「ルール調整」のタイミングの目安

ドキュメント PR は集計 PR を一旦締めくくるためのもの。これ以降「ルールを実際に変える PR」を立てるときは:

1. n ≥ 30 が pool で集まったか確認
2. 1 通貨だけでなく **複数通貨で同方向の傾向**が出ているか確認
3. min_pct / max_pct で外れ値依存していないか確認
4. ルール変更前の `cross_stats` snapshot を保存 (将来差分を見るため)
5. ルール変更後の同 fixture で再 smoke、`final_action_by_outcome` の改善を pin

---

## 8. 関連ファイル参照

| 役割 | ファイル |
|---|---|
| schema 本体 | `src/fx/decision_trace.py` |
| trace ビルダー | `src/fx/decision_trace_build.py` |
| backtest 配線 | `src/fx/backtest_engine.py` |
| ディスク export | `src/fx/decision_trace_io.py` |
| 集計 (single + multi) | `src/fx/decision_trace_stats.py` |
| CLI | `src/fx/cli.py` (`backtest-engine`, `trace-stats`, `trace-stats-multi`) |
| events.json | `data/events.json` (PR #9 で seed) |
| Risk Gate | `src/fx/risk_gate.py` (event_high / spread_abnormal / ...) |
| event 読み込み | `src/fx/calendar.py` |
| HANDOVER 上位資料 | `docs/HANDOVER.md` |

### 関連テスト

| テストファイル | カバー内容 |
|---|---|
| `tests/test_decision_trace.py` | schema / rule_check 19 種 / 不変性 |
| `tests/test_decision_trace_export.py` | export_run / strict overwrite / sibling summary.json |
| `tests/test_decision_trace_stats.py` | aggregate_stats / aggregate_many / cross_stats 4 種 |
| `tests/test_events_json_seed.py` | events.json 公式照合 / Risk Gate 発火 |

