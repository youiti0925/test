# Royal Road Gap Audit v1

Created: 2026-05-01
Branch at audit time: `docs/add-fx-masterclass-reference`
Code reference: `main` = `a2a4417` (PR #21 merged 後)
Baseline document: [`docs/technical_baseline_royal_road.md`](technical_baseline_royal_road.md)

This document is an audit only. No code is changed by this PR. No decision logic, risk gate, waveform wiring, live/OANDA path, or runs/libs is modified.

---

## 1. Executive Summary

**結論: 今のシステムは「勝てる判断システム」としては不十分。**

- royal-road 10 ステップのうち、決定ロジックに刺さっているのは **Step 1 (event window) / Step 2 (上位 TF veto) / Step 3 (trend veto + neckline 未割れ veto) / Step 7 (indicator 4 票投票) / Step 9 (ATR stop / RR floor / 連敗ブレーカー)** の **5 ステップだけ**
- 売買方向 (BUY / SELL) を生む唯一の源泉は `technical_signal()` の **4 indicator 投票** (RSI / MACD / SMA20-50 / BB position)。LLM と waveform は **veto しかできず、方向を提案できない**
- Step 4 (S/R) と Step 6 (candlestick) は **完全に欠落**。これは royal-road の根幹で、欠けたまま勝率を上げるのは構造的に難しい
- Step 8 (confluence) は **存在しない**。今ある 4 票投票は indicator 内部の majority vote であって、macro / structure / S/R / pattern / candle を横串で評価する集約レイヤーがない
- Step 7 の RSI 70/30 / BB position 0.2-0.8 を **regime に関係なく** 使っており、これは baseline が Step 7 で名指しで警告している項目そのもの
- **stop_atr_mult 1.5 vs 2.0 の検証は royal-road 上では Step 9 内のごく一部の最適化**。entry 品質 (Step 4-6) と confluence 品質 (Step 8) を放置したまま stop 幅だけ詰めても、勝てるシステムにはならない

### 勝つために足りないもの (重み順)

1. 水平 S/R 検出 (Step 4)
2. ローソク足トリガー (pin bar / engulfing) (Step 6)
3. 多源コンフルエンス・スコアリング (Step 8)
4. RSI regime filter / BB lifecycle (Step 7 の正しい使い方)
5. 構造ベース stop (recent swing / neckline) (Step 9 の核心)

### 今すぐやるべきこと

1. PR #22 の処遇を user 側で決定 (放置しても main には影響しないが、open のままだと後の handover branch 群とコンフリクトの種になる)
2. USDJPY n_traces mismatch の **再現条件** を 1 ファイルに整理 (再実行はまだしない)
3. 90d A/B 結果を「stop_atr_mult のみ差分」と注記して **暫定確定**
4. 王道手順を **observation-only** で trace に落とす設計 (`technical_confluence_v1`) — まだ実装しない、設計だけ
5. 売買判断ロジックは **触らない** (Step 8 のスコアが prove されるまで)

---

## 2. Current Branch / Repo State

| 項目 | 状態 |
|---|---|
| Current branch | `docs/add-fx-masterclass-reference` |
| HEAD (audit time) | `9b17504 docs: add reference materials README` |
| 1 個前 | `744c92d docs: add FX royal road technical baseline` |
| 2 個前 (= main 相当) | `a2a4417 Merge pull request #21` |
| 作業ツリー | clean (audit time) |
| `docs/technical_baseline_royal_road.md` | 存在 (15919 bytes) |
| `docs/reference/README.md` | 存在 (806 bytes) |
| `docs/reference/*.pptx` | **存在せず**。今回は追加しない方針で確定 |
| PR #22 | open (head=main / base=`claude/create-handover-docs-SpYQB` の逆方向、main には影響しない)。本監査では触らない |

---

## 3. Current Decision Logic

### BUY/SELL/HOLD を発生させているもの

唯一の方向源は `src/fx/indicators.py:141-182` の `technical_signal()`:

```
RSI < oversold (default 30)         → +1 BUY
RSI > overbought (default 70)       → +1 SELL
MACD hist > 0 AND MACD > signal     → +1 BUY
MACD hist < 0 AND MACD < signal     → +1 SELL
SMA20 > SMA50                       → +1 BUY
SMA20 < SMA50                       → +1 SELL
BB position < 0.2                   → +1 BUY
BB position > 0.8                   → +1 SELL

≥ 3 票同方向 → BUY/SELL  / その他 → HOLD
```

### HOLD として止めているもの (decision_engine.py:87-251)

| 順序 | 条件 | ファイル:行 |
|---|---|---|
| 1 | data_quality (≥50 bars, finite close) | risk_gate.py |
| 2 | calendar_freshness (live のみ) | risk_gate.py |
| 3 | event_high (FOMC ±6h 等の窓内) | risk_gate.py:87-99 |
| 4 | spread_abnormal (>0.05%) | risk_gate.py:217-249 |
| 5 | daily_loss_cap | risk_gate.py:252-266 |
| 6 | consecutive_losses (≥3 default) | risk_gate.py:269-282 |
| 7 | rule_unverified (rule age > 24h) | risk_gate.py |
| 8 | sentiment_spike (crisis keywords) | risk_gate.py:304-307 |
| 9 | technical_signal == HOLD | indicators.py:141-182 |
| 10 | pattern 未割れ (top/bottom で neckline 未 break) | decision_engine.py:128-148 |
| 11 | higher-TF counter-trend | decision_engine.py:151-165 |
| 12 | RR < 1.5 | decision_engine.py:168-174 |
| 13 | LLM advisory: confidence < 0.6 or 不一致 | decision_engine.py:182-195 |
| 14 | waveform bias 不一致 (directional) | decision_engine.py |

### Observation-only (decide_action から見えていない)

- `MacroContextSlice` 全フィールド (DXY trend / zscore, yield delta bp, 株価) — `decision_trace.py:503-603`、本体に「ALL observation-only — never read by decide_action / risk_gate」明記
- `FundamentalSlice` (warning events, news evidence)
- `LongTermTrendSlice` (monthly trend など)
- waveform 全フィールドは veto には使われるが direction には使われない
- `r_candidates_v1` 全 11 仮説 (R1〜R11)

### Metadata-only

- `parameter_defaults.PARAMETER_BASELINE_V1` (literature_baseline_v1) — PR #19 で metadata に乗せただけ
- PR #20 までの baseline は **decide_action に一切刺さっていない**

### Summary-only / R-candidate-only

- R1〜R11 の hypothesis 集計 (`decision_trace_r_candidates.py`)
- `summary.json` の rule_results / gate_effects / hold_reasons 集計 (`decision_trace_stats.py`)
- これらは **post-hoc 分析専用**、decide_action は読まない

### 名前だけある / 実体が弱い

- head_and_shoulders / inverse_head_and_shoulders: `PatternResult.detected_pattern` enum にあるが **検出関数なし**
- `fractional_kelly`: `risk.py:74` に実装ありだが **どこからも呼ばれていない**
- `signal_voting` (PARAMETER_BASELINE_V1 内 vote_threshold): runtime に未接続 (`runtime_parameters.SKIPPED_SECTIONS_PR21` に明記)

---

## 4. Royal Road Gap Audit Table

凡例:
- `→ BUY/SELL?`: decide_action の方向決定に直接読まれているか
- `HOLD blocker?`: HOLD 化に効くか
- `Trace?`: `BarDecisionTrace` JSONL に出るか
- `Metadata?`: `run_metadata.json` に出るか
- `Summary / R?`: `summary.json` または R-candidate に出るか
- `Obs-only first?`: まず observation-only で入れるべきか

| Step | Concept | Status | File / Function | → BUY/SELL? | HOLD blocker? | Trace? | Metadata? | Summary / R? | Risk if missing | Next step | Obs-only first? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 金利 (rate diff) | missing | — | No | No | No | No | No | macro 大方向不在 | カレント yield delta だけ既出。spread 計算追加 | Yes |
| 1 | DXY level / trend / zscore | obs-only | macro.py / decision_trace.py:503-603 | No | No | Yes | Yes | r_candidates 一部 | direction 判断に未使用 | regime label 追加 | Yes |
| 1 | VIX | obs-only | macro.py | No | No | Yes | Yes | No | risk-on/off 判定不在 | risk_on/off ラベル trace 化 | Yes |
| 1 | FOMC/BOJ/ECB/BOE/CPI/NFP/GDP | implemented | calendar.py / risk_gate.py:87-99 | No | **Yes** (event_high) | Yes | Yes | Yes | — | サプライズ大小 (forecast 比) 追加 | Yes |
| 1 | 要人発言 / speeches | partial | events.json seed | No | event_high 経由 | Yes | Yes | Yes | hawkish/dovish 評価不在 | tone 分類は後段 | Yes |
| 1 | 通貨強弱 | missing | — | No | No | No | No | No | 同方向通貨被り検出不在 | G-8 strength rank 観測 | Yes |
| 1 | risk-on/off | missing | — | No | No | No | No | No | regime 切替検出不在 | VIX + 株価 → ラベル | Yes |
| 2 | 上位 TF trend | implemented | higher_timeframe.py:18-50 | No | **Yes** | Yes | Yes | Yes | — | スタック化 (4H+1D+1W) | partial done |
| 2 | 上位 TF 押し戻し率 | missing | — | No | No | No | No | No | 押し目水準不明 | Fib retracement 観測 | Yes |
| 2 | SMA50 / SMA200 | obs-only | indicators.py | No | No | Yes (PR #20) | No | r_candidates 一部 | trend filter 弱い | label 化のみ | Yes |
| 3 | swing high / low 検出 | implemented | patterns.py:95-159 | No | indirect | Yes | No | No | — | stop に流用 (Step 9 へ) | already obs |
| 3 | HH / HL / LH / LL | implemented | patterns.py:171-189 | No | trend_state 経由 Yes | Yes | No | No | — | BOS フラグ追加 | already obs |
| 3 | trend_state (UP/DOWN/RANGE/VOLATILE) | implemented | patterns.py:192-211 | No | **Yes** (counter-trend) | Yes | No | No | — | TRANSITION ラベル追加 | partial done |
| 3 | break of structure (BOS) | missing | — | No | No | No | No | No | 構造転換検出不在 | swing 比較で算出 | Yes |
| 3 | 直近押し安値 / 戻り高値ラベル | missing | — | No | No | No | No | No | structure stop 不在 | swing 値の名前付け | Yes |
| 4 | 水平 support / resistance | **missing** | — | No | No | No | No | No | **エントリ位置質ゼロ** | level detector + 距離 ATR | Yes |
| 4 | trendline | missing | — | No | No | No | No | No | 角度情報なし | swing 接続線 | Yes |
| 4 | breakout / pullback | missing | — | No | No | No | No | No | 上昇継続判定不在 | level break + retest | Yes |
| 4 | role reversal | missing | — | No | No | No | No | No | 旧抵抗→支持 判定不在 | level 履歴記録 | Yes |
| 4 | fake breakout | missing | — | No | No | No | No | No | 騙し検出不在 | break + close 戻り | Yes |
| 4 | distance to level (ATR) | missing | — | No | No | No | No | No | 「中途半端な位置」検出不在 | level 検出後算出 | Yes |
| 5 | double_top / double_bottom | implemented | patterns.py:223-284 | No | **Yes** (未 break で HOLD) | Yes | No | No | — | confidence スコア化 | partial done |
| 5 | triple_top | implemented (信頼度付き) | patterns.py:244-269 | No | Yes | Yes | No | No | — | — | done |
| 5 | head_and_shoulders | **name-only** | enum 名のみ | No | No | No | No | No | 検出関数なし | 実装 | Yes |
| 5 | inverse H&S | name-only | enum 名のみ | No | No | No | No | No | 検出関数なし | 実装 | Yes |
| 5 | flag / wedge / triangle | missing | — | No | No | No | No | No | continuation 検出不在 | 検出ロジック | Yes |
| 5 | retest after breakout | missing | — | No | No | No | No | No | 二段確認なし | level + close 戻り | Yes |
| 6 | pin bar | **missing** | — | No | No | No | No | No | trigger 不在 | wick/body 比 | Yes |
| 6 | engulfing | missing | — | No | No | No | No | No | trigger 不在 | 前後足比較 | Yes |
| 6 | harami | missing | — | No | No | No | No | No | trigger 不在 | 内包足検出 | Yes |
| 6 | strong body / rejection wick | missing | — | No | No | No | No | No | momentum 質不在 | body/range 比 | Yes |
| 6 | 下位 TF trigger 設計 | missing | — | No | No | No | No | No | 1h で 15m 確認なし | 別 PR (data path) | 後段 |
| 7 | RSI (default 14) | implemented | indicators.py:18-25 | **Yes** (1 票) | indirectly | Yes | Yes (used 値) | r_candidates | regime-blind 危険 | regime filter 追加 | obs-only filter |
| 7 | RSI regime filter (range vs trend) | missing | — | No | No | No | No | No | trend で 70/30 反対サイン化 | trend_state × RSI ラベル | Yes |
| 7 | RSI divergence | partial | patterns.py: 検出関数あり | **No** (technical_signal に未接続) | No | Yes (PatternResult) | No | No | divergence 未活用 | technical_signal に組み込みは後段 | Yes (まず可視化) |
| 7 | MACD divergence | partial | patterns.py:319 macd_weakening | No | No | Yes | No | No | 同上 | 同上 | Yes |
| 7 | MACD line / signal / hist | implemented | indicators.py:28-34 | **Yes** (1 票) | — | Yes | Yes | r_candidates | 正負だけ評価 | 強度評価追加 | Yes |
| 7 | BB position | implemented | indicators.py:37-42 | **Yes** (1 票) | — | Yes | Yes | r_candidates | lifecycle 不在 | squeeze/walk ラベル | Yes |
| 7 | BB squeeze / expansion / band-walk | missing | — | No | No | No | No | No | breakout 前夜検出不在 | band 幅 history | Yes |
| 7 | SMA20 vs SMA50 | implemented | indicators.py | **Yes** (1 票) | — | Yes | No | r_candidates | slope なし | slope ラベル | Yes |
| 7 | SMA50 / SMA200 | obs-only (PR #20) | — | No | No | Yes | No | r_candidates | filter 弱 | trend filter 化 | Yes |
| 7 | signal_voting (vote_threshold) | metadata-only | parameter_defaults.py | No | No | No | Yes | No | runtime に未接続 | runtime_parameters の SKIPPED に明記済 | 後段 |
| 8 | macro × structure × S/R × pattern × candle 集約 | **missing** | — | No | No | No | No | No | **王道核欠落** | technical_confluence_v1 設計 | Yes (絶対) |
| 8 | bullish/bearish reasons 列挙 | missing | — | No | No | No | No | No | 何で BUY したかブラックボックス | trace 追加 | Yes |
| 8 | conflicting → HOLD | partial | LLM/waveform veto のみ | indirect | Yes | Yes | No | No | indicator 同士の conflict は見ていない | reason 列挙後判定 | Yes |
| 9 | ATR stop / TP | implemented | risk.py:28-56 | indirect (RR floor) | Yes (RR<1.5) | Yes | Yes | Yes | — | 構造 stop と並列出力 | obs first |
| 9 | structure-based stop | **missing** | swing 値はあるが未使用 | No | No | No | No | No | **大損抑制の核欠落** | swing low/high stop 計算 → trace | Yes |
| 9 | invalidation_price (Analyst) | partial | analyst output にあり | No (decide に流れない) | No | partial | No | No | LLM 出力が未活用 | trace に出すだけ先 | Yes |
| 9 | RR floor 1.5 | implemented | decision_engine.py:168-174 | indirect | Yes | Yes | Yes | Yes | — | 動的 RR 検討は後段 | done |
| 9 | OCO 表現 | missing | — | No | No | No | No | No | 注文戦略の表現力不足 | live 側、後段 | 後段 |
| 9 | position sizing (Kelly) | name-only | risk.py:74 (未呼出) | No | No | No | No | No | 一定サイズ | 呼び出しは決定変更扱い | NO |
| 9 | daily loss cap | implemented | risk_gate.py:252-266 | No | Yes | Yes | Yes | Yes | — | — | done |
| 9 | consecutive loss cap | implemented | risk_gate.py:269-282 | No | Yes | Yes | Yes | Yes | — | — | done |
| 10 | future outcome | implemented | decision_trace.py:740-785 | No (post-hoc) | — | Yes | — | Yes | — | — | done |
| 10 | summary.json | implemented | decision_trace_io.py | — | — | — | — | Yes | — | — | done |
| 10 | R-candidates | implemented | decision_trace_r_candidates.py R1-R11 | No | No | — | — | Yes | 王道カバー不足 | confluence 系を R12+ で追加 | Yes |
| 10 | sufficient_n | implemented | r_candidates schema | No | No | — | — | Yes | — | 確認のみ | done |
| 10 | win/loss attribution | partial | postmortem.py / attribution.py | No (post-hoc) | No | partial | — | partial | 粒度粗 | entry 質 × stop 質の二軸追加 | Yes |
| 10 | blocked / missed opportunity | partial | hold_reasons 集計あり | — | — | — | — | Yes | missed トレースの outcome 追跡なし | second-pass 拡張 | Yes |

---

## 5. Critical Missing Pieces

### High (これが無いと royal-road 上「勝てる構造」にならない)

1. **水平 S/R 検出 (Step 4 全部)** — エントリ位置の質を測る基準が一切ない
2. **ローソク足トリガー (Step 6 全部)** — 上位足ゾーン到達後の反応確認手段がない
3. **構造ベース stop (Step 9 核)** — ATR-only のため、構造的に意味のある invalidation 線を使っていない
4. **多源 confluence スコアリング (Step 8 全部)** — 4 indicator 票投票止まり、horizontal 集約なし
5. **RSI regime filter (Step 7 警告対象)** — トレンド中の RSI 70/30 を逆サインで使う設計のまま

### Medium (勝率の天井を上げる)

6. **BB lifecycle (squeeze / expansion / band-walk)** — Step 7、breakout 前検出に効く
7. **MACD / RSI divergence の technical_signal 接続** — patterns.py に検出ロジックはあるが decide_action に流れていない
8. **role reversal / fake breakout** — Step 4、騙し回避
9. **head_and_shoulders / flag / wedge / triangle** — Step 5、現状 enum 名のみ
10. **risk-on / risk-off ラベル** — Step 1、VIX + 株価 から計算可

### Low (持っていれば良い)

11. **通貨強弱ランキング** — 同方向取引重複の検出
12. **CPI/NFP のサプライズ大小** — 現状 binary
13. **押し戻し率 (Fib)** — Step 2 補強
14. **下位 TF trigger 設計 (1h で 15m 確認)** — データパス変更を伴うので別軸

---

## 6. `technical_confluence_v1` Proposal (実装はまだしない)

### 設計方針 (baseline §4 と整合)

- **observation-only** で開始。`decide_action` / `risk_gate` には接続しない
- 既存の 4 票投票結果は **可視化するだけ** (今は内部で消費されて捨てられている)
- すでに計算済みのデータ (trend_state / pattern / higher TF / macro) は **reuse**、新規計算は最小限
- Step 4 (S/R) と Step 6 (candlestick) は新規の検出ロジックが必要 — これは PR を分ける

### Trace schema 案 (`BarDecisionTrace` 新スライス)

```json
{
  "technical_confluence_v1": {
    "policy_version": "technical_confluence_v1",
    "market_regime": "TREND_UP|TREND_DOWN|RANGE|VOLATILE|TRANSITION|UNKNOWN",
    "dow_structure": {
      "last_swing_high": null,
      "last_swing_low": null,
      "structure_code": "HH|HL|LH|LL|MIXED",
      "bos_up": false,
      "bos_down": false
    },
    "mtf_context": {
      "upper_trend": "UP|DOWN|RANGE|UNKNOWN",
      "lower_trigger": "BULLISH|BEARISH|NONE|UNKNOWN",
      "alignment": "ALIGNED|MIXED|COUNTER|UNKNOWN"
    },
    "support_resistance": {
      "nearest_support": null,
      "nearest_resistance": null,
      "distance_to_support_atr": null,
      "distance_to_resistance_atr": null,
      "near_support": false,
      "near_resistance": false,
      "breakout": false,
      "pullback": false,
      "role_reversal": false,
      "fake_breakout": false
    },
    "candlestick_signal": {
      "bullish_pinbar": false,
      "bearish_pinbar": false,
      "bullish_engulfing": false,
      "bearish_engulfing": false,
      "harami": false,
      "strong_bull_body": false,
      "strong_bear_body": false,
      "rejection_wick": false
    },
    "chart_pattern": {
      "double_top": false, "double_bottom": false,
      "triple_top": false, "triple_bottom": false,
      "head_and_shoulders": false, "inverse_head_and_shoulders": false,
      "flag": false, "wedge": false, "triangle": false,
      "neckline_broken": false, "retested": false
    },
    "indicator_context": {
      "rsi_value": null, "rsi_regime_valid": null,
      "rsi_trend_continuation": null, "rsi_range_reversal": null,
      "macd_momentum_up": null, "macd_momentum_down": null,
      "macd_divergence_bull": null, "macd_divergence_bear": null,
      "rsi_divergence_bull": null, "rsi_divergence_bear": null,
      "bb_squeeze": null, "bb_expansion": null, "bb_band_walk": null,
      "ma_trend_support": null
    },
    "risk_plan_obs": {
      "atr_stop_distance_atr": null,
      "structure_stop_distance_atr": null,
      "structure_stop_price": null,
      "rr_atr_based": null,
      "rr_structure_based": null,
      "invalidation_clear": null
    },
    "vote_breakdown": {
      "indicator_buy_votes": null,
      "indicator_sell_votes": null,
      "voted_action": "BUY|SELL|HOLD",
      "macro_alignment": "BUY|SELL|NEUTRAL|UNKNOWN",
      "htf_alignment": "BUY|SELL|NEUTRAL|UNKNOWN",
      "structure_alignment": "BUY|SELL|NEUTRAL|UNKNOWN",
      "sr_alignment": "BUY|SELL|NEUTRAL|UNKNOWN",
      "candle_alignment": "BUY|SELL|NEUTRAL|UNKNOWN"
    },
    "final_confluence": {
      "label": "STRONG_BUY_SETUP|WEAK_BUY_SETUP|STRONG_SELL_SETUP|WEAK_SELL_SETUP|NO_TRADE|AVOID_TRADE|UNKNOWN",
      "score": null,
      "bullish_reasons": [],
      "bearish_reasons": [],
      "avoid_reasons": []
    }
  }
}
```

### `summary.json` 追加案

- `confluence_label_distribution`: STRONG_BUY_SETUP / WEAK_BUY_SETUP / ... の出現回数
- `confluence_score_histogram`: bin 化したスコア分布
- `vote_breakdown_distribution`: indicator_buy_votes ごとの件数

### `r_candidates_summary` 追加案 (R12〜R20 候補)

| ID | 仮説 | 計算 |
|---|---|---|
| R12 | 「STRONG_BUY_SETUP」 entry の WR は WEAK_BUY_SETUP より高い | label 別 verdict 集計 |
| R13 | S/R 近接 entry (distance_to_level_atr < 0.5) は WR が高い | label 別 |
| R14 | bullish_pinbar + near_support 同時 entry の WR | 同時集計 |
| R15 | RSI 30 を「regime=RANGE」でだけ採用すると WR が改善 | 条件別 |
| R16 | structure_stop と ATR_stop の R-multiple 比較 | per-trade |
| R17 | bb_squeeze 直後の breakout entry の WR | label 別 |
| R18 | macd_divergence 出現バーから N 本以内の entry の WR | conditional |
| R19 | confluence_score ≥ N の entry vs < N の entry | threshold sweep |
| R20 | counter-trend setup (htf_alignment=COUNTER) の WR | 既存だが集計対象に追加 |

### sufficient_n の考え方

- 既存 (n ≥ 30) を踏襲
- ただし confluence ラベル別はサンプルが薄くなりやすい → **label × symbol × period クロスで n ≥ 30** を要求
- ペアごとに足りなければ **「サンプル不足」明記**、結論扱いしない (baseline §7 の「low-n を結論扱いしない」と整合)

### 勝ち / 負け分析方法

- 各 trace を `outcome ∈ {WIN, LOSS, OPEN}` でフィルタ
- confluence_label × outcome のクロス集計 → `WR_by_label`
- bullish_reasons の Counter / bearish_reasons の Counter を WIN/LOSS で別集計 → 「勝ち trade で頻出する reason」「負け trade で頻出する reason」
- distance_to_support_atr / distance_to_resistance_atr の WIN/LOSS 分布比較 (ヒストグラム or KS test)

### blocked / missed opportunity 分析

- HOLD trace についても same future outcome を計算 (PR #20 で shadow_outcome 入っているはず)
- 「STRONG_BUY_SETUP だが technical_signal != BUY だったため HOLD」のケースで、shadow が WIN → **missed opportunity**
- 「technical_signal == BUY だが confluence_label = AVOID_TRADE」で実際は WIN → **lucky win** (rule の正当性は弱い)
- 上記を `summary.json` の `missed_strong_setups` / `avoided_lucky_wins` として件数集計

### 最初に observation-only で入れるべき項目 (優先度順)

1. `vote_breakdown` (4 票の中身を露出するだけ — **新規計算なし、既存値を保存**)
2. `dow_structure` (swing / BOS フラグ — 既存 swing 値の整形で済む)
3. `mtf_context` (既存 higher_timeframe + lower_trigger 枠だけ確保)
4. `risk_plan_obs` の `atr_stop_distance_atr` と `structure_stop_distance_atr` 並列出力 (既存 swing 値を使うだけ)
5. `chart_pattern` の現状検出済みパターン整形 (既存)

→ ここまでは **既存データの整形のみ** で実装可能

### 次に実装する (新規ロジック必要)

6. `support_resistance` (新規 detector)
7. `candlestick_signal` (新規 detector)
8. `indicator_context` の RSI regime / BB lifecycle / divergence (一部 patterns.py 流用)
9. `final_confluence` (集約スコアリング — 上記が揃ってから)

### 「まだ decision に接続してはいけない」項目

- **すべて**。`technical_confluence_v1` は **observation-only** で固定
- `decide_action` から読むのは confluence_label が **複数 symbol × 複数期間で WIN を分離する根拠を持った後** (baseline §4 step 6)
- 接続するなら別 PR (PR-D 相当)、しかも `--apply-confluence-filter` 等の opt-in flag 越しで現行と byte-identical を pin

---

## 7. Recommended Next Steps

ユーザー提示の順序:

```
A. PR #22 open状態の確認
B. USDJPY n_traces mismatch 確認
C. 90d A/B 確定
D. Royal road gap audit 完了
E. technical_confluence_v1 設計
F. technical_confluence_v1 observation-only 実装
G. 180d A/B
H. stop_atr_mult 継続検証
I. 売買ルール変更
```

### 評価

| 順 | 項目 | 妥当性評価 | 理由 |
|---|---|---|---|
| A | PR #22 確認 | **妥当 (本監査で完了)** | head=main, base=handover branch なので main は脅かさないが open のままだと累積コンフリクト元になる。**user 判断 (close / そのまま) のみで OK、merge 不要** |
| B | USDJPY n_traces mismatch 確認 | **妥当** | yfinance 1h の非決定性 (data_snapshot_hash 差) が一次仮説。**再現条件の特定だけ先**、再実行は不要。条件を docs に書き残す方が ROI 高い |
| C | 90d A/B 確定 | **妥当** | stop_atr_mult のみ差分という empirical 事実が出ている。注記付きで「暫定確定」、interpretation を fix |
| D | Royal road gap audit 完了 | **本ファイルで暫定完了** | 残り: file 化するなら `docs/AUDIT.md` 追記 or 新規 `docs/royal_road_gap_audit_v1.md` (= 本ファイル) |
| E | technical_confluence_v1 設計 | **妥当 (本ファイル §6 で着手)** | 詳細化は 1〜2 PR 分の規模。step 4 / step 6 detector は別 PR |
| F | observation-only 実装 | **妥当だが分割推奨** | §6 の優先 1〜5 (既存データ整形のみ) を **PR-C1**、優先 6〜9 (新規 detector) を **PR-C2 〜 C4** に分ける |
| G | 180d A/B | **延期推奨** | confluence 観測が入る前の 180d は同じ 4-票投票の長尺評価にしかならない。entry 品質を変える前にサンプル増やす意味が薄い。180d 完了済みなら結果は読むだけ |
| H | stop_atr_mult 継続検証 | **延期推奨** | royal-road Step 9 のごく一部。Step 4 + Step 6 (entry 質) が放置のままなら stop 幅最適化のリターンは小さい |
| I | 売買ルール変更 | **保留** | confluence label が WIN を分離する根拠が出るまで触らない。baseline §4 と §7 で明文化されている方針 |

### 推奨される真の実行順 (微調整版)

```
1. PR #22 を user 判断で処理 (close 推奨、ただし指示待ち)
2. USDJPY n_traces mismatch の再現条件と原因仮説を docs に整理 (再実行なし)
3. 90d A/B を「stop_atr_mult のみ差分の暫定確定」として docs にロック
4. 本レポートを docs/royal_road_gap_audit_v1.md として保存 (= 本ファイル)
5. technical_confluence_v1 設計詳細化 (本レポート §6 を雛形に DESIGN.md に追記)
6. PR-C1: confluence trace の "既存データ整形" 部分のみ observation-only 実装
   (vote_breakdown, dow_structure obs, mtf_context obs, risk_plan_obs, chart_pattern obs)
7. PR-C2: 水平 S/R detector 観測実装 (Step 4)
8. PR-C3: candlestick detector 観測実装 (Step 6)
9. PR-C4: confluence label 集約 + R12-R20 R-candidate 追加
10. (ここまで揃って) 90d / 180d / 365d で WIN/LOSS 分離を確認
11. (有意な分離が出たら) PR-D で opt-in flag 越しに decide_action 接続検討
12. stop_atr_mult や rule 変更は (10) のあとで
```

このうち 1〜5 は **コード変更ゼロ**、6〜9 は **decide_action 一切触らず**。
ユーザー判断ゲートは (1), (4 を作るか), (10 の解釈), (11 を発動するか) の 4 点。

---

## 8. Final Recommendation

### Q1: 今のシステムは勝つための判断として十分か

**No、十分ではない。**
売買方向を決める源が `technical_signal()` の 4 indicator 投票だけで、Step 4 (S/R) と Step 6 (candlestick) が完全欠落、Step 8 (confluence) は集約レイヤー自体が無い。これは「indicator vote が trend を当てた時だけ偶然勝つ」構造で、敗ける条件 (中途半端な位置でのエントリ、S/R 直前のエントリ、レンジ中の RSI 逆張りでトレンド開始) を見抜けない。

### Q2: 今のまま stop 幅だけ詰めるべきか

**No、勝率の天井に到達できない。**
stop_atr_mult 1.5 vs 2.0 の効果は royal-road Step 9 のごく一部の最適化で、entry 品質 (Step 4-6) と confluence 品質 (Step 8) を放置したまま stop を詰めても、TP 到達前に stop に当たる構造的な負けは減らない。90d で +0.61pp 出ているのは事実だが、これは **「現在の勝てる範囲の中での効率改善」** であって **「勝てる範囲を広げる」** 改善ではない。

### Q3: 王道手順に寄せるべきか

**Yes、寄せるべき。**
ただし baseline §4 / §7 通り、**順序を厳守**:

1. observation-only で trace に出す (まず可視化)
2. WIN/LOSS の分布差を検証 (本当に分離するか)
3. 単一 symbol / 単一期間で良く見えただけの指標を採用しない (multi-symbol × multi-period)
4. 上記を満たした指標だけを decide_action に接続する

### Q4: 新しい別手法を探すべきか

**No、現基盤を捨てる必要は皆無。**
既存の Risk Gate / backtest engine / decision trace / future outcome / R candidates / parameter A/B / event determinism は、まさに王道手順 (Step 8 confluence と Step 10 verification) を **計測するためのインフラ**。捨てると同じものを作り直すだけになる。

ただし、**売買判断の核は indicator-vote-centered から royal-road confluence-based に寄せる**。これは新手法を探すというより、既存の観測インフラの上に Step 4 / Step 6 / Step 8 を載せて confluence 判断に **段階的にシフトする** という設計。

### Q5: どの順番が一番勝てるシステムに近いか

```
[手前の片付け]
1. PR #22 をユーザー判断で処理
2. USDJPY mismatch 原因を docs に固定 (再実行なし)
3. 90d A/B を「stop_atr_mult 差分の暫定確定」として fix

[観測レイヤーの拡張]
4. 既存データ整形だけで confluence trace の 50% を出す (PR-C1)
5. 水平 S/R 検出を入れる (PR-C2) ← 王道で最大の欠落
6. candlestick 検出を入れる (PR-C3) ← 王道で 2 番目の欠落
7. confluence label と R12-R20 を出す (PR-C4)

[検証]
8. 90d / 180d / 365d で confluence label が WIN/LOSS を分離するか確認
9. multi-symbol / multi-period で頑健性確認

[判断ロジック更新]
10. 有意な分離が出た指標を opt-in flag で decide_action に統合 (PR-D)
11. stop_atr_mult や RR floor の動的化はこの後

[ペース感]
- 4-7 は各 PR 200-400 行規模、観測のみなので低リスク
- 8-9 は backtest 流すだけ、コード変更なし
- 10 は royal-road 上の意思決定なので user 承認ゲート
```

この順序の核は、**"検証のための検証で沼化しない"** という user の明示原則と整合する:

- 各 PR-C は明確な観測フィールド追加 (沼化しない)
- 8-9 は出力固定の集計 (やる/やらない明確)
- 10 はゲート (user 承認まで決定変更ゼロ)

---

## Non-Negotiable Guardrails (本監査時点で守られていること)

- コード変更: なし
- PR 作成 / 更新 / merge / close / comment: 一切なし (PR #22 含む)
- main への push: なし
- decision_engine / decide / decide_action / risk_gate / waveform / live: 触っていない
- USDJPY 専用ルールなし、live / OANDA / paper 触らず、runs/ libs/ コミットなし
- 低 n の結果を結論扱いしていない
- current_runtime と literature_baseline_v1 を混同していない
