# Audit Report — 2026-04-25

仕様書 §16 の最初のタスク: 既存コードのうち、最終判断に**本当に効いているもの**と**飾り**を区別する。

## 結論

**現状は危険な状態。** 最終判断 (`combine()` in `strategy.py`) は **テクニカル × LLM の合意ルールのみ**で、Risk Gate は実装されていない。経済イベント・スプレッド・データ品質・RR は LLM のプロンプトに混ぜているだけで、強制 HOLD 条件として機能していない。

## モジュール別分類

| モジュール | 状態 | 詳細 |
|-----------|------|------|
| `src/fx/strategy.py` (`combine`) | ⚠ **これが現在の決定エンジン** | `cli.py:135` から呼ばれる。tech vs LLM の合意のみ。Risk Gate ナシ |
| `src/fx/calendar.py` | 📺 **表示されるだけ** | LLM プロンプトに渡る。**最終決定で参照されていない** |
| `src/fx/risk.py` (`plan_trade`) | 🟢 効いている (発注計画のみ) | `cmd_trade` で計画生成。決定ゲートではない |
| `src/fx/sentiment.py` (FX側リーダー) | 📺 **表示されるだけ** | LLM プロンプトに渡る。最終決定で参照されていない |
| `src/fx/prediction.py` | 🟢 効いている (事後評価) | リアルタイム決定には使わない (設計通り) |
| `src/fx/analyst.py` | 📺 LLM 呼び出し | TradeSignal を返すが、`combine()` が無視可能 |
| `src/fx/web/pipeline_stream.py` | 🟢 表示構造はOK | 各ステップを SSE で返す。決定は `combine()` を呼ぶだけ |
| `src/fx/indicators.py` | 🟢 効いている | テクニカル指標。**ただし波形認識ナシ** |

## 重大な欠落

仕様書 §5 の警告に対応する欠落:

| 仕様 | 現状 | 必要な対応 |
|------|------|-----------|
| FOMC/日銀会合時の強制 HOLD | ❌ ナシ | `risk_gate.py` を新設、Decision Engine で強制 HOLD |
| 波形認識 (HH/HL/LH/LL、三山等) | ❌ ナシ | `patterns.py` を新設 |
| ネックライン割れ確認 | ❌ ナシ | `patterns.py` で実装 |
| AI が Risk Gate を上書きできない構造 | ❌ ナシ | `decision_engine.py` を新設、AI シグナルを参考扱い |
| スプレッド異常検知 | ❌ ナシ | `risk_gate.py` |
| データ欠損による HOLD | ❌ ナシ | `risk_gate.py` |
| 1日損失上限・連敗上限 | ❌ ナシ | `risk_gate.py` |
| RR < 1.5 の HOLD | ❌ ナシ | Decision Engine |
| Postmortem 拡張ログ項目 | ⚠ 一部のみ | DB スキーマ拡張 |

## 改善計画

仕様書 §16 の順序に従う。本作業ブランチで:

1. ✅ **このレポート** — 現状監査
2. 🔨 `src/fx/patterns.py` — 波形/構造認識
3. 🔨 `src/fx/risk_gate.py` — Risk Gate(強制 HOLD 条件の集合体)
4. 🔨 `src/fx/decision_engine.py` — 固定ルール Decision Engine
5. 🔨 `src/fx/context.py` — 拡張 Snapshot を抱えるデータクラス
6. 🔨 `src/fx/strategy.py:combine` を Decision Engine 呼び出しに置き換え
7. 🔨 Storage に Postmortem 拡張カラムを追加
8. 🔨 必須テスト 11 本(§15)
9. 🔨 `analyst.py` のシステムプロンプトを「**助言者**としての立ち位置」に書き換え
10. 🔨 README / ARCHITECTURE.md / Web ダッシュボードに判断フロー表示

実発注は引き続き OFF。OANDABroker は 3 重ロックを維持。
