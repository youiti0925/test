# Royal-road focus chart layout — 引き継ぎ資料

**新セッションにそのまま貼り付けて、続きを依頼してください。**

---

## 1. 現状サマリ

**タスク**: ユーザー mockup の見やすさを取り込んだ「royal-road focus chart layout」を visual_audit に実装中。実装は完了し、テストも全部 PASS している。残りは commit / push + preview 再生成 + 最終報告のみ。

**ブランチ状態**:
```
feat/royal-road-complete-v2:    head = dd3bd31 (origin と同期)
artifact/visual-audit-mobile-preview: head = 826de93 (origin と同期)
```

**作業中ブランチ**: `feat/royal-road-complete-v2` (working tree に未 commit の変更あり)

**未 commit ファイル**:
```
modified:   src/fx/visual_audit.py                       (+649 -70 行)
modified:   tests/test_visual_audit_structural_lines.py  (16 行更新)
new file:   tests/test_visual_audit_marker_anchoring.py
new file:   tests/test_visual_audit_royal_road_mobile_layout.py
```

`git diff > /tmp/royal_road_focus_wip.patch` で patch を保存済み (47 KB)。
別マシンに引っ越す場合はこの patch + 2 つの未 tracked test ファイルを持ち出してください。

---

## 2. 実装済みの内容 (9 項目)

すべて `src/fx/visual_audit.py` の中:

1. **`royal-road-focus-layout` グリッドラップ**
   `_mobile_case_section_html` の `g5-chart-row` を `royal-road-focus-layout` に拡張。
   `g5-chart-cell` に `royal-road-main-chart-card` クラスを追加。
   CSS で grid 2 カラム → モバイルで stack。

2. **グループラベル統合**
   `_wave_derived_lines_svg_fragment` を改修。同じ price 帯の WNL/ENTRY/SNL1, WSL/STOP/SIL1, WTP/TP/STP1 を 1 つの `<g class='price-label-group ...'>` 内の単一バッジに統合。
   構造ライン側 (SNL/SIL/STP) は同じ価格に独立 `<text>` を出さなくなり、id だけ `<line ...stroke-opacity='0'>` placeholder で残して `data-structural-line-id="SNL1"` 等の grep を保つ。

3. **royal-road wave-line polyline (P1→NL→P2→BR / B1→NL→B2→BR)**
   `_wave_overlay_svg_fragment` で matched_parts から canonical 順に visit する `<polyline class='wave-skeleton-line royal-road-wave-line' data-rr-sequence='P1,NL,P2,BR' ...>` を描画。

4. **DT P1↔P2 / DB B1↔B2 構造線の強化**
   structural-lines block で `kind=structural_trendline + role=top_structure_line` のとき `double-top-topline` + `royal-road-main-structure-line` クラス追加、stroke-width 3.0 / opacity 0.85、ラベルを「STL1 P1-P2」(DB なら「STL1 B1-B2」) に変更。`data-anchor-parts="P1,P2"` + `data-anchor-quality='EXACT'`。

5. **WNL に royal-road-neckline クラス**
   `_wave_derived_lines_svg_fragment` で role=`entry_confirmation_line` の `<line>` に `royal-road-neckline neckline-line entry-trigger-line` を追加。

6. **BREAK / RETEST / CONFIRM markers の anchor**
   marker block を全面書き換え:
   - BREAK: `pattern_levels.parts.BR.index` があれば `EXACT` でその x、なければ `n*0.6` 付近で `APPROX` (ラベルは `BREAK~`)
   - RETEST: BR + a few bars (`EXACT`) または `n*0.78` 付近 (`APPROX`)
   - CONFIRM: 最終 visible bar (`EXACT`, `data-anchor-part='last_bar'`)
   全マーカーが `data-anchor-quality='EXACT|APPROX'` + `data-anchor-part='...'` + `data-source='entry_plan'` を持つ。

7. **WAIT_BREAKOUT のとき BR projected 表示**
   `_wave_overlay_svg_fragment` に `breakout_confirmed: bool` 引数追加。`False` のとき BR ラベルを `BR?` に置換し、class に `wave-part-label-projected` 追加、`data-projected='true'` `data-reason='breakout_not_confirmed'`。
   BREAK marker は `if breakout_confirmed:` の中なので自動的に出ない。

8. **T1 primary / T2/T3 secondary 区別**
   trendline-selected loop で `tl_idx == 1` のときだけ `trendline-primary` (opacity 0.9, stroke 2.6)、それ以外 `trendline-secondary` (opacity 0.18, stroke 1.2)。

9. **コンパクトパネル + 詳細 details**
   `_render_g5_royal_road_procedure_html` に `structural_lines` / `entry_plan` 引数追加。
   先頭に `royal-road-compact-panel` を出力 — `<dl class='royal-road-compact-checklist'>` で 環境認識/ダウ/波形/ネックライン/トレンドライン/ブレイク/リターンムーブ/確認足/RR/補助線 を sentence 形式で表示、最後に `royal-road-compact-result` で結論。
   既存 14 行表は `<details class='royal-road-detail-checklist'><summary>詳細14項目を見る</summary>...</details>` に折りたたみ。

CSS 追加 (`/* Royal-road focus chart layout */` ブロック):
- `.royal-road-focus-layout` (grid)
- `.royal-road-main-chart-card`
- `.royal-road-compact-panel`
- `.royal-road-compact-checklist` (dl grid)
- `.royal-road-compact-result`
- `.royal-road-detail-checklist`
- `.wave-part-label-projected` (opacity 0.6 + italic)
- `.royal-procedure-marker-approx` (italic)

---

## 3. テスト状況

新規テスト:
- `tests/test_visual_audit_royal_road_mobile_layout.py`: 7 tests (layout / DT canonical lines / DB canonical / 結合ラベル / T1-T2 区別 / 補助線サマリ)
- `tests/test_visual_audit_marker_anchoring.py`: 4 tests (anchor 属性 / quality 値 / WAIT_BREAKOUT 時 projected / CONFIRM=last_bar)

更新したテスト:
- `tests/test_visual_audit_structural_lines.py`: SNL1 を独立ラベルから「グループラベル + data-structural-line-id placeholder」に変更

PASS 確認済み:
```
tests/test_visual_audit_royal_road_mobile_layout.py:  7 / 7  PASS in ~110s
tests/test_visual_audit_marker_anchoring.py:          4 / 4  PASS
tests/test_visual_audit_structural_lines.py:          9 / 9  PASS in ~167s
```

未確認 (次セッションで実行):
```
tests/test_visual_audit_chart_geometry_visibility.py
tests/test_visual_audit_royal_road_procedure.py
tests/test_visual_audit_entry_candidates.py
tests/test_entry_candidates.py
tests/test_structural_lines.py
tests/test_royal_road_procedure_checklist.py
tests/test_royal_road_integrated_decision.py
tests/test_decision_trace.py
tests/test_royal_road_decision_v2.py
```

これらは layout 変更で影響を受ける可能性が低いはずだが、念のため全部走らせて確認。

---

## 4. 残タスク (新セッションで実施)

```
1. working tree 復元 (引っ越した場合のみ)
   - feat/royal-road-complete-v2 を head=dd3bd31 で checkout
   - /tmp/royal_road_focus_wip.patch を git apply
   - 2 つの新規 test ファイルを配置
2. 全 test スイートを再実行 (上記未確認分含む)
3. すべて PASS したら feat へ commit + push:
     git add src/fx/visual_audit.py \
             tests/test_visual_audit_structural_lines.py \
             tests/test_visual_audit_royal_road_mobile_layout.py \
             tests/test_visual_audit_marker_anchoring.py
     git commit -m "feat: add royal-road focus chart layout"
     git push origin feat/royal-road-complete-v2
4. artifact branch で 8 mobile preview を再生成 (~3 分、並列)
5. preview を docs/royal_road_current_preview/ にコピー
6. index.html の changelog 更新 (生成元 commit を新 sha に)
7. artifact へ commit + push:
     git commit -m "docs: refresh royal-road focus chart preview"
     git push origin artifact/visual-audit-mobile-preview
8. 最終報告 (Royal-road focus chart layout report フォーマットで)
```

---

## 5. 守るべき制約

絶対にやらないこと:
- `entry_plan.py` の READY 条件を変えない
- final action を変えない
- `current_runtime` を変えない
- `detect_trendlines` / `trendline_context` / `support_resistance_v2` / `structural_lines` を消さない
- Daily10k に進まない
- AI監査に進まない
- live / OANDA / paper に触らない
- 90d / 180d / 365d 比較しない
- `scripts/compare_90d_royal_v1.py` を実行しない
- `main` に push しない
- 新規 PR を作らない
- PR #22 に触らない
- PR #23 を merge / close しない
- force push しない

---

## 6. 開始前確認コマンド

```bash
git fetch origin
git switch feat/royal-road-complete-v2
git status
git rev-parse --short HEAD       # → dd3bd31
git rev-parse --short origin/artifact/visual-audit-mobile-preview  # → 826de93

# patch を当てる場合:
git apply /tmp/royal_road_focus_wip.patch
# または別マシンなら patch の内容を持ち越して同じファイルを再現

# 未 tracked test を配置 (patch には含まれない):
ls tests/test_visual_audit_marker_anchoring.py
ls tests/test_visual_audit_royal_road_mobile_layout.py

# 確認:
python3 -m pytest tests/test_visual_audit_royal_road_mobile_layout.py tests/test_visual_audit_marker_anchoring.py -q
# 期待: 11 passed
```

---

## 7. 完了報告フォーマット

```
Royal-road focus chart layout report

1. Branch / commit
feat/royal-road-complete-v2:
  before: dd3bd31
  after:  <new sha>
  commit: feat: add royal-road focus chart layout
artifact/visual-audit-mobile-preview:
  before: 826de93
  after:  <new sha>
  commit: docs: refresh royal-road focus chart preview

2. Implemented
royal_road_focus display mode:        yes (default in mobile preview)
compact right panel:                  yes
details 14-step checklist preserved:  yes
grouped labels:
  WNL/ENTRY/SNL1: yes
  WSL/STOP/SIL1:  yes
  WTP/TP/STP1:    yes
double-top structure:
  P1->NL->P2->BR wave line:           yes (royal-road-wave-line + data-rr-sequence)
  P1-P2 top structure line:           yes (double-top-topline + royal-road-main-structure-line)
  neckline:                           yes (royal-road-neckline class)
BREAK/RETEST/CONFIRM anchoring:
  breakout: BR (EXACT) or right-side approx
  retest:   post_BR (EXACT) or right approx
  confirm:  last_bar (EXACT)
WAIT_BREAKOUT projected BR:           yes (BR? + data-projected='true')
numeric trendline focus:              T1 primary, T2/T3 secondary muted

3-7. (実行後に埋める)

8. Final classification
PASS
```

---

## 8. ハンドオフを使った再開フレーズ案

新セッションで以下を貼ってください:

> 続き: feat=`dd3bd31` の上に royal-road focus chart layout を実装中。working tree に未 commit の変更あり (visual_audit.py + 3 test ファイル)。新規テスト 11/11 + structural_lines test 9/9 PASS 確認済み。
>
> 残タスク:
> 1. 全 test 再実行で regression 確認
> 2. feat へ commit + push: `feat: add royal-road focus chart layout`
> 3. artifact で 8 preview 再生成
> 4. artifact へ commit + push: `docs: refresh royal-road focus chart preview`
> 5. 最終報告
>
> 詳細は `docs/handover/royal_road_focus_layout_handover.md` を参照してください。
