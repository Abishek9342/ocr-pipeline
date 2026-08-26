# Routing V2 Readiness Audit

Mission section 13: for each of the 11 degradation presets, where does
`select_primary_engine`'s current rule set (`ocr_resilience/engine_selection.py`)
already do well, where is there a real, measured gap, and — for each real
gap — what would it take to close it? This is a prioritization exercise
built entirely from evidence already collected this session
(`benchmark/results/condition_engine_table.csv`,
`benchmark/results/routing_regret_raw.csv`, `benchmark/results/raw_results.csv`),
not new speculation.

**The single most important methodological finding in this audit**: a
regret number computed by `analyze_routing.py` in isolation (single-engine
choice vs. the best RAW baseline engine, no preprocessing) can look
dramatically worse than what the actually-assembled pipeline achieves,
because preprocessing and ensemble-routing can compensate for an
imperfect engine pick. Two presets below (`salt_pepper`, `motion_blur`)
have a near-worst-possible isolated regret score AND a genuinely
excellent assembled-pipeline result, for two different reasons (a
preprocessing gate rescuing one, ensemble routing rescuing the other).
Reading only the regret table would have flagged both as urgent bugs;
reading the actual pipeline output shows they mostly aren't. This is
exactly why this audit checks BOTH layers before prioritizing anything.

## Prioritized table

| Condition | Current single-engine rule | Isolated regret (single-engine only) | Actual `ours` CER (assembled pipeline) | Real gap? | Signal needed | Difficulty | Priority |
|---|---|---:|---:|---|---|---|---|
| `salt_pepper` | none (falls through to first-available) | 0.979 (worst possible — picks Tesseract, which is CER 1.0 raw) | **0.0831** (beats every raw baseline) | No — already well handled, by `median_denoise` preprocessing, not engine choice | n/a | n/a | None |
| `motion_blur` | none (falls through to `paddleocr` via low-contrast rule on some images) | 0.178 (picks PaddleOCR 20/20 times; best is RapidOCR) | 0.087 (13/20 images route to full ensemble via the "likely handwritten" flag) | **Partial** — the ~7/20 images that stay on the single-engine path get PaddleOCR (baseline CER 0.233) instead of RapidOCR (0.120) | A directional/motion-blur-specific signal distinct from `blur_score` (which doesn't currently distinguish motion streaks from defocus blur) | Medium — needs a new quality metric (e.g. directional-gradient anisotropy), not just a threshold tweak | **Medium** |
| `smudged` | none (falls through to `tesseract`, 19/20 images) | not separately isolated | **0.0270**, vs. 0.0081 achievable with PaddleOCR | **Yes** — real, measured, ~3.3x CER left on the table, not compensated by ensemble (only 1/20 images ensemble here) | **Checked directly**: a spot-check of `assess()` on 5 smudged images found 4 of 5 raise NO quality flags at all (the 5th only `is_low_contrast`) — confirming this is NOT a "just add a rule using an existing field" fix. A genuinely new signal (e.g. local-contrast variance / ink-bleed detection) would need to be designed and validated before a rule could use it | Medium — new quality metric required, same category of work as the `motion_blur` row, not a quick rule addition | **Medium** |
| `light_blur` | mixed: `paddleocr` (10/20) / `tesseract` (9/20) via fallthrough, 1/20 ensemble | not separately isolated | 0.0143 (matches Tesseract's baseline, not PaddleOCR's 0.0043) | **Yes** — real, small-in-absolute-terms gap (~0.01 CER) on the ~9 fallthrough images | **Checked directly, and this is NOT the low-difficulty fix it first looked like**: a spot-check of `assess()` on 6 `light_blur` images found only 1 of 6 even sets `is_blurry` (blur_score mostly sits at 96-181, i.e. barely under to well above the existing 100.0 `is_blurry` threshold) — most `light_blur` images raise NO blur flag at all, so a "mild blur tier of the existing rule" would only catch a small minority of them, not the ~9 fallthrough images this row is about | Medium, not Low as first assumed — `blur_score`'s existing threshold doesn't separate `light_blur` from `clean` well enough to route on; would need a lower/differently-calibrated threshold, validated against evidence this session didn't collect (risks overfitting to this 20-image corpus, which this project's own hard limits warn against) | **Low** (real gap, but no cheap fix currently supported by the evidence) |
| `combo_hard` | `paddleocr` (via noise/contrast/blur/skew rules, whichever fires) | 0.018 (already much improved vs. the OLD router's 0.416) | 0.0681 (up from 0.0581 pre-`denoise()`-fix — see below) | **Yes, but the opposite direction from the others**: a recent fix (removing `is_noisy -> denoise()`) measurably improved `noisy` but measurably hurt `combo_hard`, because Tesseract's ensemble contribution here specifically benefits from denoising even though its single-engine `noisy` performance doesn't | Would need per-engine (not per-image) preprocessing to fix cleanly — the current architecture applies ONE preprocessed image to every engine in an ensemble | High — requires an architecture change (per-engine preprocessing), not a rule tweak | **Low** (real, but the fix is architecturally expensive relative to the size of the regression) |
| `heavy_blur` | `paddleocr` (severe-blur rule) when single-engine; 18/20 images route to full ensemble | 0.051 (low top-1 rate, 35%, on the single-engine subset) | **0.0065** — best result of any preset for `ours` | No — the isolated regret looks concerning, but 90% of images take the ensemble path where it doesn't matter, and PaddleOCR/Tesseract are nearly tied (0.0695 vs. 0.0701) on the few that don't | n/a | n/a | None |
| `jpeg_compressed` | none (falls through to `tesseract`, 17/20 — happens to be the actual best baseline engine here) | 0.015 | 0.0142 (matches best baseline) | **Fragile, not urgent** — currently optimal only because registration order happens to put Tesseract first; an unrelated future change to engine registration order could silently break this | JPEG-artifact/blockiness detection — no existing quality field captures this | Medium (new signal) | **Low** (works today, but should become an explicit rule for robustness, not left implicit) |
| `clean`, `skewed`, `low_contrast`, `noisy` | explicit rules (or, for `clean`, no rule needed) | 0.005–0.017, top-1 75-90% | All at or near the best achievable CER | No | n/a | n/a | None |

## What this changes about tomorrow's priorities

Every remaining real gap in this table (`smudged`, `light_blur`,
`motion_blur`) turned out, once actually checked against `assess()`'s
real output rather than assumed, to need a NEW or recalibrated quality
signal — none of them had a cheap "just add a rule using an existing
field" fix waiting to be written. This is itself the headline finding of
this audit: the easy wins from this session's earlier work
(`docs/NEXT_PHASE_REPORT.md`'s "just more rules written against evidence
that already exists") are exhausted. **`smudged` is the highest-priority
target for tomorrow** (3.3x CER left on the table, no ensemble
compensation) but the actual next task is designing and validating a new
signal, not writing a routing rule — writing one against an existing
field that doesn't actually correlate would repeat exactly the "guess,
don't measure" mistake `engine_selection.py`'s own docstring already
warns against for impulse noise. `combo_hard`'s
regression is the one case in this table where the CORRECT next step is
explicitly NOT a quick rule change — it would need a real architecture
change (per-engine preprocessing) to fix without trading it against
`noisy`'s win again, and that tradeoff is not obviously worth making for
a 0.01 CER swing on one preset.

## Reproducing the evidence behind this table

```bash
python -m benchmark.analyze_routing   # condition_engine_table.csv, routing_regret_raw.csv/summary.csv
python -m benchmark.run_benchmark --engines tesseract,easyocr,paddleocr,rapidocr,ours --presets all  # raw_results.csv
```

The per-preset `engine_used` breakdown used throughout this table comes
directly from `raw_results.csv`'s `engine_used`/`routing_reason` columns
for `system == "ours"`, grouped by `preset`.
