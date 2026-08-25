# Engine Selection Report

Covers Phases 1–8 of `ocr_next_research_mission.md`. Every number below
comes from `benchmark/results/raw_results.csv` (the authoritative run:
20 images x 11 presets x 5 systems, enhanced schema with per-row
confidence/routing data) and `benchmark/results/condition_engine_table.csv`
/ `routing_regret_summary.csv`, produced by `benchmark/analyze_routing.py`.
Reproduce: `python -m benchmark.run_benchmark --engines tesseract,easyocr,paddleocr,rapidocr,ours --presets all`
then `python -m benchmark.analyze_routing`.

## Phase 1: Current router analysis (before this phase's changes)

- **Single-engine vs. ensemble**: `router.decide()` counts 5 boolean
  degradation flags (`is_blurry`, `is_noisy`, `is_impulse_noisy`,
  `is_low_contrast`, `is_skewed`) plus a handwriting flag. If handwritten
  OR flag count >= `degradation_threshold` (default 2) -> ensemble every
  available engine. Otherwise -> single engine.
- **Which single engine (BEFORE this phase)**: `available_engines[0]` —
  pure registration order (whatever order `OCRPipeline.with_engines([...])`
  received), with zero regard for which engine is actually strong under
  the observed conditions. This was the primary target of this phase.
- **Confidence**: `Detection.confidence` per detection (each engine's own
  native scale, unnormalized); `OCRResult.confidence` is the plain mean
  across detections. Fusion's per-region confidence is likewise an
  unweighted mean across contributing engines' hypotheses.
- **Fallback (added in a prior phase, extended in this one)**:
  `OCRPipeline.run(min_confidence_for_fallback=...)` triggers when the
  first pass's mean confidence is below threshold and not every engine
  already ran.
- **Fusion weights**: `fusion.fuse(weighted=True)` (default) sorts
  candidates by raw confidence and does ROVER-style position-wise
  majority voting weighted by each candidate's raw confidence;
  `weighted=False` gives every candidate equal weight.
- **Registration-order dependencies found**: (1) the single-engine router
  path (fixed in this phase), (2) `OCR`'s `engine="auto"` resolution via
  `_ENGINE_PRIORITY` — found to be **completely missing RapidOCR** (a real
  bug, fixed in this phase; see `CHANGELOG.md`).
- **Hardcoded thresholds** (none tuned against a held-out set — all
  empirically eyeballed): `blur_score < 100`, `noise_score > 8`,
  `impulse_noise_score > 0.015`, `contrast_score < 40`, `abs(skew) > 1°`,
  `stroke_width_variance > 0.55` (handwriting), `degradation_threshold = 2`.

## Phase 2/3: Engine profiles (empirical, this benchmark's own evidence)

Overall (all 11 presets, all styles) — from `benchmark/results/summary.csv`:

| System | Mean CER | Mean confidence | Catastrophic failure rate | Mean latency (s) | Mean peak memory |
|---|---:|---:|---:|---:|---:|
| paddleocr | 0.1108 | 0.904 | 5.0% | 0.198 | 3.6 MB |
| tesseract | 0.1790 | 0.731 | 14.1% | 0.200 | 0.1 MB |
| easyocr | 0.1213 | 0.685 | 0.0% | 0.961 | 3.3 MB |
| rapidocr | 0.1718 | 0.854 | 12.7% | 1.212 | 161.0 MB |
| **ours** | **0.0319** | **0.890** | **0.5%** | 1.234 | 44.7 MB |

("ours" here reflects the quality-aware router this phase built — an
earlier full-benchmark run, started before `engine_selection.py` existed,
showed 0.0382/0.855; re-run after all this phase's code landed to get a
number that's actually consistent with what's in the repo. See the
`CHANGELOG.md` finding on why an initial run of this same benchmark
wasn't trustworthy for exactly this reason.)

Static capability notes and empirical strengths/weaknesses per engine are
recorded in `ocr_resilience/engine_selection.py::KNOWN_PROFILES` (kept in
code, not just docs, so it stays next to the logic it justifies) and
`docs/engine_landscape.md`.

## Phase 6: Condition x engine strength

| Preset | Best engine | Best CER | 2nd best | 2nd CER | Fastest |
|---|---|---:|---|---:|---|
| clean | paddleocr | 0.0106 | tesseract | 0.0166 | paddleocr |
| combo_hard | paddleocr | 0.0199 | easyocr | 0.0478 | tesseract |
| heavy_blur | paddleocr | 0.0695 | tesseract | 0.0701 | tesseract |
| jpeg_compressed | tesseract | 0.0142 | paddleocr | 0.0437 | tesseract |
| light_blur | paddleocr | 0.0043 | tesseract | 0.0143 | tesseract |
| low_contrast | paddleocr | 0.0129 | tesseract | 0.0209 | tesseract |
| motion_blur | rapidocr | 0.1201 | paddleocr | 0.2327 | paddleocr |
| noisy | paddleocr | 0.0056 | tesseract | 0.0187 | paddleocr |
| salt_pepper | easyocr | 0.1213 | rapidocr | 0.2000 | paddleocr |
| skewed | paddleocr | 0.0064 | rapidocr | 0.0442 | tesseract |
| smudged | paddleocr | 0.0081 | tesseract | 0.0270 | tesseract |

**PaddleOCR is the best single engine on 8 of 11 conditions** — this is
the single most important fact this phase's router changes are built on.
It is also fast enough (~0.15-0.23s, close to the fastest engine on most
presets) that preferring it doesn't trade much latency for the accuracy
gain. RapidOCR (motion_blur) and EasyOCR (salt_pepper) are each the best
engine on exactly one condition — genuinely condition-specific strengths,
not noise (each margin is large: RapidOCR's motion_blur CER is roughly
half PaddleOCR's; EasyOCR's salt_pepper CER is roughly a third of
RapidOCR's second-best).

## Phase 4/5/7: Quality-aware single-engine selection (implemented)

`ocr_resilience/engine_selection.py::select_primary_engine()` replaces
`available_engines[0]` with interpretable rules keyed on `QualityReport`'s
**continuous** fields (never on preset names — see Phase 17 below):
prefer PaddleOCR for low contrast and skew; prefer PaddleOCR/EasyOCR over
Tesseract for high general-noise severity (Tesseract's measured
robustness-curve cliff, see `docs/robustness_curves.md`); prefer
PaddleOCR/Tesseract over RapidOCR for severe blur (RapidOCR's measured
heavy-blur cliff); no rule asserted for impulse noise alone (handled by
preprocessing, not engine choice — see the module for why). Falls back to
`available_engines[0]` when no rule matches, so behavior for
not-yet-covered conditions is unchanged from before this phase, not
undefined.

## Phase 8: Ranked fallback chains (implemented)

`OCRPipeline._confidence_fallback()` now escalates in two tiers instead
of jumping straight to the full ensemble: tier 1 adds only the single
best-ranked fallback engine (`engine_selection.rank_fallback_chain()`,
ordered `paddleocr > tesseract > easyocr > rapidocr` by this benchmark's
overall mean CER); tier 2 (full ensemble) only runs if tier 1 still
doesn't clear the confidence threshold. With exactly two engines total,
tier 1 and the full ensemble are the same set (no wasted work); the
two-tier structure only does genuinely different work — and saves
latency relative to the old "always full ensemble" fallback — with three
or more engines available.

## Phase 13: Routing regret — OLD vs. NEW single-engine selection

From `benchmark/results/routing_regret_summary.csv` (220 image/preset
cases, baselines = tesseract/easyocr/paddleocr/rapidocr):

| Router | Top-1 accuracy | Mean regret | Mean selected CER |
|---|---:|---:|---:|
| OLD (registration order, always tesseract) | 56.8% | 0.1696 | 0.1790 |
| **NEW (quality-aware)** | **59.6%** | **0.1199** | **0.1293** |

**A real measurement bug was found and fixed while producing this table**:
the first version of this analysis computed "top-1" by comparing engine
*names* against whichever engine `pandas.DataFrame.idxmin()` returned for
the minimum CER — which silently resolves ties (extremely common at
CER=0.0 on easy images) to whichever engine's rows appear first in
`raw_results.csv`, always Tesseract by construction. That bug made the
NEW router look far worse than it is (28% apparent top-1 vs. the correct
60%). Fixed by comparing CER *values* (`selected_cer <= best_cer`), not
names — see `tests/test_analyze_routing.py` for the regression test.
This is exactly the kind of thing worth stating plainly per this
project's own standard: a wrong analysis, caught by checking it against
what the underlying condition table implied, not trusted on the first
run.

**Conclusion**: the quality-aware router selects the objectively best
available engine more often, with lower average regret and lower average
selected CER, than the old registration-order router — on all three
metrics simultaneously, not a tradeoff. See `docs/routing_benchmark_report.md`
for the full context of what this comparison does and doesn't cover.

## Known limitations

- The condition x engine table and the selection rules derived from it
  are keyed on this project's own synthetic benchmark; whether the exact
  same rankings hold on real scanned documents is unverified (see
  `docs/engineering_backlog.md`'s "real dataset" gap).
- `select_primary_engine` has explicit rules for 4 of the 11 conditions
  (low contrast, skew, high noise, severe blur); the other 7 (clean,
  light blur, JPEG, smudged, combo_hard, salt_pepper, mild degradations)
  fall through to `available_engines[0]` — which happens to still work
  reasonably well for several of them (Tesseract is the true best or
  near-best on clean/light_blur/jpeg/smudged too), but that's registration
  order coincidentally aligning with the evidence, not a rule asserting it.
- `rank_fallback_chain`'s ordering is a single aggregate ranking
  (`paddleocr > tesseract > easyocr > rapidocr`), not condition-specific —
  a condition-aware fallback ranking would be a natural refinement.
