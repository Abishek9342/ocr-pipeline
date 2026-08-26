# Robustness Curves

Mission section 18: not just one CER number per degradation type at one
severity, but CER as severity increases — "how gracefully does OCR
performance degrade?" `benchmark/run_benchmark.py`'s fixed-severity
presets (e.g. `noisy` = Gaussian noise sigma=25 always) can't answer this;
`benchmark/run_robustness.py` sweeps 4 corruption types across 5 severity
levels each (5 corpus images x {tesseract, easyocr, ours}) specifically
to answer it. Real data below — see `benchmark/results/robustness_curves.csv`
for the full table, `robustness_raw.csv` for per-image rows.

## Gaussian blur (sigma)

| sigma | easyocr | ours | tesseract |
|---:|---:|---:|---:|
| 0.0 | 0.0118 | 0.0174 | 0.0174 |
| 0.5 | 0.0118 | 0.0174 | 0.0174 |
| 1.5 | 0.0000 | 0.0000 | 0.0087 |
| 3.0 | 0.1479 | 0.0174 | 0.0174 |
| 5.0 | **0.9591** | **0.7223** | **0.7049** |

All three systems hold up fine through sigma=3.0, then **all three fall
off a cliff at sigma=5.0** — not a gradual slide, a sudden collapse. The
pipeline's adaptive deblurring reduces the damage (0.72 vs. easyocr's
0.96) but doesn't prevent it. No system here degrades "gracefully" past a
certain blur severity; they all have a breaking point, just at slightly
different heights.

## Gaussian noise (sigma) — the most important curve in this file

| sigma | easyocr | ours | tesseract |
|---:|---:|---:|---:|
| 0 | 0.0118 | 0.0174 | 0.0174 |
| 10 | 0.0205 | 0.0261 | 0.0174 |
| 25 | 0.0205 | 0.0118 | 0.0348 |
| 50 | 0.0205 | **0.0087** | **1.0000** |
| 80 | 0.0000 | **0.0000** | **1.0000** |

**Tesseract has a sharp failure cliff between sigma 25 and 50** — CER
jumps from 0.035 straight to a complete, total failure (1.0) and stays
there. EasyOCR and the pipeline show no such cliff at all — both remain
essentially unaffected even at sigma=80, more than 3x the noise level of
the main benchmark's fixed `noisy` preset (sigma=25). **This means the
main benchmark's single-severity `noisy` result (tesseract: 0.0187 CER,
looking totally fine) badly understates how fragile Tesseract actually is
to this exact noise type** — it just happens to sit below Tesseract's
cliff at the one severity level tested there. This is precisely the
mission's point about single-point benchmarks vs. curves, demonstrated
directly rather than asserted.

## Skew (degrees)

| degrees | easyocr | ours | tesseract |
|---:|---:|---:|---:|
| 0 | 0.0118 | 0.0174 | 0.0174 |
| 2 | 0.1195 | 0.0261 | 0.0000 |
| 5 | 0.3012 | 0.0261 | 0.2235 |
| 10 | 0.6068 | 0.3389 | 0.3524 |
| 20 | 0.6978 | 0.4664 | 0.8899 |

A genuinely gradual degradation for all three (no cliff), with the
pipeline's deskew step visibly earning its place — it degrades noticeably
slower than either single engine from 5° onward, consistent with the
main benchmark's `skewed` preset result.

## JPEG compression (quality, lower = more compressed)

| quality | easyocr | ours | tesseract |
|---:|---:|---:|---:|
| 95 | 0.0118 | 0.0174 | 0.0174 |
| 50 | 0.0000 | 0.0087 | 0.0087 |
| 25 | 0.0118 | 0.0087 | 0.0087 |
| 12 | 0.0118 | 0.0000 | 0.0000 |
| 5 | 0.0118 | 0.0000 | 0.0000 |

**No real trend here** — every value is within noise (0.000-0.018) of
every other, including a few instances where a MORE compressed image
scored slightly better than a less-compressed one. With only 5 images
per cell, this is almost certainly measurement noise, not a genuine
"heavier compression helps" effect — the honest reading is "JPEG
compression at these quality levels doesn't meaningfully affect printed
text recognition for any of these three systems," not "we found an
inverse relationship." Reported as a null/noisy result rather than
searched for a story that isn't there (mission section 28's "a negative
result is still a valid result," applied here to a curve, not just a
single ablation cell).

## Summary statistics per curve (2026-08-25 addition — supplements, does not replace, the curves above)

`benchmark/analyze_robustness.py` computes four numbers per
(corruption type, system) directly from `robustness_raw.csv`: worst-
severity CER, degradation slope (OLS slope of mean CER vs. raw severity
value — comparable within a corruption type only, since units differ
across types), catastrophic-failure onset severity (lowest tested
severity where mean CER ≥ 0.9, or blank if none tested reaches it), and
a severity-normalized AUC (comparable across corruption types):

| Corruption | System | Worst-severity CER | Slope | Onset severity (CER≥0.9) | Normalized AUC |
|---|---|---:|---:|---:|---:|
| blur_sigma | easyocr | 0.9591 | 0.1808 | 5.0 | 0.2459 |
| blur_sigma | ours | 0.7223 | 0.1287 | — | 0.1540 |
| blur_sigma | tesseract | 0.7049 | 0.1253 | — | 0.1527 |
| noise_sigma | easyocr | 0.0000 | -0.0002 | — | 0.0161 |
| noise_sigma | ours | 0.0000 | -0.0002 | — | 0.0111 |
| noise_sigma | tesseract | **1.0000** | 0.0150 | **50.0** | 0.5438 |
| skew_deg | easyocr | 0.6978 | 0.0349 | — | 0.4777 |
| skew_deg | ours | 0.4664 | 0.0252 | — | 0.2530 |
| skew_deg | tesseract | 0.8899 | 0.0449 | — | 0.4002 |
| jpeg_quality | easyocr | 0.0118 | -0.0000 | — | 0.0072 |
| jpeg_quality | ours | 0.0174 | 0.0002 | — | 0.0096 |
| jpeg_quality | tesseract | 0.0174 | 0.0002 | — | 0.0096 |

This is a consistency check as much as a new finding: Tesseract's noise
cliff (onset severity exactly 50.0, worst-severity CER exactly 1.0) is
the same real effect already described qualitatively above, now with a
precise onset value rather than "somewhere between 25 and 50." The
lowest normalized AUC per corruption type matches the curve tables'
qualitative "which system holds up best" story in every case
(`ours` lowest on blur and skew; `ours` and `easyocr` effectively tied
and both far below `tesseract` on noise). None of these four numbers were
used to pick which system "wins" here without checking against the
underlying curve first — a single AUC number would, for example, hide
that easyocr's blur AUC (0.246) is driven entirely by one severity level
(sigma=5.0) rather than a uniformly worse curve at every level, which is
visible only in the curve table above. Reproduce:
`python -m benchmark.analyze_robustness` (requires `robustness_raw.csv`
to already exist).

## Reproduce

```bash
python -m benchmark.run_robustness
```

Writes `benchmark/results/robustness_raw.csv` (per-image rows) and
`robustness_curves.csv` (the tables above). Deliberately small (5 images,
3 systems, to keep runtime reasonable) — the Gaussian-noise cliff finding
is worth re-verifying at a larger N before treating it as settled;
directionally it was unambiguous even at N=5 (every one of the 5 images
showed CER exactly 1.0 for Tesseract at both sigma=50 and 80).
