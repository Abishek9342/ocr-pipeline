## What this changes

## Why

## Benchmark impact
<!--
Required if this touches preprocessing/routing/fusion/post-processing:
paste before/after numbers from `python -m benchmark.run_benchmark` (or
`run_ablation`) for the presets your change plausibly affects. "N/A —
this is a docs/CLI/test-only change" is a fine answer when it's true.
-->

## Checklist
- [ ] `ruff check .` passes
- [ ] `pytest tests/` passes
- [ ] New behavior has a test (or a benchmark/ablation number, for
      accuracy-relevant changes)
