---
name: Feature / technique proposal
about: Propose a new preprocessing technique, engine adapter, or capability
title: ""
labels: enhancement
---

**What you're proposing**

**What it should improve, and on what kind of image** (a specific
degradation/document type this is meant to help with — "makes OCR better"
in general isn't falsifiable; "reduces CER on skewed receipts" is)

**How you'd measure whether it actually helps**

Per this project's engineering principle (see CONTRIBUTING.md and
`prompt.md`'s "Do not build N preprocessing techniques just to make the
code look impressive" rule): a new technique is expected to ship with (or
be followed promptly by) a benchmark/ablation run showing it helps for the
case it targets — and it's just as valuable to submit if the honest result
is "measured it, didn't help, documenting that."
