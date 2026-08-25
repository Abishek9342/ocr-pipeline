# OCR Engine Landscape (research notes, August 2026)

Before adding a fourth engine, the current open-source OCR ecosystem was
researched to identify genuinely strong candidates rather than adding a
library because it's popular. This records that research so the decision
is auditable later, and so re-evaluating candidates that were rejected for
environment-specific reasons (license, CPU latency) is a five-minute
lookup, not a re-investigation from scratch.

**Constraint that drove the decision:** this project's benchmark harness
runs on a CPU-only Windows dev machine and needs to stay lightweight
enough for GitHub Actions' CPU-only `ubuntu-latest` runners — no GPU
assumption anywhere in CI.

| Project | Version (Aug 2026) | Architecture | Detection | Recognition | Layout | CPU | GPU | License | Install | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| **RapidOCR** | active, PP-OCRv6-based models | PP-OCR CNNs exported to ONNX Runtime | Yes | Yes | Only via separate PaddleX doc models | Yes — built for CPU/edge | Optional | Apache-2.0 | `pip install rapidocr onnxruntime`, ~30MB wheel, no PyTorch/Paddle/TF dependency | **Added** — see `ocr_resilience/engines.py::RapidOCRAdapter` |
| **docTR** / **OnnxTR** | active | Modular CNN/transformer (DBNet/LinkNet det; CRNN/SAR/PARSeq/etc. rec) | Yes | Yes | Optional `detect_layout=True` | Yes — ~0.12-0.17s/page (FUNSD/CORD, mobilenet variants, official docs) | Yes | Apache-2.0 | `pip install python-doctr` (PyTorch/TF2) or `onnxtr[cpu]` (zero DL-framework dependency) | Strong second candidate — not added this round, worth revisiting |
| **Surya** (datalab-to/surya) | v0.22.1 | Single ~650M-param unified det+rec+layout VLM | Yes | Yes | Yes — layout, reading order, tables | Unverified on x86 CPU; only GPU (RTX 5090) and slow Apple-Silicon numbers found | Yes (primary target, via vLLM) | Code: Apache-2.0. **Model weights: modified Open RAIL-M — commercial license required above $5M revenue**, not a clean permissive license | `pip install surya-ocr`, downloads a multi-hundred-MB model | **Rejected**: license has a commercial paywall clause, and no verified CPU-only latency |
| **MMOCR** (OpenMMLab) | v1.0.0 (April 2023) | CNN/transformer zoo (DBNet, ABINet, SATRN, etc.) | Yes | Yes | Partial | Possible but heavy (`mmcv`+`mmdet`+`mmengine` version-locking) | Yes (primary target) | Apache-2.0 | Fragile — exact-version-matched install chain via `openmim` | **Rejected**: effectively unmaintained since 2023 (flagged inactive), fragile install |
| **Kraken** | v5.2 | VGSL-configurable CRNN + trainable segmentation | Yes | Yes | Yes — baselines, regions, reading order, ALTO/PAGE-XML export | Yes (slower) | Optional | Apache-2.0 | pip/conda, models fetched separately via `kraken get` | **Rejected (scope)**: specialized for historical/non-Latin/handwritten text, not this project's general-purpose target |
| **TrOCR** (Microsoft) | stable in `transformers` | Transformer encoder-decoder (BEiT + RoBERTa) | **No** — recognition only, needs pre-cropped line images | Yes | No | Yes ("works on CPU" per docs, GPU recommended) | Yes (recommended) | MIT | Trivial (`transformers`) | **Rejected (scope)**: no detection stage, English-centric, narrower than a full backend |
| PaddleOCR-VL, GOT-OCR2.0, dots.ocr, olmOCR, DeepSeek-OCR, GLM-OCR | 2025-2026 wave | Large VLM document parsers | Yes (unified) | Yes | Yes, strong | Not documented/likely impractical | Primary target (GPU serving) | Varies (mostly Apache-2.0/permissive) | Multi-hundred-MB to multi-GB weights, GPU-first reference deployments | **Rejected (environment)**: current SOTA on document benchmarks (~94-96% on OmniDocBench), but built and benchmarked for GPU serving, not a CPU-only CI budget |

## Decision

**RapidOCR added** (`ocr_resilience/engines.py::RapidOCRAdapter`) — the
only candidate with zero heavy ML-framework dependency (no PyTorch, no
PaddlePaddle, no TensorFlow), a small (~30MB) install, and explicit
CPU/edge design intent. It shares the PP-OCR model family with the
existing `PaddleOCRAdapter`, so this isn't a "which is better" choice so
much as "same model family, much lighter runtime" — a natural second
option for CI or resource-constrained deployment.

**docTR/OnnxTR is the strongest not-yet-added candidate** — genuinely
CPU-benchmarked, modular (swappable detection/recognition backbones), and
via OnnxTR installable with zero deep-learning framework dependency at
all. Worth adding next if a fifth engine is wanted; not added this round
simply to keep this iteration's scope to one new engine plus its
benchmark results.

**Surya is the most capable architecturally** (unified detection +
recognition + layout + reading order + table structure in one model, 90+
languages) and should be revisited if either (a) its model-weight license
becomes cleanly permissive, or (b) a verified CPU-only x86 latency number
appears — neither was true on the sources checked here.

## Re-checking this later

Re-run this research (WebSearch for each project name + "OCR" + current
year) periodically — per the mission's own instruction, this project
should stay benchmark-current rather than comparing against a frozen
snapshot of the ecosystem. `AVAILABLE_ENGINES` in `ocr_resilience/engines.py`
is the plug-in point; any new candidate implements the `OCREngine`
protocol (`name: str`, `recognize(image) -> list[Detection]`) and is
registered there, then benchmarked via `python -m benchmark.run_benchmark
--engines <new-name>,...`.
