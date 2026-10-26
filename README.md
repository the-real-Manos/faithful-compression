# Faithful Compression

A reproducible study of what quantisation actually costs. The same open
instruction-tuned model is deployed under three numerical precisions (FP16,
Q8_0, Q4_K_M) and measured under a locked evaluation protocol across four
dimensions: systems cost, task capability, output stability, and refusal
behaviour.

**Thesis under test**: average benchmark scores understate the effect of
quantisation. Tail behaviour, item-level consistency and refusal metrics can
shift when FP16 drops to 4-bit even when mean accuracy looks flat. The claim is
pre-registered in [PROTOCOL.md](PROTOCOL.md) and is falsifiable; a null result
will be reported as the headline if that is what the data shows.

## Status

| Stage | State |
|-------|-------|
| Stage 0: scaffold, configs, protocol, GPU-free modules, tests | Complete |
| Stage 1: backend (llama.cpp via Vulkan) | Complete |
| Stage 2: model freeze, engine pin, Phase 0 smoke test | Complete (2026-09-11) |
| Stage 3: Suite S (systems) | Complete (2026-09-11) |
| Stage 4: Suite C (capability) | Complete (2026-09-11) |
| Stage 5: Suites R and B (reliability, behavioural) | Complete (2026-09-11; classifier corrected 2026-09-15) |
| Stage 6: aggregation, figures, technical note | Complete (2026-09-15) |
| Follow-up F1: JailbreakBench (exploratory) | Complete (2026-09-15) |

`PROTOCOL.md` was written before the first model file was downloaded and frozen
before the capability, reliability and behavioural runs. Results are in
`results/summary/` and the write-up in `reports/technical_note.md`. The smoke test
below is a feasibility check and is not part of any suite.

## Hardware sheet

| Field | Value |
|-------|-------|
| GPU | AMD Radeon RX 9070 XT |
| VRAM | 16 GB |
| Architecture | RDNA 4 (gfx1201) |
| Driver | 32.0.31021.5001 (pinned for the study) |
| OS | Windows 10 Home 10.0.19045 |
| Engine | llama.cpp v0.4.0, build b10809 (commit 5266f24da), Vulkan, pinned in `configs/hardware.yaml` |
| Device as seen by the engine | `Vulkan0: AMD Radeon RX 9070 XT` (16304 MiB) |
| Python | 3.12 |

### Phase 0 smoke test (2026-09-11)

`python scripts/00_smoke_hardware.py`. Context 4096, all layers on GPU, greedy,
256 generated tokens from a 45-token prompt. All three model files matched their
frozen publisher SHA256 before loading.

| Precision | Load (s) | Decode (tok/s) | TTFT (ms) | GPU weights (MiB) | GPU KV (MiB) | GPU compute (MiB) | Weights / file size | Greedy repeatable |
|-----------|----------|----------------|-----------|-------------------|--------------|-------------------|---------------------|-------------------|
| f16 | 4.55 | 83.9 | 34.4 | 6128.17 | 448.00 | 64.01 | 0.999 | byte-identical x3 |
| Q8_0 | 3.02 | 142.6 | 25.5 | 3255.90 | 448.00 | 64.01 | 0.998 | byte-identical x3 |
| Q4_K_M | 2.00 | 206.3 | 25.4 | 1918.35 | 448.00 | 64.01 | 0.996 | byte-identical x3 |

Single-run feasibility figures, not the Suite S measurements (which use 10
measured runs per prompt across 24 prompts).

Measured systems-suite prompt lengths (Llama 3.2 tokenizer): short mean 55
(50 to 60), medium mean 221 (210 to 237), long mean 507 (443 to 628) tokens.

The first engine launch after installing the binary took over 2 minutes while
the AMD driver compiled GPU shaders; later launches take 2 to 5 seconds.

## Design

The whole study rests on one invariant: **evaluation code never branches on
precision tier**. Every generation goes through a single `Backend` protocol, and
precision differs only in which GGUF file is passed to `Backend.load()`. If the
eval loop could tell which precision it was running, the comparison would not
mean anything.

Two further decisions do most of the methodological work:

- **Everything is paired by item ID.** `stats/paired.py` raises rather than
  silently dropping an item missing from any precision, and the bootstrap
  resamples paired deltas rather than the two arrays independently. Comparing
  unpaired aggregates would let a different item set stand behind each
  precision's mean.
- **Every run is manifested.** Each suite run appends one line to
  `results/raw/run_manifest.jsonl` recording the model file and its SHA256, the
  complete generation config including every sampler, the backend and its
  version, the driver, and a content hash of the exact prompt set. A result file
  with no matching manifest entry is not evidence.

## Repository layout

```
PROTOCOL.md         pre-registration: RQs, metrics, statistics, deviations, limitations
CLAUDE.md           instructions for AI coding agents working in this repo
configs/            all run configuration, YAML only, loaded with yaml.safe_load
data/               fixed prompt sets committed to the repo
src/                library code
scripts/            numbered entry points, 00 through 07, no argument parsing
tests/              pytest suite, no GPU or network required
results/raw/        append-only JSONL, one file per suite per precision
results/summary/    generated CSV summary tables
results/figures/    generated figures
reports/            technical note
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

This installs everything needed to run the test suite. The inference engine and
the model files are downloaded separately (below); neither is committed.

### Inference engine

The study runs a pinned, prebuilt llama.cpp `llama-server` binary as a
subprocess and talks to it over localhost using only the standard library. No
compilation, no admin rights.

1. Download the zip at `llama_cpp_asset_url` in `configs/hardware.yaml`
   (llama.cpp v0.4.0, build b10809, Vulkan).
2. Check its SHA256 against `llama_cpp_asset_sha256` in the same file.
3. Unzip it to `tools/llama.cpp/` so that `tools/llama.cpp/llama-server.exe`
   exists.

`llama-cpp-python` is listed as an optional extra but is not used: it has no
prebuilt wheels and its Vulkan build needs the LunarG Vulkan SDK and CMake.

### Model files and data

1. Download the three files named in `configs/model_primary.yaml` from
   `huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF` into `models/`. Their
   publisher SHA256 values are frozen in that file, and every run checks the
   files against them before loading.
2. Download `harmful_behaviors.csv` from the
   [llm-attacks](https://github.com/llm-attacks/llm-attacks) repository
   (`data/advbench/harmful_behaviors.csv`) to `data/behavioural/`. The copy
   used in the study (520 rows, columns `goal` and `target`) has SHA256
   `6cd1a5c63c07610d7eb67307772ee5606017ee950b5770ab288a2c487489d3e1`. It is
   gitignored rather than redistributed.
3. Follow-up F1 only: download the file at `pair.source_url` in
   `configs/suite_jbb.yaml` to the path in `pair.file`; it is checked against
   `pair.sha256` before use. The JailbreakBench behaviours load automatically
   from Hugging Face at the pinned `behaviors.revision`. Both are gitignored.

`models/` and `tools/` are gitignored. The hashes in `configs/` are what pin the
study.

## Reproduction

Scripts run in order and each reads its configuration from `configs/`:

```bash
python scripts/00_smoke_hardware.py      # verify model hashes, load each precision, check VRAM and determinism
python scripts/01_run_systems.py         # Suite S
python scripts/02_run_capability.py      # Suite C
python scripts/03_run_reliability.py     # Suite R
python scripts/04_run_behavioural.py     # Suite B
python scripts/05_aggregate_and_plot.py  # summary CSVs and figures
python scripts/06_build_report_tables.py # markdown tables for the report
python scripts/07_run_followup_jbb.py    # follow-up F1 (exploratory), then re-run 05 and 06
```

## Development

```bash
pytest tests/      # no GPU, model or network required
ruff check .
ruff format .
```

## Scope

This is a deployment-relevant measurement study on one model, one GPU and one
quantisation family. It is not a claim of frontier AI safety research, and its
limitations are declared in advance in
[PROTOCOL.md section 11](PROTOCOL.md#11-limitations-declared-in-advance) rather
than discovered afterwards.

The behavioural suite measures whether refusal behaviour shifts across precision
tiers using AdvBench and a fixed set of published jailbreak archetypes.

## What this public copy withholds

Some safety-evaluation material is deliberately not published, because several
of the prompts succeeded against the model and could be reused to cause harm.

- **The 30 jailbreak templates** (`data/behavioural/jailbreak_templates.jsonl`)
  are withheld. Their categories are described in `PROTOCOL.md` section 6 and
  the build spec.
- **Behavioural and follow-up result rows** (`results/raw/behavioural_*` and
  `results/raw/followup_*`) keep every label, matched pattern, token count and
  timing, but the prompt and response text are replaced by SHA256 fingerprints
  (`prompt_text_sha256`, `generated_text_sha256`). Every safety table in
  `results/summary/` can be recomputed from these rows with
  `scripts/05_aggregate_and_plot.py`.
- Consequently `scripts/04_run_behavioural.py` cannot be re-run from this copy
  without supplying your own templates. `scripts/07_run_followup_jbb.py` can: its
  data comes from the public JailbreakBench release pinned in
  `configs/suite_jbb.yaml`.

The study was run in a private repository whose full history, including the
pre-registration commit order, is summarised in [PROVENANCE.md](PROVENANCE.md).
