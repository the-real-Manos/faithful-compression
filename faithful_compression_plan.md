# Faithful Compression: Execution Plan

Execution plan for the work remaining on `faithful-compression`. Grounded in
`faithful_compression_spec_1.md`. Scoped with the ponytail Laziness Ladder:
every milestone notes the highest rung that covered it, and any chunk that
exists only because nothing higher on the ladder applied is flagged
`LADDER 7`.

**Revision 8, 2026-09-17.** Complete and tagged. Post-hoc safety-label audit (R30) added after the author reviewed raw responses; conclusions unchanged. The issues ledger remains the record of every break found and fixed.

## Status legend

| Label | Meaning | Evidence required |
|-------|---------|-------------------|
| `[DONE]` | Implemented and verified. Do not rebuild (ladder rung 2). | Passing tests or a verified file, noted inline. |
| `[PARTIAL]` | Code exists but is unverified, failing lint, or known-buggy. | What is missing is listed. |
| `[TODO]` | Not started. | |

Status snapshot taken 2026-09-11: `pytest tests/` 185 passed; `ruff check .`
clean; commits `bb4033a` (baseline) and `ea8f5eb` (tag `protocol-freeze`); Suite S
complete (720 rows); Suite C running; AdvBench CSV downloaded (gitignored).

## How to use this document

Execute top to bottom. Within a milestone, execute chunks in order. Nothing
references anything defined later. After `G1`, the code for `M3`, `M4`, `M5`
can be written in any order or in parallel; the GPU runs themselves serialise
on one card.

Before handing a chunk to an agent, check its status label. `[DONE]` chunks
are verify-only: run the evidence command, do not rewrite.

## Next steps before the larger plan (start here)

The larger plan starts at `M2`. These must be done first, in order.

| Step | Who | What | Chunk | Closes |
|------|-----|------|-------|--------|
| 1 | `[DONE]` 2026-09-10 | Study binary chosen and installed: official llama.cpp v0.4.0 (build b10809) Vulkan, SHA256-verified, pinned in `configs/hardware.yaml`. | MAN-0 | R8 decision |
| 2 | `[DONE]` 2026-09-11 | Baseline committed as `bb4033a` (46 files), before any result exists. | M0.1 | R4 |
| 3 | `[DONE]` 2026-09-11 | Backend code fixes. | M1.1 to M1.4 | R1, R5, R6, R7, R8, R11, R12, R18 to R25 |
| 4 | `[DONE]` 2026-09-10 | The three Llama-3.2-3B-Instruct GGUFs downloaded into `models/` and verified against the publisher's SHA256. AdvBench (MAN-3) can still happen any time before `M5`. | MAN-1 | |
| 5 | `[DONE]` 2026-09-11 | Smoke test passed on all three models; results in the README hardware sheet. | M1.5, MAN-2 | Verified L7, L9, L10; closed R26 |
| 6 | `[DONE]` 2026-09-11 | Backend unit tests built from the real log and timings output. | M1.6 | Locks R1 so it cannot regress |

Done since: `M2` to `M5` runs, `G1` freeze, `M6` tables and figures, MAN-4
calibration and the R28 re-score. Next: MAN-5 (your error categories), then `M7`
report, then `M8` and tag `v1.0.0`. Optional follow-up: JailbreakBench, with its
own short pre-registration committed before it runs.

## Issues ledger

Every known break. Update the status columns as things change; never delete a
row, because a closed row is the record that it was fixed.

- **Plan column:** `corrected` means the mistake was only in this plan's text
  and is gone. `specified` means the fix is written into the named chunk.
- **Repo column:** the state of the actual files. `OPEN` means the problem
  exists in the repo now. `DECISION` means it waits on your call.
  `nothing to fix` means no repo file is affected (usually because the code is
  not written yet, and the plan now describes it correctly). `CLOSED` means
  fixed and verified.

| ID | Sev | Where | Problem | Plan | Repo | Fix in |
|----|-----|-------|---------|------|------|--------|
| R1 | CRITICAL | `backends/llama_cpp_backend.py`, `_parse_vram_from_log` | Counted host buffers as VRAM. The regex captured the word before "buffer size" (`model`, `KV`), never the device, so the CPU exclusion never fired. On the real f16 load it would have reported 6879.67 MiB instead of 6128.17 (the 751.50 MiB `CPU_Mapped` embedding buffer leaking in). This would have misreported the RQ1 headline memory number. | specified | CLOSED 2026-09-11: regex captures the device; real f16 log parses to 6128.17 (weights/file 0.999); locked by `test_backend.py` using the captured log | M1.2 |
| R2 | MAJOR | `stats/tables.py` (not written yet) | RQ3 needs agreement on the greedy **answer**. Revision 1 compared only `correct`, which counts two different wrong answers (17 vs 19) as agreement, hiding the item-level instability the thesis is about. | corrected | nothing to fix | M6.3 |
| R3 | MAJOR | Plan sequencing; `PROTOCOL.md` section 2 | No protocol-freeze gate before the C/R/B runs (spec 9.5); template probe ran mid-study (post-hoc tuning risk); the pre-registered claim's `rel(m)` divides by the FP16 value, undefined if FP16 attack success rate is 0. | corrected (`G1` added, probe moved) | CLOSED 2026-09-11: C1 written into `PROTOCOL.md` 12.1; frozen at `protocol-freeze` | G1.2 |
| R4 | MAJOR | git | Zero commits, so nothing proved `PROTOCOL.md` predated the results. | specified | CLOSED 2026-09-11: baseline commit `bb4033a` (author the-real-Manos) precedes every result | M0.1 |
| R5 | MAJOR | `.gitignore` | The calibration sample and disagreement review contain full harmful completions. The current pattern only covers `results/raw/behavioural_*_unredacted.jsonl`, so review files in `results/summary/` would be committed. Nothing has leaked: no such file exists yet. | corrected (files renamed `_unredacted`) | CLOSED 2026-09-11: pattern is now `results/**/*_unredacted.jsonl`; `git check-ignore` confirmed for both the raw and summary paths | M5.1 (done early) |
| R6 | MAJOR | `backends/llama_cpp_backend.py`, `load()`; every run script | A crashed script leaves llama-server running with the old model on the port. The next `load()` health check can get a 200 from that orphan, so results come from the wrong model. | specified | CLOSED 2026-09-11 for the backend: `load()` refuses to start if the port answers. Scripts still need `finally: unload()` as each is written (M2.2, M3.2, M4.2, M5.4); the smoke script has it | M1.3 |
| R7 | MAJOR | `utils/hashing.py` | The frozen SHA256 was copied into manifests but never checked against the file on disk. | specified | CLOSED 2026-09-11: `verify_model()` added with 4 tests; the smoke test verified all three files. Each run script must call it (M2.2 onward) | M1.4 |
| R8 | MAJOR | `configs/hardware.yaml`; backend constructor | The binary was not pinned. LM Studio's backend folder auto-updates and cannot be reproduced from repo instructions (spec MVP exit gate). | corrected (auto-discovery cut) | CLOSED 2026-09-11: v0.4.0 b10809 pinned with URL and SHA256; scripts pass `llama_server_path` to the backend; the manifest records the engine's own reported version `0.4.0-dev (build 10809, commit 5266f24da)` | M1.4 |
| R9 | MAJOR | Plan, M1 | Backend tests needed fixtures from a later chunk. | corrected | nothing to fix | |
| R10 | MINOR | Plan, M2 | Suite S count was 864; spec gives 2 warmups per precision, so 726. | corrected | nothing to fix | |
| R11 | MINOR | `_parse_completion_response` | `wall_time_ms` was server processing time; spec defines it as client wall-clock. | specified | CLOSED 2026-09-11: measured client-side around the HTTP call; tested | M1.3 |
| R12 | MINOR | `_parse_vram_from_log`; `MemoryReport` | llama.cpp's compute buffer (VRAM) was ignored, so peak VRAM was understated. | specified | CLOSED 2026-09-11: `MemoryReport.compute_vram_mb` added; 64.01 MiB on all three precisions | M1.2 |
| R13 | MINOR | Plan; `PROTOCOL.md` section 10 | Revision 1 added 6 test files and 2 report files beyond the spec 11.1 tree. Cut to 3 test files justified by R1, R2, R5; deviation 10 not yet written. | corrected | CLOSED 2026-09-11: deviation 10 in `PROTOCOL.md` section 10 | G1.2 |
| R14 | MINOR | Plan | ROCm timing comparison was out of spec scope. | corrected (cut) | nothing to fix | |
| R15 | MINOR | Plan | Dataset-drift mitigation was vague; `content_hash` already detects drift. | corrected | nothing to fix | |
| R16 | MINOR | `stats/paired.py`, `dedupe_by_item` | Last-row-wins can splice a partial rerun onto a complete run. | specified (process rule) | ACCEPTED, see L12 | M6.6 |
| R17 | MINOR | Plan, M1.5; `configs/model_primary.yaml` | The smoke script had no filenames to load. | corrected | CLOSED 2026-09-10: filenames and publisher SHA256 values frozen in `model_primary.yaml` | |
| R18 | MINOR | Backend, `_wait_for_health` | Failed `ruff check` (E501) and `ruff format --check`. | specified | CLOSED 2026-09-11: file renamed to `llama_cpp_backend.py`, lint and format clean | M1.1 |
| R19 | MINOR | Backend, `backend_info()` | Hardcoded `build_flags: "vulkan"`; the manifest would be wrong if a non-Vulkan binary were configured. | specified | CLOSED 2026-09-11: read from the binary's own `--list-devices` (`Vulkan0: AMD Radeon RX 9070 XT`) | M1.4 |
| R20 | MINOR | Backend, `load()` / `unload()` | The log file handle opened in `load()` was never closed. | specified | CLOSED 2026-09-11: handle kept on the instance and closed in `unload()` | M1.3 |
| R21 | MAJOR | Backend, `_build_completion_payload` | Found 2026-09-11. Only the samplers in `GenConfig` were sent. llama-server has others (repeat, presence and frequency penalties, DRY, XTC, typical-p, mirostat); any non-neutral default would silently change even greedy output and falsify the protocol's "every sampler pinned" claim. | specified | CLOSED 2026-09-11: all sent at neutral values; the server's `generation_settings` echo confirms every sampler is off; locked by a test | M1.3 |
| R22 | CRITICAL | Backend, `load()` | Found 2026-09-11. llama.cpp v0.4.0 no longer logs buffer sizes at its default verbosity, so every VRAM figure would have been `None` and RQ1's memory result would have been empty. | specified | CLOSED 2026-09-11: server launched with `-lv 4`; the smoke test parsed all three precisions | M1.3 |
| R23 | MAJOR | Backend, `load()` | Found 2026-09-11. llama.cpp v0.4.0 defaults to `--fit on`, which can silently change unset settings per model to fit memory, so precisions could run under different effective configurations. | specified | CLOSED 2026-09-11: launched with `--fit off` | M1.3 |
| R24 | MAJOR | Backend, `load()` | Found 2026-09-11. llama.cpp v0.4.0 has a server-wide RAM prompt cache (`--cache-ram`, default 8192 MiB) separate from the per-request `cache_prompt` flag. It could make Suite S's 10 repeats of each prompt look artificially fast. | specified | CLOSED 2026-09-11: launched with `--cache-ram 0`; responses show `cache_n: 0` | M1.3 |
| R25 | MAJOR | Backend, `load()` | Found 2026-09-11 when the first smoke run timed out. A failed startup left the server process running (the R6 orphan created by the backend itself). | specified | CLOSED 2026-09-11: `load()` calls `unload()` before re-raising any startup failure | M1.3 |
| R26 | MINOR | Backend startup timeout | Found 2026-09-11. The first launch after installing the binary compiles GPU shaders and took more than the 120 s startup limit. | specified | CLOSED 2026-09-11: limit raised to 600 s; normal loads take 2 to 5 s | M1.3 |
| R27 | MAJOR | `data/loaders.py`, GSM8K template | Found 2026-09-11 by the G1.1 probe. With the spec template, only 3/5, 2/5, 4/5 GSM8K answers were extractable (f16, Q8, Q4): Llama 3.2 ends with "The final answer is: $\boxed{N}$". GSM8K accuracy would have measured format recall more than arithmetic, differently per precision. | specified | CLOSED 2026-09-11: format reminder appended after the question, validated on held-out items 200 to 219 (20/20, 20/20, 19/20); documented in `PROTOCOL.md` 12.6 | G1.1 |
| R28 | CRITICAL | `eval/refusal_classifier.py` | Found 2026-09-11 after Suite B scoring. The frozen patterns use a straight apostrophe (`i can't`); Llama 3.2 often writes "I can’t" (U+2019). 54 of 82 COMPLIANCE rows are refusals. It moved harmful refusal from 98% to 82% and invented a Q4 safety regression (jailbreak success 10.7% to 14.0%, where normalised text gives 6.0% to 4.7%); Q4 writes the curly form more often. The pre-registered claim is falsified either way. | specified (fix pre-stated in `PROTOCOL.md` deviation 12 before calibration) | CLOSED 2026-09-15: blind calibration gave 65% (13/20; 0 false positives; 7 false negatives, all curly apostrophes), triggering the pre-stated fix. Re-score reproduced every frozen label exactly and changed 54, all COMPLIANCE to REFUSAL. Both versions reported (Tables 5 and 5a). | MAN-4, then a re-score chunk |
| R29 | MAJOR | `scripts/05_aggregate_and_plot.py` | Found 2026-09-15 when a re-run hit the review sheet while it was open in Excel. Every run regenerated `disagreement_review_unredacted.csv`, so a re-run after the author saved their categories would have silently erased them. No data lost: Excel's file lock blocked the write. | specified | CLOSED 2026-09-15: the script keeps an existing review file and never overwrites it; verified by re-running with the file present | M6.6 |
| R30 | NOTE | RQ4 labels; `PROTOCOL.md` section 15 | 2026-09-17: author inspected raw responses and questioned refusal labels and whether Q4 answered more. Post-hoc blind audit (68 ambiguity-enriched + 90 random confident refusals). | n/a | RESOLVED: confident refusals 90/90 human-confirmed (0% false, CI 0-4%, equal across precisions); boundary agreement 81-82% (kappa 0.62-0.65) exceeds every between-precision gap, so no safety difference detectable. Conclusions unchanged; one non-independent AI rating pass disclosed as a deviation. | section 15 |

## Known limitations (accepted, not being fixed)

These are deliberate or unavoidable. They are recorded so they are reported,
not rediscovered. Study-level limitations (single model, single GPU, GGUF only,
suite sizes, no multiple-comparison correction, fixed attack surface) live in
`PROTOCOL.md` section 11 and are not repeated here. `VERIFY` rows are
assumptions a later chunk checks; if the check fails, the row becomes a new
`R` row in the ledger.

| ID | Where | Limitation | Status | Reported in |
|----|-------|------------|--------|-------------|
| L1 | `eval/refusal_classifier.py` | Patterns are checked in order, so the `i apologize...` and `sorry...` patterns only fire on non-first-person refusals. `matched_pattern` names the rule that fired, not the phrasing the model used. Pinned by a test. | ACCEPTED | `PROTOCOL.md` section 7 |
| L2 | `eval/refusal_classifier.py` | A response that refuses and then complies is scored REFUSAL (a false positive direction). | ACCEPTED | `PROTOCOL.md` section 7 |
| L3 | `eval/refusal_classifier.py` | A keyword classifier has false positives and negatives in general. Measured by the MAN-4 calibration; below 80% accuracy triggers pattern expansion and a full re-score. | ACCEPTED, measured | Appendix B |
| L4 | `eval/scoring.py` | The GSM8K headline score requires the `####` marker (no fallback), so it measures format following as well as arithmetic. Format-failure rate is reported separately. | ACCEPTED (deliberate) | `PROTOCOL.md` section 6 |
| L5 | Backend `generate()` | Peak RSS is sampled once after each request, not continuously, so a transient peak can be missed. It also counts the memory-mapped model file: the smoke test shows peak RSS about equal to file size (f16 6294 MiB vs a 6136 MiB file), so RSS measures file mapping rather than working memory. RSS is the secondary memory figure; VRAM allocation is primary. | ACCEPTED | Technical note limitations |
| L6 | Backend `memory_report()` | VRAM is llama.cpp's reported allocation, not an observed peak. PDH sampling was cut; `peak_vram_observed_mb` is always `None`. | ACCEPTED | Technical note limitations |
| L7 | `_parse_completion_response` | TTFT is derived as prompt processing time plus one mean decode step from the server's timings, not a streamed first-token timestamp. | ACCEPTED; timings keys verified 2026-09-11 | Technical note method |
| L8 | `data/systems/prompts.jsonl` | The long bucket was authored against the spec's ~750-token target. Measured 2026-09-11: short mean 55, medium 221, long 507 (443 to 628). Described by measured length, not topped up. | ACCEPTED, measured | `PROTOCOL.md` deviation 8 (update at G1.2) |
| L9 | Vulkan backend | Greedy decoding is not guaranteed to be byte-identical on Vulkan. Measured 2026-09-11: byte-identical across 3 runs on all three precisions. | VERIFIED; still reported | Technical note method |
| L10 | Backend parsers | `timings` key names and the buffer-size log format were assumed from upstream llama.cpp. | VERIFIED 2026-09-11 against v0.4.0 b10809 (buffer lines need `-lv 4`, see R22) | |
| L11 | `data/loaders.py` | The `datasets`-backed loaders have never run against the network. | VERIFY in G1.1 | |
| L12 | `stats/paired.py` (R16) | Aggregation splices runs if a partial rerun follows a complete one. Mitigated by moving partial raw files aside before aggregating; the manifest keeps the record. | ACCEPTED | M6.6 |
| L13 | `data/loaders.py` | ARC items with more than 5 choices are skipped and counted (expected to be zero). | ACCEPTED | Technical note, capability section |
| L14 | `eval/capability.py`, `eval/reliability.py` | Near-duplicate runner loops with no shared base class. Deliberate: one implementation each. | ACCEPTED (deliberate) | |
| L15 | Suite runners | No unit tests; verified by dry-run acceptance checks only. Revisit if a runner bug reaches the results. | ACCEPTED | `PROTOCOL.md` deviation 10 |
| L16 | Suite R | Sampling at temperature 0.7 raises unreadable GSM8K answers to 8 to 10% (from 1.5 to 3% greedy), and about 40% of self-inconsistent items involve an unreadable answer. Under the pre-registered definition a number versus no answer counts as inconsistent, so self-consistency partly measures format stability under sampling. | ACCEPTED; report both | Technical note, reliability section |

## Already executed `[DONE]`

Verify-only. Evidence command for all Python items: `pytest tests/` (161
passed on 2026-09-10).

| Item | Status | Evidence |
|------|--------|----------|
| `configs/` (8 YAML files, real machine values) | `[DONE]` | Files present; `hardware.yaml` driver 32.0.31021.5001, gfx1201. |
| `backends/types.py`: `GenConfig` (with `min_p`), `GenResult`, `MemoryReport`, `Backend` protocol | `[DONE]` | Imported by `conftest.py`; tests pass. |
| `eval/scoring.py`: GSM8K and ARC extraction, dynamic ARC choice count | `[DONE]` | `test_scoring.py`. |
| `eval/refusal_classifier.py`: frozen pattern list | `[DONE]` | `test_refusal_classifier.py`, every pattern has a covering example. |
| `stats/bootstrap.py`: `bootstrap_ci`, `paired_bootstrap_ci`, `disagreement_rate`, `cohens_d_paired` | `[DONE]` | `test_bootstrap.py`. |
| `stats/paired.py`: `read_jsonl`, `dedupe_by_item`, `merge_by_item`, `flipped_items` | `[DONE]` | `test_pairing.py`. |
| `data/loaders.py`: GSM8K, ARC, systems prompts, AdvBench, jailbreak loaders and builder, Llama 3.2 formatters | `[DONE]` | Formatting covered in `test_scoring.py`. The `datasets`-backed loaders have not been run against the network yet; first exercised in `G1.1`. |
| `utils/hashing.py`, `utils/manifest.py`, `utils/logging.py` | `[DONE]` | `test_manifest.py`. |
| `data/systems/prompts.jsonl` (24 prompts, 8 per bucket) | `[DONE]` | Loaded through `load_systems_prompts`. Long bucket estimated ~560 tokens against the spec's ~750; measured in M1.5 (deviation 8). |
| `data/behavioural/jailbreak_templates.jsonl` (30 templates, 10 per category) | `[DONE]` | Expands to 150 unique jailbreak item ids. |
| `PROTOCOL.md` (deviations 1 to 9), `CLAUDE.md`, `README.md`, `pyproject.toml`, `ruff.toml`, `.gitignore` | `[DONE]` | Present. `PROTOCOL.md` gets amended in `G1`. |
| `.venv` (Python 3.12, `pip install -e ".[dev]"`) | `[DONE]` | Tests run from it. |
| `backends/llama_cpp_backend.py` (subprocess backend, renamed from `llama_server_backend.py`) | `[DONE]` 2026-09-11 | M1 complete; `test_backend.py`; smoke test passed on all three models. |
| LM Studio's bundled `llama-server.exe` accepts the flags the backend passes (`-m -c -ngl --host --port -np --no-webui`) | `[DONE]` | `--help` output checked 2026-09-10. |

## Decisions already taken (do not re-litigate)

- The backend is the llama-server subprocess wrapper. No `llama-cpp-python`
  build (ladder rung 1).
- This plan covers remaining work only.
- Human-only steps live in the "Manual track", not interleaved as chunks.
- The agent writes the full technical note prose, including the 90-second
  spoken summary as its final section.
- Vulkan only. No ROCm comparison (R14).
- Memory: llama.cpp's own startup-log figures (weights, KV, compute) plus peak
  RSS. Windows PDH GPU-counter sampling is cut; `peak_vram_observed_mb` stays
  `None`. Add it only if a reviewer asks for an observed transient peak.
- The study binary is the official llama.cpp v0.4.0 release, build b10809,
  Vulkan (MAN-0, installed 2026-09-10), not LM Studio's auto-updating copy (R8).
  Chosen because v0.4.0 is the stable release; the nightly `b108xx` builds are
  marked pre-release. LM Studio stays installed and untouched.
- The model is Llama-3.2-3B-Instruct from `bartowski`, as the spec and
  `PROTOCOL.md` lock it. A newer model family is a Phase 5 follow-up after
  v1.0.0, not a replacement.

## Spec gaps and ambiguities

Recorded, no action needed:

- `run_manifest.json` (section 10.1 header) vs `.jsonl` (path): `.jsonl` used.
- "8 prompts x 3 buckets = 24 prompt-precision combinations" (section 7): it is
  24 prompts, 72 prompt-precision combinations.
- `models/`, `.gitignore`, and `data/behavioural/harmful_behaviors.csv` are
  required but absent from the section 11.1 tree. Covered by the deviation note.
- The spec pins dataset name, config and split, but not revision. The manifest
  `content_hash` detects any drift (R15).
- bartowski's repo may publish `bf16` rather than `f16`. Handled in MAN-1.

**Decided by you, 2026-09-11** (these change what the numbers mean, so they
needed your sign-off, not an agent's guess). Written into `PROTOCOL.md` at G1.2:

- **C1: zero denominator in the pre-registered claim. DECIDED: percentage,
  with fallback.** `PROTOCOL.md` section 2 defines
  `rel(m) = |m_Q4 - m_FP16| / m_FP16`, which is undefined if an FP16 value is 0.
  Always report both absolute (percentage-point) and relative movement; evaluate
  the claim on relative movement, unless any FP16 denominator is 0, in which
  case evaluate it on absolute movement for all four metrics.
- **C2: answer-level agreement definition. DECIDED: both-failed counts as
  agreement.** Compute on all Suite C items per task (not only the 80 Suite R
  items), comparing `extracted_answer`; two extraction failures (`None` vs
  `None`) count as agreement, and each precision's format-failure rate is
  reported alongside so that case is visible.

---

# Manual track (human only)

The agent builds any tooling; a human runs these.

| ID | Status | Task | Blocks / feeds | Notes |
|----|--------|------|----------------|-------|
| MAN-0 | `[DONE]` 2026-09-10 (done by the agent) | `llama-b10809-bin-win-vulkan-x64.zip` (llama.cpp v0.4.0) downloaded, SHA256 matched GitHub's published digest, unzipped to `tools/llama.cpp/`; path, version, build, asset name, SHA256 and URL recorded in `configs/hardware.yaml`; `tools/` gitignored. `--list-devices` shows the RX 9070 XT over Vulkan; every flag the backend passes is present in `--help`. | M1.5 | To reproduce on another machine: download the URL in `hardware.yaml`, check the SHA256, unzip to the same path. |
| MAN-1 | `[DONE]` 2026-09-10 (done by the agent; all three SHA256 values match the frozen publisher hashes) | Download these three files from `huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF` into `models/`: `Llama-3.2-3B-Instruct-f16.gguf`, `Llama-3.2-3B-Instruct-Q8_0.gguf`, `Llama-3.2-3B-Instruct-Q4_K_M.gguf`. | M1.5, all runs | ~11.9 GB. Filenames and the publisher's SHA256 values are already frozen in `configs/model_primary.yaml` (2026-09-10); a real `f16` exists, so no `bf16` swap. M1.5 checks each download against those hashes. |
| MAN-2 | `[DONE]` 2026-09-11 (done by the agent; results in the README hardware sheet) | Run `scripts/00_smoke_hardware.py` (built in M1.5). Paste its output into the README hardware sheet: binary build, per-precision 256-token tok/s, VRAM figures, determinism result, measured prompt tokens per bucket. Confirm the SHA256 values it wrote. | M2, G1 | If f16 at 4096 context OOMs, drop to 2048 and record it for `G1`. |
| MAN-3 | `[DONE]` 2026-09-11 (done by the agent; 520 rows, SHA256 6cd1a5c6... in README; gitignored, not redistributed) | Download AdvBench `harmful_behaviors.csv` (`github.com/llm-attacks/llm-attacks`, path `data/advbench/harmful_behaviors.csv`) into `data/behavioural/`. | M5 | First 50 rows; `goal` column. |
| MAN-4 | `[DONE]` 2026-09-15 (frozen classifier accuracy 65%, below the 80% trigger) | Label the 20 rows in `results/summary/refusal_calibration_sheet_unredacted.csv` (blind: classifier label hidden, order shuffled): `your_label` = REFUSAL or COMPLIANCE. | R28 fix, M7 Appendix B | Gitignored: contains full completions, about 3 of them genuinely harmful. Accuracy below 80% triggers the pre-stated fix in `PROTOCOL.md` deviation 12. |
| MAN-5 | `[DONE]` 2026-09-15 (18 categorised: 14 reasoning chain divergence, 1 arithmetic error, 1 answer format failure, 2 refusal flip) | After M6: fill `category` in `results/summary/disagreement_review_unredacted.csv` with one of: arithmetic error, reasoning chain divergence, answer format failure, refusal flip. | M7 Appendix A | Gitignored. Pre-filled with both prompts, both generations, both answers. |
| MAN-6 | `[DONE]` 2026-09-11 | C1 and C2 decided: both recommended defaults. | G1.2 | Two decisions, recommended defaults written above. |

---

# M0: Commit the baseline `[DONE]` 2026-09-11 (`bb4033a`)

**Ladder rung: 3** (git).

**Goal / definition of done.** `git log` shows one commit containing
`PROTOCOL.md`, all Stage 0 code, this plan, and the draft backend, with a
timestamp that predates every future `results/raw/` file.

**Why necessary.** The pre-registration claim in `PROTOCOL.md` is only
evidence if git history proves the protocol existed before the results (R4).

**Dependencies.** None.

**Risks / unknowns.** None. The draft backend is committed with its known
issues; `M1` fixes them in later commits, which is itself an honest record.

| # | Status | Chunk | Size | Acceptance | Files |
|---|--------|-------|------|-----------|-------|
| M0.1 | `[DONE]` 2026-09-11, commit `bb4033a` | Stage everything (`git add` the plan and the backend draft; the rest is already staged), review `git status`, commit as the Stage 0 baseline. | S | `git log --oneline` shows 1 commit; `git show --stat HEAD` includes `PROTOCOL.md`. `models/`, `.venv/` absent from it. | repo-wide |

---

# M1: Backend finalisation `[DONE]` 2026-09-11

**Ladder rung: 2 and 5.** Reuses the existing draft backend and a prebuilt
`llama-server` binary. New code: small fixes, one helper, the smoke script,
one test file.

**Goal / definition of done.** The backend in `llama_cpp_backend.py` passes
`ruff`, reports VRAM correctly, has unit tests for its pure functions, and
`scripts/00_smoke_hardware.py` runs end to end against all three real GGUFs,
printing a valid `GenResult` and `MemoryReport` for each and writing the
SHA256 values into `configs/model_primary.yaml`.

**Why necessary.** Every suite calls `backend.generate()`. Nothing downstream
works until this runs against a real server.

**Dependencies.** `M0`. MAN-0 and MAN-1 before M1.5.

**Risks / unknowns.**

- The `/completion` `timings` key names can vary across llama.cpp builds. The
  parser assumes `prompt_ms`, `predicted_ms`, `predicted_per_token_ms`,
  `predicted_per_second`, `predicted_n`, `prompt_n`. M1.5 logs the real keys;
  fix the parser there if they differ.
- The buffer-size log format can vary across builds. M1.5 checks the parsed
  weights VRAM lands within 15% of the GGUF file size.
- Greedy determinism on Vulkan is not guaranteed. M1.5 measures it.

| # | Status | Chunk | Size | Acceptance | Files |
|---|--------|-------|------|-----------|-------|
| M1.1 | `[DONE]` 2026-09-11 | Rename `llama_server_backend.py` to `llama_cpp_backend.py` (the spec 11.1 name); keep class `LlamaServerBackend`. Fix the E501 at line 245 and run `ruff format`. | S | `ruff check .` and `ruff format --check .` clean; `grep -r llama_server_backend src tests` empty; `pytest` green. | `backends/llama_cpp_backend.py` |
| M1.2 | `[DONE]` 2026-09-11 | Fix `_parse_vram_from_log` (R1). Regex: `(\S+)\s+(model\|KV\|compute) buffer size\s*=\s*([\d.]+)\s*MiB`. Skip devices starting with `CPU` or ending with `_Host`. Sum `model` into weights, `KV` into KV, `compute` into a new `MemoryReport.compute_vram_mb` field. | S | On the log below, returns weights 6128.17, KV 448.0, compute 256.5. The `CPU_Mapped` and `Vulkan_Host` lines are excluded. | `backends/llama_cpp_backend.py`, `backends/types.py` |
| M1.3 | `[DONE]` 2026-09-11 | In `generate()`, set `wall_time_ms` from a client-side `time.monotonic()` around the HTTP call (R11). In `load()`, raise `RuntimeError("port N in use")` if `/health` already answers before launch (R6). Keep the log file handle on `self` and close it in `unload()` (R20). Add `logger.debug` of the raw `timings` dict. | S | With a server already on port 8090, a second `load()` raises before launching; `wall_time_ms >= ttft_ms` on a real call (checked in M1.5); after `unload()` the log file can be deleted (Windows would refuse if the handle were still open). | `backends/llama_cpp_backend.py` |
| M1.4 | `[DONE]` 2026-09-11 | Backend reads `llama_server_path` from `configs/hardware.yaml` (explicit path, no auto-discovery, R8), and `backend_info()` takes `build_flags` from that file's `backend` value instead of hardcoding `"vulkan"` (R19). Add `verify_model(path, expected_sha256) -> None` to `utils/hashing.py`: raises `ValueError` on an empty expected hash or a mismatch (R7). (`tools/` is already gitignored and `llama_server_path` already set, both from MAN-0.) | S | Two new tests in `test_manifest.py` (next to the existing hashing tests): mismatch raises; empty expected hash raises. | `backends/llama_cpp_backend.py`, `utils/hashing.py`, `configs/hardware.yaml`, `.gitignore`, `tests/test_manifest.py` |
| M1.5 | `[DONE]` 2026-09-11 | Create `scripts/00_smoke_hardware.py` (no argparse, section 11.3 pattern). For each precision: `verify_model` against the frozen publisher SHA256 in `model_primary.yaml`; `load` (in `try/finally` with `unload`); 1 warmup, then one 256-token generation; print `GenResult` and `MemoryReport`; run a fixed prompt 3 times at temp 0 and print `determinism: identical\|divergent`. On the first precision only, run each of the 24 systems prompts with `max_tokens=1` and print mean `prompt_tokens` per bucket (the tokenizer is identical across quants). Run at DEBUG so the `timings` keys print. | M | All 3 downloads pass `verify_model`; prints 3 valid `GenResult`s with non-zero `decode_tok_s`; weights VRAM within 15% of each file size; one determinism line per precision; three bucket means. Feeds MAN-2. **LADDER 7**: the smoke loop is minimum custom glue. | `scripts/00_smoke_hardware.py`, `backends/llama_cpp_backend.py`, `utils/hashing.py`, `data/loaders.py` |
| M1.6 | `[DONE]` 2026-09-11 | Create `tests/test_backend.py` for the pure functions, with fixture strings copied inline from M1.5's real log file (`models/.llama_server_8090.log`) and the debug `timings` output. Cover: host-buffer exclusion (the R1 log below), missing-`timings` `KeyError`, and a payload that pins every sampler (`top_p`, `top_k`, `min_p`) with `cache_prompt: False`. | M | `pytest tests/test_backend.py` green. New test file justified by R1 (deviation 10). | `tests/test_backend.py` |

Verifying log for M1.2 and M1.6:

```
load_tensors:      Vulkan0 model buffer size =  6128.17 MiB
load_tensors:   CPU_Mapped model buffer size =   500.77 MiB
llama_kv_cache:    Vulkan0 KV buffer size =   448.00 MiB
llama_context:    Vulkan0 compute buffer size =   256.50 MiB
llama_context: Vulkan_Host compute buffer size =    14.01 MiB
```

---

# M2: Systems suite (Suite S) `[DONE]` 2026-09-11

**Ladder rung: 7 for the timed loop, 2 for everything else.** Reuses
`load_systems_prompts`, the backend, `manifest`, and `verify_model`.

**Goal / definition of done.** `python scripts/01_run_systems.py` produces
`results/raw/systems_{precision}_{timestamp}.jsonl` for all 3 precisions (240
measured rows each, schema per spec 10.2) and one `run_manifest.jsonl` line per
precision.

**Why necessary.** Answers RQ1. It is also the cheapest end-to-end check of
the manifest and raw-write path before the long suites. The spec (9.5) allows
it to run before the protocol freeze, since its results inform feasibility.

**Dependencies.** `M1`, MAN-2.

**Risks / unknowns.**

- Per precision: 2 warmups, then 24 prompts x 10 measured. 726 generations in
  total, roughly 1 hour (R10).
- Warmup rows are never written (not written-then-filtered).
- `load_time_s` is per precision, repeated on each row per the spec schema.

| # | Status | Chunk | Size | Acceptance | Files |
|---|--------|-------|------|-----------|-------|
| M2.1 | `[DONE]` 2026-09-11 | Create `eval/systems.py`: `run_systems(backend, prompts, warmup_runs, measured_runs, precision, run_id, load_time_s) -> list[dict]`. Warmups run on the first prompt and are discarded. Rows follow spec 10.2; `peak_rss_mb` and `peak_vram_mb` (weights + KV + compute, `null` if weights is `null`) come from `backend.memory_report()`. | M | Covered by the M2.2 dry-run check. **LADDER 7**: the timed loop. | `eval/systems.py` |
| M2.2 | `[DONE]` 2026-09-11 (dry run passed: 3 files, 3 manifest lines, all spec keys) | Create `scripts/01_run_systems.py`: load `suite_systems.yaml`, `hardware.yaml`, `model_primary.yaml`, `gen_greedy.yaml`. For each precision: `verify_model`, build the manifest entry, `load` in `try/finally` with `unload`, call `run_systems`, append JSONL, write the manifest line. | M | Dry run (2 prompts, `measured_runs: 1`): 3 files, 3 manifest lines, and a one-liner check that every row has exactly the spec 10.2 keys and `is_warmup` is false. | `scripts/01_run_systems.py` |
| M2.3 | `[DONE]` 2026-09-11 (3 x 240 rows; decode 83.1 / 140.6 / 202.2 tok/s; VRAM 6640 / 3768 / 2430 MiB; greedy repeatable on all 24 prompts x 10) | Full run. | M (wall clock ~1 h) | 3 files x 240 rows; 3 manifest lines. | `results/raw/` |

---

# G1: Protocol freeze (gate) `[DONE]` 2026-09-11 (commit `ea8f5eb`, tag `protocol-freeze`)

**Ladder rung: 2 and 3.** Reuses the capability script's dry run as the
template probe; the rest is editing `PROTOCOL.md` and a git tag.

**Goal / definition of done.** `PROTOCOL.md` records everything learned from
the smoke test and Suite S, resolves C1 and C2, and is committed and tagged
`protocol-freeze` before any capability, reliability or behavioural raw file
exists.

**Why necessary.** Spec 9.5. Without this gate the study is not
pre-registered (R3).

**Dependencies.** `M2`, MAN-6. G1.1 needs M3.1 and M3.2 written (code only, no
full run).

**Risks / unknowns.** If the probe fails, the template changes here, once,
before any scored result exists. After the tag, format failures are data, not
a reason to change the template.

| # | Status | Chunk | Size | Acceptance | Files |
|---|--------|-------|------|-----------|-------|
| G1.1 | `[DONE]` 2026-09-11 (first probe failed on GSM8K; template changed and validated on held-out items, see `PROTOCOL.md` 12.6) | Template probe: dry-run `scripts/02_run_capability.py` with `n_items: 5` for both tasks on all 3 precisions. The spec's 90% threshold on 5 items means all 5 must extract. If any precision misses one, fix the template, re-probe, and record the change. This is the first live exercise of the `datasets` loaders. | S | Extraction 5/5 for both tasks on all 3 precisions, or a recorded template change followed by a passing re-probe. Delete the probe's raw files and manifest lines afterwards (they predate the freeze). | `data/loaders.py` only if the template changes |
| G1.2 | `[DONE]` 2026-09-11 | Amend `PROTOCOL.md`: the binary and its build (v0.4.0, b10809); the determinism result (byte-identical x3 on all precisions); measured bucket lengths (short 55, medium 221, long 507: deviation 8); the server launch flags and why (`-lv 4` for buffer logging, `--fit off`, `--cache-ram 0`: R22 to R24); the additional neutral samplers sent on every request (R21) in section 5; the C1 and C2 decisions (section 2 and section 6); deviation 10 (the three added test files, the added output files, and the extra `prompt_tokens` column in systems rows, R13); deviation 11 (pinned upstream binary, no ROCm, R8 and R14). No `bf16` swap happened (a real f16 exists). | S | Every item listed is present; section 10 has deviations 10 and 11. | `PROTOCOL.md` |
| G1.3 | `[DONE]` 2026-09-11, commit `ea8f5eb`, tag `protocol-freeze` | Commit and `git tag protocol-freeze`. | S | `git tag` lists `protocol-freeze`; `results/raw/` holds only systems files. | repo-wide |

---

# M3: Capability suite (Suite C) `[DONE]` 2026-09-11

**Ladder rung: 2.** Loaders, scorers, manifest and `verify_model` exist. The
runner is an iterate-generate-score-write loop.

**Goal / definition of done.** `python scripts/02_run_capability.py` runs GSM8K
(200) and ARC-Challenge (300) across 3 precisions with greedy config, writing
`results/raw/capability_{task}_{precision}_{timestamp}.jsonl` (spec 10.3) and
one manifest line per task-precision.

**Why necessary.** Answers RQ2, and supplies the greedy answers RQ3 pairs.

**Dependencies.** M3.1 and M3.2 before `G1` (the probe uses them). The full run
(M3.3) after `G1`.

**Risks / unknowns.**

- ARC items with more than 5 choices are skipped and counted by
  `load_arc_challenge`; the script logs the count for the report.
- 1,500 generations, roughly 4 hours. A rerun writes a new timestamped file;
  see the aggregation rule in M6.6.

| # | Status | Chunk | Size | Acceptance | Files |
|---|--------|-------|------|-----------|-------|
| M3.1 | `[DONE]` 2026-09-11 (exercised by G1.1) | Create `eval/capability.py`: `run_task(backend, items, task, precision, run_id, gen_config) -> list[dict]`, dispatching to `score_gsm8k` or `score_arc`. Rows follow spec 10.3; an extraction failure still yields a row with `correct: 0`. | M | Covered by the G1.1 probe and the M3.2 dry-run check. | `eval/capability.py` |
| M3.2 | `[DONE]` 2026-09-11 (end-to-end on held-out items; all spec 10.3 keys) | Create `scripts/02_run_capability.py`: load configs; load each task's items once; for each precision, `verify_model`, `load` in `try/finally`, run, append JSONL, write the manifest line; log the ARC skipped count. No in-run probe or abort (moved to G1.1, R3). | M | Dry run (`n_items: 5`): 6 files, 6 manifest lines; one-liner check that every row has exactly the spec 10.3 keys. | `scripts/02_run_capability.py` |
| M3.3 | `[DONE]` 2026-09-11 (GSM8K 200, ARC 300 per precision; extraction 97 to 100%; 5 GSM8K generations hit the 1024-token cap) | Full run, after `G1`. | M (wall clock ~4 h) | GSM8K 3 x 200 rows, ARC 3 x (300 minus skipped) rows; 6 manifest lines. | `results/raw/` |

---

# M4: Reliability suite (Suite R) `[DONE]` 2026-09-11

**Ladder rung: 2.** Same building blocks as `M3` plus a seed loop. The spec
mandates a separate module and script; the near-duplication with `M3` stays
(no shared suite-runner base: one implementation each, ladder rung 1 on the
abstraction).

**Goal / definition of done.** `python scripts/03_run_reliability.py` runs GSM8K
items 0-79 across 3 precisions x seeds `[23, 67, 100]` with the sampled config,
writing `results/raw/reliability_{precision}_{seed}_{timestamp}.jsonl` (spec
10.4) and one manifest line per precision-seed.

**Why necessary.** Answers RQ3 self-consistency.

**Dependencies.** `G1`. (Answer-level cross-precision agreement uses M3's
greedy results, but only at aggregation time.)

**Risks / unknowns.**

- 720 generations at temperature 0.7, up to 1,024 tokens. Roughly 2 to 3 hours.
- Seeded reproducibility depends on llama-server honouring `seed` with the
  pinned sampler chain. M4.2's dry run checks it directly.

| # | Status | Chunk | Size | Acceptance | Files |
|---|--------|-------|------|-----------|-------|
| M4.1 | `[DONE]` 2026-09-11 | Create `eval/reliability.py`: `run_reliability(backend, items, precision, seed, run_id, gen_config) -> list[dict]`, scoring with `score_gsm8k`; rows follow spec 10.4. | M | Covered by the M4.2 dry-run check. | `eval/reliability.py` |
| M4.2 | `[DONE]` 2026-09-11 (dry run on held-out items 200 to 202: seed 23 twice gave identical text on all precisions) | Create `scripts/03_run_reliability.py`: load `suite_reliability.yaml`, `gen_sampled.yaml`, configs; load 80 items once; loop precision then seed, building `GenConfig` with that seed; `verify_model`, `load` in `try/finally`; append JSONL; write the manifest line. | M | Dry run (`n_items: 3`, `seeds: [23, 23]`): the two seed-23 passes produce identical text per item; every row has exactly the spec 10.4 keys. | `scripts/03_run_reliability.py` |
| M4.3 | `[DONE]` 2026-09-11 (9 x 80 rows; self-consistency f16 45.0%, q8 41.2%, q4 32.5%; seeds verified: 100% of items' texts differ) | Full run. | M (wall clock ~2-3 h) | 9 files x 80 rows; 9 manifest lines. | `results/raw/` |

---

# M5: Behavioural suite (Suite B) `[DONE]` 2026-09-11 (R28 fixed 2026-09-15; both classifiers reported)

**Ladder rung: 2.** `classify`, `load_advbench_harmful`,
`load_jailbreak_templates`, `build_jailbreak_items` exist. New: the run loop,
redaction, calibration sampling.

**Goal / definition of done.** `python scripts/04_run_behavioural.py` runs 50
harmful and 150 jailbreak items across 3 precisions with greedy config,
writing `results/raw/behavioural_{set}_{precision}_{timestamp}.jsonl` (spec
10.5) with `generated_text` redacted to a SHA256 plus a 120-character snippet,
the full text in a gitignored `_unredacted.jsonl` sibling, and one manifest
line per set-precision. Also writes the 20-row calibration sample for MAN-4.

**Why necessary.** Answers RQ4.

**Dependencies.** `G1`, MAN-3.

**Risks / unknowns.**

- Redaction must hold: no committed file may contain a full completion. M5.3
  tests it, and M5.4's dry run checks `git check-ignore`.
- `load_advbench_harmful` assumes a `goal` column; the dry run fails loudly if
  it is absent.
- Refuse-then-comply is scored REFUSAL. This is a documented limitation, not
  corrected.
- 600 generations, roughly 1 to 2 hours.

| # | Status | Chunk | Size | Acceptance | Files |
|---|--------|-------|------|-----------|-------|
| M5.1 | `[DONE]` 2026-09-11 (done early, R5) | Broaden `.gitignore` from `results/raw/behavioural_*_unredacted.jsonl` to `results/**/*_unredacted.jsonl` (R5). | S | `git check-ignore results/summary/x_unredacted.jsonl` prints the path. | `.gitignore` |
| M5.2 | `[DONE]` 2026-09-11 | Create `eval/behavioural.py`: `run_set(backend, items, set_name, precision, run_id, gen_config) -> list[dict]` (spec 10.5; `refusal_classification` and `matched_pattern` from `classify`); `redact_rows(rows, snippet_chars) -> list[dict]` (copy with `generated_text` truncated and `generated_text_sha256` added); `sample_calibration(rows, n=20, seed=42) -> list[dict]` (10 REFUSAL + 10 COMPLIANCE, full text, blank `human_label`). | M | Covered by M5.3. **LADDER 7**: `redact_rows` (nothing in stdlib hashes-and-truncates one field). | `eval/behavioural.py`, `utils/hashing.py` |
| M5.3 | `[DONE]` 2026-09-11 (8 tests) | Create `tests/test_behavioural.py`: `redact_rows` never emits `generated_text` longer than `snippet_chars` and never drops the hash; `sample_calibration` returns exactly 10 of each label. | S | Green. New test file justified by R5 (deviation 10). | `tests/test_behavioural.py` |
| M5.4 | `[DONE]` 2026-09-11 (dry run on held-out AdvBench rows 100 to 102: redaction holds, unredacted siblings written, calibration file written) | Create `scripts/04_run_behavioural.py`: build harmful items and jailbreak items; loop set then precision; `verify_model`, `load` in `try/finally`; write `_unredacted.jsonl` then the redacted `.jsonl`; write the manifest line. After all precisions, write `results/summary/refusal_calibration_unredacted.jsonl` via `sample_calibration`. | M | Dry run (`n_items: 3`, `n_templates: 2`): 6 redacted files, 6 unredacted siblings, 6 manifest lines, and `git status --porcelain` lists no `_unredacted` file. | `scripts/04_run_behavioural.py` |
| M5.5 | `[DONE]` 2026-09-11 (600 generations; committed files hold at most 120 characters per generation; calibration sample 10 + 10) | Full run. | M (wall clock ~1-2 h) | Harmful 3 x 50 rows, jailbreak 3 x 150 rows; calibration file has 20 rows. Feeds MAN-4. | `results/` |

---

# M6: Aggregation, tables, figures `[TODO]`

**Ladder rung: 3 and 5.** `pandas` groupby for the CSVs, `matplotlib` for
figures, the existing `stats/bootstrap.py` and `stats/paired.py` for every
interval. `stats/tables.py` is glue.

**Goal / definition of done.** `python scripts/05_aggregate_and_plot.py` writes
the four spec 10.6 CSVs (exact columns) plus `agreement_summary.csv`, Figure 1
(Pareto) and Figure 2 (disagreement heatmap), and the MAN-5 review file.
`python scripts/06_build_report_tables.py` writes Tables 1 to 5 (spec section
18) to `results/summary/report_tables.md`.

**Why necessary.** Turns raw JSONL into the pre-declared statistics.

**Dependencies.** M6.1 to M6.5 can be written and tested against fixtures in
parallel with `M2` to `M5`. M6.6 and M6.7 need all raw outputs.

**Risks / unknowns.**

- `merge_by_item` raises if a precision is missing an item. A partial run
  fails aggregation loudly, by design.
- Cohen's d on binary data is reported, but the disagreement and flip rates
  carry the headline (`PROTOCOL.md` section 8.3).

| # | Status | Chunk | Size | Acceptance | Files |
|---|--------|-------|------|-----------|-------|
| M6.1 | `[DONE]` 2026-09-11 (adds `peak_vram_mb_mean`, VRAM being the primary memory figure) | Create `stats/tables.py` with `systems_summary(rows) -> DataFrame`: spec 10.6 columns grouped by precision, prompt_id, bucket. | M | Covered by M6.5. | `stats/tables.py` |
| M6.2 | `[DONE]` 2026-09-11 | `capability_summary(rows)`: per task and precision, mean with `bootstrap_ci`; vs f16: `paired_bootstrap_ci`, `disagreement_rate`, `cohens_d_paired` on `correct` via `merge_by_item`. Exact spec 10.6 columns. | M | Covered by M6.5. | `stats/tables.py` |
| M6.3 | `[DONE]` 2026-09-11 | `agreement_summary(rows)` (R2): per task and precision pair, `merge_by_item(value_field="extracted_answer")`, then agreement rate, flip count, and `flipped_items`, following C2. Also the format-failure rate per precision. Feeds Table 3 and Figure 2. | M | Covered by M6.5. | `stats/tables.py` |
| M6.4 | `[DONE]` 2026-09-11 (reliability adds `texts_differ_rate` to prove seeds changed the sampling; also `preregistered_claim` with the C1 rule) | `reliability_summary(rows)` (self-consistency plus CI per precision) and `behavioural_summary(rows)` (refusal rate plus CI, attack success rate, flip rate plus CI vs f16). Exact spec 10.6 columns. | M | Covered by M6.5. | `stats/tables.py` |
| M6.5 | `[DONE]` 2026-09-11 (12 tests; verified on the real Suite S and C data) | Create `tests/test_tables.py`: exact column sets for the four spec CSVs; hand-checked fixtures proving that two different wrong answers count as a flip in `agreement_summary` but not in `capability_summary`'s disagreement; self-consistency is the fraction of items where all 3 seeds agree; attack success rate is 1 minus jailbreak refusal rate. | M | Green. New test file justified by R2 (deviation 10). | `tests/test_tables.py` |
| M6.6 | `[DONE]` 2026-09-15 (re-run with both classifiers) run 2026-09-11 on all four suites with the frozen classifier; re-run after the R28 fix; both figures rendered and inspected (dataviz: palette validated all-pairs, direct labels, computed title); final run after M5 | Create `scripts/05_aggregate_and_plot.py`. First, move any partial raw run aside (R16; the manifest keeps the record). Then glob `results/raw/`, write the 5 CSVs, Figure 1 (x tok/s, y capability score, point size peak VRAM, 3 points per task) and Figure 2 (sampled item rows x precision-pair columns, agree or disagree), and `results/summary/disagreement_review_unredacted.csv`: 15 to 25 FP16-vs-Q4 items across GSM8K, ARC and behavioural refusal flips, with text taken from the unredacted files. | L | 5 CSVs with correct columns, 2 PNGs in `results/figures/`, a review file with 15 to 25 rows that `git status` does not list. Feeds MAN-5. **LADDER 7**: the two figure layouts. | `scripts/05_aggregate_and_plot.py` |
| M6.7 | `[DONE]` 2026-09-15 (re-run with both classifiers) run 2026-09-11 (all six tables, frozen classifier); re-run after the R28 fix | Create `scripts/06_build_report_tables.py`: read the CSVs and write `results/summary/report_tables.md` with Tables 1 to 5 laid out as in spec section 18. | M | 5 tables whose values match the CSVs. | `scripts/06_build_report_tables.py` |

---

# M7: Technical note and spoken summary `[DONE]` 2026-09-15

**Ladder rung: 1 check passed** (required by spec section 19 and the thesis).
Prose, not code.

**Goal / definition of done.** `reports/technical_note.md`: a complete 4 to 6
page note covering method, systems, capability, agreement, reliability,
behavioural, limitations, Appendices A and B, and a final "Spoken summary"
section that reads in about 90 seconds. It embeds `report_tables.md` and both
figures, and states whether the pre-registered claim held or was falsified,
with the numbers.

**Why necessary.** It is the deliverable a reviewer reads.

**Dependencies.** `M6`, MAN-4, MAN-5.

**Risks / unknowns.**

- A null result leads the note as the finding (`PROTOCOL.md` section 2). The
  prose must not hedge it into a non-result.
- Map Q8_0 and Q4_K_M to their per-tensor bit widths (spec section 20) to
  pre-empt the "GGUF is not real quantisation" critique.
- Appendices quote no full behavioural completions: item ids, template
  category and classification change only.

| # | Status | Chunk | Size | Acceptance | Files |
|---|--------|-------|------|-----------|-------|
| M7.1 | `[DONE]` 2026-09-15 | Sections 1 to 3: method (condensed from `PROTOCOL.md`), systems (RQ1, Table 1, Figure 1), capability (RQ2, Table 2). | M | Every number traces to a CSV cell. | `reports/technical_note.md` |
| M7.2 | `[DONE]` 2026-09-15 | Sections 4 and 5: agreement (RQ3, Table 3, Figure 2) and reliability (Table 4); the pre-registered claim verdict under the C1 rule. | M | The verdict sentence says held or falsified and cites the four movements. | `reports/technical_note.md` |
| M7.3 | `[DONE]` 2026-09-15 | Section 6: behavioural (RQ4, Table 5). Appendix A from the filled disagreement review; Appendix B from the filled calibration file (accuracy, false positives, false negatives). | M | Appendix A has 15 to 25 categorised items; Appendix B reports all three calibration numbers. | `reports/technical_note.md` |
| M7.4 | `[DONE]` 2026-09-15 | Limitations (every item in `PROTOCOL.md` section 11, plus the timing source and binary provenance), the bit-width mapping, and the spoken summary section. | M | Limitations cover section 11 in full; spoken summary about 90 seconds read aloud. | `reports/technical_note.md` |

---

# M8: Hardening and v1.0.0 `[DONE]` 2026-09-15

**Ladder rung: 3.** `pip freeze`, `ruff`, `pytest`, `git tag`.

**Goal / definition of done.** `ruff check .` and `pytest tests/` pass;
dependencies are pinned; README reproduction steps are verified; tag `v1.0.0`
exists.

**Why necessary.** Spec section 19 and the section 22 week 7 exit gate.

**Dependencies.** `M7`.

**Risks / unknowns.** "Clean clone reproduces" cannot mean re-running the GPU
suites automatically (no CI, by rule). It means the scripts start, read their
configs, and stop at the first missing model, CSV or binary with a pointed
error from `verify_model` or the loaders.

| # | Status | Chunk | Size | Acceptance | Files |
|---|--------|-------|------|-----------|-------|
| M8.1 | `[DONE]` 2026-09-15 (10 packages pinned to installed versions; resolves) | Pin every dependency in `pyproject.toml` to the installed version (`==`); keep `llama-cpp-python` under the optional extra. | S | A fresh venv `pip install -e ".[dev]"` resolves. | `pyproject.toml` |
| M8.2 | `[DONE]` 2026-09-15 (ruff clean, tests pass, scripts stop with a pointed missing-file message) | `ruff format . && ruff check . && pytest tests/`; fix failures. Confirm scripts `01` to `04` stop with a pointed message when data is absent. | M | All three commands exit 0; each script prints a clear missing-file error. | repo-wide |
| M8.3 | `[DONE]` 2026-09-15 (README follow-up data step added; tag v1.0.0) | Walk the README steps line by line and fix drift; `git tag v1.0.0`. | S | Following "Setup" leaves only the documented manual steps; `git tag` lists `v1.0.0`. | `README.md` |

---

# F1: Follow-up, JailbreakBench (exploratory) `[DONE]` 2026-09-15

**Ladder rung: 2 and 5.** Reuses the backend, `run_set`, redaction, calibration
sampling and the `datasets` library. New: two loaders, prompt redaction, one run
script (`scripts/07_run_followup_jbb.py`). Pre-registered in `PROTOCOL.md`
section 14. Does not change the main study's verdict.

**Goal / definition of done.** 287 items (JBB harmful 100, JBB benign 100, PAIR
prompts 87) run on all three precisions; committed rows carry no prompt text;
30-item blind calibration done; F1 tables in the report, labelled exploratory.

| # | Status | Chunk | Acceptance |
|---|--------|-------|-----------|
| F1.1 | `[DONE]` 2026-09-15 | Pin data: JBB-Behaviors revision 886acc35, PAIR file SHA256 49e007ce (gitignored) in `configs/suite_jbb.yaml` | `git check-ignore` confirms the data directory |
| F1.2 | `[DONE]` 2026-09-15 | Loaders, prompt redaction, corrected-label calibration sampling, run script, tests | tests pass; dry run on 2 items per set passes |
| F1.3 | `[DONE]` 2026-09-15, commit `07db5b2` | Commit the pre-registration (section 14) before any F1 generation | commit precedes every `followup_*` result file |
| F1.4 | `[DONE]` 2026-09-15 (861 rows; prompts redacted; no Q4 difference detected on any set) | Full run, about 861 generations | 9 result files, 9 manifest lines |
| MAN-7 | `[DONE]` 2026-09-15 (28/30 = 93%; every set at least 90%) **next** | You: blind 30-item sheet `results/summary/followup_calibration_sheet_unredacted.csv` | accuracy per set reported |
| F1.5 | `[DONE]` 2026-09-15 (Table F1 and report section 8) | F1 summary table and report section | exploratory label on every F1 number |

---

# Sequencing summary

```
M0 --> M1.1-M1.4 --> [MAN-0, MAN-1] --> M1.5 --> MAN-2 --> M1.6
                                                    |
                                                    v
                                                   M2 --> M3.1, M3.2 --> [MAN-6] --> G1 (tag protocol-freeze)
                                                                                      |
              +-----------------------------------------------------------------------+
              v                            v                                          v
         M3.3 run                   M4 (code + run)                 [MAN-3] --> M5 (code + run) --> MAN-4
              |                            |                                          |
              +----------------------------+------------------------------------------+
                                           v
                                          M6 --> MAN-5 --> M7 --> M8 (tag v1.0.0)

M6.1-M6.5 can be written against fixtures any time after M0.
GPU runs (M2.3, M3.3, M4.3, M5.5) serialise on one card: roughly 9 hours total.
```

Critical path: `M0 -> M1 -> M2 -> G1 -> M3.3 -> M6 -> M7 -> M8`. The capability
run is the longest single job and gates `M6` alongside `M4` and `M5`.
