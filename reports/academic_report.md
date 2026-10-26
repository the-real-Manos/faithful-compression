# Faithful Compression: an item-level evaluation of post-training quantisation

*Formal report. Companion to `technical_note.md`, which carries the full result
tables. All figures below are reproducible from `results/` at tag `v1.0.0`.*

## Introduction

Post-training quantisation is the standard method for deploying large language
models on constrained hardware, and its cost is conventionally reported as a
change in aggregate benchmark accuracy. That convention assumes the aggregate is
an adequate summary of the change in model behaviour. This study tests the
assumption directly by holding every element of the evaluation constant except
the numerical precision of the weights, and by comparing outputs at the level of
the individual item rather than the mean.

The work was pre-registered. Research questions, metrics, statistical procedures
and a falsifiable claim were committed to version control before any model file
was downloaded, and frozen under a git tag before the capability, reliability and
behavioural suites were executed. The pre-registered claim held that relative
movement in behavioural metrics, namely refusal and jailbreak susceptibility,
would be at least as large as relative movement in aggregate capability. The
study reports that this claim was falsified.

## Method

The subject model was Llama-3.2-3B-Instruct in GGUF format, obtained from a
single publisher and evaluated at three precisions: F16 (16.00 bits per weight),
Q8_0 (8.51) and Q4_K_M (5.01). Each file was verified against its publisher
SHA256 before every load. Inference used a pinned llama.cpp release (v0.4.0,
build b10809, Vulkan) executed as a local subprocess on an AMD Radeon RX 9070 XT
under Windows 10, with all layers offloaded to the GPU and a 4,096-token context.
The evaluation code accesses the model exclusively through a single interface and
never branches on precision; precision differs only in the file path supplied at
load time. Every sampler was pinned explicitly, including parameters the
configuration does not expose, because the inference server applies non-neutral
defaults to any parameter left unset. Each run appends one manifest record
containing the model hash, the complete generation configuration, the engine
version and a content hash of the exact prompt set, so that a result file without
a matching manifest entry is not treated as evidence.

Four suites were executed, totalling 4,401 retained generations: a systems suite
(24 prompts across three length bands, ten measured repetitions per prompt per
precision); a capability suite (GSM8K, 200 items; ARC-Challenge, 300 items;
greedy decoding); a reliability suite (80 GSM8K items at three sampling seeds);
and a behavioural suite (50 AdvBench harmful requests and 150 template-based
jailbreak items). A pre-registered exploratory follow-up added 287 JailbreakBench
items, comprising 100 harmful requests, 100 benign near-duplicates and 87
persuasive attack prompts. Scoring was deterministic throughout: exact numeric
match for GSM8K, option-letter match for ARC-Challenge, and a frozen keyword
regex for refusal classification, with no model used as a judge. All comparisons
were paired by item identifier; intervals are 95 percent percentile bootstrap
intervals over 1,000 item-level resamples, and a difference is reported as
detected only where the paired interval excludes zero. Two procedural controls
were applied before the freeze: a five-item template probe, which failed and
required a prompt revision validated on held-out items, and a blind manual
calibration of the refusal classifier.

## Discussion

Systems cost scales as expected: Q4_K_M decoded at 201.9 tokens per second
against 83.1 for F16, in 2,430 MB of GPU memory against 6,640 MB. Aggregate
capability moved little, and neither change was detected: GSM8K fell 3.5
percentage points (95 percent interval -2.0 to +8.5) and ARC-Challenge fell 3.0
(-0.3 to +6.7). The item-level comparison tells a different story. Comparing the
extracted answers rather than the correctness scores, 23.0 percent of GSM8K
answers (17.5 to 29.0) and 14.0 percent of ARC-Challenge answers (10.3 to 18.0)
changed between F16 and Q4_K_M. Because errors introduced and errors corrected
approximately offset one another, the mean conceals the substitution. Manual
categorisation of a random sample of disagreements attributes the majority to
divergence of the reasoning path rather than to arithmetic or formatting failure.

Behavioural metrics were stable. Harmful requests were refused at 98.0 percent at
every precision, and jailbreak attack success was 6.0 percent at F16 against 4.7
percent at Q4_K_M. The exploratory follow-up, using persuasive attack prompts
that succeeded 27.6 percent of the time, likewise found no detected difference
between precisions, and measured stable over-refusal of 11 to 12 percent on
benign near-duplicates. The pre-registered claim is therefore falsified, and is
reported as such. A post-hoc blind audit of the refusal labels, prompted
by the author's inspection of the raw responses, confirmed 90 of 90 confident
refusals and found that inter-rater disagreement on ambiguous cases (about a
fifth; Cohen's kappa 0.62 to 0.65 across three raters) exceeds every
precision-to-precision difference, so no safety difference is detectable under any
rater. Two measurement defects are material to interpretation and are
documented in full. A parser attributed host-memory buffers to GPU memory, which
would have overstated F16 memory by 751 MB. More instructively, the frozen
refusal classifier matched only a straight apostrophe, and the quantised model
produced the typographic variant more frequently; uncorrected, this artefact
presented as a safety regression in the compressed model, raising apparent
jailbreak success from 6.0 to 14.0 percent. The correction was specified in the
protocol before the calibration labels were examined, and both scorings are
reported.

## Conclusion

Aggregate benchmark scores are an insufficient basis for deciding whether a
quantised model is a safe substitute for its full-precision counterpart. On this
model, 4-bit quantisation delivered a 2.4-fold throughput gain and a 63 percent
reduction in GPU memory at no detectable cost in mean accuracy, while altering
roughly one in four individual mathematical answers. Deployment decisions that
depend on per-item consistency, such as caching, regression testing or auditing,
should measure agreement directly. Refusal behaviour proved robust under
compression on this model, including under persuasive attacks.

The study's scope is narrow: one model family at one parameter scale, one
quantisation family, one GPU and one inference backend, with suite sizes that
bound the smallest detectable effect. The generalisable contribution is
methodological. Pre-registration constrained the analysis, item-level pairing
exposed instability that the mean concealed, and a calibration procedure
committed to in advance caught a scoring defect that would otherwise have been
published as a safety finding.

## Appendix

**A. Principal results.** Full tables, including confidence intervals for every
cell, are in `results/summary/report_tables.md`; per-item records are in
`results/raw/`.

| Precision | Tokens/s | GPU memory (MB) | GSM8K | ARC-Challenge | Harmful refusal |
|-----------|----------|-----------------|-------|---------------|-----------------|
| F16 | 83.1 | 6,640 | 78.5% | 70.7% | 98.0% |
| Q8_0 | 139.5 | 3,768 | 75.5% | 70.0% | 98.0% |
| Q4_K_M | 201.9 | 2,430 | 75.0% | 67.7% | 98.0% |

| Contrast | Mean difference [95% CI] | Answer change rate [95% CI] |
|----------|--------------------------|------------------------------|
| GSM8K, F16 vs Q4_K_M | +3.5 pts [-2.0, +8.5] | 23.0% [17.5, 29.0] |
| ARC-Challenge, F16 vs Q4_K_M | +3.0 pts [-0.3, +6.7] | 14.0% [10.3, 18.0] |

**B. Paired statistical procedure.** Resampling operates on item-level
differences, which preserves the pairing; resampling the two arrays
independently would not.

```python
def paired_bootstrap_ci(values_a, values_b, n_resamples=1000, ci=0.95, seed=42):
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    if values_a.shape != values_b.shape:
        raise ValueError(f"unpaired arrays: {values_a.shape} vs {values_b.shape}")
    return bootstrap_ci(values_a - values_b, n_resamples=n_resamples, ci=ci, seed=seed)
```

**C. Classifier correction.** The frozen pattern list was retained unchanged; a
normalisation step was added ahead of matching, and the original scoring remains
reproducible through a parameter.

```python
_APOSTROPHES = str.maketrans({"‘": "'", "’": "'", "ʼ": "'"})

def classify(text, normalise_apostrophes=True):
    lowered = text.lower()
    if normalise_apostrophes:
        lowered = lowered.translate(_APOSTROPHES)
```

**D. Calibration.** First calibration, frozen classifier, 20 items: 65 percent
agreement, 0 false positives, 7 false negatives, all attributable to the
apostrophe defect. Follow-up calibration, corrected classifier, 30 items: 93
percent agreement (90 percent harmful, 90 percent benign, 100 percent persuasive
attacks). Post-hoc safety-label audit (`PROTOCOL.md` section 15): on 90
random confident refusals, human agreement 90/90 (false-refusal rate 0 percent,
95 percent CI 0 to 4 percent), equal across precisions; on 68 ambiguity-enriched
items, three-rater agreement 81 to 82 percent (kappa 0.62 to 0.65).

**E. Supplementary figures.** `results/figures/fig1_pareto.png` plots throughput
against accuracy with memory encoded by marker area.
`results/figures/fig2_disagreement.png` displays per-item answer changes by
precision pair.

**F. Provenance.** `bb4033a` protocol committed; `ea8f5eb` protocol frozen (tag
`protocol-freeze`); `07db5b2` follow-up pre-registered; `80c3b50` final results
(tag `v1.0.0`).
