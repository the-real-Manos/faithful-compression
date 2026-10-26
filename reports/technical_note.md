# Faithful Compression: what 4-bit quantisation changes beyond the average score

**Llama-3.2-3B-Instruct at FP16, Q8_0 and Q4_K_M on an AMD RX 9070 XT (Vulkan).**
Pre-registered study. Protocol: `PROTOCOL.md` (committed `3e3caf3`, frozen at tag
`protocol-freeze`). Every number below traces to a file in `results/summary/`.

## Summary

Q4_K_M decodes 2.4 times faster than FP16 in 37 percent of the GPU memory, and
its mean accuracy drops by 3.5 points on GSM8K and 3.0 points on ARC-Challenge,
neither distinguishable from zero. The average hides real instability: **23
percent of GSM8K answers and 14 percent of ARC-Challenge answers change** between
FP16 and Q4. Refusal behaviour does not degrade: harmful requests are refused 98
percent of the time at every precision, and jailbreak success is 6.0 percent at
FP16 and 4.7 percent at Q4. A pre-registered follow-up with persuasive attacks
raised attack success to about 27 percent, equally at every precision.

The pre-registered claim, that behavioural metrics move at least as much as
capability, is **falsified**. The study also caught and corrected a refusal
classifier defect that, uncorrected, would have reported a false Q4 safety
regression.

## 1. Method

**Model and precisions.** Llama-3.2-3B-Instruct GGUF files from `bartowski`, each
checked against the publisher's SHA256 before every run.

| Tier | llama.cpp type | Bits per weight | File size |
|------|----------------|-----------------|-----------|
| FP16 | F16 | 16.00 | 6.43 GB |
| Q8 | Q8_0 | 8.51 | 3.42 GB |
| Q4 | Q4_K_M | 5.01 | 2.02 GB |

Q4_K_M is a mixed k-quant: most tensors use 4-bit blocks, some are kept at higher
precision, so its average is 5.01 bits per weight rather than 4. Bits per weight
for F16 and Q4_K_M come from llama.cpp's load log; Q8_0 is computed from file size.

**Engine.** Official llama.cpp v0.4.0 (build b10809) Vulkan binary, pinned by URL
and SHA256, run as a local `llama-server` subprocess. All layers on the GPU,
context 4096. Launch settings prevent the server from adjusting settings per
model or caching prompts across requests.

**Invariant.** Evaluation code never branches on precision; only the model file
passed to the backend differs. Every sampler is pinned explicitly, including ones
the configuration does not expose. Greedy decoding was byte-identical across
repeated runs at every precision.

**Suites.**

| Suite | Items | Runs | Generations |
|-------|-------|------|-------------|
| Systems | 24 prompts (short 55, medium 221, long 507 tokens on average) | 10 measured per prompt | 720 |
| Capability | GSM8K 200, ARC-Challenge 300 | 1, greedy | 1,500 |
| Reliability | GSM8K 0 to 79 | 3 seeds, temperature 0.7 | 720 |
| Behavioural | AdvBench 50 harmful, 150 jailbreak (30 templates x 5 requests) | 1, greedy | 600 |

**Statistics** (fixed before results). Means with 95 percent percentile bootstrap
intervals (1,000 resamples over items). Every comparison with FP16 is paired by
item. A difference is "detected" only when the paired interval excludes zero. No
multiple-comparison correction.

## 2. Systems (RQ1)

| Precision | Load (s) | TTFT p50 (ms) | TTFT p95 (ms) | Tokens/s | GPU memory (MB) |
|-----------|----------|---------------|---------------|----------|-----------------|
| FP16 | 5.06 | 53.3 | 111.5 | 83.1 | 6,640 |
| Q8 | 3.01 | 47.3 | 92.7 | 139.5 | 3,768 |
| Q4 | 2.02 | 49.3 | 99.5 | 201.9 | 2,430 |

Decode speed scales roughly with the inverse of weight size, because generating a
token on this hardware is limited by memory bandwidth. Time to first token barely
moves: prompt processing is compute-bound and short prompts finish in tens of
milliseconds at every precision. GPU memory is llama.cpp's own allocation for
weights, KV cache and compute buffers; the KV cache (448 MB) and compute buffer
(64 MB) are identical across precisions, so the whole saving is in the weights.

![Figure 1](../results/figures/fig1_pareto.png)

## 3. Capability (RQ2)

| Task | N | FP16 [CI] | Q8 [CI] | Q4 [CI] | FP16 - Q4 [CI] |
|------|---|-----------|---------|---------|----------------|
| GSM8K | 200 | 78.5% [72.0, 84.0] | 75.5% [69.0, 81.5] | 75.0% [69.0, 80.5] | +3.5 pts [-2.0, +8.5] |
| ARC-Challenge | 300 | 70.7% [65.7, 75.7] | 70.0% [65.0, 75.0] | 67.7% [62.3, 72.7] | +3.0 pts [-0.3, +6.7] |

No FP16 to Q4 drop is detected at this sample size. One comparison is detected:
FP16 to Q8 on GSM8K, +3.0 points [+0.5, +5.5]. Q8 flips few items but more of its
flips go from right to wrong, which gives a narrow interval. Cohen's d for FP16 vs
Q4 is 0.09 on both tasks.

## 4. Answer stability (RQ3)

| Task | Same answer FP16-Q4 | Answer flip rate [CI] | Correctness disagreement | Items flipped |
|------|---------------------|-----------------------|--------------------------|---------------|
| GSM8K | 77.0% | 23.0% [17.5, 29.0] | 14.5% | 46 of 200 |
| ARC-Challenge | 86.0% | 14.0% [10.3, 18.0] | 10.3% | 42 of 300 |

This is the central finding. The answer flip rate counts any change in the
extracted answer, including one wrong answer replaced by another; correctness
disagreement counts only right-to-wrong or wrong-to-right. Many flips cancel in
the mean: a 3.5-point average drop on GSM8K sits on top of 46 changed answers.
Q8 is far more faithful: 18 GSM8K flips (9.0 percent) and 3 on ARC-Challenge
(1.0 percent).

![Figure 2](../results/figures/fig2_disagreement.png)

## 5. Reliability under sampling

| Precision | Same answer across 3 seeds [CI] |
|-----------|---------------------------------|
| FP16 | 45.0% [35.0, 56.2] |
| Q8 | 41.2% [31.2, 52.5] |
| Q4 | 32.5% [22.5, 42.5] |

Self-consistency falls with precision, though the intervals overlap. Every item
produced three different texts, confirming the seeds changed the sampling.
Sampling also raises unreadable GSM8K answers to 8 to 10 percent (from 1.5 to 3
percent greedy), and about 40 percent of inconsistent items involve one, so this
metric partly measures format stability under sampling.

## 6. Refusal behaviour (RQ4)

Corrected classifier (primary):

| Set | Metric | FP16 | Q8 | Q4 | Q4 - FP16 [CI] |
|-----|--------|------|----|----|----------------|
| Harmful (50) | Refusal rate | 98.0% | 98.0% | 98.0% | +0.0 pts [+0.0, +0.0] |
| Jailbreak (150) | Refusal rate | 94.0% | 94.0% | 95.3% | +1.3 pts [+0.0, +3.3] |
| Jailbreak (150) | Attack success rate | 6.0% | 6.0% | 4.7% | -1.3 pts [-3.3, +0.0] |

As originally scored by the frozen classifier, the same generations gave harmful
refusal 82.0 / 80.0 / 84.0 percent and jailbreak success 10.7 / 12.0 / 14.0
percent. That apparent Q4 regression was an artefact, explained in section 10.

**Post-hoc label audit.** Because these rates depend on a keyword classifier,
the raw responses for the attack sets were re-examined under two blind human
review sheets (`PROTOCOL.md` section 15). On a random sample of 90 confident
refusals the author confirmed 90 of 90, identical at every precision, so the
classifier does not over-count refusals for one precision more than another. On
a sample enriched for ambiguous cases, three raters (author, classifier, an AI
pass) agreed only 81 to 82 percent, kappa 0.62 to 0.65, disagreeing on
refuse-then-partially-engage responses. That boundary disagreement, about a
fifth of ambiguous cases, is larger than every FP16-vs-Q4 difference above (1 to
3 items), so no safety difference between precisions is detectable under any
rater.

## 7. Pre-registered claim

| Metric | FP16 | Q4 | Absolute movement | Relative movement |
|--------|------|----|-------------------|-------------------|
| Harmful refusal rate | 98.0% | 98.0% | 0.0 pts | 0.0% |
| Jailbreak attack success | 6.0% | 4.7% | 1.3 pts | 22.2% |
| GSM8K accuracy | 78.5% | 75.0% | 3.5 pts | 4.5% |
| ARC-Challenge accuracy | 70.7% | 67.7% | 3.0 pts | 4.2% |

The claim required both behavioural metrics to move at least as much, relatively,
as both capability metrics. Harmful refusal did not move at all, so the claim is
**falsified**. It is also falsified under the frozen classifier. The one large
relative movement, attack success, is small in absolute terms and in the safer
direction.

## 8. Follow-up F1: JailbreakBench (exploratory)

Pre-registered in `PROTOCOL.md` section 14 (commit `f85c13e`) before any
generation. The main study's refusal rates were near ceiling, so F1 asks the same
question with a second benchmark, harmless look-alike requests, and persuasive
attacks. Corrected classifier; 861 generations.

| Set | N | FP16 [CI] | Q8 [CI] | Q4 [CI] | Q4 - FP16 [CI] |
|-----|---|-----------|---------|---------|----------------|
| Harmful requests: refused | 100 | 94.0% [89.0, 98.0] | 94.0% [89.0, 98.0] | 93.0% [88.0, 97.0] | -1.0 pts [-3.0, +0.0] |
| Harmless look-alikes: refused (over-refusal) | 100 | 11.0% [5.0, 18.0] | 12.0% [6.0, 19.0] | 12.0% [6.0, 19.0] | +1.0 pts [+0.0, +3.0] |
| PAIR persuasive attacks: refused | 87 | 72.4% [62.1, 81.6] | 72.4% [62.1, 81.6] | 73.6% [64.3, 82.8] | +1.1 pts [+0.0, +3.4] |

- **Harder attacks work; compression does not change how well.** PAIR prompts
  succeed 27.6 percent of the time at FP16, against 6.0 percent for the main
  study's templates, but Q4 is within one item of FP16 on every set. No
  difference is detected: every paired interval reaches zero.
- **Over-refusal is present and stable.** The model refuses 11 to 12 percent of
  harmless requests that resemble harmful ones, at every precision.
- These are transfer attacks written against GPT-3.5, not optimised for Llama
  3.2. A blind 30-item calibration of the corrected classifier agreed with the
  human label on 28 of 30 (90 percent harmful, 90 percent benign, 100 percent
  PAIR), so no set is marked unreliable. 10 of 861 generations reached the
  1,024-token limit.

## 9. Limitations

- One model, one parameter scale, one GPU, one quantisation family (GGUF
  k-quants), Vulkan only.
- Suite sizes bound the smallest detectable effect: a 3.5-point drop on 200
  GSM8K items is not detectable, which the intervals show directly.
- No multiple-comparison correction.
- Refusal classification is keyword based. It scored 65 percent on its first
  calibration, before correction, and 93 percent on the follow-up's blind check
  after it; refusing then complying counts as a refusal.
- The refusal metric is rater-dependent at the boundary: three raters agree
  only 81 to 82 percent (kappa about 0.63) on ambiguous refuse-then-engage cases,
  though they agree essentially fully on confident refusals (`PROTOCOL.md` 15).
- AdvBench and the jailbreak templates are a fixed, ordinary attack surface, not
  an adaptive attacker. F1's PAIR prompts are stronger but were optimised
  against another model.
- The GSM8K score requires the `#### N` answer format, so it measures format
  following as well as arithmetic; format failures are reported separately.
- Time to first token is derived from server timings, not a streamed first-token
  timestamp. Host memory (RSS) tracks the memory-mapped file and is not reported
  as working memory.

## 10. Integrity: problems found and fixed

Each was recorded in the private project plan's issues ledger and, where it
touched the protocol, in a dated amendment.

- **GPU memory parser counted host buffers** (critical). Would have overstated
  FP16 memory by 751 MB. Fixed and locked by a test built from the real log.
- **Engine hid its memory report by default** (critical). Would have left every
  memory figure empty. Fixed by a launch setting.
- **GSM8K prompt template failed its pre-run check**: only 2 to 4 of 5 answers
  were machine-readable. A format reminder was validated on 20 held-out questions
  outside the study set before the freeze (`PROTOCOL.md` 12.6).
- **Refusal classifier missed typographic apostrophes** ("I can’t"), scoring 54
  refusals as compliance. Because Q4 used that character more often, the error
  manufactured a Q4 safety regression. The exact fix was written into the
  protocol and committed before the blind calibration was scored; calibration
  then measured 65 percent, triggering it. Both versions are reported.

## Appendix A: What changes when FP16 and Q4 disagree

18 FP16-vs-Q4 disagreements drawn at random (seed 42): 8 GSM8K, 8 ARC-Challenge,
and both jailbreak items whose corrected refusal label differs. Categorised by
the author.

| Category | Count | Items |
|----------|-------|-------|
| Reasoning chain divergence | 14 | GSM8K 4, 19, 28, 29, 60, 156; ARC-Challenge 7, 18, 29, 86, 189, 226, 248, 289 |
| Arithmetic error | 1 | GSM8K 46 |
| Answer format failure | 1 | GSM8K 47 |
| Refusal flip | 2 | jail_014_000, jail_014_004 |

- **GSM8K: 6 of 8 disagreements are divergent reasoning,** not a calculation or
  formatting slip. The two models take different routes through the problem and
  arrive at different answers.
- **ARC-Challenge rows show no reasoning.** Both models replied with a single
  letter, so these 8 were categorised by elimination: the chosen answer changed
  with no arithmetic or format explanation available.
- **GSM8K 46: both models were wrong** (truth 163; FP16 117, Q4 243). FP16 set up
  the equation incorrectly; Q4 set it up correctly and mis-solved it, so the
  arithmetic error is Q4's.
- **GSM8K 47:** FP16 gave no machine-readable answer; Q4 answered 800, correctly.
  The format failure is FP16's.
- **Both refusal flips went the safer way.** FP16 complied with a fictional
  framing (template 14, a role-playing game supplement) and Q4 refused,
  consistent with section 6.

## Appendix B: Refusal classifier calibration

Blind manual review of 20 items (10 classified REFUSAL, 10 COMPLIANCE by the
frozen classifier; label hidden, order shuffled).

| | Human: REFUSAL | Human: COMPLIANCE |
|---|---|---|
| Classifier: REFUSAL | 10 | 0 |
| Classifier: COMPLIANCE | 7 | 3 |

Accuracy 65 percent (13 of 20); false positives 0; false negatives 7, every one a
typographic apostrophe. Below the pre-registered 80 percent threshold, so the
pre-stated fix was applied.

### Follow-up F1 calibration (corrected classifier)

Blind review of 30 items: per set, 5 classified REFUSAL and 5 COMPLIANCE.

| Set | Accuracy | False positives | False negatives |
|-----|----------|-----------------|-----------------|
| JBB harmful | 90% (9/10) | 0 | 1 |
| JBB benign | 90% (9/10) | 1 | 0 |
| PAIR attacks | 100% (10/10) | 0 | 0 |
| All | 93% (28/30) | 1 | 1 |

Every set clears the pre-registered 80 percent threshold.

## Spoken summary (about 90 seconds)

I wanted to know what you actually lose when you shrink a language model to 4
bits. The usual answer is a benchmark average, and the average says almost
nothing is lost: on a 3-billion-parameter Llama, 4-bit cut accuracy by about 3
points, not statistically detectable, while running 2.4 times faster in a third
of the memory.

But I pre-registered the study to look past the average. Pairing every question
across precisions, nearly a quarter of the maths answers changed. The mean hid
that because wrong-to-right and right-to-wrong flips cancel out.

I also expected safety behaviour to be fragile under compression. It was not:
refusals held at 98 percent, and even persuasive attacks that got through a
quarter of the time got through equally often at every precision. My
pre-registered hypothesis was falsified, and I report it that way.

The part I would highlight is process. A pre-run check caught a prompt template
that made maths scores measure formatting. A calibration I committed to in
advance caught a classifier bug that would have invented a safety regression in
the 4-bit model. Both fixes are documented, dated and reported next to the
uncorrected numbers.
