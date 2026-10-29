# 13 — Revision plan: ICARC 2027 resubmission

**Created:** 29 October 2026
**Target:** ICARC 2027, Track 01 — Artificial Intelligence and Machine Learning (its topic list includes "Advances in GenAI and XAI")
**Dates:** submission 25 November 2026 · notification 6 January 2027 · camera-ready 21 January 2027 · conference 16–17 February 2027
**Format:** at most six A4 pages in the IEEE conference template, PDF, submitted through CMT, double-blind review

---

## 1. Where things stand

The conference paper ([paper/fisher-rao-conference-paper.md](paper/fisher-rao-conference-paper.md); the six-page version is [fisher-rao.pdf](fisher-rao.pdf)) was rejected at NSBM ICACT 2026 on two reviews.

- **Reviewer 1** scored most criteria 2/5. The paper was judged "mathematically strong", but the research question, the gap, the methodology and the analysis were not clear to the reader, and reproduction details were missing.
- **Reviewer 3** scored most criteria 3/5 and named the validation ladder, the KL–Hessian check, the reparameterisation test and the deliberately broken assembly as strengths. The objections were technical: whether the scrambled metric behind the central claim is a valid and properly controlled geometric object, and the difference between curvature on a slice and curvature of the full manifold.
- Both scored significance 2/5 and reported important literature missing.

A check made while writing this plan (§3) shows that Reviewer 3's first objection is correct and decisive. The revision therefore changes what the paper claims, not only how it is written.

## 2. Reviewer comments and planned responses

| # | Reviewer | Comment (paraphrased) | Planned response | Paper section |
|---|---|---|---|---|
| 1 | R1 | The problem and research question are not clear | Open with the problem; state three research questions; tie each contribution to one | I |
| 2 | R1, R3 | The gap is not established; important literature is missing | Related Work in four groups, each ending with what it leaves open; add the 2026 papers of §6; 35–40 references | II |
| 3 | R3 | The scramble's local probability map, derivative construction and conditioning invariance are not established | Replace it with the consistent control of §3; report both and show why the inconsistent one fails; add an untrained-model control | III, V |
| 4 | R3 | Slice curvature is not distinguished from full-manifold curvature | State the Gauss equation; measure the extrinsic term on a known answer and by nested slices (E4); call the quantity slice sectional curvature throughout | III, V |
| 5 | R1, R3 | Important claims are unsupported | Drop "curvature carries learned structure" unless E1 reverses the pilot; tie every claim to a table or figure | all |
| 6 | R1, R3 | Comparisons with baselines are missing | A controls table (real model, both scrambles, Gaussian `U`, untrained model) and an instruments table (intrinsic `K`, the three published proxies, Euclidean, `UᵀU`) | V |
| 7 | R1, R3 | Limitations and threats to validity are insufficient | A dedicated Threats to Validity section | VI |
| 8 | R1 | Reproduction details are missing | Hyperparameters, seeds, sampling, hardware, runtimes and an anonymised code link | III |
| 9 | R1 | Clarity is weak | Problem-first writing, a pipeline figure, fewer symbols before Section III | all |
| 10 | R3 | Formatting problems | Official IEEE conference template, IEEE reference style, IEEE figure and table conventions | all |
| 11 | R1, R3 | Significance is limited | Say in the first paragraph who needs the result and why: which geometric probes of a language model can be trusted for interpretability | I, VII |

## 3. The finding that changes the central claim

**The problem.** The paper's paired scramble (`fisher_metric_projected(rows=...)` in [fisherrao/metrics.py](fisherrao/metrics.py)) takes the predictive distribution `p(h)` from the real model but the Jacobian from permuted unembedding rows. At the base point that matches the paper's description. Curvature, however, is built from the first and second derivatives of the metric, and away from the base point `p` moves along the real directions while the metric measures along the permuted ones. The result is a valid Riemannian metric, but it is not the Fisher pullback of any statistical model.

**The consistent control.** [run_scramble_consistent.py](run_scramble_consistent.py) realises the same permutation as a genuine model,

    z(h) = U[idx] n(h0) + U[rows] (n(h) − n(h0)),    p(h) = softmax(z(h)),

which agrees with the paper's control exactly at the base point (same `p`, metric, frame, conditioning and row multiset; relative difference 0) and differs only in that `p` moves along the directions the metric uses. With the identity permutation it reproduces the real metric to 1.4 × 10⁻¹⁴.

**Pilot result** (SmolLM2-135M, `k` = 5, 16 planes, 30 states stratified by entropy; [results/scramble/consistent_pilot.json](results/scramble/consistent_pilot.json)):

| condition | median `K` | real > condition |
|---|---|---|
| real model | 0.2554 | — |
| paper's scramble (not a pullback) | −0.0004 | 27/30, z = +4.38 |
| consistent scramble (a pullback) | 0.2532 | 15/30, z = 0.00 |

The Gaussian-unembedding control of [08-rq3a-log.md](08-rq3a-log.md) §5.1, which is also a genuine pullback with no learned structure, had already matched the real `K` to within 0.002.

**What it implies.** The collapse reported in the paper comes from the scrambled metric not being a pullback, not from destroying the learned assignment. `K ≈ +1/4` appears to be a property of softmax-pullback geometry rather than of learning.

**What it does not affect.** The spectral results (log-volume ρ = +0.957, effective dimension ρ = +0.991) are functions of `G(h0)` alone, and at `h0` the two controls are identical. The validation ladder, the null-direction test, `K ≈ +1/4` as a measurement, the calibration of the 10⁻⁵ reading and the behavioural probe also stand.

**Caveat.** One model, 30 states and `k` = 5. E1 and E2 (§5) decide the claim. Full record: [08-rq3a-log.md](08-rq3a-log.md) §5.7 and [12-audit-log.md](12-audit-log.md) §8.

## 4. Proposed framing (to be agreed with the supervisor)

**Working title:** *What Does Fisher–Rao Geometry Measure in Transformer Representations? A Controlled Study of Curvature and Spectrum*

**Problem statement (draft).** Geometric summaries of the Fisher–Rao metric on transformer hidden states — its spectrum, effective dimension, volume and curvature — are increasingly used to describe what language models represent. Whether these quantities reflect learned structure, or are fixed by the predictive distribution and the geometry of the probability simplex, has not been tested with controls that hold the predictive distribution fixed and remain genuine pullbacks.

**Research questions.**

- **RQ1.** Can the intrinsic Riemannian curvature of the Fisher–Rao pullback metric on transformer hidden states be computed reliably? *(method, validation ladder, null-direction test)*
- **RQ2.** What value does that curvature take across architectures, and is the value produced by learning? *(`K ≈ +1/4`; Gaussian, untrained and consistent controls)*
- **RQ3.** What do the spectral summaries — log-volume and effective dimension — read? *(matched-entropy controls; the depth profile)*

**Contributions, if E1–E2 confirm the pilot.**

1. A validated pipeline for the intrinsic curvature of the Fisher–Rao pullback on transformer hidden states.
2. `K ≈ +1/4` across four architectures, shown by genuine-pullback controls to be inherited from the simplex rather than learned.
3. The spectral profiles, including the depth "hourglass", reproduced by structure-free controls at matched entropy.
4. A methodological result: a control that is not itself a pullback manufactures a spurious collapse (0.255 → 0.000), and a consistency criterion that prevents it.
5. A calibration of the published near-flat reading (10⁻⁵): the same estimator reports a unit 3-sphere as flat.

**Dropped:** "curvature carries learned structure". **Moved to future work:** the causal probe (current §VI.H). **Kept briefly if space allows:** the behavioural probe.

**If E1 differs from the pilot** on other architectures, the framing changes to report where real and consistent curvature diverge. The claim follows the data either way.
