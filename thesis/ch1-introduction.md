# Chapter 1 — Introduction

> **Draft.** Written last, from Chapters 3–7. Sources: [02-research-gap.md](../02-research-gap.md) and the results chapters.

---

## 1.1 The question

A transformer's hidden states are usually treated as vectors in `ℝᵈ`, and distances between them as Euclidean. But every hidden state `h` induces a probability distribution over the vocabulary through the model's own output head,

```
p(h) = softmax( U · norm(h) + b )
```

so hidden states are not vectors. **They are points on a statistical manifold**, and the canonical way to measure distance between them is the Fisher–Rao metric — canonical by Čencov's theorem, which makes it the unique metric invariant under sufficient statistics.

That much is known. What is not known is what that manifold looks like.

> **Is the representation manifold of a transformer flat or curved, and does the answer depend on the model or only on how confident its predictions are?**

Three published papers answer differently, and this thesis shows that all three answered with instruments that cannot distinguish a sphere from a plane at the settings they used.

---

## 1.2 The gap

The Fisher–Rao pullback metric on transformer hidden states has been derived correctly — twice, independently. Layer-wise curvature has been measured — twice, under metrics that are *constant* and therefore have identically zero intrinsic curvature.

> **The genuine intrinsic Riemannian curvature of the state-dependent Fisher–Rao metric has never been computed for a transformer.**

Three groups each hold two of the three necessary pieces (Chapter 3 §3.1). Nobody holds all three, and the reason is a single technical obstacle that the literature states incorrectly: not that the metric is singular — it is generically full rank — but that it is **catastrophically ill-conditioned**, and that the normalisation layer contributes an **exact** nullity nobody had identified.

---

## 1.3 What this thesis does

**It computes the thing.** The full Riemann tensor of the Fisher–Rao pullback on real transformer hidden states, at ~2.6 seconds per point, validated against three analytically known answers at machine precision.

The method is two steps and Chapter 4 develops both: **quotient by the exact null space, then restrict to the effective subspace.** The compensating fact that makes it work at all is that the catastrophic conditioning lives entirely in the directions being discarded — *within* the retained subspace, `cond_eff` is only 10¹–10².

**It adjudicates the disagreement with positive controls rather than by argument.** The decisive demonstration is not that this thesis's number differs from Mabrok's, but that at the setting which reproduces his published `10⁻⁵`, **his own instrument reports a unit 3-sphere as flat to 10⁻³¹**.

---

## 1.4 Principal findings

**The manifold is strongly positively curved, not flat.** `K ≈ +1/4` — the ambient categorical-simplex value — on **four architectures**, 456 points, within 0.022 of each other across changes of normalisation kind, depth, width and embedding regime.

**No published proxy tracks it, or tracks any other.** On identical activations at `n` = 360, every pairwise |ρ| ≤ 0.25 and every within-layer |ρ| ≤ 0.11, against a positive control of +0.72. The result holds on two corpora, and **the intrinsic instrument is corpus-invariant while every proxy is not** — proxies move 0.10–0.19 between corpora, the intrinsic quantities 0.03.

**The curvature carries learned structure; the spectrum does not.** Under one identical control — destroying which probability sits on which direction, at exactly matched entropy — sectional curvature collapses from **0.2546 to 0.0109** (n = 221, 201/221, z = +12.18), while layer-wise log-volume (ρ = **+0.957**) and effective dimension (ρ = **+0.991**) are reproduced almost exactly.

> **Everything derived from the metric's spectrum reads predictive concentration. Only the curvature, which needs the second derivatives, reads the model.**

This is the sharpest statement the thesis has, and it retires two shapes reported in the literature — the volume profile and the hourglass in intrinsic dimension — as largely definitional.

**The geometry is the model's, not the chart's.** Perturbing a hidden state along a direction the metric calls null leaves the output unchanged to `KL ≈ 10⁻¹³` at a step size that *doubles* the state, against 2.6–7.8 for a random step of the same size. **On all six models tested.** The metric further *ranks* directions by how much they matter, and the model's behaviour bears the ranking out.

**The metric changes a substantive answer by ~20 layers.** Where two contextual variants of a token separate: layer 4–7 under Fisher–Rao, layer 26–28 under the raw flat metrics. Reproduced on a standard sense-annotated dataset. Controls identify the mechanism as **scale-invariance** — which Fisher–Rao has by construction and the published instruments lack.

---

## 1.5 What was retracted, and why that is here

Six claims in this work were weakened or withdrawn by their own controls:

| claim | what killed it |
|---|---|
| the deviation `K − 1/4` is the model-specific signal | `k`-instability; layer ranking scrambled |
| Manson and King agree with each other (ρ = +0.685) | `n` = 40 → 360; fell to +0.248 |
| the curvature–entropy link is mostly definitional | the control was the *identity* on this metric |
| "with wording matched, sense separates" | the wording was not matched — 2.00× overlap difference |
| Ricci predicts departure at matched Fisher norm | `n` = 40 → 105, then a `k` sweep; narrowed three times |
| Pythia's volume offset is the norm gain | a 24-point convenience sample; reversed at `n` = 220 |

**These are in the thesis rather than out of it.** Each was found by a control this work built to attack its own result, and the pattern they form is itself a finding: *a result at one `n` and one `k` is a hypothesis, not a finding* (Chapter 4 §4.4.1). A reader who wants to know how far to trust Chapter 6 is better served by the list of things that did not survive than by any assurance.

---

## 1.6 Contributions

1. The **first computation** of the intrinsic Riemannian curvature of the Fisher–Rao pullback metric on transformer hidden states, validated to machine precision against three known answers.
2. A **method** for getting past the conditioning wall, with its limits measured — including a `k` ceiling that the natural selection rule violates by a factor of two.
3. An **adjudication** of a live three-way disagreement, resolved by showing that neither published instrument can distinguish curved from flat at its own operating point.
4. A **separation** of what the metric's spectrum measures from what its curvature measures, established by one control applied to three quantities.
5. A **causal falsification test** passing on six models at machine precision — the behavioural validation the prior literature names as missing.
6. A demonstration that **the choice of metric changes a substantive empirical answer** by ~20 layers, with the mechanism identified by control rather than asserted.

---

## 1.7 Structure of the thesis

| | |
|---|---|
| **Chapter 2** | Background: statistical manifolds, the Fisher–Rao metric, Riemannian conventions fixed once, and the two different Fishers |
| **Chapter 3** | Literature across five streams, the gap, and the disagreement being adjudicated |
| **Chapter 4** | Methodology: the real obstacle, the quotient-and-restrict method, `k` selection, and the controls |
| **Chapter 5** | **Validation** — the ladder, both gates, and what validation does *not* cover |
| **Chapter 6** | Layer-wise results, the instrument comparison, and the spectrum-versus-curvature split |
| **Chapter 7** | Behavioural and causal analysis |
| **Chapter 8** | Discussion, limitations and future work |

> **Chapter 5 is not preliminary matter.** Curvature errors are silent and produce plausible numbers; the reference implementation in this work initially returned `−0.250000` for the simplex — right magnitude, wrong sign, and constant across every sample point, which made it look *more* trustworthy. Only a known answer caught it. Chapter 5 is what makes Chapters 6 and 7 worth reading.
