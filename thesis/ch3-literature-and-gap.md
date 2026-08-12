# Chapter 3 — Literature review and the gap

> **Draft.** Sources: [01-literature-review.md](../01-literature-review.md), [02-research-gap.md](../02-research-gap.md), and Chapter 6's replication results.

---

## 3.1 Five streams, and what each one is missing

The work this thesis sits between falls into five streams. None of them is wrong; each holds part of what is needed.

| Stream | Representative work | Has the Fisher–Rao metric on hidden states | Computes intrinsic curvature | On transformers |
|---|---|---|---|---|
| **A** Fisher–Rao pullbacks on activations | Mabrok 2026 (`2603.22301`) | ✅ derives `G(h)` exactly | ❌ PCA proxies; calls it intractable | ✅ |
| | FishBack 2026 (`2605.17231`) | ✅ same metric + layer recursion | ❌ eigenvalues only, explicitly no curvature | ✅ |
| **B** Layer-wise trajectory curvature | Manson 2025 (`2507.21107`) | ❌ constant `UᵀU` | ❌ Frenet curvature in a **flat** space | ✅ |
| **C** Contextual curvature | King et al. 2026 (`2604.23985`) | ❌ Euclidean angles | ❌ path bending, not space bending | ✅ |
| **D** Intrinsic curvature of pullback metrics | Zavatone-Veth et al. 2025 | ❌ generic pullback | ✅ Ricci scalar | ❌ CNNs on images |
| **E** Manifold structure of LM representations | Valeriani et al. 2023 | ❌ ambient point clouds | ❌ intrinsic *dimension*, not curvature | ✅ |

---

## 3.2 The gap

> **The Fisher–Rao pullback metric on transformer hidden states has been correctly derived, and layer-wise curvature has been measured under flat metrics. The genuine intrinsic Riemannian curvature of the state-dependent Fisher–Rao metric has never been computed for a transformer.**

Three groups each hold two of the three necessary pieces. Nobody holds all three.

### 3.2.1 A distinction that must be made early

Two objects share the name "Fisher information" and conflating them is the most likely misreading of this thesis.

| | parameter-space Fisher | **the pullback used here** |
|---|---|---|
| lives on | model weights | **hidden states** |
| dimension | billions | **`d` = 576–1600** |
| cost | intractable; K-FAC approximates it | **closed form, one forward pass** |
| literature | optimisation, natural gradient | information geometry of the output distribution |

The K-FAC literature's intractability results say nothing about the object studied here.

---

## 3.3 Why the gap stayed open — and why the usual explanation is wrong

Not because the question is uninteresting. Because of one technical obstacle, which the literature states incorrectly.

**The loose version, widely repeated.** *`Σ_p = diag(p) − ppᵀ` annihilates the all-ones direction, so `G(h) = JᵀΣ_pJ` is singular and `G⁻¹` does not exist.*

**Why it does not follow.** `null(G) = {v : Jv ∈ span{1_N}}`, which requires `1_N ∈ range(J)` — a `d`-dimensional subspace of `ℝᴺ` with `d ≪ N`. Generically it is not, and `G(h)` is **full rank**. Verified directly: forcing `1_N` into `range(J)` recovers a deficiency of exactly 1, and otherwise there is none.

**The correct version — two obstacles, both measured on real activations rather than assumed:**

1. **Conditioning.** A language-model softmax is sharply peaked, so the spectrum of `G` decays steeply. Across 31 layers, `λ_min` underflows to exactly 0 at **22 of them**, and only **0.5–14.9%** of eigendirections hold 99% of the trace — independently replicating the 2–17% FishBack reports. `G⁻¹` formally exists and is numerically useless. Curvature needs *second* derivatives, roughly doubling the digit loss.

2. **An exact nullity from the normalisation layer, which the literature does not identify at all.** RMSNorm and LayerNorm are scale-invariant, so their Jacobians are projectors and `h` itself is an **exact** null direction of `G(h)` — LayerNorm contributes a second, from the mean subtraction. Doubling a hidden state changes the model's prediction by `KL ≈ 10⁻¹³` (Chapter 5 §5.5.3).

> This second obstacle is also the resolution. **The predictive map depends only on the direction of `h`**, so the semantic manifold is a space of directions, and the quotient formulation is the *correct* formulation rather than a workaround.

**Every prior author stopped at this wall:** Mabrok calls true Riemann curvature "computationally intractable" and retreats to local-PCA proxies; FishBack regularises the metric but never differentiates it; Zavatone-Veth et al. report the Ricci scalar "numerically challenging" even for small networks and mostly retreat to volume elements.

---

## 3.4 The live disagreement this thesis adjudicates

Two published results are in tension, and this is what makes the gap worth closing now rather than an exercise in applying method X to domain Y.

| source | instrument | result |
|---|---|---|
| Mabrok 2026 | local-PCA residual variance, ambient coordinates | curvature ≈ **10⁻⁵**, flat, stable across layers |
| Manson 2025 | Frenet curvature under constant `UᵀU` | curvature **varies** with semantics and depth |
| King et al. 2026 | Euclidean angles between difference vectors | curvature **correlates** with next-token entropy (r ≈ 0.15) |

**Both instruments are wrong for the question asked.** Mabrok's is extrinsic and embedding-dependent. Manson's and King's metrics are *constant*, so their spaces have **identically zero** intrinsic curvature — they can only measure how a *path* bends, never how the *space* bends.

### 3.4.1 What this thesis found when it checked them

Chapter 6 does not merely assert that the proxies are the wrong instrument; it measures them on identical activations and shows why.

- **They do not track the intrinsic quantity, or each other.** Every pooled |ρ| ≤ 0.25, every within-layer |ρ| ≤ 0.11, against a positive control of +0.72 (§6.3).
- **Mabrok's `10⁻⁵` reproduces — and is his threshold.** At the setting that yields it, the same code reports a **unit 3-sphere as flat to 10⁻³¹** (§6.4).
- **His second proxy is worse, not safer.** At `q` = 5 it reports a perfectly flat plane as *more* curved than a sphere (§6.4.2).
- **The proxies are corpus-dependent and the intrinsic quantity is not** — proxies move 0.10–0.19 between corpora, intrinsic quantities 0.03 (§6.3.2).

> The disagreement is resolved not by preferring one published number to another, but by showing that **neither instrument can distinguish curved from flat at the settings its authors used.**

---

## 3.5 Contributions claimed

1. **The first computation of the intrinsic Riemannian curvature of the Fisher–Rao pullback metric on transformer hidden states**, validated against three analytically known answers at machine precision.
2. **A method for getting past the conditioning wall** — quotient by the exact null space, then restrict to the effective subspace — together with the measured limits of that method, including a `k` ceiling the natural selection rule violates.
3. **An adjudication of a live disagreement**, with positive controls showing the published proxies cannot see the curvature of a sphere at their own operating points.
4. **A separation of what the metric's spectrum measures from what its curvature measures**: under one identical control, log-volume and effective dimension are reproduced by a structure-free scramble (ρ = +0.97, +0.98) while sectional curvature collapses.
5. **A causal falsification test** passing on six models at machine precision, answering the behavioural-validation gap Manson's own future work names.
6. **A demonstration that the metric changes a substantive answer** — where two contextual variants of a token separate — by ~20 layers, on two corpora, with the mechanism (scale-invariance) identified by control rather than asserted.

---

## 3.6 Scope boundaries

Stated so the thesis is not read as claiming them.

- **α-connections and the dually-flat structure** are not treated; only the Levi-Civita connection of the Fisher–Rao metric.
- **Fine-tuning and training dynamics** are out of scope; all measurements are on trained, frozen models.
- **Multimodal and encoder-decoder architectures** are not covered.
- **Models are 70M–160M parameters.** The largest models in the comparison literature are not included.
- **Curvature is computed on a `k` ≤ 8 affine slice**, not on the full manifold.
