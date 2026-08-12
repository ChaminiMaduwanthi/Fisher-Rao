# Literature Review

**Project:** Information-Geometric Curvature Analysis of Transformer Representations Using Fisher–Rao Metrics
**Review date:** 10 August 2026
**Status:** Draft 1 — evidence base for the gap statement in [02-research-gap.md](02-research-gap.md)

---

## 0. How to read this document

The field splits into **four research streams** that have each solved part of the problem but have never been joined:

| Stream | What it has | What it lacks |
|---|---|---|
| A. Fisher pullback metrics on transformers | The correct metric `G(h)` | Any true curvature computation |
| B. Curvature of LM trajectories | Layer-wise curvature measurements | A statistically correct metric (uses flat/Euclidean) |
| C. Intrinsic curvature of pullback metrics | Real Riemann/Ricci computation | Not transformers — CNNs on images |
| D. Manifold structure of LM representations | Dimension, anisotropy, manifold evidence | Curvature only via proxies |

**The thesis sits precisely at the intersection of A and C.** Section 6 states this formally.

> ⚠️ **Verification note.** Entries marked 🟢 were read at full-text or methods-section level. Entries marked 🟡 were read at abstract/summary level and **must be re-read in full before the review chapter is finalised** — do not cite a claim from a 🟡 entry in the thesis without checking the original.

---

## Stream A — Fisher–Rao pullback metrics on transformer activations

This is the stream closest to the thesis title. Two papers matter.

### A1. 🟢 Mabrok (2026), *Latent Semantic Manifolds in Large Language Models*
`arXiv:2603.22301` — submitted 17 March 2026, single author (Mohamed A. Mabrok)

**This is the single most important paper for the project. Read it line by line before anything else.**

What it does:
- Treats LLM hidden states as points on a Riemannian submanifold equipped with the Fisher information metric; tokens become Voronoi regions partitioning that manifold.
- **Derives the exact metric the thesis needs** (their Definition 5.1 / Proposition 5.2):

  ```
  G(h) = Wᵀ (diag(p) − p pᵀ) W  =  Wᵀ Σ_p W
  ```

  where `p = p(·|h) ∈ ℝᴺ` is the next-token distribution and `W` the unembedding matrix. In local semantic coordinates: `g^F(z) = J_Φ(z)ᵀ Wᵀ Σ_p(z) W J_Φ(z)`.
- Introduces the *expressibility gap* (geometric distortion from finite vocabulary) and proves a rate–distortion lower bound plus a linear volume scaling law via the coarea formula.
- Validates on six architectures, 124M–1.5B params.

Findings:
- Universal **hourglass intrinsic-dimension profile**; representations occupy only ~1–3% of ambient dimensions.
- Linear expressibility-gap scaling, slopes 0.87–1.12, R² > 0.985.
- A scale-invariant "hard core" of boundary-proximal representations, giving a geometric decomposition of perplexity.

**Critical weakness — and the opening for this thesis:**
- The paper *defines* the Riemann curvature tensor (Def. 9.1) and sectional curvature (Def. 9.2) **but never computes them.**
- Empirical curvature is measured by **two proxies only**: (i) local-PCA curvature — residual variance orthogonal to the dominant tangent subspace; (ii) the norm of the second fundamental form ‖II‖, estimated by tangent-plane rotation between neighbours.
- Reported result: PCA curvature is **uniformly ~10⁻⁵ across all layers** and "broadly stable"; ‖II‖ bounded. The paper's own reading is that the manifold is locally well-approximated by its tangent plane.
- The paper concedes that true Riemann curvature on estimated manifolds "remains computationally intractable."
- **No explicit Limitations or Future Work section.** Section 15.1 ("When the Manifold Hypothesis Breaks Down") is speculative.
- The framework as presented **assumes tied embedding/unembedding weights**; behaviour with untied matrices is unclear. Many modern models are untied — this matters.

> **Why the ~10⁻⁵ finding is an opportunity, not a closed door.** Local-PCA residual variance is an *extrinsic, embedding-dependent* quantity in the **ambient Euclidean coordinates**. It is not the intrinsic sectional curvature of the Fisher metric, and there is no theorem making one a bound on the other. A manifold can be extrinsically near-flat in ambient coordinates while the induced Fisher metric has large intrinsic curvature, because `G(h)` varies sharply with `h` even where the point set looks flat. **Whether the ~10⁻⁵ result survives a genuine intrinsic computation is an open empirical question — and answering it is a publishable contribution either way.**

### A2. 🟢 Wang & Zhao (2026), *FishBack: Pullback Fisher Geometry for Optimal Activation Steering in Transformers*
`arXiv:2605.17231v1` — 17 May 2026

- Uses the **same pullback Fisher metric** `G^(ℓ)(h) = Jᵀ H J` on intermediate activation space, described as closed-form, layer-specific, and input-dependent.
- **Theorem 3.3 — layer-wise recursion:** `G^(ℓ) = A_ℓᵀ G^(ℓ+1) A_ℓ` with `A_ℓ = I + DF_ℓ` the Jacobian of residual block ℓ. A congruence transformation, giving depth-dependent condition-number bounds; anisotropy compounds exponentially with depth. **This recursion is directly reusable and is the cheapest correct way to propagate the metric backwards through layers — adopt it.**
- Two numbers worth quoting in the thesis introduction: the Fisher metric deviates from Euclidean by **>97% in relative spectral norm** on GPT-2, and effective dimensionality is only **2–17% of ambient**.
- Shows existing steering methods each implicitly assume a particular *approximate* metric.

Computational recipe (reusable):
- Jacobian `J` via **768 backward passes** for GPT-2 small (d = 768).
- Fisher `H` estimated from the **top-5000 tokens by probability mass** — not the full 50k vocabulary.
- **Adaptive Tikhonov regularisation** `α = c · λ_median(G)` to handle the singular metric.

**Critical weakness:** the paper **explicitly does not compute Christoffel symbols, the Riemann tensor, or sectional/scalar curvature.** All analysis is spectral — eigenvalues, condition numbers, effective rank. Confined to GPT-2 small; the authors flag that Jacobian computation becomes prohibitive at larger scale.

### A3. 🟡 Zaher, Trzaskowski, Nguyen & Roosta (2026), *The Geometric Wall: Manifold Structure Predicts Layerwise Sparse Autoencoder Scaling Laws*
`arXiv:2605.09887` — 12 May 2026

- States the **rank bound** for the pullback: `rank G(h) ≤ min(N − 1, rank J)`, where N is vocabulary size.
- ⚠️ **Note the bound is non-binding in the regime of interest.** With `d` ≈ 768 and `N` ≈ 50 000, `min(N−1, rank J)` = `d` whenever `J` has full column rank — i.e. the bound is satisfied trivially. It bites only when `J` is *itself* low-rank, which is the interesting case for deep pullbacks through many residual blocks. Do not cite it as evidence that `G` is singular; see §6 and `conditioning_check.py`.
- PDF text extraction failed on this source. **Action: obtain the HTML or LaTeX version and re-read**, and check whether the authors intend the bound as a statement about `J` degenerating with depth.

---

## Stream B — Curvature of layer-wise representation trajectories

These papers measure curvature *of transformer trajectories* — but under a **flat metric**, which is the flaw the thesis corrects.

### B1. 🟢 Manson (2025), *Curved Inference: Concern-Sensitive Geometry in Large Language Model Residual Streams*
`arXiv:2507.21107` — 8 July 2025. Also a Feb 2026 consolidated framework paper (Bon View Press, *AIA*).

Definitions:
```
salience    S(t) = ‖x_{t+1} − x_t‖_G                    (first-order, layer-wise step size)
curvature   κᵢ  = √(‖aᵢ‖²_G ‖vᵢ‖²_G − ⟨aᵢ,vᵢ⟩²_G) / ‖vᵢ‖³_G
```
with `v` velocity, `a` acceleration, 3-point central finite differences across layers.

**The decisive detail: the metric is `G = Uᵀ U`, the unembedding Gram matrix.**

> 🔴 **`G = UᵀU` is a constant, state-independent metric.** It does not depend on `h`. A constant metric has **identically zero Riemann curvature** — the space is flat, isometric to Euclidean space after a linear change of coordinates. So `κᵢ` here is the **Frenet curvature of a curve in a flat space**, not curvature of the representation manifold. It is a legitimate and useful quantity, but it is *categorically not* what the thesis title promises, and it is **not the Fisher–Rao metric** — `UᵀU` omits the state-dependent `Σ_p` factor entirely.
>
> Equivalently: Manson measures how much the *path* bends. This thesis measures how much the *space* bends. Making that distinction crisply in the introduction is one of the cheapest and strongest framing moves available.

Findings:
- LLaMA 3.2-3b: curvature and salience scale significantly with concern intensity (p = 0.006, p = 0.016). Gemma 3-1b: weak separation.
- Concern-induced curvature begins at concern tokens and propagates downstream with "turbulence."
- Strong salience↔curvature anticorrelation, r = −0.89 (LLaMA).
- Structure appears only in the residual stream.

Limitations (author-stated) — **all of them are things this thesis can improve on**: only 20 curated prompts across 7 domains; models limited to 1–3B; **no behavioural validation** linking curvature to output; 3-point finite differences unstable on longer sequences; unresolved early-layer artefacts in Gemma.

Author's own future work list includes *"singularity analysis of embedding space combined with contextual curvature"*, force attribution, causal interventions on high-curvature layers, and alignment monitoring via curvature spikes.

### B2. 🟢 King, Fedorenko & Hosseini (2026), *Representational Curvature Modulates Behavioral Uncertainty in Large Language Models*
`arXiv:2604.23985v1` — 27 April 2026. MIT; Hosseini now at Google DeepMind.

- **Contextual curvature**: angle between adjacent first-order difference vectors of consecutive-token activations, averaged over a 3-token backward window. **Plain Euclidean angles — no metric at all.**
- Models: GPT-2 XL (48 layers), Pythia-2.8B (32 layers). Data: LAMBADA (long context), Universal Dependencies (short, 10–30 tokens).
- Findings: middle-layer curvature correlates with next-token entropy (**r ≈ 0.15**), peaking where trajectories are straightest; the coupling **emerges during pretraining** alongside trajectory straightening; only trajectory-aligned perturbations modulate entropy while random ones do not; training with a curvature penalty modestly reduces entropy without hurting validation loss.
- Author-stated limitations: curvature collapses complex geometry to one scalar; entropy misses distributional structure; effects were **small** and at limited scale; unclear whether entropy reduction improves calibration or downstream performance.

**Why this paper is strategically valuable:** it establishes that curvature–behaviour coupling is *real and measurable*, which de-risks the thesis premise. But r ≈ 0.15 is weak — and a plausible reason is the **wrong metric**. "Does the Fisher–Rao metric strengthen the curvature–entropy relationship relative to the Euclidean baseline of King et al.?" is a sharp, pre-registerable, falsifiable hypothesis with a ready-made baseline number to beat. **Make this the headline empirical claim.**

### B3. 🟡 *Beyond Scalars: Evaluating and Understanding LLM Reasoning via Geometric Progress and Stability* — `arXiv:2603.10384`
### B4. 🟡 *Truth as a Trajectory: What Internal Representations Reveal About LLM Reasoning* — `arXiv:2603.01326`

Both apply trajectory geometry to reasoning quality and truthfulness. Relevant to the *application* stage (Stage 4) rather than the method. **To be read in Stage 1.**

---

## Stream C — Intrinsic curvature of neural-network pullback metrics

This stream has the mathematical machinery the thesis needs. It has never been applied to transformers.

### C1. 🟢 Zavatone-Veth, Yang, Rubinfien & Pehlevan, *How does training shape the Riemannian geometry of neural network representations?*
`arXiv:2301.11375v4` — **NeurIPS 2025 Workshop on Symmetry and Geometry in Neural Representations (NeurReps)**, oral. Harvard/Yale. (Proceedings appear via PMLR — *not* JMLR; confirm the exact citation format from OpenReview `id=BaVIDhh7bj` before submitting.)

**This is the methodological template. Stream A gives the metric; this gives the curvature machinery.**

- Pullback metric from a feature map `Φ`: `g_μν = ∂_μΦ_i ∂_νΦ_i`.
- Computes **two** geometric quantities: the volume element `dV = √(det g) dᵈx`, and the **Ricci scalar** `R = g^{βν} R^α_{βαν}` — genuine intrinsic curvature.
- Gives a simplification: when `∂_α g_μν` is symmetric under index permutation,
  `R = −¾ g^{ρρ}(∂_α g^{μρ} ∂_β g^{νρ} − ∂_β g^{μρ} ∂_α g^{νρ})`. **Check whether this symmetry condition holds for the Fisher pullback — if it does, it is a large computational saving.**
- Metric differentiability needs activations `Cᵏ, k ≥ 3` → **GELU/SiLU fine, ReLU problematic.** Transformers use GELU/SiLU, so the thesis is well-placed here; note it explicitly as a favourable condition.
- Handles the degenerate/rank-deficient case: singular semi-Riemannian manifold, volume element on the quotient `ℳ/∼` from the product of **non-zero eigenvalues**. **Directly applicable to the singular Fisher pullback.**

Findings: infinite-width nets at init give spherically symmetric, task-agnostic metrics; trained finite-width nets develop volume-element peaks **precisely at decision boundaries**, strengthening with depth, across shallow nets, ResNets and self-supervised methods (Barlow Twins, SimCLR); the lazy/kernel regime keeps geometry static.

Limitations, stated honestly by the authors and **directly transferable as risks to this project**:
- Ricci scalar computation is *"computationally expensive and numerically challenging"* even for small networks — most of their analysis retreats to volume elements.
- Analysis confined to toy 2D tasks or low-dimensional slices; the authors call this a "fundamental limitation."
- **No transformers, no language models** — CNNs and MLPs on MNIST/CIFAR-10 only.
- No causal test of whether geometry drives generalisation.
- Their future-work list explicitly names extension to *"pre-trained networks for which one does not have access to training classes"* — **which is exactly a pretrained LLM.** Quote this: it is an invitation from an established group at a top venue.

### C2. 🟡 Hauser & Ray, *Principles of Riemannian Geometry in Neural Networks* — NeurIPS 2017
Early treatment of pullback metrics across layers; decomposes computation into discretising continuous features plus logic on discrete variables. Historical framing.

### C3. 🟡 *Riemannian Geometry with differentiable ambient space and metric operator* — `arXiv:2105.01583`
**Practically the most useful item in this stream.** Reports that computing Christoffel symbols directly and then forming the Ricci scalar is *significantly less efficient* than alternatives, and that with autodiff one only needs the contracted form `Γᵏ_ij vⁱ vʲ` rather than the full symbol array. **Read this before writing a single line of curvature code — it may decide the entire implementation strategy.**

### C4. 🟡 *Emergent Riemannian geometry over learning discrete computations on continuous manifolds* — `arXiv:2512.00196`
### C5. 🟡 *RNNs perform task computations by dynamically warping neural representations* — `arXiv:2512.04310`

Sequence-model geometry; closer to transformers than C1 but on RNNs. **Read in Stage 1.**

---

## Stream D — Manifold structure of LM representations (context, not method)

### D1. 🟡 Valeriani et al. (2023), *The geometry of hidden representations of large transformer models* — NeurIPS 2023, `arXiv:2302.00294`
Establishes the layer-wise intrinsic-dimension profile with a local minimum around 0.3–0.4 relative depth and a second shallower peak near the end. **The canonical citation for "representations live on a low-dimensional manifold" — needed to justify the manifold hypothesis in the thesis.**

### D2. 🟡 Modell, Rubin-Delanchy & Whiteley (2025), *The Origins of Representation Manifolds in Large Language Models* — `arXiv:2505.18235`
Argues features are **manifolds**, not just linear directions, and that cosine similarity may encode intrinsic feature geometry via shortest **on-manifold (geodesic) paths**. Challenges the linear representation hypothesis. **Theoretical justification for caring about geodesics rather than straight lines — cite early in the thesis.**

### D3. 🟡 *The Shape of Learning: Anisotropy and Intrinsic Dimensions in Transformer-Based Models* — `arXiv:2311.05928`
Anisotropy of contextual embeddings. Supports the claim that Euclidean coordinates are the wrong frame.

### D4. 🟡 *Symmetry in language statistics shapes the geometry of model representations* — `arXiv:2602.15029`
Continuum features (years, number lines) form compact 1D manifolds with "ripples" — **extrinsic curvature**. A clean, low-dimensional, well-understood test case. **Strong candidate for the Stage 3 validation suite: a place where the correct answer is partly known in advance.**

### D5. 🟡 *A Comparative Study of Learning Paradigms in LLMs via Intrinsic Dimension* — `arXiv:2412.06245`; *The representation landscape of few-shot learning and fine-tuning* — `arXiv:2409.03662`
Fine-tuning vs in-context learning produce different geometry. Relevant if the thesis extends to fine-tuning dynamics (optional Stage 6).

---

## Stream E — Computational tooling and Fisher approximation

### E1. 🟢 K-FAC and Fisher approximation family
Forming or inverting the full Fisher is infeasible when it couples all parameters. K-FAC approximates layerwise blocks by Kronecker products of two smaller factors. Related: Matrix-free Fisher Factorization (MFF), which avoids explicit storage; Fisher-Weighted SVD (FWSVD) for compression; Bayesian LoRA exploiting wide-then-narrow adapter shapes for cheap K-FAC.

> ⚠️ **Do not confuse two different Fishers.** K-FAC and the natural-gradient literature concern the Fisher over **parameters θ** (dimension = billions). This thesis concerns the Fisher pulled back onto **hidden states h** (dimension = 768–4096). The thesis object is *vastly* smaller and admits closed form. **State this distinction explicitly in the thesis — reviewers and examiners will otherwise assume the project is intractable, and it is the single most likely misunderstanding at the proposal defence.**

### E2. 🟢 Le Brigant, Deschamps, Collas & Miolane (2023), *Parametric information geometry with the package Geomstats*
*ACM Transactions on Mathematical Software*; `arXiv:2211.11643`

Implements Fisher–Rao Riemannian manifolds for normal, gamma, beta, Dirichlet families, as child classes of `RiemannianMetric`, with sectional curvature via the Riemann tensor.

**Use as the ground-truth oracle for the validation ladder in [03-methodology.md](03-methodology.md) §5.** Independently implemented, peer-reviewed, and it covers families with analytically known curvature.

⚠️ **Check the sectional-curvature sign and normalisation convention before comparing.** Conventions differ across sources in two independent ways: the index order of the lowered Riemann tensor (flips the sign) and whether the simplex is normalised to a radius-2 or unit sphere (a factor of 4). A disagreement of exactly −1 or 4 is a convention mismatch, not a bug — diagnose before "fixing." See `validate_curvature.py`, where exactly this sign error occurred and was caught.

### E3. 🟡 Rieoptax — Riemannian optimisation in JAX, `arXiv:2210.04840`
Exponential/log maps, Riemannian distance. Useful if geodesic integration is needed; not Fisher-specific.

### E4. 🟢 Known-curvature reference cases (for validation)
From the Fisher–Rao geometry literature:
- **Čencov's theorem**: Fisher–Rao is the *unique* metric (up to scale) invariant under sufficient statistics. **This is the thesis's central justification for choosing this metric over any other — it is not one option among many, it is the canonical one.**
- Categorical simplex under Fisher–Rao ≅ positive orthant of a sphere of radius 2 → **constant curvature +1/4**. Closed-form distance `d(p,q) = 2·arccos(Σᵢ√(pᵢqᵢ))`.
- **Pareto family** ≅ Poincaré half-plane, constant **K = −1**.
- **Dirichlet and beta** families: everywhere **negative** sectional curvature (Hadamard property ⇒ unique Fréchet means).
- Gaussian random field manifolds: curvature computed explicitly (`arXiv:2109.09204`).
- Fisher–Rao as the pullback of the L² metric under the square-root transform — the cleanest route to the constant-curvature result.
- Known hard case: no closed form for mixtures of Gaussians.

### E5. Foundational texts
- Amari, *Information Geometry and Its Applications* — the standard reference; α-connections, dually flat structure.
- Amari & Nagaoka, *Methods of Information Geometry*.
- Nielsen, information-geometry survey papers.
- Scholarpedia, "Fisher-Rao metric" — fast orientation.

---

## 6. Synthesis: what nobody has done

Laying the streams side by side:

| Requirement | Mabrok A1 | FishBack A2 | Manson B1 | King B2 | Zavatone-Veth C1 | **This thesis** |
|---|---|---|---|---|---|---|
| Applied to transformers | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| State-dependent metric `G(h)` | ✅ | ✅ | ❌ constant `UᵀU` | ❌ none | ✅ | ✅ |
| Fisher–Rao specifically | ✅ | ✅ | ❌ | ❌ | ❌ generic pullback | ✅ |
| Christoffel symbols | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Riemann tensor / sectional curvature | ❌ defined only | ❌ | ❌ | ❌ | ✅ Ricci scalar | ✅ |
| Layer-wise curvature profile | proxy only | ❌ | ✅ flat-metric | ✅ Euclidean | ❌ | ✅ |
| Behavioural / causal validation | ❌ | steering | ❌ | ✅ partial | ❌ | ✅ |

**Every column has a gap. No row has all ticks. The final column is the thesis.**

### The gap in one sentence

> The Fisher–Rao pullback metric on transformer hidden states has been **correctly derived** (Mabrok 2026; Wang & Zhao 2026) and layer-wise curvature has been **measured under flat metrics** (Manson 2025; King et al. 2026), but **the genuine intrinsic Riemannian curvature of the state-dependent Fisher–Rao metric — Christoffel symbols → Riemann tensor → sectional and scalar curvature — has never been computed for a transformer, at any scale, at any layer.**

### The empirical conflict that makes this urgent

Two published results point in opposite directions:

- **Mabrok (A1):** proxy curvature ≈ **10⁻⁵**, "broadly stable across layers" → *the manifold is essentially flat.*
- **Manson (B1) and King et al. (B2):** curvature **varies systematically** with semantic content, layer depth, and predictive uncertainty → *geometry is doing work.*

These cannot both be the whole story. The most likely resolution is that **both used the wrong instrument**: Mabrok used an extrinsic proxy in ambient coordinates; Manson and King used metrics with identically zero intrinsic curvature. **Computing the correct quantity adjudicates a live disagreement in the literature.** That framing is far stronger than "we apply method X to domain Y," and it is what makes this a thesis rather than an exercise.

### Three things that make the project feasible rather than aspirational

1. **The metric is closed-form.** `G(h) = Wᵀ Σ_p W` needs no estimation, no sampling, no training — one forward pass and a matrix product (A1, A2).
2. **The dimension is small.** `d` = 768–4096, not billions (E1). Effective rank is 2–17% of that (A2), so the working dimension may be **under 100** — well within reach of full Riemann tensor computation.
3. **The machinery exists and is validated.** C1 computes Ricci scalars of pullback metrics; C3 gives the efficient autodiff route; E2 provides an independent oracle with analytically known answers.

### The one obstacle that must be solved first

It is tempting to argue: *`Σ_p = diag(p) − ppᵀ` annihilates the all-ones direction, therefore `G(h)` is singular, therefore `G⁻¹` does not exist.* **The first clause is true; the conclusion does not follow.** Since

```
null(G) = { v : W v ∈ span{1_N} }
```

an exact null direction exists only if `1_N` lies in `range(W)` — a `d`-dimensional subspace of `ℝᴺ` with `d ≪ N`. Generically it does not, and **`G(h)` is full rank `d`.** Verified numerically in `conditioning_check.py`; forcing `1_N` into `range(W)` recovers a deficiency of exactly 1, confirming the mechanism.

**The real obstacle is conditioning, not nullity.** Because a language-model softmax is sharply peaked, `Σ_p` has few significant eigenvalues and the spectrum of `G` decays fast. Measured on **synthetic** data at `N`=2000, `d`=64 (`conditioning_check.py`, seed 0):

| `p` | formal rank | cond(G) | directions holding 99% of trace |
|---|---|---|---|
| near-uniform | 64/64 | 2.0 | 64/64 (100%) |
| moderately peaked | 64/64 | 4.1 × 10¹ | 61/64 (95%) |
| **sharply peaked (realistic)** | **64/64** | **1.8 × 10⁸** | **4/64 (6%)** |

Formal rank stays full throughout while conditioning degrades by eight orders and the effective direction count collapses to single digits — independently reproducing the mechanism behind A2's reported 2–17% effective dimensionality. `G⁻¹` formally exists and is **numerically useless.**

⚠️ These are synthetic figures whose exact values depend on the seed and assumed sharpness; the order of magnitude is the claim. **Rerun on real GPT-2 activations before quoting anything in the thesis** ([04-stage-plan.md](04-stage-plan.md) task 3.10b).

Two consequences:

1. **float64 is mandatory, not advisable.** `cond(G)` ≈ 10⁸ already costs ~8 of float64's ~16 significant digits; curvature needs *second* derivatives of `G`, roughly doubling the loss. float32 (~7 digits) leaves nothing at all. This is a quantitative argument, and a much sharper one than "the metric is singular."
2. **The cutoff is a scientific decision, not preprocessing.** Where the spectrum is this steep, the retained rank `k` *silently sets the curvature values*. Naive pseudo-inversion is not acceptable, and curvature must always be reported as a function of `k`.

Handling this in a principled, documented, sensitivity-tested way is **the core methodological contribution of the thesis.** See [03-methodology.md](03-methodology.md) §4.

---

## 7. Reading order

**Before writing any code (Stage 0–1):**
1. A1 Mabrok `2603.22301` — full text. The metric derivation is the foundation.
2. A2 FishBack `2605.17231` — full text. The recursion and the computational recipe.
3. C1 Zavatone-Veth `2301.11375` — full text. The curvature machinery.
4. C3 `2105.01583` — the efficient autodiff strategy.
5. B1 Manson `2507.21107` — to state the contrast precisely.
6. Amari, chapters on Fisher metric and α-connections.

**During Stage 1:** A3, B2, D1, D2, E2 docs.
**Deferred to Stage 4:** B3, B4, D3, D4, D5, C4, C5.

---

## 8. Outstanding actions

- [ ] Obtain HTML/LaTeX of A3 (`2605.09887`) — PDF extraction failed. Confirm the rank bound and its proof.
- [ ] Re-read every 🟡 entry at full text before finalising the review chapter.
- [ ] Confirm whether A1's framework **requires** tied embeddings or merely assumes them — this determines the model selection in Stage 4.
- [ ] Check whether C1's index-symmetry simplification of the Ricci scalar applies to the Fisher pullback.
- [ ] **Search for work published after 10 August 2026** — this field is producing relevant papers monthly (2602, 2603, 2604, 2605, 2606 all appeared within six months). Re-run the search before submission; set up arXiv alerts now (see [04-stage-plan.md](04-stage-plan.md) Stage 1).
- [ ] Email Mabrok and Manson. Both are recent single/small-author works; both left the exact next step undone. **Manson's own future-work list names "singularity analysis of embedding space combined with contextual curvature"** — which is this thesis. A short, specific email is low cost and high expected value.
