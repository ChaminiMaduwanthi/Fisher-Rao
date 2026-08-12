# Research Gap, Questions and Contributions

**Project:** Information-Geometric Curvature Analysis of Transformer Representations Using Fisher–Rao Metrics
**Date:** 10 August 2026
**Depends on:** [01-literature-review.md](01-literature-review.md)

---

## 1. The gap

> The Fisher–Rao pullback metric on transformer hidden states has been correctly derived, and layer-wise curvature has been measured under flat metrics. **The genuine intrinsic Riemannian curvature of the state-dependent Fisher–Rao metric on transformer representations has never been computed.**

Three groups each hold two of the three necessary pieces:

```
        Fisher–Rao metric        Intrinsic curvature        Transformers
        on hidden states         (Riemann tensor)
────────────────────────────────────────────────────────────────────────────
Mabrok 2026        ✅                    ❌ (proxies)              ✅
FishBack 2026      ✅                    ❌ (spectra only)         ✅
Manson 2025        ❌ (const. UᵀU)       ❌ (curve, flat space)    ✅
King et al. 2026   ❌ (Euclidean)        ❌ (curve, flat space)    ✅
Zavatone-Veth      ❌ (generic)          ✅ (Ricci scalar)         ❌ (CNNs)
────────────────────────────────────────────────────────────────────────────
THIS THESIS        ✅                    ✅                        ✅
```

### Why the gap has stayed open

Not because it is uninteresting — because of one specific technical obstacle. It is worth stating precisely, because the loose version of the argument is wrong.

**The loose version.** *`Σ_p = diag(p) − ppᵀ` annihilates the all-ones direction, so `G(h) = Wᵀ Σ_p W` is singular, so `G⁻¹` does not exist.*

**Why that does not follow.** `null(G) = { v : Wv ∈ span{1_N} }`, so an exact null direction requires `1_N ∈ range(W)` — a `d`-dimensional subspace of `ℝᴺ` with `d ≪ N`. Generically it is not, and **`G(h)` is full rank `d`.** (Verified in `conditioning_check.py`; forcing `1_N` into `range(W)` recovers a deficiency of exactly 1.)

**The correct version — the obstacle is conditioning.** A language-model softmax is sharply peaked, so `Σ_p` has few significant eigenvalues and the spectrum of `G` decays steeply. Measured on synthetic data at `N`=2000, `d`=64: near-uniform `p` gives `cond(G)` ≈ 2 with all 64 directions active; **realistic sharply-peaked `p` gives `cond(G)` ≈ 10⁸ with only ~6% of directions holding 99% of the trace** — independently reproducing the mechanism behind the 2–17% effective dimensionality reported by FishBack. `G⁻¹` formally exists and is numerically useless. (Synthetic; order of magnitude is the claim. Rerun on real activations before quoting.)

This gives a sharper quantitative argument than "the metric is singular": `cond(G)` ≈ 10⁸ costs ~8 of float64's ~16 digits, and curvature requires *second* derivatives of `G`, roughly doubling the loss. **float64 is mandatory; float32 leaves nothing.** And where the spectrum is this steep, the retained rank `k` *silently determines the curvature values*.

Every prior author stopped at this wall:
- Mabrok calls true Riemann curvature "computationally intractable" and retreats to local-PCA proxies.
- FishBack applies adaptive Tikhonov regularisation but only to get a usable *metric*, never differentiating it.
- Zavatone-Veth et al. report the Ricci scalar is "computationally expensive and numerically challenging" even for small networks, and mostly retreat to volume elements.

**Getting past this wall in a principled, sensitivity-tested way is the thesis's core methodological contribution.** It is not a preprocessing step to be dispatched in a footnote.

---

## 2. The live disagreement this resolves

Two published results are in tension:

| Source | Instrument | Result |
|---|---|---|
| Mabrok 2026 | local-PCA residual variance, ambient coordinates | curvature ≈ **10⁻⁵**, flat, stable across layers |
| Manson 2025 | Frenet curvature of curve, constant metric `UᵀU` | curvature **varies** with semantic concern, propagates across layers |
| King et al. 2026 | Euclidean angles between difference vectors | curvature **correlates** with next-token entropy (r ≈ 0.15) |

**Both instruments are wrong for the question asked.** Mabrok's is extrinsic and embedding-dependent; Manson's and King's metrics have *identically zero* intrinsic curvature, so they can only measure path bending, never space bending.

There is no theorem making the extrinsic PCA residual a bound on intrinsic Fisher curvature. A point cloud can look flat in ambient coordinates while `G(h)` varies sharply with `h` — which is exactly what large intrinsic curvature means.

> **The thesis adjudicates a live empirical disagreement with the correct instrument.** This framing is materially stronger than "apply method X to domain Y," and it is what elevates the work from exercise to contribution.

---

## 3. Research questions

Ordered so that each is answerable independently. **RQ1–RQ2 alone constitute a defensible thesis**; RQ3–RQ5 are upside.

### RQ1 (Methodological — the core) — *Can it be computed at all?*
**Can the intrinsic Riemannian curvature of the Fisher–Rao pullback metric on transformer hidden states be computed tractably and verifiably, given that the metric is severely ill-conditioned?**

Sub-questions:
- 1a. What is the principled treatment of the steep spectrum — restriction to the effective eigen-subspace, quotient manifold, or regularised inverse? How sensitive are curvature values to that choice?
- 1b. Does the autodiff-contracted route (`Γᵏ_ij vⁱ vʲ` without materialising full symbol arrays) make the computation feasible at `d` = 768?
- 1c. Can the implementation be validated against families with **analytically known** curvature?

**Deliverable:** a validated, open-source curvature computation pipeline. *Even if every empirical result is null, this is a contribution.*

### RQ2 (Empirical — the headline) — *What is the answer?*
**How does intrinsic Fisher–Rao curvature vary across layers, models, and inputs — and does it agree with the published proxy results?**

- 2a. What is the layer-wise sectional/scalar curvature profile? Does it show the hourglass structure known from intrinsic dimension (Valeriani et al. 2023; Mabrok 2026)?
- 2b. **Is Mabrok's "essentially flat, ~10⁻⁵" conclusion an artefact of the PCA proxy?** *(Directly falsifiable.)*
- 2c. Is the profile universal across architectures (GPT-2 / Pythia / Llama / Gemma), or architecture-specific?
- 2d. Sign of curvature: positive (sphere-like, as the full simplex would give at +1/4), negative (hyperbolic, as Dirichlet/beta families give), or mixed by direction?

### RQ3 (Behavioural — the value proposition) — *Does it matter?*
**Does Fisher–Rao curvature predict model behaviour better than the flat-metric baselines?**

- 3a. **Does it beat r ≈ 0.15 for the curvature–entropy correlation reported by King et al. 2026?** *(A pre-registerable hypothesis with a published baseline number to beat — the cleanest possible empirical claim.)*
- 3b. Does curvature localise the layer at which lexical ambiguity resolves? (Test case: polysemous tokens in contrasting contexts.)
- 3c. Do curvature signatures differ between factual and hallucinated generations?

### RQ4 (Causal) — *Is it cause or correlate?*
**Do interventions along high- versus low-curvature directions affect output differently?**

Extends King et al.'s finding that only trajectory-aligned perturbations modulate entropy, and answers Manson's stated limitation of "no behavioural validation." Method: perturb `h` by matched Fisher-norm steps along the top and bottom sectional-curvature eigendirections; compare KL divergence of output distributions.

### RQ5 (Optional — only if Stages 0–5 finish early) — *Does it transfer?*
Does curvature structure change under fine-tuning, and can low-curvature layers be pruned with less damage than high-curvature ones?

---

## 4. Claimed contributions

| # | Contribution | Type | Risk |
|---|---|---|---|
| C1 | First computation of intrinsic Riemannian curvature (Christoffel → Riemann → sectional/scalar) of the Fisher–Rao pullback metric on transformer representations | Methodological | Low |
| C2 | A principled, sensitivity-tested treatment of the severely ill-conditioned Fisher pullback, enabling curvature computation where prior work declared it intractable — including the correction that the metric is generically full-rank, so conditioning rather than nullity is the true obstacle | Methodological | Low |
| C3 | A validation ladder against analytically-known Fisher–Rao curvature (categorical simplex +1/4; gamma and beta/Dirichlet negative), cross-checked against Geomstats | Methodological | **Very low** |
| C4 | Layer-wise intrinsic curvature profiles across multiple architectures, and adjudication of the Mabrok-vs-Manson/King disagreement | Empirical | Medium |
| C5 | Quantified comparison of Fisher–Rao vs Euclidean vs `UᵀU` metrics as predictors of next-token entropy, against a published baseline | Empirical | Medium |
| C6 | Open-source library + reproducible artefacts | Engineering | Very low |

**Note the risk column.** C1–C3 and C6 are *within the student's control*: they depend on correct implementation, not on nature cooperating. C4–C5 depend on what the data shows. This is deliberate — see §6.

---

## 5. Scope boundaries

**In scope:** decoder-only transformers, 124M–3B; the final-token next-token distribution as the statistical model; the hidden-state (residual stream) space as the manifold; layer-wise and token-wise analysis; English text.

**Out of scope** (state these explicitly at the proposal defence — unbounded scope is the most common reason a geometry thesis stalls):
- Fisher over **parameters** θ — the natural-gradient/K-FAC object. *Different Fisher, different thesis.* This distinction must be stated in the introduction; it is the single most likely examiner misunderstanding.
- Encoder-only models (BERT) — no next-token distribution, so no natural statistical model. Possible extension via masked-token distributions; not attempted.
- Models above ~3B — Jacobian cost is prohibitive (FishBack already flags this at GPT-2 scale).
- Training-time geometry evolution — requires checkpoint series; deferred to RQ5.
- α-connections beyond α = 0 (Levi-Civita). Amari's dually-flat ±1 structure is a natural extension; out of scope for the main thesis.
- Multilingual and multimodal representations.

---

## 6. Honest risk assessment

**The single biggest scientific risk:** the answer to RQ2 may be *"curvature is small and uniform"* — i.e. Mabrok was right after all. Prior evidence gives this real probability: the PCA proxy says 10⁻⁵, King et al.'s effects were "small," and effective dimensionality is only 2–17% of ambient, which is consistent with a nearly-flat low-dimensional structure.

**Why the thesis survives this outcome:**

1. **A rigorous null is publishable when it overturns an assumption.** "Intrinsic Fisher–Rao curvature of transformer representations is near-zero, established with the correct instrument for the first time, therefore flat-metric methods are justified and the field can stop worrying" — that is a real, citable result. It *licenses* Manson's and King's simpler methods rather than merely criticising them.
2. **C1–C3 and C6 do not depend on the empirical outcome.** The pipeline, the singularity treatment, and the validation ladder are contributions regardless of what the numbers say.
3. **A guaranteed fallback exists.** Even under flat curvature, the *metric* `G(h)` is emphatically non-trivial: >97% deviation from Euclidean in relative spectral norm (FishBack 2026). Anisotropy, condition number, effective rank, volume element, and geodesic deviation are all substantive quantities that survive. Zavatone-Veth et al. found the **volume element** more informative and far more numerically stable than the Ricci scalar — the same is likely true here. **Instrument the volume element from Stage 2 onward, not as a rescue measure but as a parallel primary quantity.**

**Design implication:** structure the thesis so Chapters 3–4 (method + validation) stand alone, and Chapters 5–6 (empirical) report whatever is found. **Do not stake the thesis on curvature being large.**

**The competition risk is real and should be stated plainly.** Relevant papers appeared in Feb, Mar, Apr, May and Jun 2026 — roughly monthly. Mabrok's paper already contains the metric derivation and *defines* the Riemann tensor without computing it; someone will compute it. Mitigations: (i) set up arXiv alerts in Stage 1, not later; (ii) target a workshop paper on C1–C3 by month 6 to establish priority; (iii) if scooped mid-project, pivot to RQ3/RQ4 — behavioural and causal validation — where the field is emptiest and Manson explicitly flags the absence of behavioural validation.

---

## 7. Working title options

1. *Information-Geometric Curvature Analysis of Transformer Representations Using Fisher–Rao Metrics* (as assigned — keep for administrative purposes)
2. *Is the Semantic Manifold Flat? Intrinsic Fisher–Rao Curvature of Transformer Representations* — better for a paper; foregrounds the falsifiable question
3. *Beyond Flat Metrics: Intrinsic Curvature of the Fisher–Rao Pullback in Language Models* — foregrounds the methodological contrast
