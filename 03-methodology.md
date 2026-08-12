# Methodology

**Project:** Information-Geometric Curvature Analysis of Transformer Representations Using Fisher–Rao Metrics
**Date:** 10 August 2026
**Depends on:** [02-research-gap.md](02-research-gap.md)

---

## 1. The statistical model

A decoder-only transformer with hidden dimension `d` and vocabulary size `N` defines, for every hidden state `h ∈ ℝᵈ`, a categorical distribution over next tokens:

```
ℓ(h) = W h + b            logits,        W ∈ ℝ^{N×d}    (unembedding)
p(h) = softmax(ℓ(h))      distribution,  p ∈ Δ^{N−1}    (probability simplex)
```

So the map `h ↦ p(h)` sends hidden-state space into the simplex. Hidden states are not abstract vectors — **each one is a point in a statistical manifold.** That is what licenses information geometry here, and it is the sentence to open the methodology chapter with.

**Why Fisher–Rao and not any other metric.** By **Čencov's theorem**, the Fisher–Rao metric is the *unique* Riemannian metric on the simplex (up to scale) invariant under sufficient statistics — i.e. under reparameterisation and under merging/splitting of outcomes. Every alternative choice, including the Euclidean metric of King et al. and the `UᵀU` metric of Manson, breaks that invariance and therefore depends on arbitrary coordinate conventions. **This is a uniqueness argument, not a preference. State it early and precisely; it is the strongest single justification in the thesis.**

---

## 2. The metric

The Fisher information metric on the simplex, pulled back to hidden-state space through the softmax + unembedding map:

```
G(h) = Wᵀ Σ_p W ,        Σ_p = diag(p) − p pᵀ
```

`G(h) ∈ ℝ^{d×d}`, symmetric, positive **semi**-definite. Independently derived by Mabrok (2026, Prop. 5.2) and Wang & Zhao (2026); the same object appears in Zaher et al. (2026).

> 🔴 **The final norm belongs in the pullback, and this is not a detail.** The predictive map is `h → norm(h) → U → softmax`, so `J = U · ∂norm/∂h` and the norm's Jacobian is part of the metric. For RMSNorm with gain `g` and `r = √(mean(h²) + eps)`:
>
> ```
> A = diag(g) · (1/r) · ( I − h hᵀ / (r² d) )        G(h) = Aᵀ ( Ũᵀ Σ_p Ũ ) A
> ```
>
> Measured on SmolLM2-135M ([05-stage0-log.md](05-stage0-log.md) §4.1), omitting `A` changes the metric in two distinct ways, and they must not be conflated: an **overall `1/r²` scale factor** (`‖G_without‖/‖G_with‖` = 107.74 against `r²` = 107.72 at layer 20) plus a **rank-1 projector correction of only 2.0%** in Frobenius norm. Both matter — sectional curvature is not scale-invariant (`K → K/c` under `g → c·g`), and the small projector term is what produces the exact radial nullity. Omitting `A` is an *error*, not an approximation.
>
> **Mabrok's published `G(h) = Wᵀ Σ_p W` and Manson's `G = UᵀU` both appear to omit it.** Verify this against their full texts — if confirmed it is a second, independent methodological gap, and a plausible partial explanation for proxy curvature coming out at 10⁻⁵.

**Properties to verify numerically as unit tests, before trusting anything downstream:**

| Property | Test |
|---|---|
| Symmetry | `‖G − Gᵀ‖ < ε` |
| Positive semi-definiteness | `λ_min(G) ≥ −ε` |
| `Σ_p 1 = 0` | the all-ones direction is annihilated **in `ℝᴺ`** — note this does *not* imply `G` is singular (§4) |
| Generic full rank | `rank G = d` for random `W`; and deficiency exactly 1 when `1_N` is forced into `range(W)` |
| Conditioning profile | `cond(G)` and 99%-of-trace rank as functions of softmax sharpness — the quantity that actually constrains the project |
| Reparameterisation invariance | apply a smooth reparameterisation; check induced distance is unchanged |
| Agreement with finite-difference KL | `KL(p(h) ‖ p(h+εv)) ≈ ½ ε² vᵀG(h)v + O(ε³)` |

> The last test is the most valuable one in the whole project. The Fisher metric **is** the Hessian of KL divergence at coincidence — so it can be checked against a quantity computed by a completely independent code path (two forward passes and a KL). **If this test does not pass to third order, nothing downstream is trustworthy.** Write it first.
>
> **✅ Already implemented and passing:** `gate_a_kl_test.py`. Relative error falls as `O(ε)` — the predicted third-order remainder — from `2.25e-3` at `ε=1e-1` down to `1.90e-6` at `ε=1e-4`.
>
> ⚠️ **But it then gets worse again, and this will waste your time if you don't expect it.** Below `ε ≈ 1e-5` the error *rises*: `9.5e-5` at `1e-5`, and **18% off at `ε=1e-7`**. That is not a problem with `G`. KL between two nearly identical distributions is a difference of nearly equal logs, and at `ε=1e-7` the true KL is `~1e-16` — float64 epsilon. **A tiny `ε` chosen "for accuracy" makes a correct implementation fail its own gate.** Sweep `ε` and report the `O(ε)` plateau plot as the evidence, never a single number. The same discipline applies to every finite difference in the project (§6.3).

**Interpretation for the thesis text:** `vᵀ G(h) v` is the rate at which the model's *predictions* change when the hidden state moves along `v`. Directions with large `vᵀGv` are semantically loaded; directions in the null space change the internal state without changing the output at all. Fisher–Rao geometry measures distance in *predictive consequence*, not in coordinates.

**Layer-wise propagation** (FishBack Thm 3.3) — use this rather than recomputing from scratch at each layer:

```
G^(ℓ) = A_ℓᵀ G^(ℓ+1) A_ℓ ,       A_ℓ = I + DF_ℓ
```

with `DF_ℓ` the Jacobian of residual block ℓ. A congruence transformation, so PSD and the rank bound are preserved down the stack.

---

## 3. Curvature: what exactly is being computed

Given a Riemannian metric `g_ij(x)` on a `d`-dimensional space:

```
Christoffel symbols   Γᵏ_ij = ½ g^{kl}( ∂_i g_jl + ∂_j g_il − ∂_l g_ij )

Riemann tensor        R^ρ_σμν = ∂_μ Γᵖ_νσ − ∂_ν Γᵖ_μσ + Γᵖ_μλ Γ^λ_νσ − Γᵖ_νλ Γ^λ_μσ

Sectional curvature   K(u,v) = ⟨R(u,v)u, v⟩ / (⟨u,u⟩⟨v,v⟩ − ⟨u,v⟩²)

Ricci tensor          R_ij = R^k_ikj              Scalar curvature  R = g^{ij} R_ij

Volume element        dV = √(det g) dᵈx
```

**Three separate quantities to keep distinct in the write-up** — conflating them is the most common conceptual error in this literature:

1. **Intrinsic curvature of the space** — Riemann tensor of `G(h)`. Requires `G` to *vary with* `h`. **← the thesis contribution.**
2. **Extrinsic curvature of the manifold** in its ambient embedding — second fundamental form ‖II‖. *← what Mabrok's proxies measure.*
3. **Frenet curvature of a trajectory** through the space — how a path bends. *← what Manson and King et al. measure.*

A constant metric gives **zero** intrinsic curvature but can still give non-zero (2) and (3). This is precisely why `G = UᵀU` cannot answer the thesis question, and it is worth one clear paragraph plus a diagram in the thesis.

---

## 4. The central obstacle: `G(h)` is catastrophically ill-conditioned

**State this correctly — the intuitive version of the argument is wrong.**

`Σ_p 1 = 0` always. It is tempting to conclude that `G(h) = Wᵀ Σ_p W` is therefore singular. But

```
null(G) = { v : W v ∈ span{1_N} }
```

so an exact null direction exists only if `1_N ∈ range(W)` — a `d`-dimensional subspace of `ℝᴺ` with `d ≪ N`. Generically it is not, and **`G(h)` has full rank `d`.** Run `conditioning_check.py`: case [4] forces `1_N` into `range(W)` and recovers a deficiency of exactly 1, confirming the mechanism, while cases [1]–[3] are all full rank.

**The obstacle is conditioning.** Because the softmax is sharply peaked, `Σ_p` has few significant eigenvalues and the spectrum of `G` decays steeply. Synthetic measurement, `N`=2000, `d`=64, seed 0 (`conditioning_check.py`):

| `p` | formal rank | `cond(G)` | dirs holding 99% of trace | participation ratio |
|---|---|---|---|---|
| near-uniform | 64/64 | 2.0 | 64/64 (100%) | 61.9 |
| moderately peaked | 64/64 | 4.1 × 10¹ | 61/64 (95%) | 27.9 |
| **sharply peaked (realistic)** | **64/64** | **1.8 × 10⁸** | **4/64 (6%)** | **1.78** |

Formal rank stays full while conditioning degrades by eight orders — independently reproducing the mechanism behind FishBack's 2–17% effective dimensionality. `g^{kl}` formally exists and is **numerically useless.**

⚠️ Synthetic figures; exact values are seed- and sharpness-dependent, and the order of magnitude is the claim. Rerun on real activations (task 3.10b) before quoting in the thesis.

> **Two consequences, both load-bearing.**
>
> **(i) float64 is mandatory, not advisable.** `cond(G)` ≈ 10⁸ already costs ~8 of float64's ~16 significant digits. Curvature needs *second* derivatives of `G`, roughly doubling the loss — leaving ~0 digits in float32. Quote this number in the thesis; it is a far sharper argument than "the metric is singular," and it justifies the numerical-hygiene requirements in §6.3 quantitatively.
>
> **(ii) The cutoff is a scientific decision.** With a participation ratio near 1, the retained rank `k` *silently sets* the curvature values. An undocumented cutoff makes the whole result unfalsifiable.

Three principled options, to be **implemented and compared, not chosen a priori**:

> 🔴 **Revised after Stage 0 (see [05-stage0-log.md](05-stage0-log.md) §6.2).** On *real* activations the situation is worse than the synthetic study suggested: **λ_min underflows to exactly 0.0 at 22 of 31 layers**, so `cond(G) = ∞`, and the working-precision rank falls as low as 288/576. Real softmaxes are far more peaked than the synthetic logit scale used (measured entropies 0.0–1.8 nats).
>
> **Therefore float64 is necessary but *not sufficient*, and Option A is not one of three alternatives — it is the only viable primary route.** Options B and C remain as cross-checks and sensitivity instruments only.
>
> 🟢 **The compensating good news:** conditioning *within* the retained subspace, `cond_eff = λ_max/λ_{k_eff}`, is only **10¹–10³**. Once restricted, the metric is well conditioned and leaves ample float64 headroom for second derivatives. And with `k_eff ≤ 86` measured, the **full Riemann tensor on the restricted subspace is affordable** (86⁴ ≈ 5.5 × 10⁷), so the random-2-plane sampling in §6.1 may prove unnecessary.

> 🔴 **Revised again after the Stage 3 measurement ([06-stage3-log.md](06-stage3-log.md) §4.3). Two rules replace what this section originally said.**
>
> **Rule 1 — select `k` by a conditioning ceiling, not a trace fraction.** Choosing `k` = `k_eff` (99% of trace) admits eigendirections small enough to destroy second derivatives: `dR/R_ref` then swings from −8% to +583% and the layer ordering scrambles (Spearman ρ = 0.20 between `k`=4 and `k`=7). The diagnostic that tracked the failure was `cond_eff = λ_max/λ_k` of the *retained* subspace, which jumped an order of magnitude (to ~5×10²) exactly where instability set in. **Take the largest `k` with `cond_eff ≤ ~10²`** (`curvature.select_k`). Measured on SmolLM2-135M this gives `k` = 3–5 for layers 1–25 and 9–13 for the last layers, always retaining ≥ 92% of the trace.
>
> **Rule 2 — the volume element is the primary layer-wise quantity; Riemann-derived quantities are secondary.** Verified, not assumed: per-dimension log-volume preserves the layer ordering across `k` with Spearman ρ = **0.93–0.98** for `k`=3–7 and **0.73** even between `k`=3 and `k`=10, against **0.20** for the curvature deviation. Use `volume_element(..., per_dim=True)` whenever comparing across different `k`, since the raw sum grows with `k` by construction. Same conclusion Zavatone-Veth et al. reached for CNNs.

> 🔴 **Rule 2b — measure the volume ON THE SPHERE, not in ambient coordinates.** `λᵢ ∝ 1/r²`, so `log_vol_per_dim` carries a `−log r` term *by construction*, and the residual norm grows enormously with depth. Regressing on `log r` gives slope **−0.937** against the mandated **−1**: an uncorrected layer profile is essentially all scale. Since the semantic manifold is the sphere of directions, the metric there is `r²·G_ambient`, adding exactly `+log r` per dimension. `volume_element(..., on_sphere=True)` is the default. **This is not cosmetic — the uncorrected profile showed monotone contraction with a late recovery, the corrected one is U-shaped with a minimum at layer 20.** Different shape, different conclusion ([07-stage4-log.md](07-stage4-log.md) §2.1–2.2).
>
> Absolute sectional curvature remains robust (`K` ≈ +1/4, stable across `k`=3–7); it is the *deviation* from it that is not identifiable.

### Option A — Effective subspace restriction *(the primary route — see revisions above)*
Eigendecompose `G(h) = V Λ Vᵀ`. Keep the top `k` eigendirections capturing a fixed fraction (e.g. 99%) of the trace. Restrict to that `k`-dimensional subspace and compute curvature of the restricted metric.

- ✅ Directly justified by the empirical 2–17% effective dimensionality; likely `k` < 100, making the **full Riemann tensor** affordable.
- ⚠️ The subspace **rotates with `h`**, so differentiating the restricted metric requires care — a moving frame, not a fixed basis. This subtlety is easy to get silently wrong and is worth its own section in the thesis. Handle it by differentiating in ambient coordinates and projecting, or by using a smoothly-varying frame; verify by checking that results are invariant to an arbitrary rotation of the retained basis.

### Option B — Quotient manifold *(co-primary; the structurally correct route)*
Follow Zavatone-Veth et al.: treat it as a singular semi-Riemannian manifold, work on `ℳ/∼`, and use the product of **non-zero** eigenvalues for the volume element.

> 🟢 **Upgraded after Stage 0 ([05-stage0-log.md](05-stage0-log.md) §4.2, §6.6).** An earlier draft downgraded this option for lack of an exact null distribution. **That was wrong.** There is one, and it is as well-behaved as it could possibly be:
>
> **RMSNorm is scale-invariant.** Since `mean(h²) = |h|²/d`, the bracket in `A` is `I − hĥᵀ|ĥ` — a projector — so **`h` itself is an exact null direction of `G(h)`.** Measured: `KL(p(h)‖p(2h)) = 1.6 × 10⁻¹⁶` versus `1.15` for a random perturbation of the same magnitude.
>
> - The null distribution is **1-dimensional** ⟹ Frobenius integrability is **automatic**. The check in task 3.9 is trivially satisfied.
> - Its integral curves are **radial rays** ⟹ the quotient is the **sphere of directions**.
>
> **The predictive map factors through the unit sphere, so the semantic manifold is a `(d−1)`-dimensional space of directions, not a `d`-dimensional vector space.** Computing curvature on the quotient is not a workaround — it is working on the correct manifold. This also supplies an architectural mechanism for why cosine similarity carries geometric meaning (Modell et al. 2025) and why Euclidean coordinates are the wrong frame (anisotropy literature) — a theoretical contribution beyond the curvature computation itself.

**A and B compose, and should be used together:** quotient by the radial direction first (exact, structural), then restrict to the top-`k_eff` subspace (numerical, sensitivity-tested).

### Option C — Tikhonov regularisation
`G_α = G + α I` with `α = c · λ_median(G)` (FishBack's scheme).

- ✅ Trivial to implement; makes everything invertible; good as a cross-check.
- ⚠️ Introduces a bias that **does not vanish** — the regulariser adds an artificial flat direction, which biases curvature *toward zero*. Never report Option C values as primary.

**Mandatory sensitivity analysis.** Report curvature as a function of `k` (Option A) and `c` (Option C). **If conclusions depend on the cutoff, the honest finding is that curvature is not identifiable from this metric — and reporting that clearly is itself a contribution.** Decide the reporting format *before* looking at the results.

---

## 5. Validation ladder — build this before touching a transformer

> **This is the most important section of the methodology and the part most likely to be skipped under time pressure. Do not skip it.** Curvature code is notoriously easy to get wrong and the errors are silent: sign errors, index-ordering errors, and factor-of-two errors all produce plausible-looking numbers. Without an oracle there is no way to know. A thesis whose curvature code is unvalidated is not defensible, and the failure surfaces at the viva.

Climb these rungs in order. Each has a **known correct answer**.

| Rung | Test case | Known answer | Purpose |
|---|---|---|---|
| 0 | Flat ℝ² Euclidean | `R = 0` exactly | catches gross implementation errors |
| 1 | 2-sphere radius r | `K = 1/r²` | validates sign and normalisation |
| 2 | Poincaré half-plane | `K = −1` | validates negative curvature |
| 3 | **Categorical simplex, full Fisher–Rao** | **`K = +1/4` constant** (isometric to the positive orthant of a radius-2 sphere) | ✅ **the key rung — same metric family as the thesis, analytically known** |
| 4 | **Gamma family** | negative; **cross-checked against Geomstats** | independent Fisher–Rao case |
| 5 | **Beta / Dirichlet families** | everywhere negative sectional curvature | multi-parameter Fisher–Rao |
| 6 | Univariate Gaussian family | `K = −1/2` | cross-check against Geomstats |
| 7 | Synthetic linear `W`, tiny `N`, `d` | closed form derivable by hand | isolates the pullback step specifically |
| 8 | Known 1-D "ripple" manifolds (years, number lines — `arXiv:2602.15029`) | qualitatively known extrinsic curvature | first contact with real LM data, still partly checkable |

**Cross-check rungs 2–6 against Geomstats' `information_geometry` module** (Le Brigant et al., *ACM TOMS* 2023 — independently implemented and peer-reviewed). Two independent implementations agreeing is the standard of evidence to aim for. Run in `validate_geomstats.py`; **it succeeds on three families and fails on two, and the failures are diagnosed rather than patched** — see §5.2 below.

### 5.1 Status: rungs 2, 3, 4, 5, 6 and 7 pass; cross-checked against geomstats

`validate_curvature.py` in this directory is a working reference implementation — explicit Christoffel symbols, explicit `d⁴` Riemann tensor, central finite differences. Verified 10 August 2026:

```
RUNG 3  categorical simplex (n=4)   K = +0.250001   (expect +1/4)  ✅
RUNG 6  univariate Gaussian         K = -0.500000   (expect −1/2)  ✅
RUNG 2  Poincaré half-plane         K = -1.000000   (expect −1)    ✅
RUNG 7  synthetic pullback, d=N-1   K = +0.25       (expect +1/4)  ✅  worst 9.7e-12
        + reparameterisation invariance                            ✅  worst 1.9e-11
        + scalar R = k(k-1)/4 through the pullback                 ✅  worst 4.9e-12
        + negative control (broken assembly) CAUGHT at 0.263       ✅
RUNG 4  gamma family        K = -0.4639035  vs geomstats  2.5e-14   ✅
RUNG 5  beta family         K = -0.4567948  vs geomstats  3.2e-14   ✅
```

### 5.2 The Geomstats cross-check: 3 of 5 families agree, 2 disagree

`validate_geomstats.py` (geomstats 2.8.0, `GEOMSTATS_BACKEND=pytorch`) runs both
implementations on **identical inputs**:

| family | this work | geomstats | \|diff\| |
|---|---|---|---|
| rung 6 univariate normal | −0.50000000 | −0.50000000 | **1.1e−16** |
| **rung 4 gamma** | −0.46390348 | −0.46390348 | **2.5e−14** |
| **rung 5 beta** | −0.45679484 | −0.45679484 | **3.2e−14** |
| rung 2 Poincaré half-plane | −1.00000000 | +0.18392434 (point-varying) | ❌ |
| rung 3 categorical simplex | +0.25000000 | +2.00000000 | ❌ |

Three families agree at machine precision across eight comparisons. On the two
that disagree, the diagnostic was to compare the **metrics** rather than the
curvatures — a disagreement arising from a parameterisation convention is
harmless, one arising from a curvature routine is not:

> **In both failing cases the metrics agree exactly** (Poincaré 1.5625 both,
> to 1e−12; categorical inner products to 7e−15) **while the curvatures do
> not.** Neither disagreement is a convention; both localise to a curvature
> routine.

Which side to doubt, on the evidence available: geomstats' Poincaré curvature is
**not constant across base points**, which is impossible for hyperbolic space;
and this work's `+1/4` is corroborated by three independent routes (the analytic
radius-2 sphere isometry, rung 3's slow reference implementation, and rung 7's
entirely separate code path). The corroboration is asymmetric. The discrepancy
is nonetheless **reported, not resolved in this work's favour by assumption**.

### 5.3 Rung 7

**Rung 7 (`validate_pullback.py`) closes the last indirect link.** Until it was run, the
pullback assembly `G = J^T Sigma_p J` was validated only through Gate A and the
null-space test -- neither of which would catch a transpose, an index swap or a
dropped outer-product term. The construction: with `d = N-1` and `U`'s columns
plus `1_N` spanning `R^N`, the map `h -> softmax(Uh)` is a diffeomorphism onto
the simplex interior, so the pullback metric is the simplex metric in other
coordinates and `K = +1/4` exactly.

It is deliberately the slowest possible implementation and does not scale — that is fine. Its only job is to be an oracle you can trust while building the fast version. **Keep it in the test suite permanently** and check every optimisation against it.

### 5.4 ⚠️ The sign-convention trap — this already bit once

The first run of that script returned **`K = −0.250000` for the simplex**: correct magnitude, wrong sign. The cause was contracting the lowered Riemann tensor as `R(u,v,u,v)` instead of `R(u,v,v,u)`. Nothing warned about it — the number simply looked plausible, and constant across all sample points, which made it look *more* trustworthy rather than less.

**This is the single most likely silent bug in the project.** Fix the convention once, against a known answer, and never change it. The convention used in `validate_curvature.py`:

```
R^ρ_σij  such that  R(e_i, e_j) e_σ = R^ρ_σij e_ρ
R_μσij   = g_μρ R^ρ_σij = g( R(e_i,e_j) e_σ , e_μ )
K(u,v)   = R(u,v,v,u) / (|u|²|v|² − ⟨u,v⟩²)  =  R_μσij u^μ v^σ u^i v^j / (…)
```

Two independent convention axes cause disagreements, and both look like bugs:
- **Index order of the lowered tensor** → flips the **sign**.
- **Simplex normalisation** — radius-2 sphere (`K = +1/4`, used here) versus unit sphere with the metric scaled by 4 (`K = +1`) → a factor of **4**.

A disagreement with a library of exactly `−1×` or `4×` is a convention mismatch, not an error. **Diagnose before "fixing."**

**Rung 3 doubles as a sanity anchor for the real experiments:** if the transformer's representations filled the simplex, curvature would be exactly +1/4. Deviation from +1/4 measures how far the learned submanifold departs from the ambient statistical manifold — which is a meaningful, interpretable baseline, and a good number to put in an early figure.

---

## 6. Computational strategy

### 6.0 Three implementation traps, all found the hard way in Stage 3

These are not style notes. Each one produced either wrong numbers or a crash, and the first produced *plausible* wrong numbers.

**(a) `torch.autograd.functional.jacobian` cannot be nested.** It defaults to `create_graph=False`, so a nested call is detached from the graph and the outer derivative silently loses the `∂(∂G)` term. Symptom: the **simplex rung passed** (the missing term happens to vanish there) while the Gaussian and Poincaré rungs returned **exactly −2× the true curvature**. Had the ladder contained only the simplex, this would have shipped. **Use `torch.func.jacrev`**, which composes correctly for higher derivatives — after switching, all three rungs pass at machine precision (8×10⁻¹⁵, 4×10⁻¹⁶, 8×10⁻¹⁶).

> This is the second time a sign/factor error survived a plausible-looking result in this project. It is the argument for **multiple** rungs with **different** known answers, not one.

**(b) Top-k truncation must be frozen before differentiating.** If `G(h)` re-selects its own top-k at each evaluation, the retained token set changes discretely as `h` moves, so `G` is only *piecewise* smooth — it jumps wherever top-k membership changes. Curvature needs second derivatives, so this is not survivable. **Freeze the index set at the base point** and hold it fixed across the whole derivative computation (`metrics.topk_indices`). The approximation is then centred on the base point, which is what a local geometric quantity wants anyway.

**(c) Project into the frame *before* forming the metric.** Building the full `d×d` `G(h)` and then projecting to `k×k` costs `O(N d²)` and forces double autodiff to carry a `(d,d)` intermediate — it attempted a **20.6 GB** allocation and died. Instead collapse to `k` dimensions first:

```
M   = U_idx · (A · frame)                  (n_tok, k)
G_k = Mᵀ diag(p) M − (Mᵀp)(Mᵀp)ᵀ          (k, k)
```

`A · frame` is cheap in closed form because `A` is a scaled projector: `A·frame = diag(g)(frame − ĥ(ĥᵀframe))/r`. The largest intermediate becomes `(n_tok, k)`. Measured cost after this fix: **~9 s per Riemann tensor at `k`=6 on CPU** — slow but entirely workable.

### 6.1 Avoid materialising full tensors

The Riemann tensor has `d⁴` components — at `d = 768` that is 3.5 × 10¹¹ entries. **Never form it.**

Instead (following `arXiv:2105.01583`):
- Only the **contracted** forms are ever needed: `Γᵏ_ij vⁱ vʲ` for geodesics, `⟨R(u,v)u, v⟩` for sectional curvature in a specific plane.
- Compute directional derivatives of `G` by **JVP/VJP**, never a full `∂G` array.
- Sample sectional curvature over random 2-planes rather than computing the full tensor; report the distribution. This is honest, cheap, and statistically adequate for the research questions.
- After Option A restriction, `k` may be < 100, at which point the full Riemann tensor on the restricted subspace becomes affordable (`k⁴` = 10⁸ at `k`=100 — borderline; 10⁶ at `k`=32 — easy). **Report `k` explicitly with every curvature number.**
- Note: computing Christoffel symbols explicitly and *then* the Ricci scalar is reported to be significantly less efficient than contracting first. Benchmark both once; then commit.

### 6.2 Vocabulary cost

Forming `Σ_p` naively costs `O(N²)`; forming `G` costs `O(N d²)`. With `N` ≈ 50k this is wasteful. Note that `Σ_p` need never be materialised:

```
G(h) = Wᵀ diag(p) W − (Wᵀp)(Wᵀp)ᵀ
```

The second term is a rank-1 outer product of a single `d`-vector. The first is a weighted Gram matrix. Combined with **top-k token truncation** (FishBack uses top-5000 by probability mass), this is cheap.

> **Truncation must be justified, not assumed.** Report the error `‖G_full − G_topk‖ / ‖G_full‖` as a function of `k` on a sample where the full computation is affordable. Softmax distributions are typically sharply peaked, so top-k should be accurate — **but demonstrate it, because a reviewer will ask.** One figure settles it permanently.

### 6.3 Framework choice

**JAX** — recommended. `jax.jacfwd`/`jacrev` compose cleanly to the higher derivatives curvature needs, `jit` matters for the inner loops, and float64 is available (`jax_enable_x64`), which curvature computation needs. **PyTorch** is acceptable via `torch.func` (`jacrev`, `jvp`, `vmap`) and is easier if the model loading pipeline is already PyTorch — a reasonable hybrid is to extract hidden states and `W` with PyTorch/`transformers`, save to disk, then do all geometry in JAX.

**Numerical hygiene, non-negotiable:**
- **float64 for all geometry.** Not a style preference — `cond(G)` ≈ 10⁸ at realistic softmax sharpness (§4) costs ~8 of float64's ~16 digits before any differentiation, and curvature needs second derivatives. float32 (~7 digits) is already below the conditioning of the metric itself. Extract activations in float32/bf16, then cast to float64 before any geometry.
- Never compute softmax naively — use log-space (`logsumexp`).
- Use analytic derivatives of `G` where possible; `p` is a smooth explicit function of `h`, so `∂G/∂h` is analytically available and far more stable than finite differences.
- Where finite differences are unavoidable, do a **step-size sweep** and look for the plateau between truncation error and cancellation error. Report the chosen step and the plateau plot.
- Fix seeds; log library versions; store the exact activation tensors used in every figure.

### 6.4 Model selection

| Model | `d` | Layers | Role |
|---|---|---|---|
| GPT-2 small | 768 | 12 | primary development; matches FishBack and Mabrok for comparability |
| GPT-2 XL | 1600 | 48 | matches King et al. — needed for the r ≈ 0.15 comparison |
| Pythia-1.4B / 2.8B | 2048 / 2560 | 24 / 32 | matches King et al.; checkpoint series available for RQ5 |
| Llama 3.2-1B / 3B | 2048 / 3072 | 16 / 28 | matches Manson; modern architecture |
| Gemma 3-1B | 2048 | 26 | matches Manson; the negative case in his results |

**Check tied vs untied embeddings for each** — Mabrok's framework may assume tying. GPT-2 ties; several modern models do not. This affects `W` and therefore `G`. **Resolve before Stage 4** and record the answer per model in a table.

### 6.5 Data

- **LAMBADA** — long-context; matches King et al.
- **Universal Dependencies** — short sentences (10–30 tokens); matches King et al.
- **WikiText-103** — general-purpose layer profiles.
- **A purpose-built polysemy probe set** for RQ3b: minimal pairs where one ambiguous token is disambiguated in opposite directions (e.g. *bank* / river vs money; *bat* / animal vs cricket). Aim for ≥ 200 pairs, ≥ 4 domains. **Manson's 20 prompts across 7 domains is a stated weakness of his work — beating it on scale is cheap and directly strengthens the contribution.**

  > 🔴 **The disambiguating context MUST PRECEDE the ambiguous word.** An earlier version of this line said "disambiguated by *later* context," and the first corpus built to it was vacuous: under **causal attention** the hidden state at the target token attends only to tokens before it, so *"The **bank** was steep…"* and *"The **bank** refused…"* produce **bit-for-bit identical** states and every separation measure returns exactly 0.0 at every layer. Later context cannot reach back. Establish the sense first and place the ambiguous word last. `run_polysemy.py` asserts non-zero final separation so this cannot recur silently. ([07-stage4-log.md](07-stage4-log.md) §3b.1)
- For RQ3c, a hallucination-annotated set (e.g. TruthfulQA-derived, or generations labelled against a reference).

---

## 7. Experimental design

### E1 — Layer-wise curvature profile (RQ2)
For each model, each layer, ~10⁴ token positions: compute `G(h)`, restrict via Option A, sample sectional curvature over random 2-planes, record the distribution plus scalar curvature and volume element. Plot median and IQR against relative depth. Overlay the intrinsic-dimension profile to test the hourglass hypothesis.

### E2 — Instrument comparison (RQ2b) — *the adjudication experiment*
On identical data, compute all four quantities side by side:
1. Intrinsic Fisher–Rao sectional curvature (this work)
2. Local-PCA proxy curvature (Mabrok's method)
3. Frenet curvature under `G = UᵀU` (Manson's method)
4. Euclidean difference-vector angles (King et al.'s method)

Report pairwise correlations. **This single figure is the paper's centrepiece** — it either shows the proxies track the true quantity (vindicating them) or that they diverge (justifying the thesis). Both outcomes are results.

### E3 — Curvature vs entropy (RQ3a)
Correlate each curvature variant against next-token entropy, layer by layer, on LAMBADA and UD. **Target: beat r ≈ 0.15.** Pre-register the comparison and use the same data splits as King et al. where possible. Report confidence intervals, not just point estimates — with r this small, CI width is the whole story.

### E4 — Ambiguity localisation (RQ3b)
On the polysemy probe set, track the Fisher–Rao *geodesic* distance between the two contextual variants of the same token across layers. Identify the layer of maximum separation rate and maximum curvature. Hypothesis: curvature peaks at the disambiguation layer.

### E5 — Causal intervention (RQ4)
Perturb `h` by matched Fisher-norm steps `ε` along (i) the highest sectional-curvature direction, (ii) the lowest, (iii) a random direction, (iv) the null space of `G`. Measure KL divergence of the resulting output distributions. **Prediction: null-space perturbations produce ≈ 0 KL — this is a strong falsification test of the whole framework.** If null-space perturbations *do* change the output, the metric is wrong or the implementation is buggy. Run this test early; it is cheap and highly diagnostic.

---

## 8. Reproducibility requirements

- Every figure regenerable by one script from a fixed seed and a pinned environment.
- All curvature numbers reported with the `k` (or `c`) that produced them, plus the sensitivity curve.
- Validation-ladder results included as an appendix table with pass/fail per rung, in the thesis.
- Code released with the paper; validation tests in CI.
- Store activation tensors (or exact extraction scripts plus versions) for every experiment.
