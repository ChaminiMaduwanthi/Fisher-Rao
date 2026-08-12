# Chapter 5 — Validation

> **Draft.** Every number in this chapter is reproducible from the scripts named beside it. Sources: [05-stage0-log.md](../05-stage0-log.md), [06-stage3-log.md](../06-stage3-log.md), [10-architecture-log.md](../10-architecture-log.md).

---

## 5.1 Why this chapter comes before the results

Curvature code is unusually easy to get wrong, and its errors are silent. A sign error, an index-ordering error, or a factor of two all produce numbers that look entirely reasonable. There is no internal signal that anything is amiss: the output is a small real number, it varies smoothly with the input, and it has the units one expects.

This is not a hypothetical concern. The first run of the reference implementation in this work returned

```
K = −0.250000     for the categorical simplex, where the correct answer is +1/4
```

— the right magnitude, the wrong sign, and **constant across every sample point**, which made it look *more* trustworthy rather than less. Nothing but a known answer could have caught it.

Accordingly, this work treats validation as a precondition rather than a postscript. Chapter 6's results are only worth reading because the machinery that produced them has been checked against cases whose answers are known in advance, and this chapter is that check.

The structure of the argument is a **ladder**: a sequence of test manifolds ordered so that each rung isolates one additional source of error, and each has an analytically known curvature. A failure at rung *k* makes every rung above it uninterpretable, so the rungs are reported together and none is skipped on failure.

---

## 5.2 The ladder

| Rung | Test case | Known answer | What it isolates |
|---|---|---|---|
| 0 | Flat ℝ², Euclidean metric | `R = 0` exactly | gross implementation errors |
| 1 | 2-sphere of radius `r` | `K = 1/r²` | sign and normalisation |
| **2** | **Poincaré half-plane** | **`K = −1`** | negative curvature |
| **3** | **Categorical simplex, full Fisher–Rao** | **`K = +1/4`** | ⭐ *the key rung* — same metric family as the thesis |
| **4** | **Gamma family** | **negative, cross-checked** | an independent Fisher–Rao case |
| **5** | **Beta / Dirichlet family** | **everywhere negative** | multi-parameter Fisher–Rao |
| **6** | **Univariate Gaussian family** | **`K = −1/2`** | cross-check against Geomstats |
| **7** | **Synthetic linear `U`, `d` = `N−1`** | **`K = +1/4`** | ⭐ *the pullback step specifically* |
| 8 | Known 1-D "ripple" manifolds | qualitatively known | first contact with real LM data |

Rung 3 is the load-bearing one. The categorical simplex under the full Fisher–Rao metric is isometric to the positive orthant of a sphere of radius 2, so its sectional curvature is exactly `+1/4` everywhere — and it is the **same metric family** the thesis computes on transformer states. A curvature routine that gets the simplex right is being tested on the object it will actually be used for, not on a generic Riemannian toy.

---

## 5.3 Results — rungs 2, 3, 4, 5, 6 and 7 pass at machine precision

`validate_curvature.py` is a deliberately naive reference implementation: explicit Christoffel symbols, an explicit rank-4 Riemann tensor, central finite differences. It does not scale, and it is not meant to. Its only job is to be an oracle trustworthy enough to check the fast implementation against.

| Rung | Manifold | Expected | Measured (worst of 4 random base points) |
|---|---|---|---|
| **3** | categorical simplex, `n` = 4 | `+0.25` | **+0.250001** |
| **6** | univariate Gaussian | `−0.50` | **−0.500000** |
| **2** | Poincaré half-plane | `−1.00` | **−1.000000** |

**Scalar curvature is validated separately**, because validating `sectional()` does not validate the Ricci contraction and it is scalar `R` that carries several of the results in Chapter 6. For a manifold of constant sectional curvature `K` in `k` dimensions, `R = k(k−1)K` exactly:

| simplex `n` | `k` | expected `R` | measured | relative error |
|---|---|---|---|---|
| 4 | 3 | +1.50 | 1.500000 | **4.2e−12** |
| 6 | 5 | +5.00 | 5.000000 | **2.0e−15** |
| 7 | 6 | +7.50 | 7.500000 | **1.8e−14** |

These are **Gate B** in the project's staging (§5.7). The gate was defined in advance as "rungs 0–6 pass, **agreeing with an independent implementation**". The known-answer half is met outright; the independent-implementation half is met on three families of five, and §5.3b reports the other two rather than quietly restating the criterion as "agreeing with known answers".

---

### 5.3a Rungs 4 and 5, and an independent cross-check

The values above are **analytic** — `+1/4`, `−1/2`, `−1` taken from the literature. Agreeing with a remembered constant tests one thing; agreeing with a peer-reviewed implementation that made its own choices about parameterisation, sign and normalisation tests rather more. `validate_geomstats.py` runs both, against **geomstats 2.8.0** on identical inputs, and adds two further Fisher–Rao families as rungs 4 and 5.

| family | this work | geomstats | \|diff\| |
|---|---|---|---|
| **rung 6** univariate normal | −0.50000000 | −0.50000000 | **1.1e−16** |
| **rung 4** gamma | −0.46390348 | −0.46390348 | **2.5e−14** |
| **rung 4** gamma | −0.47313320 | −0.47313320 | 2.3e−13 |
| **rung 5** beta | −0.45679484 | −0.45679484 | **3.2e−14** |
| **rung 5** beta | −0.42604467 | −0.42604467 | 1.0e−14 |

**Three families agree at machine precision across eight comparisons, including both new rungs.** Rung 5's published answer is qualitative — sectional curvature everywhere negative for multi-parameter Fisher–Rao families — and both gamma (−0.463, −0.473, −0.463) and beta (−0.457, −0.426, −0.440) satisfy it.

### 5.3b 🔴 Two families disagree — and the disagreement is localised, not patched

geomstats returns `+2.0` for the categorical simplex where this work returns `+0.25`, and a **point-varying positive** value for the Poincaré half-plane where this work returns `−1`.

Section 5.4's rule is *diagnose before "fixing"*, and the diagnostic is to compare the **metrics** rather than the curvatures — a curvature disagreement arising from a metric convention is harmless, one arising from a curvature routine is not.

| | metrics identical? | curvatures agree? |
|---|---|---|
| Poincaré half-plane | ✅ **yes**, to 1e−12 | ❌ no |
| categorical simplex | ✅ **yes**, to 7e−15 | ❌ no |

> **In both cases the metrics agree exactly while the curvatures do not.** Neither disagreement is a parameterisation convention; both are localised to a curvature routine.

**Which side to doubt, on the evidence available:**

- **geomstats' Poincaré curvature is not constant across base points.** Hyperbolic space has constant curvature by definition, so a point-varying answer cannot be correct. This work returns `−1` at every point.
- **This work's `+1/4` for the simplex is corroborated by three independent routes**: the analytic radius-2 sphere isometry; ladder rung 3, a separate slow reference implementation; and ladder rung 7, an entirely different code path through the pullback.

The disagreements are therefore **reported rather than resolved in this work's favour by assumption** — but the corroboration is asymmetric, and it points one way. Reproducing them is a matter of running one script.

---

## 5.4 The sign-convention trap

Two independent convention axes cause disagreements with published values, and **both look like bugs**:

**Index order of the lowered Riemann tensor** flips the sign. The convention fixed here, once, against rung 3:

```
R^ρ_σij   such that   R(e_i, e_j) e_σ = R^ρ_σij e_ρ
R_μσij    = g_μρ R^ρ_σij
K(u,v)    = R(u,v,v,u) / (|u|²|v|² − ⟨u,v⟩²)
```

The original error was contracting `R(u,v,u,v)` instead of `R(u,v,v,u)`.

**Simplex normalisation** gives a factor of four. The radius-2 sphere convention used here gives `K = +1/4`; a unit-sphere convention with the metric scaled by 4 gives `K = +1`.

> **A disagreement with a reference implementation of exactly `−1×` or `4×` is a convention mismatch, not an error. Diagnose before "fixing".**

---

## 5.5 Validation on real activations

The ladder validates the *curvature* code. Four further checks validate the *pullback* — the step from a transformer's hidden state to a metric — and they are what make Chapter 6's numbers about a model rather than about an arbitrary construction.

### 5.5.1 Gate A — the metric is the Hessian of KL

The defining property of the Fisher information metric is that it is the second-order term in the KL divergence between nearby distributions:

```
KL( p(h) ‖ p(h + εv) )  =  ½ ε² vᵀ G(h) v  +  O(ε³)
```

Verified on a real model via an independent code path (`gate_a_kl_test.py`) — the left-hand side computed from logits alone, the right from the assembled metric:

| `ε` | `KL / ε²` | relative error |
|---|---|---|
| 1e−1 | 0.022341 | 7.6e−03 |
| 1e−2 | 0.022495 | 7.7e−04 |
| 1e−3 | 0.022510 | 7.7e−05 |
| **1e−4** | **0.022512** | **8.7e−06** ← plateau |
| 1e−5 | 0.022510 | 9.0e−05 |
| 1e−6 | 0.022272 | 1.1e−02 |
| 1e−7 | 0.019559 | 1.3e−01 |

against the predicted `½ vᵀGv = 0.0225120`.

**The sweep matters more than any single value.** Below `ε` ≈ 1e−5 catastrophic cancellation in the KL destroys the test even when the metric is exactly right — the error rises again, and a naive single-`ε` check at 1e−7 would report a 13% discrepancy and be believed. **The correct reading is the plateau**, and it sits at 8.7 × 10⁻⁶.

### 5.5.2 Frame invariance

Curvature is a geometric quantity and must not depend on the basis chosen for the retained subspace. Rotating that basis by an arbitrary orthogonal `Q`:

| condition | median relative change in scalar `R` | worst |
|---|---|---|
| real metric | 2.23e−14 | 1.54e−10 |
| scrambled control | 9.05e−15 | 1.90e−11 |

`n` = 24 points. **Invariant in both conditions**, which additionally establishes that the "frame sensitivity" reported at one stage of this work was a misnomer — the two conditions were two different *subspaces*, not two bases for one subspace (Chapter 6, §6.4).

### 5.5.3 Rung 7 — the pullback assembly, validated directly

§5.5.1 and §5.5.4 validate the pullback *indirectly*: Gate A checks that `G` is the Hessian of KL, and the null-space test checks that `G`'s kernel is the model's kernel. Neither checks that

```
G(h) = Jᵀ ( diag(p) − p pᵀ ) J
```

is **assembled** correctly — a transpose, an index swap, or a dropped outer-product term would survive both.

**The construction.** Take a synthetic linear model with no normalisation layer, `N` outcomes and `d = N − 1`:

```
p(h) = softmax(U h)
```

If `U`'s columns together with `1_N` span `ℝᴺ`, then `h ↦ p` is a **diffeomorphism** onto the simplex interior — the logits determine `p` up to the all-ones direction, which softmax quotients out, and `d = N − 1` is exactly the remaining freedom. **Curvature is invariant under reparameterisation**, so the pullback metric is the simplex metric in other coordinates and

> `K = +1/4` **exactly, at every `h`, for every valid `U`.**

The test exercises the project's own `fisher_metric` rather than a re-implementation — a bespoke reference metric here would test nothing.

| check | result |
|---|---|
| `K` = +1/4 through the pullback, `N` = 3, 4, 5 | worst \|err\| **1.5e−13, 2.6e−14, 9.7e−12** |
| **reparameterisation** `h = Mx`, so `U → UM` — same manifold, different coordinates, `K` must be unchanged | \|diff\| **1.9e−14, 5.2e−13, 1.9e−11** |
| scalar `R = k(k−1)/4` through the pullback | rel. err **1.3e−14, 4.9e−12, 3.0e−12** |
| **negative control** — drop the `−(Jᵀp)(Jᵀp)ᵀ` term | **caught, \|K − 1/4\| = 0.263** |

**The reparameterisation check is the one that catches index-order and transpose errors**, because it compares the code against itself under a transformation it must be blind to. **The negative control matters as much**: a deliberately broken assembly must fail, or the test proves nothing.

### 5.5.4 The null-space falsification test

The strongest available check, because it can fail outright rather than by degree. RMSNorm and LayerNorm are scale-invariant, so `h` itself is an **exact** null direction of `G(h)`: the metric asserts that rescaling a hidden state changes nothing. If the model disagrees, the metric is not the model's geometry and every number in Chapter 6 is void.

Measured at a step size that **doubles the hidden state**:

| model | `KL` along the null direction | `KL`, random direction, same size | ratio |
|---|---|---|---|
| SmolLM2-135M (RMSNorm) | 6.0e−15 | 2.59 | 1.1e−12 |
| gpt2 (LayerNorm) | 1.97e−13 | 3.86 | 3.2e−12 |
| pythia-160m (LayerNorm) | 6.88e−11 | 7.77 | 6.4e−11 |
| llama-160m (RMSNorm, untied) | — | — | 3.7e−14 |

**Passes on all six models tested**, across two normalisation kinds, four architecture families and both embedding regimes.

---

## 5.6 Architecture-level correctness

Extending beyond one model required four rungs of its own, because a pipeline that runs only on one architecture has not been tested — it has been fitted. `check_architectures.py`:

| | SmolLM2-135M | gpt2 | pythia-160m |
|---|---|---|---|
| norm kind | LlamaRMSNorm | LayerNorm | LayerNorm |
| **`apply_norm` vs the model's own norm** | 3.53e−08 | 6.02e−08 | 8.40e−08 |
| **`norm_jacobian` vs autograd** | **1.14e−16** | **1.09e−16** | **1.04e−16** |
| null directions found (expected) | 1 (1) | **2 (2)** | **2 (2)** |
| **logit lens vs the model's real output logits** | 2.75e−07 | 8.45e−08 | 8.30e−08 |
| argmax matches | ✅ | ✅ | ✅ |

**6/6 architectures pass** (the three above plus llama-160m, pythia-70m, gpt-neo-125m).

The forward-error column sits at 1e−8 because the reference runs in float32 while everything here is float64; that is the float32 noise floor, not disagreement.

**The logit-lens rung is the end-to-end one.** It fails if the residual stream is captured at the wrong point, if the final norm is applied twice, or if the unembedding is wrong — and it is what makes the other three worth trusting. It caught a double-normalisation bug at relative error **2.07** during Stage 0, an inconsistency that would not have announced itself in a curvature plot.

### 5.6.1 Two silent bugs this found

**(i) LayerNorm bias placement.** The implementation computed `g·z + g·b` instead of `g·z + b`. This is *exactly zero error* on an RMSNorm model with no bias — i.e. on the model everything had been developed against — and a plausible-looking error on every LayerNorm model. Caught by checking against the model's own norm module rather than by reading the code.

**(ii) The embedding hook.** Layer 0 was captured from a hook on the token embedding. On GPT-2 the residual stream entering block 0 is `wte + wpe`, so the positional term would have gone missing. Replaced with a forward pre-hook on block 0 — literally "whatever enters the first block", which is architecture-independent.

Neither bug was detectable without a second architecture. **This is the argument for the cross-architecture work being a validation activity, not only a generalisation one.**

---

## 5.7 The two gates

The project defined two hard gates in advance, with pass criteria fixed before the measurements were taken.

| | criterion | status |
|---|---|---|
| **Gate A** | `KL(p(h)‖p(h+εv)) ≈ ½ε²vᵀG(h)v` verified to third order via an independent code path on a real model | ✅ **passed**, plateau relative error 8.7e−06 |
| **Gate B** | validation-ladder rungs 0–6 pass, **agreeing with Geomstats** | 🟡 **passed on the known-answer arm** (machine precision; rungs 7, 4 and 5 pass too) — **the Geomstats arm is met on 3 of 5 families**, §5.3b |

---

## 5.8 What validation does not cover

Stated explicitly, because the ladder is easy to over-read.

- **Rung 8 has not been run**, and its published answer is only qualitative. Rungs 2, 3, 4, 5, 6 and 7 cover positive, negative, two independent Fisher–Rao families, the simplex and the pullback.
- **The geomstats cross-check succeeds on three of five families** (§5.3a). On the other two the metrics agree exactly and the curvatures do not, which localises the disagreement to a curvature routine; the corroboration is asymmetric and points to this work's values, but the discrepancy is unresolved and is reported as such (§5.3b).
- **The ladder validates the machinery, not the sampling.** It says nothing about whether the retained subspace dimension `k`, the top-`k` truncation, or the point sample are appropriate. Those are Chapter 4's concerns and their sensitivity is reported in Chapter 6.
- **`riemann()` is validated only in the regime it is used.** It cannot be evaluated above `k` ≈ 8 at all (Chapter 4, §4.6), so no rung tests it at large `k`.
