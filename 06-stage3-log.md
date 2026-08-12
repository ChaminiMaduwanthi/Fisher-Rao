# Stage 3 — Progress Log: the first intrinsic Fisher–Rao curvature measurement

**Date:** 10 August 2026
**Status:** ✅ **Gate B passed. RQ1 answered. RQ2b answered provisionally.**
**Reproduce:** `python run_stage3.py` then `python run_control.py` → `results/stage3/`

> ⚠️ **Read §5 before quoting anything here.** These numbers come from **one model, one prompt, one token position**. They are a proof of feasibility and a strong provisional answer, not an empirical study.

---

## 1. ✅ Gate B — the validation ladder passes at machine precision

```
rung 3  categorical simplex (n=4)   K = +0.25   worst |err| = 3.10e-10   PASS
rung 6  univariate Gaussian         K = -0.50   worst |err| = 3.89e-16   PASS
rung 2  Poincaré half-plane         K = -1.00   worst |err| = 7.77e-16   PASS
```

Two independent implementations now agree: the finite-difference reference (`validate_curvature.py`) and the autodiff version (`fisherrao/curvature.py`). **The gate is met.** Per [04-stage-plan.md](04-stage-plan.md), the workshop paper on the method can be written now.

### 1.1 🔴 Three implementation traps, recorded because each cost real time

**(a) `torch.autograd.functional.jacobian` cannot be nested.** It defaults to `create_graph=False`, so the inner derivative is detached and the outer one silently loses the `∂(∂G)` term. Symptom: **the simplex rung passed** while the Gaussian and Poincaré rungs returned **exactly −2× the true value**.

> Had the ladder contained only the simplex — the obvious single rung to build, since it is the thesis's own metric family — this would have shipped. **This is the argument for multiple rungs with different known answers.** It is the second time in this project that a factor/sign error survived a plausible-looking result.

Fix: `torch.func.jacrev`, which composes correctly for higher derivatives.

**(b) Top-k truncation must be frozen before differentiating.** Re-selecting the top-k at each evaluation makes `G(h)` only *piecewise* smooth — it jumps where top-k membership changes. Curvature needs second derivatives, so this is fatal. Fix: freeze the index set at the base point (`metrics.topk_indices`).

**(c) Project into the frame before forming the metric.** Building the full `576×576` metric and then projecting to `6×6` made double autodiff carry a `(d,d)` intermediate — it attempted a **20.6 GB** allocation and died. Fix: `fisher_metric_projected`, which collapses to `k` dimensions first via `M = U_idx·(A·frame)`, largest intermediate `(n_tok, k)`. Cost after the fix: **~9 s per Riemann tensor at `k`=6** on CPU.

---

## 2. ✅ Task 3.8 — frame invariance

Curvature must not depend on an arbitrary rotation of the retained basis. This is what licenses holding the frame fixed while differentiating.

| layer | scalar `R` | rotated | rel. diff |
|---|---|---|---|
| 10 | +7.319908 | +7.319908 | 2.6e-11 |
| 20 | +11.08268 | +11.08268 | 1.1e-11 |
| 30 | +8.575225 | +8.575225 | 2.3e-13 |

**PASS.** The moving-frame handling is correct.

---

## 3. 🎯 The measurement

`k = 6`, radial direction quotiented, `top_k = 512`, 48 random 2-planes per layer.

| layer | K median | K p10 | K p90 | K>0 | scalar R | log vol | k_eff |
|---|---|---|---|---|---|---|---|
| 1 | 2.4975e-01 | 2.368e-01 | 2.756e-01 | 100% | 16.93 | −17.39 | 7 |
| 5 | 2.5008e-01 | 2.494e-01 | 2.509e-01 | 100% | 8.08 | −14.42 | 5 |
| 10 | 2.5003e-01 | 2.498e-01 | 2.502e-01 | 100% | 7.32 | −14.26 | 5 |
| 15 | 2.5081e-01 | 2.503e-01 | 2.542e-01 | 100% | 7.81 | −17.08 | 5 |
| 20 | 2.5050e-01 | 2.501e-01 | 2.534e-01 | 100% | 11.08 | −19.30 | 4 |
| 25 | 2.5110e-01 | 2.502e-01 | 2.668e-01 | 100% | 18.57 | −23.36 | 5 |
| 28 | 2.5735e-01 | 2.515e-01 | 2.733e-01 | 100% | 10.07 | −21.07 | 12 |
| 29 | 2.5266e-01 | 2.450e-01 | 2.570e-01 | 100% | 7.04 | −20.76 | 27 |
| 30 | 2.4683e-01 | 1.983e-01 | 2.812e-01 | 100% | 8.58 | −21.71 | 86 |

**Sectional curvature is ≈ +1/4 at every layer, and positive in 100% of sampled planes.**

`+1/4` is exactly the constant curvature of the categorical simplex under Fisher–Rao. That coincidence is the first thing that has to be ruled out.

---

## 4. ✅ The control — `+1/4` is not an artifact of the construction

Replace the real unembedding with random matrices at several scales, which changes how concentrated `p` is (`run_control.py`, layer 20):

| unembedding | entropy (nats) | K median | scalar R |
|---|---|---|---|
| **real U** | **0.81** | **+0.2506** | 11.08 |
| random ×3 | 3.29 | +0.2626 | 8.43 |
| random /√d | 9.52 | **+0.1377** | 3.90 |
| random ×0.3 | 10.68 | **−0.0176** | −0.47 |
| exact simplex reference | — | +0.250000 | — |

**Diffuse predictive distributions give curvature far from +1/4 — even negative.** So the value is not produced by the restriction. What it tracks is the **concentration of `p`**.

### 4.1 The mechanism, stated carefully

A trained LM's next-token distribution is extremely peaked: entropy **0.8–1.8 nats** across layers, versus **~10.7** for a random unembedding on the same 49 152-token vocabulary. Its representations therefore occupy a **near-low-dimensional face of the simplex**, and on such a face the induced Fisher geometry is essentially the ambient simplex geometry: constant `+1/4`.

**So the honest headline is not "the semantic manifold has curvature 1/4."** It is:

> Under the correct Fisher–Rao metric, the transformer's representation manifold is **strongly positively curved, pinned near the ambient simplex value +1/4**, because the model's predictive distribution is sharply concentrated.

Since `+1/4` is the ambient value, absolute curvature is a **weak discriminator by construction**, and the obvious next move is to look at the deviation from it. That was tried — and it failed its own sensitivity check. See §4.2.

### 4.2 🔴 RETRACTED: the deviation from +1/4 is **not identifiable**

An earlier draft of this section claimed the excess of `scalar R` over `k(k−1)/4` was "the geometry the model itself contributes." **That claim does not survive its own sensitivity check and is withdrawn.**

The re-baselined measurement looked compelling — `dR/R_ref` spanning −6.1% to +147.6% across layers at `k`=6, where absolute `K` spanned a boring 0.2468–0.2574. But `dR/R_ref` is **not stable in `k`** (`check_krank.py`):

| k | L5 | L10 | L20 | L25 | L29 | layer ranking |
|---|---|---|---|---|---|---|
| 4 | +7.4% | −7.9% | +7.3% | +0.6% | +2.4% | L5 > L20 > L29 > L25 > L10 |
| 5 | +3.4% | +1.0% | +29.2% | +70.5% | +1.2% | L25 > L20 > L5 > L29 > L10 |
| 6 | +7.8% | −2.4% | +47.8% | +147.6% | −6.1% | L25 > L20 > L5 > L10 > L29 |
| 7 | **+583.5%** | +48.5% | +15.3% | +208.0% | −1.6% | L5 > L25 > L10 > L20 > L29 |

Spearman rank correlation of the layer ordering: **ρ = +0.20 between `k`=4 and `k`=7.** Essentially no agreement. Neither the magnitudes nor even the ordering of layers survives a change in an arbitrary cutoff.

This is precisely the outcome [03-methodology.md](03-methodology.md) §4 committed to reporting:

> *"If conclusions depend on the cutoff, the honest finding is that curvature is not identifiable from this metric — and reporting that clearly is itself a contribution."*

**So it is reported. The deviation is not a usable signal at this `k` range.**

### 4.3 Diagnosing why — and a concrete methodological fix

The natural hypothesis was that instability is confined to `k > k_eff`, where retained directions have near-zero eigenvalues. **Partly right, not sufficient** (`check_keff.py`):

| layer | `k_eff` | k=3 | k=4 | k=5 | k=6 | k=7 |
|---|---|---|---|---|---|---|
| 5 | 5 | +34.8% | +7.4% | +3.4% | +7.8%\* | **+583.5%**\* |
| 10 | 5 | +11.2% | −7.9% | +1.0% | −2.4%\* | +48.5%\* |
| 20 | 4 | +12.7% | +7.3% | +29.2%\* | +47.8%\* | +15.3%\* |
| 25 | 5 | +20.4% | +0.6% | +70.5% | +147.6%\* | +208.0%\* |

\* = `k` exceeds `k_eff`

The worst blow-ups do sit at `k > k_eff`. But the deviation is unstable **even within** `k ≤ k_eff` — layer 25 gives +20.4%, +0.6%, +70.5% at `k` = 3, 4, 5. **The `k ≤ k_eff` rule reduces the damage but does not rescue the measurement.**

The informative diagnostic is `cond_eff` of the *retained* subspace, which jumps by an order of magnitude at `k`=6 across every layer (2.8e1 → 4.6e2 at layer 5; 1.5e1 → 4.6e2 at layer 10):

> **Methodological finding: `k` must be selected by a conditioning criterion, not a trace-fraction criterion.** `k_eff` (99% of trace) admits directions whose eigenvalues are small enough to destroy second derivatives. A `cond_eff` ceiling — provisionally ~10², where the instability sets in here — is the right rule. This supersedes the trace-threshold selection in [03-methodology.md](03-methodology.md) §4 Option A and should be treated as a genuine, if unwelcome, result of Stage 3.

### 4.4 What survives

| claim | status |
|---|---|
| `K ≈ +1/4`, 100% of planes positive, every layer | ✅ **robust** — stable across `k`=3–7 (0.250–0.264) and across layers |
| `+1/4` is not a construction artifact | ✅ **robust** — control gives −0.02 to +0.14 for diffuse `p` |
| The manifold is strongly positively curved, not flat at 10⁻⁵ | ✅ **robust** (RQ2b) |
| Curvature tracks the concentration of `p` | ✅ **robust** — control, monotone in entropy |
| Deviation from +1/4 encodes model-specific structure | ❌ **not established** |
| Layer-to-layer ordering by anomalousness | ❌ **not established** |

**The volume element is the promising alternative.** `log vol` across layers (−17.39, −14.42, −14.26, −17.08, −19.30, −23.36, −21.07, −20.76, −21.71) is smooth and well-behaved, with no sign of the `k`-fragility that afflicts the Riemann-derived quantities. This is exactly what Zavatone-Veth et al. reported — the volume element is both more stable and more informative than the Ricci scalar — and [02-research-gap.md](02-research-gap.md) §6 already designated it a parallel primary quantity. **Promote it to the primary layer-wise measurement.**

> ⚠️ **QUALIFIED 11 August 2026.** "More stable" survives; **"more informative" does not.** Under the paired scramble control ([08-rq3a-log.md](08-rq3a-log.md) §5.5) the layer-wise volume profile is **largely definitional** — a structure-free scramble at matched entropy reproduces its shape at ρ = +0.883 — whereas sectional `K` under the same control collapses from 0.259 to 0.005. **Stability and informativeness came apart.** The volume element remains the right primary quantity for anything needing large `k` or numerical robustness; it is the *weaker* of the two as evidence about what the model has learned.

---

## 5. ⚠️ Sensitivity, and what these numbers do *not* establish

### 5.1 `k`-sensitivity (task 3.12), layer 20

| k | K median | scalar R | `k(k−1)/4` |
|---|---|---|---|
| 3 | 2.636e-01 | 1.69 | 1.5 |
| 4 | 2.525e-01 | 3.22 | 3.0 |
| 5 | 2.544e-01 | 6.46 | 5.0 |
| 6 | 2.506e-01 | 11.08 | 7.5 |
| 7 | 2.501e-01 | 12.11 | 10.5 |

`K median` is **stable in `k`** (2.50–2.64e-01) — the qualitative conclusion does not depend on the cutoff, which is the outcome [03-methodology.md](03-methodology.md) §4 demanded. `scalar R` necessarily grows with `k` (it sums over `k(k−1)` plane pairs), and it exceeds the constant-curvature prediction at every `k`, most strongly at `k`=6.

### 5.2 Limits — all of these must be fixed before any claim leaves this file

- **One model** (SmolLM2-135M), **one prompt**, **one token position**. Nothing here is an empirical study; it is a feasibility proof plus a provisional answer.
- **`k` = 6** while `k_eff` ranges 3–86. Layer 30 has `k_eff` = 86, so `k`=6 captures a small part of its geometry.
- **Plane sampling is coordinate-uniform, not metric-uniform.** `u, v ~ N(0, I)` in the eigenframe weights each retained eigendirection equally; a metric-uniform sample would use `N(0, G⁻¹)`. With `cond_eff` of 10¹–10³ these differ. `scalar R` is a full trace and carries no such dependence — prefer it as the summary.
- **Layer 0 excluded**: with tied embeddings the logit lens there is degenerate (entropy 0.000).
- The `+1/4` pinning means **absolute curvature is a weak discriminator by construction.** Design Stage 4 around the deviation quantities from §4.2.

---

## 6. What this says about the research questions

**RQ1 — can it be computed?** ✅ **Yes.** Validated to machine precision against three known answers, frame-invariant to 10⁻¹¹, ~9 s per point at `k`=6 on a CPU. The singularity/conditioning obstacle that stopped Mabrok ("computationally intractable") and Zavatone-Veth et al. ("expensive and numerically challenging") is passable via quotient-then-restrict. **This is contribution C1–C3, delivered.**

**RQ2b — is Mabrok's ~10⁻⁵ an artifact of the PCA proxy?** ✅ **Yes in kind; the magnitude claim needs work.** Intrinsic curvature is `+2.5e-01`, so **the manifold is not "essentially flat" — it is strongly positively curved.** Stage 4 then settled the *comparison* properly: running both instruments on identical activations gives Spearman **ρ = −0.24** — the proxy does not detectably track the intrinsic quantity ([07-stage4-log.md](07-stage4-log.md) §1).

> ⚠️ **Softened.** An earlier draft claimed the two differ "by four orders of magnitude." That compared this project's intrinsic value against Mabrok's *published* number, computed on a different, much larger point cloud. Reimplementing his proxy on this corpus gives **2.4e-2**, not 1e-5 — the neighbourhoods here are not local enough. The magnitude gap is **not established** until his setup is faithfully replicated. The correlation result needs no cross-paper magnitude comparison and is the stronger claim.

**RQ2d — sign of curvature?** **Positive, unambiguously** — 100% of sampled planes at every layer. Not the negative curvature of Dirichlet/beta families; sphere-like, consistent with the simplex embedding.

**RQ4 — null-space falsification.** Already passed in Stage 0 (radial/random KL ratio 10⁻¹⁶).

---

## 7. Next actions

1. **Re-baseline the measurement** on `K − 1/4` and `R − k(k−1)/4` (§4.2). This is the single highest-value change and should come before any scaling up.
2. **Scale to a corpus** — WikiText-103, ≥10⁴ token positions, all layers. Until then §3 is one data point.
3. **Add metric-uniform plane sampling** alongside the coordinate-uniform one and report both.
4. **Raise `k` toward `k_eff`** at the final layers; measure the cost curve (`k`=7 already at ~15 s).
5. **Verify Mabrok's proxy on the same data** — implement local-PCA curvature and confirm it returns ~10⁻⁵ where the intrinsic computation returns +0.25. That head-to-head on identical activations is experiment E2, the paper's centrepiece figure, and it is now cheap to run.
6. **Write the workshop paper** on C1–C3 (Gate B is met). Do not wait for the empirical study.
7. Unblock GPT-2/Pythia (TLS issue, [05-stage0-log.md](05-stage0-log.md) §0) for cross-architecture checks and the King et al. `r ≈ 0.15` comparison.
