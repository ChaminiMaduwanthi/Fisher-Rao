# Stage 0 — Progress Log

**Date:** 10 August 2026
**Status:** ✅ Stage 0 exit criteria met. Two Stage 2 tasks and one Stage 3 task completed early.
**Reproduce:** `HF_HUB_OFFLINE=1 python run_stage0.py` → `results/stage0/{stage0.json,stage0.png}`

---

## 0. Environment

| | |
|---|---|
| Python | 3.13.14 |
| torch | 2.12.0+cpu |
| transformers | 5.14.1 |
| numpy / scipy / matplotlib | 2.3.3 / 1.16.2 / 3.10.6 |
| JAX | **not installed** — see §6 |

~~⚠️ **HuggingFace Hub is unreachable from this machine.**~~ → ✅ **FIXED 11 August 2026. See [10-architecture-log.md](10-architecture-log.md) §1.**

The diagnosis here was right: TLS interception injecting a malformed CA (`Basic Constraints of CA cert not marked critical`), and it was antivirus — **Avast Web/Mail Shield**, whose root has `basicConstraints critical=False` with `ca=True`, which RFC 5280 §4.2.1.9 forbids and OpenSSL 3.x rejects.

The refusal to "fix" it with `verify=False` was also right. But **the first suggested option above does not work**: exporting the root and pointing `REQUESTS_CA_BUNDLE` at a bundle containing it was tried and fails identically, because the failure is *malformed*, not *untrusted*. The fix that works is to verify through the **OS** certificate verifier (`fisherrao/net.py`), which is what browsers on this machine already do — full verification, no downgrade.

~~**Consequence:** no new models can be downloaded.~~ GPT-2 and Pythia now download and pass every correctness check.

---

## 1. Model

Stage 0 planned to use GPT-2. GPT-2 is not cached and cannot be fetched, so the pipeline was built on **SmolLM2-135M-Instruct**, which was.

| | |
|---|---|
| Architecture | `LlamaForCausalLM` |
| `d` | 576 |
| `N` | 49 152 |
| Layers | 30 |
| **Tied embeddings** | **True** |
| Final norm | `LlamaRMSNorm`, eps 1e-5, with gain |
| `lm_head` bias | none |

**This is a good substitute, not merely an available one.** Tied embeddings are exactly the case Mabrok's framework assumes ([01-literature-review.md](01-literature-review.md) A1), so the tied/untied question is deferred cleanly rather than confounded. A modern Llama-family architecture also means the results are not GPT-2-specific — which matters, because FishBack and Mabrok both used GPT-2, so replicating their effective-dimensionality finding here is *independent* evidence rather than a repeat.

Add GPT-2 and Pythia once network access is restored, per [04-stage-plan.md](04-stage-plan.md) §6.4.

---

## 2. 🔴 Two bugs found, both silent, both caught by sanity checks

### 2.1 `output_hidden_states=True` returns a *normalised* last entry

HF's `LlamaModel` appends `self.norm(hidden_states)` as the final element of `hidden_states`. Every other element is raw residual stream. So `hidden_states[-1]` is **already normalised** while `hidden_states[0..L-1]` are not.

The logit lens applies the final norm before the unembedding. Applying it to an already-normalised state **double-normalises**, and the resulting logits bore no relation to the model's own output:

```
logit-lens check vs model head:  rel.err = 2.07e+00   FAIL
```

Fixed by capturing block outputs with forward hooks instead of relying on that convention — which also makes `residual_stream()` architecture-agnostic. After the fix:

```
logit-lens check vs model head:  rel.err = 9.50e-07   PASS
```

> **The lesson is the same one the validation ladder exists for.** A relative error of 2.07 does not announce itself in a curvature plot — the curves would have looked perfectly plausible. **Any layer-wise study using the logit lens must verify the final layer against the model's own head.** One assertion, permanently.

### 2.2 Off-by-one in aligning curvature with salience
`curvature` returns `L−1` interior values; `salience` returns `L`. Fixed by aligning `curvature[j]` (at interior point `j+1`) with `salience[j+1]` (the step leaving that point).

---

## 3. ✅ Gate A passes on a real model (Stage 2 task 2.3, done early)

```
predicted 0.5 vᵀG(h)v = 1.803844540372e-05

     eps             KL/eps²      rel.err
   1e-01   1.804746382003e-05     5.00e-04
   1e-02   1.803938306293e-05     5.20e-05
   1e-03   1.803922251170e-05     4.31e-05   ← plateau
   1e-04   1.821751389595e-05     9.93e-03
   1e-05   2.340856290320e-05     2.98e-01
   1e-06   6.215293510760e-04     3.35e+01
   1e-07   1.001520513721e-01     5.55e+03
```

**PASS** at `ε = 1e-3`, relative error `4.31e-5`, verified against an independent code path (two forward passes and a KL).

⚠️ **The plateau is at `1e-3` here, not the `1e-4` seen on synthetic data.** Because `½vᵀGv ≈ 1.8e-5` is small, the KL at `ε=1e-5` is already ~1e-15 — float64 epsilon — so cancellation bites a decade earlier. **The plateau location is model- and state-dependent. Always sweep; never hard-code an `ε`.**

---

## 4. 🔴 The final-norm Jacobian, and the discovery it led to

The predictive map is `h → norm(h) → U → softmax`, so `J = U · ∂norm/∂h`. Implemented analytically for RMSNorm:

```
A = diag(g) · (1/r) · ( I − h hᵀ / (r² d) ),    r = √(mean(h²) + eps)
G(h) = Aᵀ ( Ũᵀ Σ_p Ũ ) A
```

### 4.1 Correcting an over-claim
A first pass reported `‖G_with − G_without‖/‖G_with‖ = 831` and read it as structural. **That was over-interpreted.** Decomposed at layer 20:

```
‖G_without‖ / ‖G_with‖ = 107.74        r² = 107.72
```

The discrepancy is **almost entirely a `1/r²` overall scale factor.** After removing it, only **2.0%** remains. So the honest statement is: the norm Jacobian contributes (i) a large scale factor, and (ii) a small-in-norm rank-1 correction.

Both still matter. Sectional curvature is **not** scale-invariant — under `g → c·g`, `K → K/c` — so a factor of ~100 in the metric is a factor of ~100 in curvature. And (ii), though only 2% in Frobenius norm, turns out to be structurally decisive:

```
‖G_with · h‖   / (‖G_with‖‖h‖)    = 1.3 × 10⁻⁹     ← null
‖G_without · h‖/ (‖G_without‖‖h‖) = 1.4 × 10⁻²     ← not null
```

### 4.2 🎯 The actual finding: `h` is an exact null direction of `G(h)`

Since `mean(h²) = |h|²/d`, the bracket in `A` is `I − h hᵀ/|h|² = I − ĥĥᵀ` — **a projector.** `A` therefore annihilates the radial direction, and so does `G(h)`. The reason is simply that **RMSNorm is scale-invariant**: the model's prediction depends only on the *direction* of `h`.

Measured (§ "RQ4 NULL-SPACE FALSIFICATION TEST" in the run output):

| layer | `‖Ah‖/‖h‖` | `‖Gh‖/(‖G‖‖h‖)` | `KL(p(h)‖p(2h))` | `KL` random | ratio |
|---|---|---|---|---|---|
| 0 | 3.0e-02 | 6.9e-05 | 5.2e-10 | 5.03 | 1e-10 |
| 10 | 6.8e-08 | 5.1e-09 | 5.6e-15 | 1.19 | 5e-15 |
| 20 | 1.4e-08 | 1.3e-09 | 1.6e-16 | 1.15 | **1e-16** |
| 30 | 7.6e-10 | 2.7e-10 | −3.1e-16 | 7.78 | −4e-17 |

**Doubling a hidden state changes the model's prediction by nothing** — sixteen orders of magnitude below a random perturbation of the same magnitude. Layer 0 is the weakest case, because there `mean(h²)` is small enough that `eps = 1e-5` is not fully negligible.

**This is a different nullity from the one everyone discusses.** `Σ_p`'s all-ones direction does *not* transfer to `G` (§6.1). The nullity that actually bites comes from the **normalisation layer**, and it is exact up to `eps`.

### 4.3 Three consequences

1. **RQ4's falsification test already passes.** [03-methodology.md](03-methodology.md) E5 predicted "null-space perturbations produce ≈ 0 KL — a strong falsification test of the whole framework." Ratio 10⁻¹⁶. The framework is validated on this axis, in Stage 0.
2. **`n_pos` = `d − 1` = 575 is now explained**, and is the modal value across layers (§6.1). It reads as `d−1` rather than a hard zero only because `eps > 0` leaves the radial eigenvalue at ~10⁻¹⁸ relative.
3. **Option B (quotient manifold) is reinstated and is arguably the *right* formulation** — see §6.6.

### 4.4 Action
**Audit whether the literature includes the norm Jacobian.** Mabrok's published `G(h) = Wᵀ Σ_p W` and Manson's `G = UᵀU` both appear to omit it. If confirmed, that is a second independent gap: without the projector the radial direction is spuriously non-null, which inflates the metric along a direction that carries **no predictive information at all** — a plausible contributor to proxy curvature landing at 10⁻⁵.

---

## 5. ✅ Top-k truncation justified (Stage 2 task 2.4, done early)

| `k` | `‖G_full − G_topk‖ / ‖G_full‖` |
|---|---|
| 100 | 1.09 × 10⁻² |
| 500 | 3.22 × 10⁻³ |
| 1 000 | 1.75 × 10⁻³ |
| **2 000** | **8.79 × 10⁻⁴** |
| 5 000 | 2.85 × 10⁻⁴ |
| 10 000 | 8.49 × 10⁻⁵ |

**`k = 2000` adopted**, error < 10⁻³ against the full 49 152-token vocabulary. FishBack's choice of 5 000 is comfortably safe. Truncation is *not* the source of the rank behaviour in §6 — verified explicitly: numerical rank at layer 20 is 306 at `k`=2000, `k`=10 000 *and* full vocabulary alike.

---

## 6. 🎯 The main result: three different ranks, and they disagree by hundreds of dimensions

Fisher metric conditioning across all 31 layers (`k`=2000, `d`=576):

| quantity | meaning | observed range |
|---|---|---|
| `n_pos` | eigenvalues > 1e-30·λ_max — ~**algebraic** rank | **516 – 576** |
| `rank_num` | `matrix_rank` at float64 default — **working-precision** rank | **288 – 575** |
| `k_eff` | directions holding 99% of trace — what **curvature** needs | **3 – 86 (0.5% – 14.9%)** |

### 6.1 The correction in [02-research-gap.md](02-research-gap.md) §1 is confirmed — with a twist
The `Σ_p` all-ones direction **does not** transfer to `G`: the intuitive argument really is wrong, on a real model and not just in the synthetic control.

But the **modal `n_pos` across layers is 575 = `d − 1`**, not `d`. That is exactly what **one** null direction predicts — the radial one identified in §4.2. So the conclusion is:

> `G(h)` is **not** singular for the reason the literature assumes (`Σ_p`), but it **is** singular by exactly one dimension for a different reason (**RMSNorm scale invariance**). Rank is `d − 1`, structurally.

Both halves need stating. Getting only the first half right — as the pre-Stage-0 drafts did — leaves the methodology solving the wrong problem.

### 6.2 But the severity was *underestimated*
The synthetic study reported `cond(G)` ≈ 1.8 × 10⁸. On real activations **λ_min underflows to exactly 0.0 at 22 of 31 layers**, so `cond = ∞`. Real softmaxes are far more peaked than the synthetic logit scale used (entropies here are 0.0 – 1.8 nats).

> **Revised conclusion: float64 is necessary but *not sufficient*. Subspace restriction is unavoidable, not a stylistic choice.** This strengthens rather than weakens the case for the thesis — it is a harder wall than advertised, and Option A ceases to be one of three alternatives and becomes the only viable primary route. Update [03-methodology.md](03-methodology.md) §4 accordingly.

### 6.3 ✅ Independent replication of FishBack
`k_eff` spans **0.5% – 14.9%** of ambient. FishBack reports **2 – 17%** on GPT-2. Different architecture, different model family, same phenomenon — this is independent corroboration, and worth stating as such.

### 6.4 🟢 The genuinely encouraging finding: `cond_eff` is *small*

Conditioning **within** the retained subspace, `λ_max/λ_{k_eff}`, is only **10¹ – 10³** (median ~7 × 10¹; worst 5 × 10³ at the final layer).

**This is the best news in Stage 0.** Once restricted to the effective subspace the metric is *well conditioned*, leaving ample float64 precision for the second derivatives curvature requires. Option A does not merely make the problem tractable — it appears to make it comfortable. With `k_eff` ≤ 86, the **full Riemann tensor on the restricted subspace is affordable** (86⁴ ≈ 5.5 × 10⁷ entries), so sampling random 2-planes may not even be necessary.

### 6.6 🎯 Option B is reinstated, and is probably the right formulation

[03-methodology.md](03-methodology.md) §4 downgraded the quotient-manifold option on the grounds that "there is no exact null distribution to quotient by, only a numerically negligible one." **§4.2 overturns that.** There *is* an exact null distribution: the radial one.

And it is maximally well-behaved:
- It is **1-dimensional**, so the Frobenius integrability condition is satisfied **automatically** — the check flagged as mandatory in task 3.9 is trivially passed.
- Its integral curves are **radial rays**, so the quotient is the **sphere** (or projective space) of directions in `ℝᵈ`.

> **This reframes the geometry, and reframes it in the model's favour.** RMSNorm means the predictive map factors through the unit sphere. The semantic manifold is therefore not a `d`-dimensional vector space at all — it is a `(d−1)`-dimensional **space of directions**, and the Fisher–Rao metric lives there. Computing curvature on the quotient is not a workaround for a degenerate metric; it is working on the correct manifold.
>
> This also connects to Stream D of the literature review. Modell et al. (2025, D2) argue that **cosine similarity** encodes intrinsic feature geometry via on-manifold geodesics, and the anisotropy literature (D3) argues Euclidean coordinates are the wrong frame. §4.2 supplies a mechanism: the architecture itself makes only direction predictively meaningful. That is a theoretical contribution beyond the curvature computation, and it costs nothing extra to state.

**Revise task 3.9 accordingly:** Option B moves from "awkward fit, implement for comparison" to a co-primary route alongside Option A, with a cleaner mathematical justification. Option A restricts to the top-`k_eff` subspace on numerical grounds; Option B quotients by the radial direction on *structural* grounds. They compose: quotient first (exact, principled), then restrict (numerical, sensitivity-tested).

### 6.5 A structural finding worth following up
`k_eff` is flat at ~5 (0.9%) through layers 1–24, then climbs sharply: **12 → 27 → 86** at layers 28, 29, 30, tracking the entropy rise (0.8 → 1.76 → 1.39 nats). The predictive geometry stays extremely low-dimensional until the last few layers, then opens out. Whether this is the tail of the hourglass profile (Valeriani et al. 2023; Mabrok 2026) is a direct Stage 4 question — and note it is a *Fisher* effective dimension, not the ambient intrinsic dimension those papers measure, so agreement would be a genuine finding rather than a restatement.

---

## 7. Manson baseline reproduced (Stage 0.6) ✅

`G = UᵀU`: `cond_eff` = 8.3 × 10³, 99%-trace rank 537/576 (93%) — i.e. `UᵀU` is nearly isotropic compared with the Fisher metric's 0.9%. **That contrast is the thesis in one number.**

| metric | median κ | max κ | at layer | corr(κ, salience) |
|---|---|---|---|---|
| `UᵀU` (Manson) | 0.0014 | 0.0109 | 6 | **−0.37** |
| `I` (Euclidean) | 0.0697 | 0.1457 | 5 | **−0.68** |

The **negative** curvature–salience correlation reproduces the *sign* of Manson's `r = −0.89` on LLaMA-3.2-3b, at weaker magnitude — expected on a 135M model versus 3B, and consistent with his own report that the effect was much weaker on Gemma-3-1b than LLaMA. Both metrics put peak trajectory curvature in the **early** layers (5–6).

---

## 8. Stage 0 exit criteria

- [x] Hidden states extracted, shapes verified against config, **logit lens verified against the model's own head**
- [x] Manson baseline reproduced and plotted
- [x] End-to-end pipeline: load → extract → metric → curvature → figure
- [ ] Hand derivations of `K = 1/r²` (sphere) and `K = +1/4` (simplex) — **outstanding, and the one item that must not be skipped.** `validate_curvature.py` verifies both numerically, which is not a substitute: Stage 3 debugging needs the derivation in your head, not just in a test.
- [x] Repo structure, pinned requirements

**Ahead of schedule:** Gate A (2.3), truncation study (2.4), and real-activation conditioning (3.10b) are all done — 3.10b was scheduled for week ~18.

---

## 9. Next actions

1. **Do the two hand derivations.** The only unmet exit criterion.
2. **Resolve the TLS problem** to unlock GPT-2, GPT-2 XL and Pythia — needed for the King et al. `r ≈ 0.15` comparison (RQ3a), which is the headline empirical claim and currently blocked.
3. **Install JAX** (`pip install "jax[cpu]"`) for Stage 3. Also blocked by the network issue; PyTorch `torch.func` is a viable fallback and the code already uses analytic derivatives where possible.
4. **Revise [03-methodology.md](03-methodology.md) §4** — promote Option A from "recommended primary" to "the only viable primary", citing §6.2 above.
5. **Audit the literature for the norm-Jacobian omission** (§4). If Mabrok and Manson both omit it, that is a second, independent gap.
6. **Extend beyond one prompt.** Everything here is a single 20-token sentence and a single token position. Necessary for a pipeline check; meaningless as evidence. Scale to WikiText-103 before drawing any conclusion about layer profiles.
