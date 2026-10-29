# 14 — Revision experiments, ICARC 2027

**Started:** 29 October 2026 · **Plan:** [13-revision-plan.md](13-revision-plan.md)

**Scope, agreed 29 October 2026.** The thesis topic is unchanged; what changes is what the paper claims. Required: the consistent control on SmolLM2-135M at the paper's own sample with `k` = 4–6 (E1), and a precise treatment of slice against manifold curvature (E4). Added: the consistent control on LLaMA-160M, which needs no new code. Only if time allows: the LayerNorm models (GPT-2, Pythia-160M) and the untrained-model control (E2).

---

## 1. 🔴 The paper's n = 221 counts 35 states twice

`results/scramble/within_points.jsonl` holds 221 rows but **186 distinct states** (`_i`). The 35 extra rows recur across checkpoint resumes and are identical to their first occurrence in `K_real`, `K_within` and entropy. [12-audit-log.md](12-audit-log.md) §3 noted the recurrence but recommended quoting n = 221, which counts the duplicates as independent observations in the sign test.

| | n | `K_real` | `K_within` | real > within | z |
|---|---|---|---|---|---|
| as quoted (rows) | 221 | 0.2546 | 0.0109 | 201/221 | +12.18 |
| **distinct states** | **186** | **0.2547** | **0.0096** | **168/186** | **+11.00** |

The direction and size of the `within` effect are unchanged, but every figure derived from these rows — the abstract, Section VI.B and Tables III, V and VI of the paper — must be recomputed on the 186 distinct states. The E1 runs below use the distinct states only.

---

## 2. Slice against manifold curvature (E4, Reviewer 3 (b))

**The question.** Every Riemann-derived number in this project is the sectional curvature of a `k`-dimensional linear slice `S = h0 + span(F)` under the induced metric. By the Gauss equation, for orthonormal `u, v` in `S`,

```
K_S(u, v) = K_M(u, v) + <II(u,u), II(v,v)> − |II(u,v)|²
```

where `M` is the full quotient manifold and `II` the second fundamental form of `S` in `M`. On a real model `K_M` is unknown, so the extrinsic term cannot be read off directly.

**The check** ([check_slice_gauss.py](check_slice_gauss.py), [results/slice/slice_gauss.json](results/slice/slice_gauss.json)). On ladder rung 7 — `p(h) = softmax(Uh)` with a random `U` and `d = N − 1` — `M` is the simplex and `K_M = +1/4` exactly, so `K_S − 1/4` *is* the extrinsic term. Two frames: the transformer pipeline's (top-`k` eigenvectors of `G(h0)`) and random orthonormal ones. 20 base points per scale, 16 planes each, curvature through `fisherrao.curvature`.

**N = 9, d = 8** — median `|K_S − 1/4|`; at `k = d` the slice is the manifold:

| k | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| eigen | 0.125 | 0.111 | 0.092 | 0.048 | 0.026 | 0.009 | **0** |
| random | 0.196 | 0.142 | 0.098 | 0.057 | 0.051 | 0.017 | **0** |

**N = 512, d = 511, k = 5** — the transformer runs' retained-vocabulary size, by predictive entropy:

| entropy (nats) | n | eigen `K_S` | eigen \|ext\| | random `K_S` | random \|ext\| |
|---|---|---|---|---|---|
| [0, 0.5) | 21 | +0.450 | 0.200 | −1.288 | 2.311 |
| [0.5, 1) | 9 | +0.255 | **0.005** | +0.209 | 0.041 |
| [1, 2) | 13 | +0.264 | **0.014** | +0.146 | 0.104 |
| [2, 4) | 20 | +0.278 | **0.028** | +0.091 | 0.159 |
| [4, ∞) | 37 | +0.222 | 0.120 | +0.005 | 0.245 |

Eigen frames, all entropies: median `K_S` = 0.281, 0.273, 0.266, 0.269, 0.260 at `k` = 2–6. (True medians across points; the first version of this section used `torch.median` — §8.)

**Reading.**

1. **The extrinsic term is real, and in general not small.** Random 5-dimensional slices of a manifold with `K_M = 1/4` report anything from −1.29 to +0.21 depending on entropy.
2. **The pipeline's eigen-frame slices sit much closer to the manifold** — within 0.005–0.028 of 1/4 between 0.5 and 4 nats. At the extremes the term grows: 0.20 below 0.5 nats and 0.12 above 4.
3. **The check passes where it must.** At `k = d` the slice is the manifold and the term is exactly 0 for both frames.
4. **A random `U` already gives the trained models' value.** With no learning at all, eigen-frame slices report `K` ≈ 0.25–0.28 across 0.5–4 nats, as SmolLM2 and LLaMA do. That agrees with the consistent control: `+1/4` comes from the simplex and the slicing, not from learning. The one exception is below 0.5 nats, where the synthetic gives 0.45 against about 0.28 for the trained models.

**For the paper.** Name the quantity *slice sectional curvature*, state the Gauss equation, report this check, and rest every comparison on like-for-like slices (a model and its controls use the same construction). Absolute values below 0.5 or above 4 nats carry an extrinsic term of order 0.1–0.2 and should not be read as manifold curvature. The paragraph is now in Section III.C of [the paper](paper/fisher-rao-conference-paper.md).

⚠️ Synthetic `U`, no normalisation layer, `d = N − 1`. This sizes the term where the answer is known; it does not measure it on the transformers.

---

## 3. Is each control a valid geometric object? (Reviewer 3 (a))

The reviewer's three words were *local probability map*, *derivative construction* and *conditioning invariance*. Each is tested directly ([check_control_validity.py](check_control_validity.py), [results/scramble/control_validity.json](results/scramble/control_validity.json); SmolLM2-135M, the 30 pilot states, `k` = 5).

### 3.1 Probability map and derivative construction — Gate A off the base point

A Fisher pullback `G` of a map `x → p(x)` is the Hessian of that map's KL divergence, `KL(p(x₁) ‖ p(x₁ + εv)) = ½ ε² vᵀG(x₁)v + O(ε³)`, so the ratio of the two sides should be 1 up to O(ε) (here ε = 10⁻⁴ ‖h₀‖). Tested at the base point and at `x₁`, a step of 5 % of ‖h₀‖ along the slice:

| metric | against the KL of | median \|ratio − 1\| at `h₀` | at `x₁` |
|---|---|---|---|
| real | `p_real` | 1.0e−3 | 8.6e−4 |
| consistent | `p_cons` | 1.1e−3 | 1.1e−3 |
| `within` (the paper's) | `p_real` | 0.86 | 0.93 |
| `within` (the paper's) | `p_cons` | 1.1e−3 | **0.66** |

The real and consistent metrics are the KL Hessians of their own maps at every point tested. The paper's `within` metric agrees with the consistent map at the base point only — where the two are identical by construction — and with no map once it leaves it (`p_real` and `p_cons` themselves differ there by a median KL of 0.68 nats). Its derivatives, which are what curvature is built from, belong to no probability map. **This is the defect the reviewer named, now measured.**

### 3.2 The RMSNorm Jacobian

`fisher_metric_projected` uses the projector `I − ĥĥᵀ`, which drops `eps`; the exact Jacobian is `diag(g)/r · (I − hhᵀ/(d r²))`. The exact form matches autodiff to 4.8e−16. The closed form differs from it by a median 2.6e−10 (relative), and recomputing `K` with the exact Jacobian changes it by a median 1.5e−8 for the real metric and by 0 for the consistent one, on 6 states. The consistent control is a pullback to that precision.

### 3.3 Conditioning

The paper states that the scramble holds "the conditioning of the retained subspace" fixed exactly. **It does not.** `λ₁/λ₅` of the top-5 block of `PGP` changes from point to point (for example 13.1 → 7.0 and 7.5 → 3.3), because permuting which probability sits on which row changes the spectrum of `UᵀΣ_pU` even though the row multiset is fixed. It is not shifted systematically: median 15.6 real against 15.1 scrambled, scrambled higher at 14/30, z = −0.37.

**For the paper.** Section V must (1) present the consistent control and say why the `within` construction is not a pullback, with the Gate A table above as the evidence, and (2) replace "held fixed exactly" for conditioning with the measured medians. [08-rq3a-log.md](08-rq3a-log.md) §5.4(d)'s conditioning-matched comparison stays as the supplementary check.

---

## 4. E1 — the consistent control on the paper's own sample (SmolLM2-135M)

The 186 distinct states behind the paper's Section VI.B, with the same frames, planes and per-point scrambles as `run_scramble_within.py` ([run_scramble_consistent.py](run_scramble_consistent.py) `--points paper --ksweep`; [results/scramble/consistent_SmolLM2-135M-Instruct_paper.json](results/scramble/consistent_SmolLM2-135M-Instruct_paper.json)). `real` and `within` reproduce the saved values (largest |ΔK| 3e−6 and 2.5e−4; the second at ill-conditioned low-entropy states where `K_within` runs from about −1 to −270).

| `k` | n | real | `within` | consistent | real > `within` | real > consistent |
|---|---|---|---|---|---|---|
| 4 | 60 | 0.2554 | 0.0110 | 0.2590 | 55/60, z = +6.45 | 22/60, z = −2.07 |
| **5** | **186** | **0.2547** | **0.0096** | **0.2565** | **168/186, z = +11.00** | **72/186, z = −3.08** |
| 6 | 60 | 0.2544 | 0.0304 | 0.2560 | 51/60, z = +5.42 | 26/60, z = −1.03 |

> 🔴 **Confirmed on the paper's own sample: the collapse does not survive a valid control.** Realised as a genuine pullback, the scramble leaves `K` at 0.2565 against the real model's 0.2547. If anything the trained model sits slightly *below* the structure-free control (real higher at only 72/186, z = −3.08), not above it.

By entropy, `k` = 5:

| entropy (nats) | n | real | `within` | consistent | real > consistent |
|---|---|---|---|---|---|
| [0, 0.5) | 33 | 0.2789 | −1.2485 | 0.3427 | 10/33, z = −2.26 |
| [0.5, 1) | 31 | 0.2570 | −0.1165 | 0.2587 | 12/31, z = −1.26 |
| [1, 2) | 60 | 0.2552 | +0.0102 | 0.2570 | 27/60, z = −0.77 |
| [2, 4) | 30 | 0.2529 | +0.1199 | 0.2482 | 14/30, z = −0.37 |
| [4, ∞) | 32 | 0.2107 | +0.1364 | 0.2379 | 9/32, z = −2.47 |

**Reading.**

1. Between 0.5 and 4 nats — 121 of the 186 states, and the band where §2 found slice curvature within 0.03 of manifold curvature — real and consistent are indistinguishable (|z| ≤ 1.26).
2. The residual difference sits at the two extremes, where the trained model is *lower* than the control: 0.279 against 0.343 below 0.5 nats, 0.211 against 0.238 above 4. Those are exactly the bands where §2 measured the largest extrinsic slice term (0.20 and 0.12), and on the 60 states swept in `k` the overall difference reaches significance at `k` = 4 only (z = −2.07, −0.77, −1.03 at `k` = 4, 5, 6). It cannot be read as a property of the manifold.
3. By layer the picture is the same. No layer has the real model above the control (largest z = +0.45, layer 10); four of nine have it slightly below (z between −2.06 and −2.24, layers 1, 5, 25 and 28).

**What the paper can now say.** `K ≈ +1/4` is not produced by learning: a structure-free pullback with the same predictive distribution, row multiset and slice construction gives the same median. The separation claim of the submitted paper is withdrawn. Its spectral half — log-volume and effective dimension track predictive concentration — is unaffected, because both are functions of `G(h₀)`, where the two scrambles coincide (§3, [08-rq3a-log.md](08-rq3a-log.md) §5.7).
