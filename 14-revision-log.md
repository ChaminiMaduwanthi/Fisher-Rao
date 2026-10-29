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

---

## 5. E1 — the consistent control on a second model (LLaMA-160M)

Entropy-stratified draw, 24 states per band, nine relative depths ([results/scramble/consistent_llama-160m_draw.json](results/scramble/consistent_llama-160m_draw.json)). Two of 120 states fail — the pseudo-inverse in `christoffel()` does not converge at entropy 10.4 nats — and are excluded.

| `k` | n | real | `within` | consistent | real > `within` | real > consistent |
|---|---|---|---|---|---|---|
| 4 | 30 | 0.2592 | -3.0151 | 0.3375 | 30/30, z = +5.48 | 7/30, z = -2.92 |
| **5** | **118** | **0.2565** | **-0.0056** | **0.2794** | **116/118, z = +10.49** | **33/118, z = -4.79** |
| 6 | 30 | 0.2549 | -8.3461 | 0.3823 | 30/30, z = +5.48 | 5/30, z = -3.65 |

By entropy, `k` = 5:

| entropy (nats) | n | real | `within` | consistent | real > consistent |
|---|---|---|---|---|---|
| [0, 0.5) | 24 | 0.3740 | -8.7943 | 0.4238 | 8/24, z = -1.63 |
| [0.5, 1) | 24 | 0.2564 | -0.3659 | 0.2818 | 5/24, z = -2.86 |
| [1, 2) | 24 | 0.2574 | -0.0056 | 0.2687 | 7/24, z = -2.04 |
| [2, 4) | 24 | 0.2518 | +0.0269 | 0.2699 | 7/24, z = -2.04 |
| [4, ∞) | 22 | 0.2438 | +0.0780 | 0.2694 | 6/22, z = -2.13 |

**Reading.**

1. The paper's scramble collapses on this model too (116/118). The consistent one does not: 0.2794 against the real 0.2565.
2. Unlike SmolLM2, the trained model sits *below* the structure-free control in every band, including 0.5–4 nats where §2 found slice curvature reliable (z between −2.04 and −2.86). The gap is about 0.01–0.03 at `k` = 5 and larger at `k` = 4 and 6.
3. Across both models, then, learning leaves `K` near +1/4. Where it moves it at all, it moves it *down towards* +1/4 relative to a structure-free pullback — the opposite direction to the submitted claim, and of model-dependent size (0.002 on SmolLM2, 0.023 on LLaMA-160M). This is a small, secondary observation, not a replacement headline.

---

## 6. A learning-free pullback, entropy-matched state by state (Gaussian `U`)

[run_gaussian_matched.py](run_gaussian_matched.py) replaces the trained unembedding with one fixed Gaussian matrix and, at each state, scales it until the predictive entropy equals the real model's (largest mismatch 9.4e−14). The map is a genuine pullback with no learned structure at all. Frames, planes and `k` = 5 are the pipeline's; the real `K` comes from §4–§5 on the same states.

| model | n | real | consistent scramble | Gaussian `U` | real > Gaussian |
|---|---|---|---|---|---|
| SmolLM2-135M | 186 | 0.2547 | 0.2565 | 0.2730 | 40/186, z = −7.77 |
| LLaMA-160M | 118 | 0.2565 | 0.2794 | 0.2846 | 35/118, z = −4.42 |

By entropy (real / Gaussian; real > Gaussian):

| entropy (nats) | SmolLM2-135M | LLaMA-160M |
|---|---|---|
| [0, 0.5) | 0.2789 / 0.2556; 17/33, z = +0.17 | 0.3740 / 0.3625; 9/24, z = −1.22 |
| [0.5, 1) | 0.2570 / 0.2755; 8/31, z = −2.69 | 0.2564 / 0.2726; 7/24, z = −2.04 |
| [1, 2) | 0.2552 / 0.2726; 12/60, z = −4.65 | 0.2574 / 0.2782; 6/24, z = −2.45 |
| [2, 4) | 0.2529 / 0.2785; 1/30, z = −5.11 | 0.2518 / 0.2821; 8/24, z = −1.63 |
| [4, ∞) | 0.2107 / 0.2739; 2/32, z = −4.95 | 0.2438 / 0.2780; 5/22, z = −2.56 |

**Reading.**

1. **`K ≈ +1/4` does not need learning.** A pullback through a random unembedding, matched in entropy at every state, gives 0.273 and 0.285, within 0.035 of +1/4, as the consistent scramble does. With §2's synthetic case this settles the first half of RQ2: the value is inherited from the simplex and the slicing.
2. **The trained models are the lowest of the genuine pullbacks.** In both models and against both controls, the trained `K` sits slightly below the structure-free one: by 0.002–0.018 on SmolLM2 and 0.023–0.028 on LLaMA-160M. On SmolLM2, against the Gaussian control, the gap holds in every band from 0.5 nats up (z between −2.69 and −5.11).
3. **A possible reading, not yet a measurement.** In Gauss-equation terms (§2), a `K` closer to +1/4 is what slices with a smaller extrinsic term produce, so the trained slices may sit closer to totally geodesic faces of the simplex than random ones do. §2 measured that term only where the full curvature is known, so this stays a hypothesis for the discussion, not a result.

This replaces the submitted claim with a smaller one of the opposite sign: learning does not create the curvature; if anything it pulls it slightly towards the simplex value.

---

## 7. Duplicated states across all results

[audit_duplicates.py](audit_duplicates.py) scans every checkpoint and recomputes each affected paper number twice: on the rows as stored, which reproduces the submitted paper exactly, and on distinct states ([results/audit/duplicates.json](results/audit/duplicates.json)).

| file | rows | identical copies |
|---|---|---|
| `results/scramble/within_points.jsonl` | 221 | 35 |
| `results/profiles/profile_points.jsonl` | 463 | 31 |
| `results/corpus_compare/points.jsonl` | 702 | 49 |
| `results/negplanes/points.jsonl` | 135 | 48 |
| `results/e5/e5_points_llama-160m_k6.jsonl` | 129 | 1 |

The summary `.json` files written by `run_profiles.py`, `run_corpus_compare.py`, `run_stage4.py`, `run_e5.py` and `check_negative_planes.py` are keyed by state and hold no copies, so the 653 corpus states, the 360 E2 states and the E5 tables are clean. The damage is confined to numbers read straight from raw checkpoints:

| paper number | as submitted (rows) | distinct states |
|---|---|---|
| Section VI.B scramble, `k` = 5 | 0.2546 → 0.0109, 201/221, z = +12.18 | 0.2547 → 0.0096, 168/186, z = +11.00 |
| `k` = 4 | 0.2543 → 0.0137, 63/69, z = +6.86 | 0.2554 → 0.0110, 55/60, z = +6.45 |
| `k` = 6 | 0.2532 → 0.0512, 58/69, z = +5.66 | 0.2544 → 0.0304, 51/60, z = +5.42 |
| Table V, layer 25 retained | −95.5 %, 20/20 | −127.9 %, 19/19 |
| Table VI, [4, ∞) | 0.2064 / +0.1238, 33/42 | 0.2107 / +0.1364, 25/32 |
| Section VI.A and Fig. 1, states | 456 | 425 |
| Pythia-160M median `K` | 0.2738 (130) | 0.2721 (108) |
| GPT-2 median `K` | 0.2599 (117) | 0.2601 (108) |
| spread across the four models | 0.019 | 0.017 |
| negative planes, Section VII | 49/456; `k_eff` 248 vs 122; AUC 0.618, z = +2.69 | 48/425; 250 vs 123; AUC 0.624, z = +2.79 |

No conclusion changes, but every count and z-statistic must be quoted on distinct states. `make_figures.py` Fig. 1 read the raw checkpoint; it now counts each state once.

---

## 8. Tables for the revision, and what rebuilding them exposed

[make_tables.py](make_tables.py) rebuilds every table the revised paper needs from per-state rows, on distinct states, with true medians and midrank Spearman: controls, architectures, instrument agreement, two corpora and behaviour ([results/tables/tables.md](results/tables/tables.md)). Its 30 checks against the stored summaries all agree once two causes are allowed for, and each cause is itself a correction to the submitted paper:

1. **Table VII's medians are lower medians.** The 32-pair medians were taken with `torch.median`, which returns the lower of the two middle values. True medians: Fisher–Rao 0.774 → 1.537 (1.98×, not 2.03×); Euclidean 262.4 → 559.1 (2.13×); `UᵀU` 10 630 → 17 953 (1.69×, not 1.71×). Sign counts and z are unchanged.
2. **The stored correlations predate the midrank Spearman** of [12-audit-log.md](12-audit-log.md) §1. The local-PCA proxy has exact ties, so they move by up to 1e−3; at three decimals only the within-layer `K`–PCA value changes (+0.000 → −0.001).

Three more errors surfaced while tracing the numbers to their sources:

3. **"Pairs involving intrinsic curvature fall at or below 0.144" holds for `K` only.** Scalar curvature `R` reaches −0.273 against the local-PCA proxy and +0.203 against Manson's, pooled, so the largest cross-family |ρ| is 0.273, not 0.248. Within layers every cross-family pair stays at or below 0.112.
4. **The corpus is WikiText-103, not WikiText-2.** `fisherrao/corpus.py` reads `Salesforce/wikitext`, `wikitext-103-raw-v1`, validation split.
5. **The half-separation depths are means** — 4.53, 26.56 and 26.28 layers over 64 pairs. The paper should say "mean".

**And one of mine.** The scripts added for this revision first summarised across states with `torch.median`, the convention the audit had already retired ([12-audit-log.md](12-audit-log.md) §4). They now use `fisherrao.stats.median`, and §2–§4 above and [08-rq3a-log.md](08-rq3a-log.md) §5.7 have been corrected. The changes are in the third or fourth decimal, except in the pilot's six-state bands. Per-state `K` keeps `torch.median` over its 16 planes, the definition behind every saved `K`. `make_figures.py` also prints lower medians to the console; the plotted boxes use matplotlib's true median.

---

## 9. Status at the end of step 1 (29 October 2026)

| task | status | where |
|---|---|---|
| consistent control, SmolLM2 at the paper's sample, `k` = 4–6 | ✅ | §4 |
| consistent control, LLaMA-160M | ✅ | §5 |
| learning-free control, entropy-matched Gaussian `U`, both models | ✅ | §6 |
| slice against manifold curvature | ✅ | §2; paper Section III.C |
| control validity: probability map, derivatives, conditioning | ✅ | §3 |
| duplicated states across all results | ✅ | §7 |
| every paper table rebuilt and checked, 30/30 | ✅ | §8; [results/tables/tables.md](results/tables/tables.md) |
| reproducibility record | ✅ | [15-reproducibility.md](15-reproducibility.md) |
| LayerNorm models, untrained-model control | optional, not run | [13-revision-plan.md](13-revision-plan.md) §5 |

**The answers the paper can now give.** RQ1: yes, and validated. RQ2: `K ≈ +1/4` on every model and every genuine pullback (0.255–0.285); it is inherited from the simplex, and learning shifts it slightly towards +1/4 rather than creating it. RQ3: the spectral summaries track predictive concentration. Seven errors in the submitted paper are listed in [12-audit-log.md](12-audit-log.md) §8.6 for the rewrite.
