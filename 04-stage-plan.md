# Staged Research Plan

**Project:** Information-Geometric Curvature Analysis of Transformer Representations Using Fisher–Rao Metrics
**Date:** 10 August 2026
**Assumed duration:** 48 weeks (~11 months) full-time-equivalent. Scale proportionally if part-time.

---

## Overview

```
STAGE 0  Foundations & environment          Weeks  1–4    ┐
STAGE 1  Literature review & positioning    Weeks  3–8    ┘ overlap
STAGE 2  Metric layer — G(h)                Weeks  7–13
STAGE 3  Curvature layer — CORE             Weeks 12–24   ← highest risk, highest value
STAGE 4  Empirical study                    Weeks 23–33
STAGE 5  Behavioural & causal validation    Weeks 32–39
STAGE 6  Write-up & dissemination           Weeks 36–48
```

**Two hard gates.** Do not pass either without the exit criteria met.

- **Gate A (end of Stage 2, ~week 13):** the KL–Hessian test passes to third order. *If `G(h)` is not verifiably correct, nothing built on it means anything.*
- **Gate B (end of Stage 3, ~week 24):** the validation ladder passes rungs 0–6. *If curvature code is unvalidated, the empirical chapters are worthless and this surfaces at the viva.*

**Critical path:** Stage 2 → Stage 3 → Stage 4. Stages 1 and 6 run in parallel with everything. Stage 5 is compressible; Stage 3 is not.

---

## STAGE 0 — Foundations & Environment
**Weeks 1–4** · *Goal: be able to read the papers and run the models.*

### Tasks
| # | Task | Output |
|---|---|---|
| 0.1 | Differential geometry: manifolds, tangent spaces, metric tensors, Christoffel symbols, Riemann/Ricci tensors, sectional curvature, geodesics | Personal notes; worked derivation of `K = 1/r²` for the sphere **by hand** |
| 0.2 | Information geometry: Fisher information, Fisher–Rao metric, Čencov's theorem, KL–Fisher relation, α-connections | Notes; hand derivation of the simplex `K = +1/4` result |
| 0.3 | Transformer internals: residual stream, unembedding, `W`, hooks for hidden-state extraction | Working script dumping per-layer hidden states from GPT-2 |
| 0.4 | Environment: Python, JAX (x64 enabled), PyTorch + `transformers`, Geomstats, git repo | Pinned `requirements.txt`; repo with CI |
| 0.5 | Read the five priority papers (A1, A2, C1, C3, B1 — see literature review §7) | Annotated summaries |
| 0.6 | **Reproduce Manson's curvature** with `G = UᵀU` on GPT-2 | Working baseline + first curvature heatmap |

### Exit criteria
- [ ] Can derive `K` for the 2-sphere and the categorical simplex without reference
- [ ] Hidden states extracted from ≥ 2 models, shapes verified against config
- [ ] Manson's baseline reproduced and plotted

> **Why 0.6 matters more than it looks.** It is the cheapest possible end-to-end pipeline: load model → extract states → apply a metric → compute a curvature → plot. Every later stage swaps one component of that pipeline. It also produces the *contrast figure* the thesis needs. **Do not skip it to "save time" — it saves time.**

> **On 0.1 and 0.2:** doing these derivations by hand is not academic ritual. Stage 3 debugging requires knowing what the right answer looks like. Students who skip this cannot distinguish a bug from a finding, and that is the most expensive failure mode in this project.

---

## STAGE 1 — Literature Review & Positioning
**Weeks 3–8** (overlaps Stage 0) · *Goal: a defensible, supervisor-approved gap statement.*

### Tasks
| # | Task | Output |
|---|---|---|
| 1.1 | Systematic search: Fisher–Rao + transformers, representation curvature, information geometry + deep learning; forward/backward citation chase from A1, A2, B1, C1 | Search log with queries and dates |
| 1.2 | Read all 🟡 entries in the literature review at full text | Upgrade every entry to 🟢 |
| 1.3 | Obtain HTML/LaTeX of `2605.09887`; verify the rank bound and its proof | Confirmed constraint |
| 1.4 | **Set up arXiv alerts** — `cs.LG`, `cs.CL`, `stat.ML` on "Fisher information", "information geometry", "representation curvature", "pullback metric" | Live alerts |
| 1.5 | Resolve open questions: does A1 require tied embeddings? does C1's index-symmetry simplification apply to the Fisher pullback? | Written answers with justification |
| 1.6 | Write the review chapter (~6–10k words) | **Thesis Chapter 2 draft** |
| 1.7 | Email Mabrok and Manson with specific, short questions | Sent; log any replies |
| 1.8 | Supervisor meeting: present the gap and the four-stream synthesis table | **Signed-off gap statement** |

### Exit criteria
- [x] Gap statement approved by supervisor — **the supervisor proposed this direction**, so R2 (wrong gap) is retired
- [ ] Chapter 2 draft complete
- [ ] arXiv alerts running
- [ ] No unresolved 🟡 entries among priority papers

> **Do 1.4 in week 3, not week 8.** Five directly relevant papers appeared between February and June 2026. Being scooped is a live risk, and alerts cost ten minutes.

---

## STAGE 2 — Metric Layer: computing `G(h)` correctly
**Weeks 7–13** · *Goal: a verified, fast implementation of the Fisher–Rao pullback.* **→ GATE A**

### Tasks
| # | Task | Output |
|---|---|---|
| 2.1 | Implement `G(h) = Wᵀdiag(p)W − (Wᵀp)(Wᵀp)ᵀ` (never materialise `Σ_p`) | `fisher_metric.py` |
| 2.2 | **Unit tests:** symmetry, PSD, `Σ_p1 = 0`, rank bound, reparameterisation invariance | Passing test suite in CI |
| 2.3 | **The KL–Hessian test:** verify `KL(p(h)‖p(h+εv)) ≈ ½ε²vᵀG(h)v + O(ε³)` via an independent code path. ✅ **`gate_a_kl_test.py` already does this** — port it to the real model. Sweep `ε`; expect the plateau near `1e-4` and degradation below `1e-5` from cancellation | Convergence plot vs `ε` — **the single most important early result** |
| 2.4 | Top-k truncation error study: `‖G_full − G_topk‖/‖G_full‖` vs `k` | Justification figure; chosen `k` |
| 2.5 | Spectral characterisation: eigenvalue spectra, effective rank, condition number per layer. Attempt to reproduce FishBack's ">97% deviation from Euclidean" and "2–17% effective dimensionality" | Spectral profile figures; **independent replication of two published numbers** |
| 2.6 | Implement the layer recursion `G^(ℓ) = A_ℓᵀG^(ℓ+1)A_ℓ`; verify against direct computation | Verified propagation |
| 2.7 | Implement the **volume element** `√(det⁺ G)` (product of non-zero eigenvalues) | `volume_element.py` — the Stage-3 insurance policy |
| 2.8 | Compute and cache `G(h)` at scale; benchmark and profile | Cached tensors; timing table |

### 🚪 GATE A exit criteria
- [ ] **KL–Hessian test passes to third order** — non-negotiable
- [ ] All unit tests green in CI
- [ ] Truncation `k` chosen with documented error bound
- [ ] Effective-rank numbers consistent with the published 2–17%
- [ ] Volume element implemented and sanity-checked

> **If 2.5 reproduces FishBack's numbers, that is a strong signal the implementation is right and worth reporting in the thesis as an independent replication.** If it does *not*, stop and find out why before Stage 3 — a discrepancy here is a bug, not a discovery.

---

## STAGE 3 — Curvature Layer: the core contribution
**Weeks 12–24** · *Goal: validated intrinsic curvature of a singular Fisher metric.* **→ GATE B**

**This is the thesis. Budget generously; expect the singularity handling to consume more time than planned. If any stage overruns, take time from Stage 5, never from here.**

### Phase 3A — Curvature on well-conditioned metrics (weeks 12–16)
Build and validate the machinery *before* confronting ill-conditioning. Two hard problems at once is how projects stall.

> **Head start: rungs 2, 3 and 6 already pass.** `validate_curvature.py` is a verified reference implementation (simplex `+0.250001`, Gaussian `−0.500000`, Poincaré `−1.000000`). Task 3.1 is to make it *fast*, not to make it work. Read its sign-convention warning before touching anything — that error already occurred once and was caught only by having a known answer.

| # | Task | Output |
|---|---|---|
| 3.1 | Christoffel symbols via autodiff, contracted form `Γᵏ_ij vⁱvʲ` only; match `validate_curvature.py` to 6 decimal places | `curvature.py` |
| 3.2 | Sectional curvature `K(u,v)` for a given 2-plane; scalar curvature by sampling | Curvature functions |
| 3.3 | **Validation ladder rungs 0–2**: flat ℝ² (`R=0`), 2-sphere (`K=1/r²`), Poincaré half-plane (`K=−1` ✅ done) | Passing tests |
| 3.4 | ✅ **Rungs 2–7**: categorical simplex (`K=+1/4`), Poincaré (`K=−1`), gamma (negative), beta/Dirichlet (negative), Gaussian (`K=−1/2`), pullback (`K=+1/4`) — all pass at machine precision | Passing tests |
| 3.5 | ✅ **Cross-check rungs 2–6 against Geomstats**, allowing for the sign and factor-of-4 convention axes | Agreement table — **3 of 5 families to 1e−14; 2 disagree with identical metrics** (`validate_geomstats.py`) |
| 3.6 | Benchmark contract-first vs explicit-Christoffel; commit to the faster route | Timing decision, recorded |

### Phase 3B — Handling the ill-conditioned metric (weeks 16–21) — *the hard part*

> **Head start:** `conditioning_check.py` already establishes that `G` is generically **full rank** and that the obstacle is conditioning (`cond ≈ 4×10⁸`, 3% of directions holding 99% of trace), not nullity. Do not re-derive this; build on it.

| # | Task | Output |
|---|---|---|
| 3.7 | Implement **Option A** (effective subspace restriction), handling the rotating frame correctly | `conditioning.py` |
| 3.8 | Verify Option A is invariant to arbitrary rotation of the retained basis | Invariance test |
| 3.9 | Implement **Option B** (approximate quotient); check the Frobenius condition, and note it is really Option A with a different `k`-selection rule since there is no exact null distribution | Option B + integrability report |
| 3.10 | Implement **Option C** (Tikhonov) as cross-check only | Option C |
| 3.10b | Extend `conditioning_check.py` to **real GPT-2 activations** — confirm the synthetic spectrum profile transfers, and record `cond(G)` and effective `k` per layer | Real-data conditioning profile |
| 3.11 | **Rung 7**: synthetic tiny `W`, small `N`/`d`, curvature derivable by hand — isolates the pullback + singularity handling together | Hand-verified test case |
| 3.12 | **Sensitivity analysis:** curvature vs `k`, curvature vs `c`. Decide the reporting format *before* seeing transformer results | Sensitivity figures |

### Phase 3C — First real measurements (weeks 21–24)
| # | Task | Output |
|---|---|---|
| 3.13 | **Rung 8**: known 1-D "ripple" manifolds (years, number lines) — first LM data, partially checkable | Qualitative validation |
| 3.14 | Curvature on GPT-2 small, one layer, small sample. **Compare against the +1/4 simplex baseline** | First real curvature number |
| 3.15 | **The null-space falsification test** (E5, cheap subset): perturb along `null(G)` — output KL must be ≈ 0 | Pass/fail — **run this the day the pipeline works** |
| 3.16 | Numerical robustness: float64 confirmed, finite-difference step-size plateau, seed stability | Robustness appendix |

### 🚪 GATE B exit criteria
- [x] **Validation ladder rungs 0–6 pass** — ✅ known-answer arm at machine precision (plus rungs 4, 5, 7). 🟡 Geomstats arm agrees on 3 families of 5; the 2 disagreements have identical metrics and are reported, not patched.
- [ ] Rung 7 hand-verified
- [ ] All three singularity options implemented; sensitivity curves produced
- [ ] Null-space perturbation test passes
- [ ] One real curvature value on GPT-2, with `k` and error bars

### Deliverable
**Workshop paper submission on C1–C3** (method + singularity treatment + validation) to establish priority. Target a NeurIPS/ICML workshop on geometry in neural representations — the venue where C1 appeared. **Do not wait for empirical results to publish the method.**

> **Reality check.** If the ladder does not pass by week 24, the correct move is to descend the ambition ladder: sectional curvature → Ricci scalar only → volume element + anisotropy only. **The volume element (2.7) is numerically far more stable and Zavatone-Veth et al. found it more informative than the Ricci scalar.** A thesis on the Fisher–Rao volume element and anisotropy structure of transformer representations is a real thesis. Decide by week 24; do not drift.

---

## STAGE 4 — Empirical Study
**Weeks 23–33** · *Goal: answer RQ2. Produce the adjudication figure.*

| # | Task | Maps to | Output |
|---|---|---|---|
| 4.1 | **E1** — layer-wise curvature profiles, all models, ~10⁴ positions | RQ2a, 2c | Profile figures with IQR bands |
| 4.2 | Overlay the intrinsic-dimension profile; test the hourglass hypothesis | RQ2a | Combined figure |
| 4.3 | **E2 — the adjudication experiment**: all four instruments on identical data, pairwise correlations | **RQ2b** | ⭐ **Centrepiece figure** |
| 4.4 | Curvature sign analysis: positive, negative, or mixed by direction; compare to the +1/4 simplex and negative Dirichlet references | RQ2d | Sign distribution figure |
| 4.5 | Cross-architecture comparison; resolve tied vs untied embeddings per model | RQ2c | Comparison table |
| 4.6 | Full sensitivity re-run on real data | — | Robustness section |

### Exit criteria
- [ ] Layer-wise profiles for ≥ 3 models
- [ ] **A defensible answer to "was Mabrok's 10⁻⁵ an artefact?"** — either direction is a result
- [ ] E2 correlations with confidence intervals

> **Write Chapter 5 as you go, not after.** The figures are the chapter.

---

## STAGE 5 — Behavioural & Causal Validation
**Weeks 32–39** · *Goal: answer RQ3–RQ4 — show it matters.* **Compressible if Stage 3 overran.**

| # | Task | Maps to | Output |
|---|---|---|---|
| 5.1 | Build the polysemy probe set: ≥ 200 minimal pairs, ≥ 4 domains | RQ3b | Dataset (release it) |
| 5.2 | **E3** — curvature vs next-token entropy, all four instruments, LAMBADA + UD. **Target: beat r ≈ 0.15** | **RQ3a** | ⭐ Correlation table with CIs |
| 5.3 | **E4** — geodesic separation of polysemous variants across layers; locate the disambiguation layer | RQ3b | Layer-localisation figure |
| 5.4 | **E5 full** — interventions along high/low/random/null curvature directions at matched Fisher norm | RQ4 | Causal table |
| 5.5 | *(Optional, only if ahead of schedule)* hallucination-signature analysis | RQ3c | Exploratory section |

### Exit criteria
- [ ] E3 complete with CIs; explicit statement of whether Fisher–Rao beats the Euclidean baseline
- [ ] E5 complete, including the null-space control
- [ ] Probe set documented and released

> **Priority order if time is short: 5.2 > 5.4 > 5.3 > 5.1 > 5.5.** 5.2 is the headline claim; 5.4 is the strongest scientific move (it answers Manson's stated "no behavioural validation" limitation). 5.5 is attractive but is a project in itself — do not start it before week 37.

---

## STAGE 6 — Write-up & Dissemination
**Weeks 36–48** (drafting begins much earlier)

### Thesis structure
| Ch | Title | Drafted during |
|---|---|---|
| 1 | Introduction, motivation, contributions | Stage 5–6 |
| 2 | Background: differential + information geometry; transformers | Stage 0–1 |
| 3 | Literature review & the gap | Stage 1 |
| 4 | Methodology: the Fisher–Rao pullback, singularity handling | Stage 2–3 |
| 5 | **Validation** — the ladder, Geomstats agreement, robustness | Stage 3 |
| 6 | Layer-wise curvature results & instrument comparison | Stage 4 |
| 7 | Behavioural and causal analysis | Stage 5 |
| 8 | Discussion, limitations, future work | Stage 6 |

**Chapter 5 is not optional padding.** It is what makes Chapters 6–7 believable, and it is the first thing an examiner will probe.

### Tasks
- 6.1 Assemble chapters; unify notation across the whole thesis (do this once, deliberately, in a notation table)
- 6.2 Regenerate every figure from pinned scripts and seeds
- 6.3 Release code + probe dataset + validation suite
- 6.4 Conference paper from Stages 3–5 (ICLR / ACL / NeurIPS depending on emphasis)
- 6.5 **Re-run the literature search** — six months will have passed since Stage 1; cite anything new
- 6.6 Viva preparation: rehearse the three questions in §"Anticipated examiner questions"

---

## Milestones

| Week | Milestone | Type |
|---|---|---|
| 4 | Manson baseline reproduced; foundations solid | Checkpoint |
| 8 | Gap statement signed off; Chapter 2 drafted | Supervisor |
| 13 | **GATE A** — `G(h)` verified via KL–Hessian | 🚪 Hard gate |
| 24 | **GATE B** — validation ladder passes; workshop paper submitted | 🚪 Hard gate |
| 33 | Layer-wise profiles complete; adjudication figure done | Deliverable |
| 39 | Behavioural + causal results complete | Deliverable |
| 44 | Full thesis draft to supervisor | Supervisor |
| 48 | Submission + conference paper | Final |

---

## Risk register

| # | Risk | P | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Ill-conditioning defeats curvature computation | Med | High | Three options in parallel (§3.7–3.10); float64 throughout; fall back to volume element + anisotropy (2.7). **Decide by week 24.** |
| R2 | Curvature is genuinely near-zero everywhere | Med | Med | Publishable null if rigorous — it *licenses* flat-metric methods. Contributions C1–C3, C6 are outcome-independent. |
| R3 | Numerical instability swamps the signal | Med | High | float64 throughout; analytic derivatives of `G`; step-size plateau study; validation ladder catches it early |
| R4 | **Scooped** — someone computes this first | Med | Med | arXiv alerts from week 3; workshop paper by week 24; pivot to RQ3/RQ4 where the field is emptiest |
| R5 | Compute insufficient for larger models | Med | Low | GPT-2 small suffices for the core claim and matches two prior papers; scale is a bonus |
| R6 | Mathematical prerequisites underestimated | Med | High | Stage 0 is 4 weeks for a reason; hand derivations are the checkpoint. **Escalate to supervisor early if 0.1/0.2 slip** — this is the most common silent failure |
| R7 | Scope creep into α-connections, multimodal, fine-tuning | High | Med | Scope boundaries are written down in [02-research-gap.md](02-research-gap.md) §5. Re-read them at every supervisor meeting. |
| R8 | Behavioural effects too small to detect | Med | Med | Report CIs, not just point estimates; King et al. already found r ≈ 0.15, so power-analyse the sample size *before* running E3 |

---

## Anticipated examiner questions

Rehearse these; each maps to a section already written.

1. **"Why Fisher–Rao rather than any other metric?"** → Čencov's theorem: it is the *unique* metric invariant under sufficient statistics. Not a preference — a uniqueness result. ([03-methodology.md](03-methodology.md) §1)
2. **"Isn't the Fisher information matrix intractable for LLMs?"** → That is the Fisher over *parameters*. This is the Fisher pulled back onto *hidden states*: `d×d`, closed-form, one forward pass. ([02-research-gap.md](02-research-gap.md) §5)
3. **"How do you know your curvature code is correct?"** → Chapter 5: eight-rung validation ladder with analytically known answers, cross-checked against an independent peer-reviewed implementation. Rungs 2, 3 and 6 pass to six decimal places. A sign-convention error was caught this way and is documented. ([03-methodology.md](03-methodology.md) §5)
4. **"Manson already published curvature of LLM trajectories — what is new?"** → His metric `UᵀU` is constant, hence *identically zero* intrinsic curvature. He measures how the path bends; this measures how the space bends. Different quantities. ([01-literature-review.md](01-literature-review.md) B1)
5. **"Mabrok reports curvature ≈ 10⁻⁵. Isn't the question settled?"** → That is an extrinsic PCA proxy in ambient coordinates; no theorem bounds intrinsic Fisher curvature by it. Testing it is RQ2b, and E2 is the experiment.
6. **"What if you find nothing?"** → §6 of the gap document. The method, the singularity treatment, and the validation suite stand; a rigorous null overturns a working assumption.
