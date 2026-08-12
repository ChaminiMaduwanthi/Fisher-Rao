# Information-Geometric Curvature Analysis of Transformer Representations Using Fisher–Rao Metrics

**Research planning documentation** · Created 10 August 2026

---

## 📂 Documents

| File | Contents | Audience |
|---|---|---|
| [00-SINHALA-GUIDE.md](00-SINHALA-GUIDE.md) | සිංහල මාර්ගෝපදේශය — everything below, explained simply | You |
| [01-literature-review.md](01-literature-review.md) | 20+ papers across 5 research streams; what each does and fails to do | Supervisor, Ch. 2–3 |
| [02-research-gap.md](02-research-gap.md) | The gap, 5 research questions, 6 claimed contributions, honest risk assessment | Supervisor, Ch. 1 |
| [03-methodology.md](03-methodology.md) | The mathematics, the singularity problem, the validation ladder, 5 experiments | Ch. 4–5 |
| [04-stage-plan.md](04-stage-plan.md) | 7 stages over 48 weeks, 2 hard gates, milestones, risk register | Project management |
| [references.bib](references.bib) | BibTeX for all cited work | Writing |
| [05-stage0-log.md](05-stage0-log.md) | Stage 0 results: bugs found, Gate A passed, the radial-null discovery | Progress |
| [06-stage3-log.md](06-stage3-log.md) | **Stage 3: Gate B passed, curvature ≈ +1/4; one claim retracted** | Progress |
| [07-stage4-log.md](07-stage4-log.md) | **Stage 4: E2 — no instrument tracks any other; RQ3b — scale-invariance is what matters; both re-run on WikiText** | Progress |
| [08-rq3a-log.md](08-rq3a-log.md) | **RQ3a: ρ = −0.58, and §5.4 settles it — the learned assignment, not concentration, puts `K` at +1/4** | Progress |
| [09-e5-log.md](09-e5-log.md) | **E5 causal intervention: null directions do nothing on 6 models; the Ricci-predicts-departure claim is a property of untied embeddings (3/3 vs 0/3)** | Progress |
| [10-architecture-log.md](10-architecture-log.md) | **TLS block removed; 6 models pass all four correctness rungs; `K ≈ +1/4` on all of them** | Progress |
| [run_wic.py](run_wic.py) / [run_samesense.py](run_samesense.py) | **The two same-sense controls** — WiC's loose one, and the tight minimal-pair one that reversed it | Stage 4 |
| [run_corpus_compare.py](run_corpus_compare.py) | E2 and RQ3a on WikiText-103 vs the hand-written corpus | Stage 4 |
| [11-mabrok-replication-log.md](11-mabrok-replication-log.md) | **Mabrok's 10⁻⁵ reproduced on his corpus — and his proxy shown to read a unit 3-sphere as flat at that setting** | Progress, Ch. 6 |
| [12-audit-log.md](12-audit-log.md) | 🔴 **Full end-to-end audit — one headline claim withdrawn (ρ = +0.42 was a rank-tie artefact), two restated, everything else reproduced** | Progress, Ch. 5 & 8 |
| [fisherrao/](fisherrao/) | `model.py`, `metrics.py`, `trajectory.py`, `curvature.py`, `corpus.py`, `net.py` | Code |
| [run_stage0.py](run_stage0.py) / [run_stage3.py](run_stage3.py) / [run_control.py](run_control.py) | Reproduce every number in the logs | Code |
| [gate_a_kl_test.py](gate_a_kl_test.py) | **Working** — the Gate A KL–Hessian test, plus the `ε`-cancellation trap | Stage 2 |
| [validate_curvature.py](validate_curvature.py) | **Working** reference implementation — validation-ladder rungs 2, 3, 6 pass | Stage 3 |
| [validate_pullback.py](validate_pullback.py) | **Rung 7** — the pullback assembly against a known answer, with reparameterisation invariance and a negative control | Stage 3 |
| [validate_geomstats.py](validate_geomstats.py) | **Rungs 4, 5 + independent cross-check** vs geomstats 2.8.0 — 3 of 5 families agree to 1e−14; the other two are diagnosed, not patched | Stage 3 |
| [conditioning_check.py](conditioning_check.py) | **Working** — establishes the real obstacle is conditioning, not singularity | Stage 2–3 |
| [check_krank.py](check_krank.py) / [check_keff.py](check_keff.py) / [check_volume.py](check_volume.py) | Sensitivity checks: **retracted** the deviation claim, **validated** the volume element | Stage 3 |
| [run_stage4.py](run_stage4.py) / [run_polysemy.py](run_polysemy.py) | E2 adjudication; RQ3b ambiguity localisation | Stage 4 |
| [run_control.py](run_control.py) / [check_polysemy.py](check_polysemy.py) | Controls: **+1/4 is not a construction artifact**; **narrowed** the RQ3b claim to scale-invariance | Stage 3–4 |

**Start here:** [00-SINHALA-GUIDE.md](00-SINHALA-GUIDE.md) → [02-research-gap.md](02-research-gap.md) → [04-stage-plan.md](04-stage-plan.md) Stage 0.

---

## The one-page summary

### The object

Every transformer hidden state `h` induces a next-token distribution `p(h) = softmax(Wh + b)`. So hidden states are **points on a statistical manifold**, and the canonical way to measure distance between them is the **Fisher–Rao metric** — canonical not by preference but by **Čencov's theorem**, which makes it the *unique* metric invariant under sufficient statistics.

Pulled back to hidden-state space:

```
G(h) = Wᵀ (diag(p) − p pᵀ) W
```

Closed-form. `d × d` where `d` = 768–4096. One forward pass. **Not** the intractable parameter-space Fisher of the K-FAC literature.

### The state of the art

| Work | Has the Fisher metric | Computes real curvature |
|---|---|---|
| Mabrok 2026 (`2603.22301`) | ✅ derives `G(h)` exactly | ❌ PCA proxies only; calls it "intractable" |
| FishBack 2026 (`2605.17231`) | ✅ same metric + layer recursion | ❌ eigenvalues only, explicitly no curvature |
| Manson 2025 (`2507.21107`) | ❌ constant `UᵀU` | ❌ Frenet curvature in a **flat** space |
| King et al. 2026 (`2604.23985`) | ❌ Euclidean angles | ❌ path bending, not space bending |
| Zavatone-Veth et al. 2025 | ❌ generic pullback | ✅ Ricci scalar — but **CNNs on images, not transformers** |

### The gap

> The Fisher–Rao pullback metric on transformer hidden states has been correctly derived, and curvature has been measured under flat metrics. **The genuine intrinsic Riemannian curvature of the state-dependent Fisher–Rao metric has never been computed for a transformer.**

### Why the gap is still open

Christoffel symbols need `G⁻¹`, and `G(h)` will not cooperate — but state the reason carefully, because the intuitive version is wrong.

`Σ_p` annihilates the all-ones direction, which *suggests* `G` is singular. **It isn't — for that reason.** `null(G) = {v : Wv ∈ span{1_N}}` requires `1_N ∈ range(W)`, a `d`-dimensional subspace of `ℝᴺ`, and generically it is not. Two things bite instead, both established on real activations rather than assumed:

**1. Conditioning.** A sharply peaked softmax collapses the spectrum. Measured across 31 layers: **λ_min underflows to exactly 0 at 22 of them**, so `cond(G) = ∞`, and only **0.5–14.9%** of eigendirections hold 99% of the trace — independently replicating FishBack's 2–17%. **float64 is necessary but not sufficient; subspace restriction is unavoidable.** ([05-stage0-log.md](05-stage0-log.md) §6)

**2. RMSNorm scale invariance.** The predictive map is `h → norm(h) → U → softmax`, and RMSNorm's Jacobian is a *projector*: `A = diag(g)(I − ĥĥᵀ)/r`. So **`h` itself is an exact null direction of `G(h)`** — doubling a hidden state changes the prediction by `KL = 1.6 × 10⁻¹⁶`, versus `1.15` for a random perturbation of the same size. The semantic manifold is therefore a space of **directions**, not vectors, and the quotient formulation is the correct one rather than a workaround. ([05-stage0-log.md](05-stage0-log.md) §4.2)

Every prior author stopped at this wall. **Quotient by the radial direction, then restrict to the effective subspace** — that combination gets past it, and is the thesis's core methodological contribution.

> 🟢 The compensating good news: *within* the retained subspace `cond_eff` is only 10¹–10³, leaving ample float64 headroom for the second derivatives curvature needs.

### Why it matters now

Two published results conflict:

- **Mabrok:** proxy curvature ≈ 10⁻⁵, flat, stable across layers.
- **Manson / King et al.:** curvature varies with semantics, depth, and predictive uncertainty.

Both used the wrong instrument — one extrinsic and embedding-dependent, the others with *identically zero* intrinsic curvature. **This thesis adjudicates a live disagreement with the correct instrument.** That is a materially stronger framing than "apply method X to domain Y."

### Feasibility

1. Metric is closed-form — no estimation, no training.
2. Working dimension after effective-subspace restriction may be **under 100**, putting the full Riemann tensor in reach.
3. Machinery exists (Zavatone-Veth for curvature, Geomstats as an independent oracle with analytically known answers).

---

## Plan at a glance

```
STAGE 0  Foundations & environment       Weeks  1–4
STAGE 1  Literature review & positioning Weeks  3–8
STAGE 2  Metric layer — G(h)             Weeks  7–13  → 🚪 GATE A
STAGE 3  Curvature layer — CORE          Weeks 12–24  → 🚪 GATE B
STAGE 4  Empirical study                 Weeks 23–33
STAGE 5  Behavioural & causal validation Weeks 32–39
STAGE 6  Write-up & dissemination        Weeks 36–48
```

**Gate A (wk 13):** `KL(p(h)‖p(h+εv)) ≈ ½ε²vᵀG(h)v` verified to third order via an independent code path.
**Gate B (wk 24):** validation ladder rungs 0–6 pass, agreeing with Geomstats. Then submit the workshop paper.
🟡 **Known-answer arm passed** at machine precision (rungs 2, 3, 6, plus 4, 5 and 7). **Geomstats arm: 3 of 5 families agree to 1e−14; 2 disagree with identical metrics, diagnosed not patched** — see `03-methodology.md` §5.2.

**Critical path is Stage 2 → 3 → 4.** If anything overruns, take time from Stage 5 — never from Stage 3.

---

## Headline falsifiable claims

1. **RQ2b** — Is Mabrok's "essentially flat, 10⁻⁵" conclusion an artefact of the PCA proxy?
2. **RQ3a** — Does Fisher–Rao curvature beat the **r ≈ 0.15** curvature–entropy correlation published by King et al. 2026?
3. **RQ4** — Do perturbations along `null(G)` leave the output unchanged? *(Strong falsification test of the whole framework — run it the day the pipeline works.)*

---

## Three things not to get wrong

1. **Two different Fishers.** Parameter-space Fisher (K-FAC, billions of dims) ≠ this pullback Fisher on hidden states (`d×d`, closed-form). State the distinction in the introduction — it is the most likely examiner misunderstanding.
2. **Validation is not optional.** Curvature bugs are silent and produce plausible numbers. `validate_curvature.py` initially returned `−0.250000` for the simplex — right magnitude, wrong sign, constant across all sample points, which made it look *more* trustworthy. Only a known answer caught it. The 8-rung ladder is Chapter 5.
3. **Don't over-read `+1/4`, and don't over-read the deviation either.** Curvature turned out large, not null, so the original risk (a flat answer) did not materialise. But `+1/4` is the *ambient simplex* value and absolute curvature is a weak discriminator by construction. The obvious next move — measure the deviation `K − 1/4` and `R − k(k−1)/4` — was tried and **failed its sensitivity check**: unstable in `k`, layer ranking not preserved. ([06-stage3-log.md](06-stage3-log.md) §4.2–4.4)

4. **Stability and informativeness are different axes, and they came apart here.** Stage 3 promoted the volume element over Riemann quantities because it is smooth and `k`-robust. It is. But under an identical paired control the volume profile turns out **largely definitional** (ρ = +0.957 per point with a structure-free scramble) while sectional `K` **collapses** (0.2546 → 0.0109, n = 221). The fragile quantity is the one carrying learned structure. Don't let "well-conditioned" stand in for "meaningful". ([08-rq3a-log.md](08-rq3a-log.md) §5.5)

6. **A control that changes two things at once answers neither question.** Three of them in this project so far: a row-shuffle that was provably the identity; a Gaussian `U` that changed spectrum, norms and assignment together; and a scramble that swapped the retained row set as well as the pairing. Each looked clean and each needed a specific arithmetic or measured check to expose. **Before quoting a control, state what it holds fixed and verify that it does.** ([08-rq3a-log.md](08-rq3a-log.md) §5.0, §5.2, §5.4(b′))

7. 🎯 **Everything from the metric's SPECTRUM reads concentration; only CURVATURE carries learned structure.** Under one identical paired control: sectional `K` collapses 0.2546 → 0.0109 (n = 221), while layer-wise log-volume (ρ = **+0.957**) and layer-wise `k_eff` (ρ = **+0.991**) are reproduced almost exactly by a structure-free scramble (both per point, n = 360). **Three quantities, one control, a clean split** — and it is the sharpest form of the thesis's central claim. ([08-rq3a-log.md](08-rq3a-log.md) §5.4–5.5, [10-architecture-log.md](10-architecture-log.md) §3b.2b)

6. **Log-volume carries artifacts that are not geometry.** The `1/r²` residual-norm term (fixed by measuring on the sphere) and its concentration dependence (a structure-free scramble reproduces the whole layer profile at ρ = +0.957 per point). Use it within a model across layers; treat cross-model comparisons as open. ([10-architecture-log.md](10-architecture-log.md) §3b.1, [08-rq3a-log.md](08-rq3a-log.md) §5.5)

5. **`riemann()` cannot be run at the `k` the conditioning rule asks for.** `cond_eff ≤ 10²` selects `k` = 11–16 on real activations; `k` = 16 attempts a **77 GB** allocation and the process dies with no traceback. Three runs were lost to this. The ceiling is a conditioning diagnostic and a valid `k` selector for the volume element — **not** for Riemann-derived quantities, which are capped at `k` ≤ 8.

---

## Status as of 11 August 2026

Both gates are **passed**, the core question has an answer, and the framework has now passed a **causal** test rather than only descriptive ones.

### What changed on 11 August

| | |
|---|---|
| **RQ3a's open question — closed** | At matched entropy, matched conditioning **and an identical direction set**, a paired scramble collapses `K` from **0.2546 → 0.0109** (201/221, z = +12.18), and it holds at `k` = 4, 5 and 6 — the one Riemann-derived result here that is not `k`-fragile. The blocking "frame sensitivity" was **subspace selection**, not a frame effect — rotation invariance holds at 1e−14 in both conditions. ([08-rq3a-log.md](08-rq3a-log.md) §5.4) |
| **…and the sharper form of it** | `K_real` stays at **0.20–0.28 across the whole entropy range**; destroy the assignment and `K` swings from **−2.15** (near-deterministic) to **+0.09** (diffuse). So the learned assignment does not merely add curvature — **it stabilises it at the simplex value across concentration regimes**, and the effect is strongest exactly where §5.3 suspected: the low-entropy tail (12/12 unanimous in all three low bins). ([08-rq3a-log.md](08-rq3a-log.md) §5.3) |
| **…and its control was itself audited** | The scramble was found to swap the retained **row set** (overlap 5/512, ‖U row‖ 2.44→3.10), not just the pairing. The clean within-set control gives a **larger** collapse, so the conclusion strengthened. Applied to the volume profile the same correction pushed it the *other* way (ρ +0.883 → **+0.967** over layer medians; +0.957 per point — [12-audit-log.md](12-audit-log.md) §4). ([08-rq3a-log.md](08-rq3a-log.md) §5.4(b′), §5.5) |
| **RQ4 answered causally, on 3 architectures** | Null perturbations: `KL` ≈ 1e−13 at a step that *doubles* the hidden state, vs 2.6–7.8 for a random step of the same size — **passes on all three models at machine precision**. ([09-e5-log.md](09-e5-log.md)) |
| ⚠️ **…and a companion claim narrowed three times** | *Ricci predicts which direction the output moves at matched Fisher norm* collapsed on two models when `n` went 40 → ~105; a 2×2 then suggested **tied vs untied embeddings**; sweeping `k` broke that too — a *tied* model is significant at `k` = 4, and only **Pythia is significant at every `k` tested**. **Robust on the GPT-NeoX family only.** ([09-e5-log.md](09-e5-log.md) §2.1–2.2b) |
| **The network block is gone** | Avast TLS interception, diagnosed and fixed without weakening verification. **GPT-2 and Pythia now pass all four correctness rungs** — LayerNorm Jacobian at 1e−16, logit lens at 1e−8, and the two-null-direction structure LayerNorm implies. ([10-architecture-log.md](10-architecture-log.md)) |
| **RQ3b strengthened** | Probe set 10 → **64 pairs**, 8 sense domains. The gap **grew**: 21.75 layers, 95% CI [20.45, 22.94], `d_FR` earlier in **64/64** pairs, no subgroup carrying it. Read with §3b.3's correction — it is a gap against the *raw* flat metrics. |
| **A `k`-selection trap, found the hard way** | The `cond_eff` ceiling selects `k` = 11–16 on real activations, where `riemann()` attempts a **77 GB** allocation and the process dies silently. `riemann` now refuses `k` > 8 with an explanation. The ceiling is a conditioning *diagnostic*, not a `k` selector for Riemann quantities. |

| | |
|---|---|
| **Gate A** — `G(h)` verified via KL–Hessian on a real model | ✅ rel.err 4.3e-5 |
| **Gate B** — validation ladder, rungs 2–7 vs known answers | ✅ machine precision |
| …and vs an independent implementation (geomstats 2.8.0) | 🟡 **3 of 5 families to 1e−14**; 2 disagree with *identical metrics* — diagnosed, not patched |
| Frame invariance (task 3.8) | ✅ 2.6e-11 |
| RQ4 null-space falsification | ✅ radial/random KL ratio 1e-16 |
| **RQ1 — can intrinsic curvature be computed?** | ✅ **yes**, ~9 s/point at `k`=6 |
| **RQ2b — do the proxies track intrinsic curvature?** | ✅ **no** — pooled ρ = −0.14, +0.13, +0.14; within-layer +0.00, −0.00, +0.11 (n=360) |
| RQ2b (magnitude) — is Mabrok's 10⁻⁵ wrong by 10⁴? | ✅ **the question is void** — at the threshold that produces 10⁻⁵, the same proxy reports a **unit 3-sphere as flat to 10⁻³¹** ([11-mabrok-replication-log.md](11-mabrok-replication-log.md)) |
| **RQ3b — where do two contextual variants of a token separate?** | ✅ **layer 4–8** (Fisher–Rao) vs **26–28** (raw flat metrics) — a ~20-layer gap, reproduced on **WiC** as well as the hand-written set |
| …and does that read *sense*, or just *context*? | ✅ **both, and cleanly separated.** With lexical overlap equalised by construction, different senses separate **2.03×** further than same senses (30/32, z = +4.95) at **the same depth** (4.69 vs 4.72). **Sense is magnitude; the metric gap is timing.** ([07-stage4-log.md](07-stage4-log.md) §3b.6b) |
| **RQ3a — is the curvature–entropy link definitional?** | ✅ **no** — clean paired scramble at matched entropy collapses `K` **0.2546 → 0.0109**, z = **+12.18** (n=221), stable in `k` |
| **RQ4 — do null perturbations do nothing?** | ✅ **confirmed causally on 6 models** at machine precision. The Ricci-predicts-departure companion claim is **robust on Pythia only** — significant at `k` = 4, 5 and 6; every other model is `k`-dependent or null |
| Hourglass hypothesis (task 4.2) | 🟡 **`k_eff` dips at relative depth 0.40** — matching Valeriani/Mabrok — **but `k_eff` is ρ≈+0.82 with entropy**, so it may be the concentration profile rather than a dimension profile ([10-architecture-log.md](10-architecture-log.md) §3b.2a) |
| Curvature sign (task 4.4) | ✅ 88–94% of points all-positive; the **6–12% with a negative plane are not conditioning failures** and have **2× the effective dimension** ([10-architecture-log.md](10-architecture-log.md) §3b.3a) |
| Cross-architecture (Stage 4 exit criterion) | ✅ **met and exceeded** — **4 architectures**, 3 families, both embedding regimes, 456 points; `K` = 0.255 / 0.256 / 0.260 / 0.272 |
| Curvature sign (task 4.4) | ✅ 88–94% of points have every sampled plane positive. The **6–12% with a negative plane are not conditioning failures** (`cond_eff` 70.9 vs 80.6, `K`–`R` agreement +0.534) and have **2× the effective dimension** at identical entropy ([10-architecture-log.md](10-architecture-log.md) §3b.3a) |
| Hourglass hypothesis (task 4.2) | 🔴 **the dip is real but definitional.** A structure-free scramble at matched entropy reproduces the whole `k_eff` profile — same minimum layer, 99.5% of the range, **ρ = +0.987**. The agreement with Valeriani/Mabrok's 0.3–0.4 depth is then evidence that **both track predictive concentration** ([10-architecture-log.md](10-architecture-log.md) §3b.2b) |

**Headline finding.** Under the correct Fisher–Rao metric the representation manifold is **strongly positively curved — `K ≈ +0.25`, and the median point has every sampled plane positive at every layer** — not "essentially flat" at 10⁻⁵. `+1/4` is the ambient simplex value, and a control with random unembeddings confirms it is *not* an artifact of the construction (diffuse distributions give `K` = −0.02 to +0.14). Curvature tracks the **concentration** of the predictive distribution.

**E2, the adjudication experiment (Stage 4).** All four instruments on identical activations, **n = 360** stratified at 40 points per layer (±0.10 error bars): **no instrument tracks any other.** Every pooled |ρ| ≤ 0.25, every within-layer |ρ| ≤ 0.11. In particular the three proxies do not detectably track intrinsic Fisher–Rao curvature — pooled ρ = −0.14, +0.13, +0.14; within-layer +0.00, −0.00, +0.11. A positive control confirms the null is informative rather than vacuous (`K` vs scalar `R`, same geometry via a different contraction: **ρ = +0.720**). *An n=40 draft had reported that Manson and King agree at +0.685; that did not survive n=360 (+0.248 pooled, +0.063 within-layer) and is retracted — they measure path bending along **different axes**, layers vs tokens.*

**RQ3b — the clearest demonstration that the metric matters.** On **64** polysemy minimal pairs across 8 sense domains, the layer at which the model has *half-resolved* the ambiguity is **layer 4–8** under the Fisher–Rao metric and **layer 26–28** under Euclidean or `UᵀU`. **A 21.75-layer disagreement on identical activations**, 95% CI [20.45, 22.94], `d_FR` earlier in **64/64** pairs, reproduced on WiC. *(An earlier 10-pair draft reported 19 layers; the larger set moved it up, not down.)* The flat metrics are late because residual norm grows to ~3.7×10⁴ by layer 29, so their distances keep rising as the vectors get bigger; Fisher–Rao is invariant to that radial rescaling and reports that the model has effectively decided by mid-network. The Fisher–Rao distance here is closed form — `d = 2·arccos(Σ√(pq))` — so it needs no `k`, no frame, no cutoff, and is immune to every sensitivity problem in §E2.

**The layer-wise volume profile — corrected, then qualified.** An earlier draft reported "monotone contraction to layer 28 then recovery." **That shape was a `1/r²` scale artifact**: regressing ambient log-volume on `log r` gives slope −0.937 against the construction-mandated −1. Measured on the **sphere of directions** (the actual manifold), the profile is **U-shaped with a minimum at layer 20**. `volume_element(on_sphere=True)` is now the default. `K ≈ +1/4` does hold at scale (per-layer medians 0.241–0.264, n=40 each).

🔴 **But the U-shape is largely definitional** ([08-rq3a-log.md](08-rq3a-log.md) §5.5, n=360). A structure-free scramble at exactly matched entropy and an identical direction set puts the minimum at the same layer 20, keeps 95% of the range, and agrees at **ρ = +0.957** per point. **The same control collapses sectional `K` from 0.2546 to 0.0109** (n = 221, z = +12.18). So the two primary quantities came apart: curvature is fragile and carries learned structure; the volume profile is stable and mostly reads predictive concentration. **Stability and informativeness are not the same axis**, and Stage 3's promotion of the volume element on stability grounds needs that qualification.

**A negative result, equally important.** The *deviation* from `+1/4` looked like the model-specific signal (`dR/R` spanning −6% to +148% across layers) but **does not survive its own sensitivity check**: it is unstable in the retained rank `k`, and the layer ranking scrambles (Spearman ρ = +0.20 between `k`=4 and `k`=7). It is reported as not identifiable, per the pre-committed rule in [03-methodology.md](03-methodology.md) §4. The diagnostic points to a concrete fix — **select `k` by a `cond_eff` ceiling, not a trace fraction** — and to the **volume element** as the stable primary quantity.

**Two discoveries that changed the methodology.** (i) `G(h)` is *not* singular for the reason the literature assumes — but RMSNorm's scale invariance makes `h` itself an **exact null direction**, so the semantic manifold is a space of *directions*, and the quotient formulation is the correct one. (ii) The final-norm Jacobian belongs in the pullback; Mabrok and Manson both appear to omit it.

⚠️ **Read the per-result scope, not this line.** `K ≈ +1/4` is now **4 architectures**, 456 points ([10-architecture-log.md](10-architecture-log.md) §3b); E2 and RQ3a are now **2 corpora** ([07-stage4-log.md](07-stage4-log.md) §1.3); RQ4/E5 is **3 architectures** ([09-e5-log.md](09-e5-log.md)). What remains single-source: RQ3b's probe set, and every `n` below ~120.

---

## Immediate next actions

**Not code — and these are now the critical path, because the code is ahead of them:**

- [x] ~~Supervisor sign-off on the gap statement~~ ✅ **not outstanding — the supervisor proposed this direction and has approved it.** Risk R2 (wrong gap) is retired; keep the scope boundaries in [02-research-gap.md](02-research-gap.md) §5 on the agenda at review meetings instead.
- [ ] Set up arXiv alerts (`cs.LG`, `cs.CL`, `stat.ML`) — five relevant papers appeared Feb–Jun 2026
- [ ] **Draft Chapter 5 (validation).** Every number it needs already exists. It is the first thing an examiner probes.
- [ ] Submit the workshop paper — Gate B passed some time ago and the plan says to submit on passing

**Code, in priority order:**

- [x] ~~Layer-wise curvature profiles on **GPT-2 and Pythia**~~ ✅ **done** — `K ≈ +1/4` holds on all three ([10-architecture-log.md](10-architecture-log.md) §3b)
- [x] ~~**Re-run E5 on GPT-2 and Pythia**~~ ✅ **done** — null test passes on all three; the Ricci prediction is significant on two
- [x] ~~**Raise `n` on GPT-2's E5**~~ ✅ **done — it retracted the claim on two of three models** ([09-e5-log.md](09-e5-log.md) §2.1)
- [x] ~~**Is the surviving Pythia effect about untied embeddings?**~~ ✅ **yes — and the volume offset is a separate fact.** An untied Llama has the E5 effect but not the volume offset ([09-e5-log.md](09-e5-log.md) §2.2–2.3)
- [x] ~~**Why is Pythia the volume outlier?**~~ ✅ **there is no outlier** — with six models the highest log-volume is a *tied* GPT-Neo, and "Pythia is the outlier" was a four-model artefact ([10-architecture-log.md](10-architecture-log.md) §3b.2)
- [x] ~~Pull a **standard corpus**~~ ✅ **done for E2 and RQ3a** — both survive on WikiText-103 at n=360, and the intrinsic instruments turn out **corpus-invariant (move 0.03) while the proxies are not (0.10–0.19)** ([07-stage4-log.md](07-stage4-log.md) §1.3). ✅ **RQ3b now checked on WiC too**, and against a purpose-built **same-sense minimal-pair** control that WiC is too loose to provide ([07-stage4-log.md](07-stage4-log.md) §3b.5–3b.6)
- [ ] Raise `n` on the [08-rq3a-log.md](08-rq3a-log.md) §5.4 paired control and break it down by layer — it is load-bearing and rests on 17–19 points
- [ ] Re-run **E5 across architectures**, and sweep `k` — [09-e5-log.md](09-e5-log.md) is `k`=5 only, and `k`-fragility is this project's known failure mode
- [ ] Apply the §5.4 paired control to the **volume profile**, which has not been tested for the same confound
- [ ] Replicate Mabrok's proxy faithfully — or **drop the magnitude claim**; it is currently unsupported
- [ ] Read `arXiv:2603.22301` (Mabrok) and `arXiv:2605.17231` (FishBack) line by line
- [ ] Derive `K = 1/r²` (sphere) and `K = +1/4` (categorical simplex) **by hand**
