# RQ3a — Curvature vs next-token entropy

**Date:** 10 August 2026 · **§5 rewritten and re-verified 11–12 August 2026**
**Status:** ✅ **Both questions answered.** Intrinsic Fisher–Rao curvature tracks entropy far more strongly than any proxy (ρ = −0.58 vs −0.20…+0.28), **and** the relationship is not definitional: at exactly matched entropy, matched conditioning **and an identical direction set**, destroying the learned token→direction assignment collapses `K` from **0.2546 to 0.0109** (201/221 points, z = +12.18), stably across `k` = 4, 5, 6. *(The n = 60 and n = 126 figures quoted below are earlier checkpoints of this same run; all shared rows agree to 1e−12 — [12-audit-log.md](12-audit-log.md) §3.)* The "frame sensitivity" that blocked this conclusion was a misnomer — it was **subspace selection** (§5.4) — and the control itself had a confound that was found on audit and corrected (§5.4 b′).
**Reproduce:** `python run_scramble_within.py 12 <budget> --ksweep` (the decisive one, checkpointed — re-run until full) · `python diag_frame.py 40` · `python run_frame_resolution.py {main,rot,ksweep} 24` · `python run_volume_scramble.py 40` · `python run_tail.py` · `python run_definitional.py`

> **Read §5.0 before quoting anything from §5.** The "shuffled" control used in the first draft is provably vacuous — it is the identity on this metric — and every inference drawn from it has been withdrawn. §2's instrument comparison is unaffected.

---

## 1. How this was found — via a claim that deflated

Stage 4 flagged a "7.2% high-curvature tail" as *"the most promising target identified so far"* for model-specific geometry ([07-stage4-log.md](07-stage4-log.md) §1.1c). `run_tail.py` was written to test cheap, deflationary explanations first.

Its **automated verdict said no cheap covariate explained the tail** (every global Spearman |ρ| ≤ 0.4, the largest being log-volume at −0.382). **That verdict was wrong**, and the median table immediately below it contradicted it:

| covariate | core median | tail median | ratio |
|---|---|---|---|
| **entropy** | 1.3753 | **0.1259** | **0.09** |
| **top-1 probability** | 0.5224 | **0.9757** | 1.87 |

Tail points have **11× lower entropy** and near-certain top-1 predictions. Global rank correlation missed it because the tail is a **threshold effect** at one end of the entropy range, and correlating over the whole range dilutes it away. Binning shows it at once:

| entropy (nats) | n | tail | tail rate | median `K` |
|---|---|---|---|---|
| [0.00, 0.05) | 11 | 8 | **72.7%** | **0.4197** |
| [0.05, 0.20) | 18 | 8 | 44.4% | 0.3332 |
| [0.20, 0.50) | 27 | 6 | 22.2% | 0.2822 |
| [0.50, 1.00) | 66 | 3 | 4.5% | 0.2593 |
| [1.00, 2.00) | 141 | 0 | **0.0%** | 0.2559 |
| [2.00, ∞) | 97 | 1 | 1.0% | 0.2444 |

By top-1 probability the gradient is just as clean: 0% tail in [0.30, 0.60), **75%** in [0.99, 1.01).

> **The "tail" is not idiosyncratic structure. It is one end of a monotone curvature–entropy relationship** — which is a better result, because it is a law rather than an anomaly, and because it is exactly RQ3a.
>
> `run_tail.py` now runs the binned test and prints a warning that its global-correlation line must not be used for the verdict. **A rank correlation over a full range is the wrong instrument for a threshold effect**, and it nearly produced a false negative here.

---

## 2. 🎯 The result — all five instruments, identical points

`n` = 360 (stratified 40/layer), Spearman against next-token entropy.

| instrument | pooled ρ | **within-layer ρ** | 95% CI (pooled) |
|---|---|---|---|
| **intrinsic scalar `R`** | **−0.578** | **−0.454** | [−0.681, −0.474] |
| **intrinsic Fisher–Rao `K`** | **−0.485** | **−0.428** | [−0.588, −0.381] |
| local-PCA residual (Mabrok) | +0.281 | −0.102 | [+0.177, +0.385] |
| Frenet `UᵀU` (Manson) | −0.252 | +0.018 | [−0.362, −0.142] |
| Euclidean angle (King et al.) | −0.204 | −0.131 | [−0.311, −0.097] |

> **On identical activations, intrinsic Fisher–Rao curvature tracks predictive entropy 2–3× more strongly than any of the three published proxies** — and the gap widens under the within-layer control, where the proxies fall to −0.13…+0.02 while the intrinsic quantities hold at −0.43…−0.45.
>
> **Scalar `R` is the best single instrument**, not sectional `K`. It is a full trace rather than a plane sample, so it carries no sampling dependence, and it is the more stable of the two under every check below.

**Sign: negative.** Curvature is *higher* where the prediction is *sharper*. This is the same relationship as the entropy-bin table in §1, expressed as a correlation.

---

## 3. ✅ Robustness

**Not driven by the tail.** Excluding the low-entropy points entirely:

| instrument | all (n=360) | entropy ≥ 0.2 (n=331) | entropy ≥ 0.5 (n=304) |
|---|---|---|---|
| intrinsic scalar `R` | −0.578 | **−0.584** | **−0.600** |
| intrinsic `K` | −0.485 | −0.429 | −0.362 |
| local-PCA | +0.281 | +0.269 | +0.262 |
| Manson | −0.252 | −0.295 | −0.350 |
| King | −0.204 | −0.271 | −0.261 |

All remain significant at ±0.11–0.12. **Scalar `R` actually strengthens** as the tail is removed — so the relationship is general, not an artefact of the most confident points.

---

## 4. ⚠️ The comparison to King et al. is NOT like-for-like

The original plan set the target as *"beat the `r ≈ 0.15` curvature–entropy correlation published by King et al. 2026."* Three reasons that framing must be qualified:

**(a) They report Pearson; the intrinsic quantities have heavy tails that destroy Pearson.**

| instrument | Spearman | Pearson |
|---|---|---|
| intrinsic `K` | −0.485 | **−0.110** |
| intrinsic scalar `R` | −0.578 | **−0.093** |
| King angle | −0.204 | −0.227 |

`K` ranges over [−0.51, 16.7] ([07-stage4-log.md](07-stage4-log.md) §1.1c), so a handful of points dominate any product-moment statistic. **On the like-for-like Pearson comparison the intrinsic measure is weaker than their published 0.15, not stronger.** Spearman is the appropriate statistic for a heavy-tailed quantity, and it must be reported as the primary one *with this caveat stated*, not quietly substituted.

**(b) Different model, corpus and sample.** SmolLM2-135M on 40 hand-written sentences versus GPT-2 XL / Pythia-2.8B on LAMBADA and Universal Dependencies. No cross-study number is comparable across those.

**(c) The sign differs.** King et al. report curvature *increasing* with entropy; the intrinsic measure *decreases*. Not a contradiction — their quantity is path bending and this one is space bending, and E2 showed the two are uncorrelated ([07-stage4-log.md](07-stage4-log.md) §1) — but it means the two results are not two estimates of one thing.

> **The defensible claim is the within-study one:** on identical activations, under an identical protocol, intrinsic Fisher–Rao curvature tracks predictive entropy substantially better than the three published proxies. That needs no cross-paper comparison and is not vulnerable to any of (a)–(c).
>
> **Answering the original RQ3a as literally posed requires re-running on GPT-2 XL / Pythia with LAMBADA and Pearson statistics** — blocked on the TLS problem ([05-stage0-log.md](05-stage0-log.md) §0).

---

## 5. THE DEFINITIONAL CONTROL — three attempts, and the third one answers it

**Read this section in order; the first two attempts are wrong and are kept because their failures are instructive.**

| | control | verdict |
|---|---|---|
| §5.0 | permute `U`'s **rows** | 🔴 **vacuous** — provably the identity on this metric |
| §5.1–5.3 | **Gaussian** `U` | ⚠️ suggestive, but changes everything at once |
| **§5.4** | **paired scramble** — real `p`, permuted directions | ✅ **decisive: learned structure, not concentration** |
| **§5.5** | the same control on the **volume profile** | 🔴 **that one *is* largely definitional** |

Entropy and curvature are both computed from the same `p(h)`. As `p` concentrates, the reachable simplex region shrinks toward a low-dimensional face and the induced geometry changes **whether or not the model learned anything**. So the correlation could be a mathematical consequence of concentration rather than a fact about the network.

### 🔴 5.0 THE "SHUFFLED" CONTROL WAS VACUOUS — this section's conclusion is retracted

`run_definitional.py` used a condition described as *"real `U` with rows permuted — destroys every token-to-direction assignment while preserving the spectrum exactly."* **It destroys nothing. It is mathematically the identity on this metric.**

Permuting `U`'s rows permutes the logits, hence permutes `p` by the **same** permutation, so each token's probability travels with its own direction:

```
(PU)ᵀ ( diag(Pp) − (Pp)(Pp)ᵀ ) (PU)
    = Uᵀ Pᵀ P ( diag(p) − p pᵀ ) Pᵀ P U
    = Uᵀ ( diag(p) − p pᵀ ) U                    since PᵀP = I
```

Verified numerically: `‖G_real − G_shuffled‖ / ‖G_real‖` = **1.5 × 10⁻¹²**, entropies identical to 10 decimals.

**So the "shuffled" row below was a re-run of the real model on a different random subsample** (the generator advanced between conditions), and the −0.555 vs −0.672 difference was **pure sampling noise**. The inference "shuffling retains 83%, therefore the relationship is definitional" **does not follow and is withdrawn.**

| condition | what it actually was | ρ(K, H) | ρ(R, H) |
|---|---|---|---|
| **real** | the trained unembedding | −0.462 | −0.672 |
| ~~shuffled~~ | ⚠️ **identical to real**; different subsample only | −0.397 | −0.555 |
| **random** | Gaussian `U`, scale swept — *a genuine control* | −0.737 | −0.899 |

The **random** condition remains valid, and the matched-entropy comparison below still stands on its own. But it is a weaker instrument than a paired design, because a Gaussian `U` differs from the real one in every respect at once, not just in the learned structure.

> **Status of the definitional question: ANSWERED — see §5.4.** A corrected paired control (`run_scramble_control.py`) keeps `p` exactly as the real model produces it and pairs those probabilities with permuted directions — matching entropy to machine precision while destroying only the learned assignment. Its first results looked **strongly frame-dependent**, which is why the draft above drew no conclusion. That dependence has since been diagnosed: it was not a frame effect, and the control stands.

### 5.1 Matched-entropy comparison — the decisive test

Comparing median `K` at equal entropy isolates anything the trained model contributes beyond concentration:

| entropy (nats) | real | shuffled | gap | min n |
|---|---|---|---|---|
| [0, 0.5) | 0.3332 | 0.2817 | **+0.0514** | 17 |
| [0.5, 1) | 0.2574 | 0.2591 | −0.0017 | 15 |
| [1, 2) | 0.2552 | 0.2573 | −0.0021 | 32 |
| [2, 3) | 0.2530 | 0.2534 | −0.0004 | 7 |
| [3, 5) | 0.2224 | 0.2486 | −0.0263 | 7 |
| [5, ∞) | 0.2026 | 0.1837 | +0.0189 | 7 |

**Across the bulk of the range ([0.5, 3) nats) the largest gap is 0.0021** — against a total effect size of ~0.18 (`K` running 0.24 → 0.42 across entropy). **Real and structure-free agree to 1.2% of the effect.**

### 5.2 What the Gaussian control does and does not settle

The matched-entropy comparison in §5.1 uses the **random** condition, which is a genuine control but a blunt one:

> A structure-free Gaussian unembedding reproduces the real model's median `K` to within 0.002 across the bulk of the entropy range. Taken alone that suggests the curvature–entropy relationship reflects concentration geometry.
>
> ⚠️ **It cannot settle the question**, because a Gaussian `U` differs from the real one in every respect at once — spectrum, row norms, alignment — not solely in the learned assignment. Isolating the learned component requires the **paired** scramble control, which changes one thing only. That is §5.4, and it points the other way.

**The instrument-quality claim is independent of all of this and is the safest thing to report:** *intrinsic Fisher–Rao curvature is a far better readout of predictive concentration than any of the three published proxies* (−0.58 vs −0.20…+0.28; −0.43 vs −0.13…+0.02 within-layer). It holds however the definitional question resolves.

### 5.3 The one residual — ✅ powered, and the suspicion was right

The lowest-entropy bin showed a real gap: **+0.0514**, or 29% of the effect size, with n = 17–19, while every other bin in the bulk sat at 0.002. The reading was *"if any model-specific geometry exists, it is in the near-deterministic regime"* — but one bin at that `n` was not evidence.

**Now powered properly**, with the clean `within` control of §5.4(b′) and an entropy-**stratified** sample (12 points per bin, so the low-entropy regime is covered by design rather than by luck):

| entropy (nats) | n | `K` real | `K` within | retained | sign test |
|---|---|---|---|---|---|
| **[0, 0.5)** | 12 | 0.2833 | **−2.1475** | **−758%** | **12/12** |
| [0.5, 1) | 12 | 0.2555 | −0.1769 | −69% | **12/12** |
| [1, 2) | 12 | 0.2554 | +0.0044 | +2% | **12/12** |
| [2, 4) | 12 | 0.2545 | +0.0884 | 35% | 10/12 |
| [4, ∞) | 12 | 0.2028 | +0.0346 | 17% | 9/12 |

> ## 🎯 **`K_real` is nearly constant across the entropy range; `K_within` is not, at all.**
>
> The trained model holds `K` between **0.20 and 0.28** whether the prediction is near-deterministic or maximally diffuse. Destroy the assignment and the geometry becomes violently entropy-dependent, swinging from **−2.15** to **+0.09**.
>
> **So the learned assignment is not merely *adding* curvature — it is *stabilising* it at the simplex value across concentration regimes.** That is a sharper claim than "structure matters", and it is the one the data supports.

**§5.3's suspicion is confirmed and inverted in emphasis.** The model-specific effect is strongest exactly where it was suspected — the near-deterministic regime — but it shows up as the scrambled geometry *collapsing into strong negative curvature* rather than as a small positive gap. At high entropy the two conditions come closest (17–35% retained, sign test 9–10/12), which is what one would expect: a diffuse `p` gives the assignment less to encode.

⚠️ The n = 12 per bin is small, and the lowest bin's median of −2.15 has a wide spread. **The unanimous sign tests, not the medians, are what carry the low-entropy result.**

---

### 5.4 ✅ RESOLVED — it was never a frame problem, and the control stands

The paired scramble control gave `K_scr` = 0.014 in the scrambled metric's own eigenframe and 5.21 in the real metric's frame, on the same points. That 370× disagreement was labelled "frame sensitivity" and the conclusion was withheld. **The label was wrong**, and once the right question is asked the disagreement stops being a problem.

**(a) Rotation invariance holds in BOTH conditions.** Task 3.8 verified it only for the real metric. Rotating the retained basis by an arbitrary orthogonal `Q`, n = 24 points:

| condition | median rel. change in scalar `R` | worst |
|---|---|---|
| real | 2.23e−14 | 1.54e−10 |
| **scrambled** | **9.05e−15** | **1.90e−11** |

> **Rotating the frame changes nothing, in either condition.** So "frame sensitivity" is a misnomer. `F_real` and `F_scr` are not two bases for one subspace — they are two **different subspaces**, and since `metric_in_frame` induces the metric on the affine slice `h + span(F)`, the two numbers are the curvatures of two different submanifolds. Two measurements, not one inconsistency.

**(b) The two subspaces barely overlap, and each metric lives almost entirely in its own** (`diag_frame.py`, n = 40, k = 5, all linear algebra — no curvature involved):

| quantity | real in own frame | scr in own frame | real in scr frame | scr in real frame | **scr1 in scr2 frame** | random frame |
|---|---|---|---|---|---|---|
| trace fraction captured | **0.879** | **0.876** | 0.043 | 0.039 | **0.064** | 0.0086 |
| conditioning of the (k×k) metric | 21.2 | 17.9 | 290 | 189 | **161** | 30.3 |

Principal angles between `F_real` and `F_scr`: smallest 64.4°, largest 89.35°.

**Three things this table settles, none of which the point estimates alone could:**

1. **The random-subspace control works.** A random 5-plane captures 0.0086 of the trace against an isotropic floor of `k/d` = 0.0087 — dead on. So the 0.04 figures are real overlap (≈5× the floor), not noise, and the instrument is calibrated.
2. **Measuring a metric off its own support is catastrophic for conditioning, and *a random subspace is better behaved than the cross-frame one*** (30 vs 189–290). The cross-frame arm is not "a different but equally valid slice"; it lands on precisely the directions where the other metric's eigenvalue ratio is most extreme, and curvature inverts that metric.
3. **🎯 The decisive control: two *independent scrambles* do the same thing to each other** — 0.064 capture, conditioning 161. **So the cross-frame blow-up is not a fact about real-versus-scrambled at all.** It is what happens between *any* two independently selected subspaces. It carries no information about learned structure, and the arm is retired.

**(b′) 🔴 THE CONTROL ITSELF HAD A CONFOUND — found on audit, fixed, and the result got stronger.**

The scramble was described throughout as *"pairs those probabilities with permuted directions… destroying only the learned assignment."* **It destroyed more than that.** With a whole-vocabulary permutation the retained rows become `perm[idx]` — measured, those overlap the real top-512 by **5/512**, and median ‖U row‖ shifts from **2.44 to 3.10**. So the condition swapped the direction **set** as well as the pairing, and the collapse could have come from either.

The clean control keeps the row multiset **identical** and permutes *within* the retained index set: `rows = idx[randperm(len(idx))]`. Three conditions, same points, `k` = 5, **n = 60** stratified by entropy:

| condition | what it destroys | median `K` | median `R` | vs real |
|---|---|---|---|---|
| **real** | nothing | **0.2555** | 5.757 | — |
| **within** | **the pairing ONLY (same rows)** | **+0.0003** | −0.223 | 55/60, **z = +6.45** |
| global | the pairing **and** the row set | −0.0029 | −0.159 | 56/60, **z = +6.71** |

> **The conclusion survives, and the clean control gives the same collapse** (0.2555 → **+0.0003**). So the effect is the **learned assignment**, not the change of direction set — which is the stronger and more specific claim, and the one §5.4 had been asserting without having earned it.

**And it is `k`-stable.** The `k` sweep in (c) below was run on the compound `global` condition; repeated under the clean `within` control on the same 60 points:

| `k` | n | `K` real | `K` within | sign test | z |
|---|---|---|---|---|---|
| 4 | 60 | 0.2554 | **+0.0097** | 55/60 | **+6.45** |
| 5 | 60 | 0.2555 | **+0.0003** | 55/60 | **+6.45** |

> **Superseded by a larger run.** The table above is the n = 60 checkpoint. The full point file has **n = 221**: `K` **0.2546 → 0.0109, 201/221, z = +12.18**, with `k` = 4 (+6.86) and `k` = 6 (+5.66) on the 69-point subset where all three `k` were computed. Every row shared with the smaller checkpoints agrees to 1e−12. [12-audit-log.md](12-audit-log.md) §3.

| 6 | 60 | 0.2537 | **+0.0297** | 51/60 | **+5.42** |

`K_real` moves by 0.002 across `k`; `K_within` stays within 0.03 of zero. **This is the one Riemann-derived result in the project that is not `k`-fragile**, which matters given that `k`-fragility is what retracted the deviation statistic in Stage 3.

### 🎯 The layer breakdown — a separate, layer-stratified run

The entropy-stratified sample leaves 3–12 points per layer, too few to say anything per layer. `run_scramble_within.py --bylayer` fills layer cells instead: **14 points at each of 9 layers, n = 126, 113/126 positive, z = +8.91.**

| layer | 1 | 5 | 10 | 15 | 20 | 25 | 28 | 29 | 30 |
|---|---|---|---|---|---|---|---|---|---|
| `K` real | 0.2518 | 0.2549 | 0.2536 | 0.2537 | 0.2570 | 0.2668 | 0.2276 | 0.2469 | 0.2393 |
| **`K` within** | **0.0568** | 0.0059 | 0.0060 | 0.0034 | −0.0136 | **−0.4970** | **0.1279** | 0.0455 | **0.1584** |
| retained | 22.6% | **2.3%** | **2.4%** | **1.4%** | −5.3% | −186% | 56.2% | 18.4% | 66.2% |
| sign test | 12/14 | **14/14** | 13/14 | 13/14 | 13/14 | **14/14** | 12/14 | 12/14 | 10/14 |

> ## **The collapse is present at every layer — but it is a U-shape in magnitude.**
>
> **Layers 5–25 are where the learned assignment does almost all the work**: destroying it leaves ≤2.4% of the curvature, or drives it negative. **The two ends are where it matters least** — layer 1 retains 22.6%, layers 28–30 retain 18–66%.

That shape is what one would expect and is worth stating as a prediction met rather than a surprise: at **layer 1** the context has barely been integrated, so there is less learned assignment to destroy; at **layers 28–30** the prediction is nearly committed and `p` is concentrated enough that the geometry is closer to being fixed by concentration alone — which is exactly the regime §5.3 found the *Gaussian* control agreeing with the real model.

⚠️ `n` = 14 per layer. The per-layer *sign tests* (10/14 to 14/14) are what carry this; the per-layer medians, especially layer 25's −0.497, are single-cell values with wide spread and should not be quoted individually.

*Each point's scramble is seeded from its own index, so the whole table is reproducible: `python run_scramble_within.py 12 <budget> --ksweep`, re-run until the checkpoint fills.*

This is the **third** control in this document that changed more than one thing at once: §5.0's row-shuffle changed nothing, §5.2's Gaussian `U` changed everything, and this one changed two things. `metrics.fisher_metric_scrambled` now documents both modes and `fisher_metric_projected` takes an explicit `rows=` argument for the clean one.

**(c) The same sweep under the *compound* `global` condition, for the record.** Superseded by the `within` sweep in (b′) above, which is the clean control at n = 60; kept because the two agree and that agreement is itself reassuring:

| `k` | n | `K_real` | `K_scr` | ratio | sign test | z |
|---|---|---|---|---|---|---|
| 4 | 17 | 0.2528 | 0.0106 | 24.0× | 16/17 | **+3.64** |
| 5 | 19 | 0.2594 | 0.0048 | 53.8× | 17/19 | **+3.44** |
| 6 | 17 | 0.2556 | 0.0127 | 20.2× | 16/17 | **+3.64** |

**(d) And it is not conditioning.** Selecting `k` per point per condition so both arms are matched on `cond_eff` (21.4 real vs 25.8 scrambled) gives `K_real` = 0.2556 vs `K_scr` = 0.0127, **15/17 positive, z = +3.15, significant**.

> ## 🎯 The conclusion
>
> **At exactly matched entropy, matched conditioning, matched direction set, and stably across `k` = 4, 5, 6, destroying the learned token→direction assignment collapses intrinsic curvature from `K ≈ +0.255` to `K ≈ 0`.** The geometry is **not** a function of predictive concentration alone. **Which probability sits on which direction** is what puts the manifold at the simplex value.
>
> This does not contradict §5.1–5.2. A Gaussian `U` reproduces the real median `K`; a *paired scramble* does not. The difference is informative: a Gaussian unembedding has, by symmetry, no relationship between probability mass and direction to destroy, whereas the real one does — and it is that relationship, not the spectrum or the row norms, that the curvature is reading.

**(e) The cross-frame arm, recorded rather than hidden** (n = 17):

| measurement | median `K` | \|K\| p90 |
|---|---|---|
| `K_scr` in the **real** frame | 33.7 | **3213** |
| *(its own-frame value)* | 0.0078 | 0.44 |
| `K_real` in the **scrambled** frame | 0.0342 | 0.90 |
| *(its own-frame value)* | 0.2640 | 0.33 |

Both arms are wrong, but **not symmetrically**: the scrambled-in-real arm has a catastrophic heavy tail (p90 = 3213) while the real-in-scrambled arm is merely deflated. Report the p90, not the median — the median hides the failure mode. The earlier "5.21" was a small-sample draw from this tail.

### 5.5 🔴 The same control applied to the VOLUME profile — and it fails

§7.4 asked for this control to be applied retrospectively to the layer-wise volume element, since it is a function of the same `p`. Done (`run_volume_scramble.py`, `n` = 40 per layer × 9 layers = 360 points, `k` = 5, on the sphere). **It is far cheaper than the curvature version** — the volume element needs one eigendecomposition, not a Riemann tensor — so it runs at proper `n` rather than the 17–19 points §5.4 could afford.

Both scramble conditions from §5.4(b′) are run, so this is not vulnerable to the confound found there.

| layer | 1 | 5 | 10 | 15 | **20** | 25 | 28 | 29 | 30 |
|---|---|---|---|---|---|---|---|---|---|
| **real** | −0.201 | −0.001 | −0.189 | −0.266 | **−0.725** | −0.469 | −0.018 | +0.014 | +0.058 |
| **within** (clean) | −0.045 | +0.110 | −0.085 | −0.114 | **−0.622** | −0.239 | +0.009 | +0.029 | +0.122 |
| global | −0.158 | +0.109 | −0.086 | −0.175 | **−0.664** | −0.288 | −0.160 | +0.023 | +0.064 |

| | real | **within** | global |
|---|---|---|---|
| minimum of the profile | **layer 20** | **layer 20** | **layer 20** |
| range (max − min) | 0.783 | **0.744** (95%) | 0.774 (99%) |
| **profile agreement vs real** | 1.000 | **ρ = +0.967** | ρ = +0.900 |

> ⚠️ These are ρ over 9 layer medians, and they depend on `torch.median`'s lower-middle convention: with a true median the `within` value is **+0.867**. **Per point (n = 360, convention-free) it is +0.957** — quote that one. [12-audit-log.md](12-audit-log.md) §4.
| ρ(log-volume, entropy) | +0.862 | **+0.782** | +0.783 |

> ## 🔴 **The U-shaped volume profile is largely definitional.**
>
> A structure-free scramble, at exactly matched entropy **and an identical direction set**, **reproduces the shape at ρ = +0.967, puts the minimum at the same layer 20, and retains 95% of the range.** The volume–entropy correlation survives at +0.782 against the real +0.862.
>
> **The clean control agrees with the real profile *more* than the compound one does** (+0.967 vs +0.900) — i.e. correcting the confound makes the volume result *more* definitional, exactly as it made the curvature result *more* structural. The two moved in opposite directions under the same correction, which is what makes the contrast below trustworthy rather than an artefact of one control.

**This is the opposite of the curvature result, and the contrast is the finding.** The same control, on the same points, at the same matched entropy:

| quantity | real | scrambled (clean `within` control) | retained |
|---|---|---|---|
| sectional `K` (§5.4, **n = 221**) | **0.2546** | **+0.0109** | **~4%** |
| layer-wise log-volume shape | ρ = 1 (by definition) | **ρ = +0.967** | **97%** |

So the two primary quantities are not interchangeable in the way Stage 3 assumed when it promoted the volume element to *"the stable primary quantity"* on the grounds that it was better conditioned. **Better conditioned it is; more informative about the model it is not.** Curvature is fragile and carries learned structure; the volume profile is stable and mostly reads concentration.

There *is* a real level effect — scrambled log-volume is systematically **higher** (real exceeds `within` at only 62/360 points, z = −12.44, median paired difference −0.112 nats/dim). The learned assignment does compress volume, and unlike the shape this **is** a genuine structural signal. But 0.112 against a profile range of 0.783 is **14% of the effect the profile is used to report**, and it is a *level* shift, not the U-shape the result is quoted for.

**Consequences that need to reach Chapter 6:**

- The **"U-shaped, minimum at layer 20"** result must be reported *with this control*. It is not wrong — the shape is real — but it is substantially a statement about where the model's predictive distribution is most concentrated, which is a much weaker claim than a statement about representational geometry.
- The Stage 3 recommendation to *"build on the volume element"* ([06-stage3-log.md](06-stage3-log.md) §4.4, and repeated in the README) needs the qualification that stability and informativeness came apart here.
- The comparison is only fair because both were tested with the *same* paired control. This is the argument for having a single control protocol rather than a different one per quantity.

⚠️ The curvature side of this comparison is `n` = 17–19 and the volume side is `n` = 360. The asymmetry is a cost limit, not a design choice, and the curvature `n` is the one to raise.

### 5.6 ⚠️ A methodological finding that came out of this, and it is not minor

The plan was to select `k` by a `cond_eff` ceiling (06-stage3-log.md §4.3) rather than by trace fraction. **On real activations at `cond_max` = 10², that rule returns `k` = 11–16** — the retained spectrum is simply flatter than the ceiling is tight.

**`riemann()` cannot be evaluated there.** Measured warm on an idle machine (`d` = 576, `top_k` = 512):

| k | 3 | 4 | 5 | 6 | 7 | 8 | 11 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| time / memory | 0.39 s | 1.00 s | 2.61 s | 8.11 s | 16.9 s | 45.7 s | 13.8 GB | 4.8 GB | **77 GB** |

Every entry of the Riemann tensor is a second derivative taken by nested `jacrev`, so the intermediates grow far faster than the `k⁴` of the tensor itself — roughly **×2.7 per unit of `k`**. **Three runs of `run_frame_resolution.py` were killed by this — silently, with no traceback and no output**, which is exactly how it wasted an afternoon before being diagnosed.

⚠️ *A first draft of this table quoted 1.6 / 4.6 / 17.2 / 27.0 s for `k` = 3–6. Those were measured cold and under contention and are ~5× too high. Warm up before timing anything in this codebase — the first `torch.func` call carries a large fixed cost.*

Consequences, now enforced in code:

- `curvature.riemann` **refuses `k` > 8** with an explanatory error rather than attempting the allocation (`K_RIEMANN_MAX`).
- Any caller choosing `k` automatically must take `k = min(select_k(...), K_RIEMANN_MAX)` and **report both numbers**. `run_frame_resolution.py` records `k_want` (11) alongside `k_sel` (6).
- **The `cond_eff` ceiling remains the right conditioning *diagnostic*, and is directly usable for the volume element and spectral diagnostics, which have no `k` ceiling. It is not usable as a `k` selector for Riemann-derived quantities.** That distinction belongs in Chapter 4.

---

## 6. Limits

- One model (135M), 40 hand-written sentences, 90–360 points per condition.
- Spearman is primary by necessity, not preference (§4a).
- The matched-entropy bins at the extremes have n = 7–19; only the [1,2) bin (n=32) is comfortably powered.
- **§5.4's clean `within` result is n = 60** (stratified 12 per entropy bin), with per-layer cells of only 3–11 points because the stratification is by entropy. The `k` = 4/5/6 sweep has now been run under `within` as well (§5.4), and the effect holds at all three.
- **`K_within` is near zero on average but strongly negative at low entropy** (−2.15 in the lowest bin, n = 12). The scrambled geometry is not uniformly "flat"; the sign of its departure depends on concentration.

---

## 7. Next

1. **Raise `n` on the §5.4 paired control and break it down by layer.** It is the load-bearing result of this document and it rests on 17–19 points. The measurement is ~44 s/point at `k`=5; `run_frame_resolution.py main` checkpoints, so this is just wall time.
2. Re-run on GPT-2 / Pythia — **no longer blocked**, the TLS problem is fixed and all three architectures pass their checks ([10-architecture-log.md](10-architecture-log.md)).
3. Replicate Mabrok's proxy faithfully so the magnitude comparison can be made or dropped ([07-stage4-log.md](07-stage4-log.md) §1.2).
4. ~~**Apply the §5.4 control retrospectively.**~~ ✅ **done** — §5.5. `K ≈ +1/4` is not a concentration artifact; **the volume profile largely is.** What remains is to propagate that qualification into [07-stage4-log.md](07-stage4-log.md) §2.2 and Chapter 6, and to revisit [06-stage3-log.md](06-stage3-log.md) §4.4's recommendation to build on the volume element.
5. **Power the low-entropy bin** (§5.3) — still the weakest part of the Gaussian comparison. Needs ~100 points at entropy < 0.5 per condition.
