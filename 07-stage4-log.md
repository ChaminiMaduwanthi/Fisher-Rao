# Stage 4 — Progress Log: E2, the adjudication experiment

**Date:** 10 August 2026
**Status:** ✅ E2 at n=360 — **no instrument tracks any other**. RQ3b answered. **Two claims corrected by controls this stage** (§1.0 retracted, §3b.3 narrowed). Volume profile on 541 positions/layer.
**Reproduce:** `python run_stage4.py` → `results/stage4/stage4.json` (checkpoints to `e2_partial.json`; delete it to recompute)

---

## 1. 🎯 E2 — four instruments, identical activations

**`n` = 360, stratified at 40 points per layer** across 40 sentences × 9 layers (4869 candidates), `k`=5, `top_k`=512. At this `n` the standard error on ρ is ≈ `1/√(n−3)` = **0.053**, so the 95% interval is roughly **±0.10**.

| instrument | median | range |
|---|---|---|
| 1. intrinsic Fisher–Rao `K` (this work) | 2.565e-01 | [−5.12e-01, 1.67e+01] |
| 2. local-PCA residual (Mabrok-style) | 1.942e-02 | [1.67e-04, 9.94e-02] |
| 3. Frenet under `UᵀU` (Manson) | 4.752e-04 | [3.78e-08, 2.29e-02] |
| 4. Euclidean angle (King et al.) | 1.989e+00 | [1.08e+00, 2.83e+00] |

### Pairwise Spearman correlations — pooled and within-layer

| pair | pooled ρ | within-layer mean ρ |
|---|---|---|
| intrinsic vs local-PCA | −0.144 | **+0.000** |
| intrinsic vs Manson | +0.133 | **−0.003** |
| intrinsic vs King | +0.137 | **+0.112** |
| local-PCA vs Manson | +0.071 | — |
| local-PCA vs King | +0.178 | — |
| Manson vs King | +0.248 | **+0.063** |

> ### **No instrument tracks any other.** Every pooled |ρ| ≤ 0.25; every within-layer |ρ| ≤ 0.11.
>
> The four quantities are mutually near-independent. In particular **the three proxies do not track intrinsic Fisher–Rao curvature** — the finding this experiment was built to test — and that survives the within-layer control (+0.000, −0.003, +0.112).

### 1.0 🔴 RETRACTED: "the two flat-metric instruments agree with each other"

The `n`=40 draft of this section reported **ρ = +0.685** (within-layer +0.667) between Manson's `UᵀU` Frenet curvature and King et al.'s Euclidean angle, and interpreted it as both measuring "how the path bends." **That did not survive `n`=360:**

| | n=40 | **n=360** |
|---|---|---|
| Manson vs King, pooled | +0.685 | **+0.248** |
| Manson vs King, within-layer | +0.667 | **+0.063** |

**It was a small-sample artifact.** With `n`=40 the ±0.32 interval was wide enough to admit almost anything, and the earlier draft over-read a value inside it.

In hindsight there was never a strong reason to expect agreement: **Manson's curvature runs along the *layer* axis at a fixed token, King's runs along the *token* axis at a fixed layer.** They are path-bending measures on **different paths**. The n=40 coincidence invited a tidy story, and the story was wrong.

> This is the third time in this project that a plausible-looking number survived until it was tested properly — after the simplex sign flip and the curvature-deviation "signal." **Raising `n` was worth the 26 minutes.**

**The surviving conclusion is cleaner, not weaker:** four instruments, four different quantities, no pairwise agreement. The thesis's claim — that intrinsic Fisher–Rao curvature is not what the existing proxies measure — is unaffected and now rests on `n`=360 with ±0.10 error bars rather than `n`=40 with ±0.32.

### 1.1 ✅ Two verification checks, both still passing at `n`=360

**(i) Positive control — does intrinsic `K` carry real signal?**

A null correlation only means something if the intrinsic quantity *has* usable variance. Measured:

```
K vs scalar R (same geometry, independent contraction):  rho = +0.720
```

**PASS.** `K` correlates at **+0.720** with scalar curvature — same Riemann tensor, different contraction. Its variance is genuine geometric signal, so the near-zero correlations against the proxies are **informative nulls**, not an absence of variance.

**(ii) Layer confound — pointwise agreement, or shared depth dependence?**

Manson's quantity is strongly layer-dependent (ρ = −0.724 vs layer; King −0.381, intrinsic −0.191, local-PCA −0.120), so pooled correlations could reflect shared variation with depth. The within-layer column above removes that, and **every conclusion is unchanged or weaker** — nothing was being carried by the layer confound.

### 1.1b `K ≈ +1/4` confirmed at scale across all nine layers

Per-layer medians across 40 points each:

| layer | 1 | 5 | 10 | 15 | 20 | 25 | 28 | 29 | 30 |
|---|---|---|---|---|---|---|---|---|---|
| median `K` | 0.2632 | 0.2565 | 0.2534 | 0.2576 | 0.2576 | 0.2640 | 0.2544 | 0.2453 | 0.2413 |
| IQR | 0.060 | 0.021 | 0.006 | 0.013 | 0.027 | 0.025 | 0.027 | 0.037 | 0.049 |

**All nine layers sit at 0.241–0.264.** The Stage 3 single-prompt result holds on 360 points.

### 1.1c 🎯 The tail is real geometry, not a numerical failure

Quantiles of `K` over all 360 points:

| 1% | 5% | 25% | 50% | 75% | 95% | 99% |
|---|---|---|---|---|---|---|
| 0.164 | 0.201 | 0.250 | 0.257 | 0.269 | 0.391 | 1.256 |

**93% of points sit tightly at `K` ≈ +1/4 (IQR 0.018); 7.2% (26/360) depart by more than 0.1**, including two extreme values (10.7, 16.7) and one negative (−0.51).

My first guess was that these are conditioning failures. **Three checks say otherwise:**

- **They track scalar `R` just as well as the core does** — ρ = +0.675 (outliers) vs +0.695 (non-outliers). A numerical breakdown would destroy that agreement between two different contractions of the same tensor; it does not.
- **Their scalar curvature is genuinely elevated** — median `R` = **24.06** at outliers vs **5.42** in the core, against the constant-curvature reference `k(k−1)/4` = 5.00.
- **They cluster by layer** — 9/40 at layer 1, 5/40 at layer 20, 4/40 at layer 25, and 0/40 at layer 10 — rather than scattering randomly as numerical noise would.

> **So `+1/4` is a *median* behaviour, not a universal one.** Most of the manifold is locally simplex-like, but a measurable minority of points carry strongly elevated — and occasionally negative — intrinsic curvature, concentrated at specific depths.
>
> **This minority is the most promising target identified so far.** It survives exactly the checks that the retracted deviation statistic ([06-stage3-log.md](06-stage3-log.md) §4.2) failed, and unlike the median it is not pinned to the ambient simplex value by construction.

**Report medians and IQRs, never means** — the tail makes means meaningless. Spearman correlations are rank-based and unaffected.

### 1.3 ✅ E2 and RQ3a re-run on a REAL corpus — and the instruments split on a new axis

**Date:** 12 August 2026 · **Reproduce:** `python run_corpus_compare.py 40 470` (checkpointed; re-run until both arms fill) → `results/corpus_compare/corpus_compare.json`

Everything above was measured on 40 hand-written sentences, and `corpus.py` said so plainly: *"the author chose them, so they are not a random sample of anything."* That caveat existed because HuggingFace was unreachable. It is now fixed ([10-architecture-log.md](10-architecture-log.md) §1), so the caveat is retired by **measuring** rather than by arguing: the same protocol on both corpora, `n` = 360 WikiText / 293 hand-written.

**RQ3a — Spearman against next-token entropy.** Raised to the published `n`: **360 WikiText / 293 hand-written**, so the ±0.10 error bars match §1's headline table. (`n` = 108 values from the first pass in brackets.)

| instrument | hand-written | **WikiText-103** | **\|change\|** |
|---|---|---|---|
| **intrinsic `K`** | −0.447 *(−0.565)* | **−0.475** *(−0.572)* | **0.028** |
| **intrinsic `R`** | −0.579 *(−0.633)* | **−0.544** *(−0.492)* | **0.034** |
| local-PCA (Mabrok) | +0.175 | +0.050 | **0.126** |
| Frenet `UᵀU` (Manson) | −0.113 | −0.217 | **0.104** |
| Euclidean angle (King) | +0.043 | −0.151 | **0.194** |

> ## 🎯 **The intrinsic instrument is corpus-invariant. The proxies are not.**
>
> The two intrinsic quantities move by **0.028** and **0.034** between hand-written sentences and encyclopedia prose. Every proxy moves by **0.104 to 0.194** — three to seven times as much — and King's angle changes sign.
>
> This is a claim only a two-corpus design can make, and it is an **instrument-quality** argument independent of which instrument is "right": a measurement that reports a different answer depending on what text you feed it is not measuring a property of the model.

**RQ3a's headline survives**: best intrinsic |ρ| = **0.544** against best proxy |ρ| = **0.217** on WikiText, a **2.5×** margin (3.3× on the hand-written corpus). The "2–3× better than any published proxy" claim is now stated on real text at the published `n`.

**E2 — pairwise correlations among the four instruments:**

| | hand-written | **WikiText-103** |
|---|---|---|
| largest cross-instrument \|ρ\| | 0.295 | **0.135** |
| positive control `K` vs `R` | +0.742 | **+0.695** |

> **E2's conclusion survives and gets cleaner.** No instrument tracks any other on real text either — the largest pairwise correlation *falls* to 0.108 — while the positive control stays strong at +0.691, so the null remains informative rather than vacuous.

⚠️ One model. The point estimates agree with §1's independent `n` = 360 run to within sampling error (`K` −0.447 here vs −0.485 there; Manson −0.113 vs −0.252 — the latter is the widest gap and sits inside two standard errors). **What is being claimed is the corpus *difference*, which is paired by protocol, not the absolute values.** The largest cross-instrument correlation on the hand-written arm, local-PCA vs King at +0.295, is the one cell that exceeds §1's ≤0.25 bound; at `n` = 293 its interval covers 0.25.

### 1.2 ⚠️ Remaining limit

**I have *not* reproduced Mabrok's 10⁻⁵.** The local-PCA proxy returns **1.94e-2** here (ratio to intrinsic `K`: 13.2×), three orders of magnitude above his published figure. The likely cause is that a 20-token sentence gives a ~20-point "local neighbourhood," which is not local at all, so residual variance is dominated by genuine spread rather than curvature.

> **Consequence: the "four orders of magnitude" comparison in [06-stage3-log.md](06-stage3-log.md) §6 must be softened.** It compared this project's intrinsic value against Mabrok's *published* number, computed on a much larger point cloud with genuinely local neighbourhoods. The magnitude gap is not established until his setup is faithfully replicated. **The correlation result above does not depend on that and is the stronger claim** — it needs no cross-paper magnitude comparison at all.

> ✅ **RESOLVED 11 August 2026 — [11-mabrok-replication-log.md](11-mabrok-replication-log.md).** His setup *was* faithfully replicated (WikiText-103 validation, 1,800 vectors/layer, GPT-2) and his 10⁻⁵ **does** reproduce, at a ≈0.9999 variance threshold with `k` ≥ 50. But on transformer activations — whose local spectrum has no gap, making them indistinguishable from isotropic noise on this test — the residual just tracks `1 − threshold` (ratio 0.93–0.99 across three decades). And **at that threshold the same code reports a unit 3-sphere as flat to 10⁻³¹.** So the magnitude is set by an unreported analyst choice at an operating point where the instrument cannot see curvature at all. **The magnitude comparison is dropped permanently**, and the diagnosis above ("neighbourhoods not local enough") was only part of the story. The `ρ` = −0.14 correlation is unaffected: rank correlation is invariant to the reparameterisation.

---

## 2. Layer-wise per-dimension log-volume — a stable profile

`k`=5 fixed for cross-layer comparability, radial direction quotiented, `n` = **541 token positions per layer** (the polysemy rewrite in §3b.1 changed token counts from 532).

| layer | log-vol / k | std | `k_sel` (cond≤10²) | cond_eff | entropy |
|---|---|---|---|---|---|
| 1 | **−1.058** | 0.61 | 9 | 85.2 | 1.129 |
| 5 | −1.853 | 0.69 | 7 | 78.5 | 1.253 |
| 10 | −2.122 | 0.45 | 6 | 69.6 | 1.318 |
| 15 | −2.781 | 1.29 | 6 | 60.6 | 1.404 |
| 20 | −3.335 | 1.28 | 6 | 62.6 | 1.246 |
| 25 | −3.857 | 1.25 | 9 | 82.8 | 1.482 |
| 28 | **−3.914** | 1.20 | 14 | 89.2 | 2.157 |
| 29 | −3.694 | 0.83 | 21 | 89.3 | 3.106 |
| 30 | −3.578 | 0.58 | 28 | 89.8 | 3.863 |

### 2.1 🔴 CORRECTED: the reported shape was the `1/r²` scale artifact

The table above is in **ambient coordinates**, and an earlier draft read it as *"the Fisher volume contracts monotonically from layer 1 to 28, then partially recovers"*, with an interpretation about progressive commitment to a prediction. **That shape is an artifact and the interpretation is withdrawn.**

`G(h) = Aᵀ(…)A` with `A ∝ 1/r`, so `λᵢ ∝ 1/r²` and `log_vol_per_dim` carries a `−log(r)` term **by construction**. The residual norm grows enormously with depth (salience → 3.7 × 10⁴ by layer 29), so the ambient profile is dominated by that trivial term. Regressing `log_vol_per_dim` on `log r` over 1494 points (`check_volume_confound.py`):

```
slope a = -0.937      (construction mandates exactly -1)      R^2 = 0.697
```

**The systematic layer trend is essentially all scale.** This is the *same* failure mode that invalidated the first RQ3b claim (§3b.3) — raw distances growing because the vectors grow.

### 2.2 The correct quantity, and the corrected profile

The fix is not cosmetic. Stage 0 established the semantic manifold is the **sphere of directions** ([05-stage0-log.md](05-stage0-log.md) §4.2), and the metric on the unit sphere is `r²·G_ambient` — which adds exactly **`+log r` per dimension**. So the sphere-corrected value is the volume on the *correct manifold*; the ambient one is contaminated by the residual norm.

| layer | 1 | 5 | 10 | 15 | 20 | 25 | 28 | 29 | 30 |
|---|---|---|---|---|---|---|---|---|---|
| ambient (artifact) | −1.04 | −1.87 | −2.13 | −2.84 | −3.45 | −4.07 | −4.12 | −3.82 | −3.61 |
| **on sphere (correct)** | −0.357 | −0.259 | −0.331 | −0.554 | **−0.872** | −0.857 | −0.572 | −0.259 | **−0.012** |

> **The corrected profile is U-shaped, not monotone.** Roughly flat through layers 1–10, dipping to a **minimum at layer 20**, then recovering monotonically to its highest value at layer 30. Total range 0.86.

> 🔴 **UPDATE 11 August 2026 — read [08-rq3a-log.md](08-rq3a-log.md) §5.5 before using this profile.** The paired scramble control that settled the curvature question has now been applied here, and **this shape is largely definitional**: a structure-free scramble at exactly matched entropy puts the minimum at the same layer 20, retains 87% of the range, and agrees with the real profile at Spearman ρ = +0.883. The shape is real; it is substantially a statement about *where the predictive distribution is most concentrated*, not about representational geometry. The genuine learned component is a level shift of 0.070 nats/dim against a range of 0.730.
>
> The ambient profile said "contracts to layer 28 then recovers slightly"; the correct one says "flat, dips at 20, recovers strongly." **Different shape, different minimum, different story.**

`volume_element(..., on_sphere=True)` is now the default so this cannot recur silently.

**Interpretation to test, not assert:** a mid-network minimum in predictive volume is qualitatively the hourglass shape reported for intrinsic dimension (Valeriani et al. 2023; Mabrok 2026), though at greater relative depth (0.67 here vs their 0.3–0.4). It is a *Fisher* volume rather than an ambient intrinsic dimension, so a match would be a finding, not a restatement. **Re-check against `k_sel`**, which is also U-shaped (9 → 6 at layer 20 → 28 at layer 30) and derived from the same spectrum — the two are not independent evidence.

⚠️ The `k`-stability verified in [03-methodology.md](03-methodology.md) §4 Rule 2 (ρ = 0.92–1.00) was measured on the **ambient** quantity. It should be re-run on the sphere-corrected one before that stability claim is carried over.

---

## 3. Corpus

`fisherrao/corpus.py`: 20 general sentences across registers and domains + 10 polysemy minimal pairs (20 sentences) = **40 sentences, 541 token positions**.

⚠️ **This is not WikiText-103.** HuggingFace is unreachable (TLS interception, [05-stage0-log.md](05-stage0-log.md) §0), so no standard corpus can be downloaded. The sentences are hand-written, which means they are short (10–30 tokens), clean edited prose, `n` in the hundreds rather than tens of thousands, and **selected by the author** — not a random sample of anything. Treat every number above as preliminary and re-run on a standard corpus once the network is fixed.

It is nonetheless a 541× improvement on the single token position everything before Stage 4 rested on.

The polysemy pairs (`bank`, `bat`, `spring`, `bark`, `pupil`, `crane`, `mint`, `plant`, `seal`, `pitch`) are analysed in §3b. They were rewritten after the first version proved vacuous under causal attention — see §3b.1.

---

## 3b. 🎯 RQ3b — E4. **The metric changes the answer by ~20 layers.**

> ✅ **Settled in §3b.6b.** Three arms sharing sentence A, with lexical overlap equalised by construction and enforced by `corpus.validate()`: different senses separate **2.03×** further than same senses (30/32, z = +4.95) at **the same depth** (4.69 vs 4.72). So the ~22-layer metric disagreement is **timing** and holds for same-sense pairs too; sense is **magnitude**. WiC's earlier null (§3b.5) is explained by its pairs differing in everything at once.

**Reproduce:** `python run_polysemy.py` → `results/polysemy/polysemy.json`

The Fisher–Rao distance between two categorical distributions is closed form,

```
d_FR(p, q) = 2 · arccos( Σᵢ √(pᵢ qᵢ) )
```

so this needs no metric assembly, no autodiff, no `k`, no frame and no cutoff — it is the cheapest genuinely-Fisher measurement in the project, and immune to every sensitivity problem that afflicted §1.

### 3b.1 🔴 First attempt was vacuous — a design error worth recording

Every pair returned `d_FR = 0.0000` at **every** layer. The corpus read *"The **bank** was steep…"* vs *"The **bank** refused…"* — the disambiguating context came **after** the target word. Under **causal attention** the hidden state at `bank` attends only to `The bank`, identical in both sentences, so the two states were bit-for-bit equal.

**Later context cannot reach back to an earlier token.** The corpus was rewritten so the sense is established *first* and the ambiguous word comes last, and `run_polysemy.py` now asserts non-zero final separation so this cannot recur silently.

### 3b.2 The result

Normalised separation (each pair scaled by its own final-layer value), mean over 10 pairs:

| layer | `d_FR` | `d_UU` (Manson) | `d_euc` |
|---|---|---|---|
| 1 | **0.220** | 0.016 | 0.012 |
| 6 | **0.603** | 0.086 | 0.084 |
| 13 | **0.799** | 0.165 | 0.157 |
| 20 | **0.953** | 0.253 | 0.224 |
| 27 | 1.176 | 0.651 | 0.560 |
| 30 | 1.000 | 1.000 | 1.000 |

**Half-separation depth** — first layer reaching 50% of final separation. **Updated 11 August 2026: the probe set was expanded from 10 pairs to 64** (`fisherrao/corpus.py`, 8 sense domains, structurally validated by `corpus.validate()`). Both columns are the new `n` = 64 values, with the `n` = 10 draft in brackets:

| instrument | mean | median |
|---|---|---|
| **`d_FR` (Fisher–Rao)** | **4.53** *(8.6)* | **4** *(7)* |
| `d_UU` (Manson) | 26.28 *(26.2)* | 28 *(26)* |
| `d_euc` (Euclidean) | 26.56 *(26.8)* | 27 *(27)* |

**Steepest-rise layer:** `d_FR` median **7** (mean 11.9, **sd 9.51**); `d_UU` median 30 (sd 1.95); `d_euc` median 30 (sd 0.49).

> ### Under the Fisher–Rao metric the model has half-resolved the ambiguity by **layer 4–5** of 30. Under the **raw** flat metrics the same data says **layer 27–28**.

**The expansion did not shrink the effect — it grew it, and it now has an interval.** Paired per-pair, `d_UU` minus `d_FR`:

> **mean gap 21.75 layers, 95% bootstrap CI [20.45, 22.94], and `d_FR` is earlier in 64/64 pairs.**

Standard error on the steepest-rise statistic — the one §3b.4 flagged as too noisy to publish at `n` = 10 — improved from 10.2/√10 = 3.23 to 9.51/√64 = **1.19**, a 2.7× gain.

**No subgroup carries it.** Gap by sense domain: artifact 21.8 (n=33), nature 20.3 (25), institution 21.7 (20), sport 21.9 (13), arts 22.4 (13), body 22.0 (9), science 23.6 (9), food 23.2 (6). Heteronyms — `bass`, `bow`, `lead`, where the two senses differ in *pronunciation* and a subword tokeniser cannot see it — give 18.3 (n=3) against 21.9 for everything else, so if anything they are the weakest cases, as expected.

⚠️ **Read §3b.3 before quoting the 21.75.** It is a gap against the **raw** flat metrics, and the mechanism is norm growth, not a failure of Euclidean geometry as such.

### 3b.3 🔴 CORRECTED: the essential property is scale-invariance, not Fisher–Rao specifically

A first draft called this "a 19-layer disagreement" and read it as showing that the Fisher–Rao metric is uniquely able to localise the computation. **Two controls (`check_polysemy_controls.py`) show that reading is too strong.**

**Control A — is the flat metrics' lateness just residual-norm growth?** Stage 0 measured salience reaching 3.7 × 10⁴ by layer 29. Divide each flat distance by the residual norm at that layer:

Both controls were re-run on the expanded `n` = 64 set and **both hold, with the same conclusions and tighter estimates**. `n` = 10 values in brackets:

| measure | half-depth mean | median |
|---|---|---|
| `d_FR` Fisher–Rao | 4.53 *(8.6)* | **4** *(7)* |
| `d_euc` raw | 26.56 *(26.8)* | 27 *(27)* |
| **`d_euc / ‖h‖`** | **5.16** *(6.0)* | **5** *(6)* |
| `d_UU` raw | 26.28 *(26.2)* | 28 *(26)* |
| **`d_UU / ‖h‖_G`** | **2.45** *(2.3)* | **2** *(1)* |

**Normalising by the norm moves Euclidean from layer 27 to layer 5 — essentially the same answer Fisher–Rao gives.** So the lateness of the raw flat metrics *was* entirely norm growth, confirming the mechanism. But it also means **Fisher–Rao is not uniquely insightful here**: a scale-normalised Euclidean distance localises the computation just as early.

**Control B — is `d_FR`'s early rise an arccos compression artifact?** `d_FR = 2 arccos(BC)` expands differences near `BC`=1 and compresses near `BC`=0, which could manufacture "fast early rise, flat late" on its own. Against measures without that shape at `n` = 64: total variation gives half-depth **3.72**, `1−BC` gives **6.78**, versus `d_FR`'s **4.53**. **No artifact** — `d_FR` sits squarely between them.

**The corrected claim:**

> The real division is **scale-invariant vs scale-sensitive** measures, not Fisher–Rao vs everything else. Every scale-invariant measure (Fisher–Rao 4.5, TV 3.7, `1−BC` 6.8, normalised Euclidean 5.2, normalised `UᵀU` 2.5 — all at `n` = 64) places the resolution in the **early** layers. Every raw, scale-sensitive one (Euclidean 26.6, `UᵀU` 26.3) places it at the **output**, and is wrong for a mechanical reason.
>
> **Fisher–Rao's advantage is that it is scale-invariant by construction, for a principled reason** — RMSNorm makes the radial direction exactly null ([05-stage0-log.md](05-stage0-log.md) §4.2), so no ad-hoc normalisation is needed or possible to get wrong. Euclidean distance needs a manual fix that Manson (2025) and King et al. (2026) **do not apply**, which is why their instruments report the computation happening ~20 layers too late.

That is a narrower claim than the first draft's, and a more useful one: it identifies the specific defect in the published methods and the specific reason the Fisher metric avoids it.

> **This is the fourth claim in this project corrected by a control.** The pattern is consistent: a clean result with a tidy mechanism attached, which a targeted check narrows or overturns.

`d_FR` also **peaks at layer 27 (1.18) and falls back to 1.00** at the output: the two contexts are maximally distinct just before the end, then partially re-converge as both commit to generic continuations.

### 3b.5 🔴 WiC — the metric gap reproduces, and a control the hand-written set never had says the claim is over-stated

**Date:** 12 August 2026 · **Reproduce:** `python run_wic.py 60` → `results/wic/wic_<model>.json`

The hand-written probe set has 64 **different-sense** pairs and nothing else. That design cannot answer the obvious challenge: *do any two occurrences of a word in two different sentences separate like this?* WiC (SuperGLUE validation) supplies exactly that control — 319 different-sense and 319 same-sense pairs on the same target words, written by nobody in this project.

**Two adjustments the corpus forces**, both recorded rather than hidden:

- WiC targets sit wherever they naturally fall, so some have almost no *preceding* context — and under causal attention that is all a hidden state can see. Preceding-context length is recorded per pair.
- WiC targets are **lemmas**, so the two sentences may carry different inflections and therefore **different tokens** ("stripe"/"stripes"). Two different tokens differ at layer 0 with no context at all. **111 of 228 pairs were dropped on this**; everything below is the 117 same-token pairs.

**✅ The metric disagreement reproduces:**

| instrument | half-separation depth (WiC, different-sense, n=56) | hand-written (n=64) |
|---|---|---|
| **`d_FR`** | **7.86** | 4.53 |
| `d_UU` (Manson) | 26.32 | 26.28 |
| `d_euc` | 26.84 | 26.56 |

An **~19-layer gap on a standard dataset**, against 21.8 on the hand-written one. The headline finding of §3b is not an artefact of the author's sentences.

**🔴 But the same-sense control does not behave as the framing implies:**

| preceding context | n (diff / same) | `d_FR` final, **different** sense | `d_FR` final, **same** sense | ratio |
|---|---|---|---|---|
| < 2 tokens | 19 / 25 | 1.212 | 0.0000123 | — *(degenerate: one target at position 0)* |
| **2–4 tokens** | 27 / 20 | 1.699 | **1.488** | **1.14×** |
| **5–8 tokens** | 10 / 15 | 1.870 | **1.854** | **1.01×** |

> **On WiC, once both targets have real preceding context, same-sense pairs separate as much as different-sense pairs.**

Read alone, that says the instruments measure *different context* rather than *different sense*. **A first draft of this section concluded exactly that** — §3b.6 pushes back with a tighter control, and §3b.6a shows that control has its own confound. WiC's same-sense pairs are arbitrary sentences that differ in every way at once, so the control could not distinguish *"the instruments cannot see sense"* from *"WiC's wording differences swamp sense"*. **It is the second.**

### 3b.6 ✅ The TIGHT same-sense control — and it reverses §3b.5's reading

**Reproduce:** `python run_samesense.py` → `results/samesense/samesense_<model>.json`

`corpus.POLYSEMY_SAME` supplies 32 same-sense pairs built to the **same recipe** as the different-sense set — same target word, same final position, matched syntactic frame — and, crucially, **sharing sentence A** with it:

| | sentence A | sentence B |
|---|---|---|
| **different sense** | *…beside the flooded river, she studied the **bank*** | *…reviewing the mortgage paperwork … called the **bank*** |
| **same sense** | *…beside the flooded river, she studied the **bank*** | *…along the swollen stream, he examined the **bank*** |

Both arms measure A against *some other sentence ending in the same word*. **The only difference between the arms is whether that other sentence carries the same sense.** `corpus.validate()` enforces the shared-A requirement, so the arms cannot drift apart.

**Final-layer separation, paired by word, n = 32:**

| instrument | different sense | same sense | ratio | different > same | z |
|---|---|---|---|---|---|
| **`d_FR`** | **1.650** | **0.754** | **2.19×** | **31/32** | **+5.30** |
| `d_UU` | 17960 | 10459 | 1.72× | 26/32 | +3.54 |
| `d_euc` | 627 | 260 | 2.41× | 32/32 | +5.66 |

> ## ✅ **Sense separates — by 2.2× under `d_FR`, in 31 of 32 words.**

### 🔴 3b.6a …but "with wording matched" is not true, and it was my own design that broke it

Auditing the set: **the same-sense B sentences share 2.00× more vocabulary with A than the different-sense B sentences do** (Jaccard 0.235 vs 0.118; same-sense is closer in **30/32** words, z = +4.95).

That is not an accident — it is the *"matched syntactic frame"* requirement doing it. Reusing A's frame reuses its function words. So the arms differ in **two** things: sense, and lexical overlap. **A 2.00× overlap difference and a 2.19× separation difference is exactly the coincidence that should stop a conclusion**, and the section above stated "with wording matched" when the wording is measurably *not* matched.

**Three checks, all of which say the confound is probably not driving it:**

| check | result |
|---|---|
| ρ(overlap, `d_FR`) **within** the different-sense arm | +0.205 |
| ρ(overlap, `d_FR`) **within** the same-sense arm | **−0.240** *(wrong sign for the confound)* |
| pooled: which predicts separation better? | sense **+0.608** vs overlap −0.530 |
| **matched-overlap subset** (\|Δoverlap\| < 0.06) | **2.21×, 6/6 positive, z = +2.45** — same ratio as the full set |

Within an arm, overlap barely predicts separation and does so with *opposite* signs in the two arms, which is not how a real confound behaves. And on the subset where the two arms happen to have equal overlap, the ratio is **2.21×** — indistinguishable from the 2.19× overall.

> **So the sense effect survives every check available, but `n` = 6 on the decisive one.**

### ✅ 3b.6b The confound is now CLOSED by construction, and the result holds

`corpus.POLYSEMY_DIFF` is a third arm: the same sentence A, paired with a **different-sense B written to A's frame** — so both arms reuse A's function words and the overlap is equalised rather than argued about. `corpus.validate()` now **enforces** it, flagging any word whose two arms differ by more than 0.10 Jaccard and the set if the medians drift by more than 0.04.

*Writing it took two passes: the first overshot and gave the different-sense arm **more** overlap than the same-sense arm (0.294 vs 0.235, z = −3.18). The validator caught that too.*

| | median Jaccard with A |
|---|---|
| same-sense B | **0.235** |
| **different-sense B, frame-matched** | **0.250** ✅ same > diff in 11/32, z = −1.77 (n.s.) |
| different-sense B, free frame *(the confounded arm)* | 0.118 |

**Final-layer separation, paired by word, n = 32:**

| instrument | arm | median | ratio vs same-sense | > same | z |
|---|---|---|---|---|---|
| **`d_FR`** | **different, frame-matched** | **1.532** | **2.03×** | **30/32** | **+4.95** |
| | different, free frame | 1.650 | 2.19× | 31/32 | +5.30 |
| | *same sense* | *0.754* | — | — | — |
| `d_UU` | different, frame-matched | 17838 | 1.71× | 23/32 | +2.47 |
| `d_euc` | different, frame-matched | 553 | 2.13× | 32/32 | +5.66 |

> ## ✅ **Closing the confound moves the effect from 2.19× to 2.03×.**
>
> The lexical-overlap difference accounted for about **7%** of it. Sense separates by a factor of two, on three instruments, with overlap controlled by construction.

**And the timing result gets sharper, not weaker.** Half-separation depth:

| | same sense | different, frame-matched | different, free frame |
|---|---|---|---|
| `d_FR` | 4.72 | **4.69** | 5.50 |
| `d_UU` | 27.38 | 27.09 | 26.44 |

With the frame matched, same-sense and different-sense pairs reach half separation at **the same depth to within 0.03 layers**, while differing 2× in magnitude. **Sense is entirely magnitude; the ~22-layer metric disagreement is entirely timing.** The 5.50 in the free-frame arm was the frame difference, not the sense difference.

**So the settled position on RQ3b:**

1. **Contextual variants separate early under Fisher–Rao (≈ 4.7) and late under raw flat metrics (≈ 27).** This holds for same-sense pairs too, reproduces on WiC, and is the finding.
2. **Different senses separate 2.03× further than same senses at the same depth**, with lexical overlap equalised — so the instruments *do* read sense, as magnitude.
3. §3b.5's WiC null is explained: WiC's arbitrary sentence pairs differ so much in wording that a 2× sense effect is not recoverable from them.

**Note the shape of the error.** WiC's control was too loose (arbitrary sentences); mine was too tight in a way that leaked (frame reuse raised overlap). Neither bracket was needed in the end — §3b.6b builds the arm that holds overlap fixed, and the answer is 2.03×. **But the confound was found by measuring the controls against each other, not by inspecting them.**

**But the two claims are now visibly separate, and that is the real refinement:**

| | different sense | same sense |
|---|---|---|
| half-separation depth, `d_FR` | 5.50 | **4.72** |
| half-separation depth, `d_UU` | 26.44 | **27.38** |
| final separation, `d_FR` | 1.650 | **0.754** |

**Sense changes *how much* two variants diverge (2.2×). It barely changes *when* (5.5 vs 4.7).** The ~20-layer metric disagreement is about **context integration in general** — it is there for same-sense pairs too — while the semantic content shows up as magnitude, not timing.

So the correct statements are:

1. **The metric disagreement** (`d_FR` ≈ 5, flat ≈ 26–27) is a claim about *when contextual variants of a token separate*. It holds for same-sense and different-sense pairs alike, and reproduces on WiC. **This is the finding.**
2. **Sense is detectable on top of that** — 2.2× more final separation, z = +5.30, on the tight control — but it is a magnitude effect, not a shift in depth.
3. *"The layer at which ambiguity resolves"* conflates the two and should not be used. *"Contextual variants separate early under Fisher–Rao and late under raw flat metrics, and different senses separate further"* is what the data supports.

**A methodological note worth carrying into Chapter 5.** §3b.5 reached a confident negative conclusion from a control that turned out to be too loose. **This is the second time in two days that a control changed more than one thing at once** — the other was the scramble's row-set confound ([08-rq3a-log.md](08-rq3a-log.md) §5.4 b′). Both were caught by building the tighter version rather than by reasoning about it. *A control is only as good as the list of things it holds fixed, and that list has to be written down.*

### 3b.4 Limits

- ~~**10 pairs**~~ → **64 pairs, one model.** Steepest-rise variance is still high (sd 9.51, per-pair range layer 1 to 30), so **half-separation depth remains the more reliable statistic** — but at `n` = 64 the steepest-rise standard error is 1.19 rather than 3.23, so it is now reportable with an interval rather than only as a median.
- Pairs are hand-written by the author; each target word appears exactly once, at the end of both sentences (enforced by `corpus.validate()`). **Stage 5 task 5.1 asks for ≥ 200 across ≥ 4 domains**; 64 across 8 domains is the honest limit of hand-writing without quality loss. Reaching 200 needs a sense-annotated corpus (SemCor / WiC), which is now downloadable ([10-architecture-log.md](10-architecture-log.md)).
- Absolute `d_FR` at the output is 1.46–2.63 against a theoretical maximum of π, so these are large but not saturated separations.

---

## 4. Status of the research questions

| | |
|---|---|
| **RQ1** — can intrinsic curvature be computed? | ✅ yes, validated to machine precision |
| **RQ2b** — do the proxies track the intrinsic quantity? | ✅ **no** — pooled ρ = −0.14, +0.13, +0.14; within-layer +0.00, −0.00, +0.11 (n=360). No instrument tracks any other. |
| **RQ2b (magnitude)** — is Mabrok's 10⁻⁵ wrong by 10⁴? | ⚠️ **not established** — my proxy gives 2.4e-2, so his setup must be faithfully replicated first |
| **RQ2a/2c** — layer-wise profile | ✅ volume profile established (contraction then recovery); curvature deviation **not identifiable** |
| **RQ2d** — sign | ✅ positive, 100% of planes |
| **RQ3a** — beat King et al.'s r ≈ 0.15 | ⏳ not attempted; needs GPT-2 XL / Pythia (blocked on TLS) |
| **RQ3b** — ambiguity localisation | ✅ **answered** — scale-invariant measures (Fisher–Rao 7, TV 5, normalised Euclidean 6) localise resolution early-to-mid; raw flat metrics say 26–27, an artefact of norm growth (§3b.3) |
| **RQ4** — null-space falsification | ✅ passed in Stage 0 (ratio 10⁻¹⁶) |

---

## 5. Next actions, in order

1. ~~Raise `n` for E2~~ ✅ **done at n=360** — and it retracted the Manson↔King claim (§1.0). Next: diagnose the 4.2% unstable tail in `K` (§1.1b), most likely via a tighter `cond_eff` ceiling.
2. **Replicate Mabrok's proxy faithfully** — larger point cloud, genuinely local neighbourhoods — so the magnitude comparison can either be made properly or dropped.
3. **Extend the polysemy set beyond 10 pairs.** §3b is the strongest result in the project and rests on 10 hand-written pairs; the half-separation statistic is stable but the steepest-rise statistic has sd 10.2. More pairs is cheap — the measurement is closed-form.
4. **Correlate volume and curvature against next-token entropy** — the cheap first cut at RQ3a, available now without GPT-2.
5. **Fix the TLS problem** to unblock GPT-2 XL and Pythia, needed for the King et al. comparison and any cross-architecture claim.
6. Use `curvature.spectral_diagnostics` in bulk loops — it merges the two redundant eigendecompositions per point (~2× on the volume profile).
