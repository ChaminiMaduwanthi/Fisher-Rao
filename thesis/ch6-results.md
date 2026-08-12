# Chapter 6 — Layer-wise results and the instrument comparison

> **Draft.** Sources: [06-stage3-log.md](../06-stage3-log.md), [07-stage4-log.md](../07-stage4-log.md), [10-architecture-log.md](../10-architecture-log.md), [11-mabrok-replication-log.md](../11-mabrok-replication-log.md).

---

## 6.1 RQ1 — can the intrinsic curvature be computed at all?

**Yes.** The full Riemann tensor of the Fisher–Rao pullback metric, on a transformer hidden state, at ~2.6 s per point for `k` = 5. This is the first such computation reported for a transformer; the two papers that derive this metric correctly both call the curvature intractable and substitute proxies.

The obstacle was never the algebra. It was (i) conditioning, (ii) an exact nullity from the normalisation layer, and (iii) the fact that the natural `k`-selection rule chooses a `k` at which the tensor cannot be evaluated. Chapter 4 addresses all three.

---

## 6.2 🎯 The manifold is strongly positively curved

**`K ≈ +1/4` on four architectures, 456 points, `k` = 5, 16 planes per point.**

| model | family | norm | tied | n | **median `K`** | IQR | \|K − 0.25\| |
|---|---|---|---|---|---|---|---|
| SmolLM2-135M | Llama | RMSNorm | ✅ | 108 | **0.2547** | [0.2507, 0.2608] | 0.005 |
| llama-160m | Llama | RMSNorm | ❌ | 101 | **0.2559** | — | 0.006 |
| gpt2 | GPT-2 | LayerNorm | ✅ | 117 | **0.2599** | [0.2411, 0.2929] | 0.010 |
| pythia-160m | GPT-NeoX | LayerNorm | ❌ | 130 | **0.2716** | [0.2607, 0.3078] | 0.022 |

All four sit within **0.022** of the ambient simplex value, across a change of normalisation kind, of depth (30 → 12 layers), of width (576 → 768) and of embedding regime.

Per-layer medians on SmolLM2 across nine depths: **0.241–0.264**, with no trend.

> **This is not "essentially flat at 10⁻⁵".** It is the ambient simplex value, and it is the first direct measurement of the quantity the field has been proxying.

### 6.2.1 What `+1/4` does and does not mean

`+1/4` is the curvature of the *ambient* categorical simplex. That the learned submanifold sits at that value says the representation is locally simplex-like — it does **not** by itself distinguish one model from another, and absolute curvature is a weak discriminator by construction.

The obvious next move — measure the **deviation** `K − 1/4` — was tried and **failed its sensitivity check**: unstable in `k`, with the layer ranking scrambling between `k` = 4 and `k` = 7 (ρ = +0.20). It is reported as **not identifiable**, per the pre-committed rule.

### 6.2.2 The sign distribution

A median `K` near `+1/4` is compatible with mixed signs. It is not:

| model | points with **every** sampled plane positive | points with ≥1 negative plane | most negative plane |
|---|---|---|---|
| SmolLM2-135M | 95/108 (88%) | 13 | −21.1 |
| gpt2 | 95/108 (88%) | 13 | −80.3 |
| pythia-160m | 101/108 (94%) | 7 | −2.7 |

> The correct statement is **"the median point has every sampled plane positively curved, and 88–94% of points do"** — not "every plane is positive". Between 6% and 12% of points carry at least one negatively curved 2-plane.

**Those points are not numerical failures.** Across all 456 points, the number of negative planes correlates with **effective dimension** at ρ = **+0.419**, with entropy at only +0.212, and with `cond_eff` at **−0.024** — i.e. the deflationary explanation contributes nothing. Restricting to a narrow entropy band leaves the dimension relationship essentially unchanged (+0.397).

**Negative curvature appears where the metric spreads over more directions.** This is the first mechanism this work has for the minority tail, and it is a geometric predictor rather than a numerical or concentration artefact.

---

## 6.3 🎯 E2 — the adjudication experiment

The centrepiece. Four instruments on **identical activations**, `n` = 360 stratified at 40 points per layer, standard error on ρ ≈ 0.053 so the 95% interval is ≈ ±0.10.

| pair | pooled ρ | within-layer mean ρ |
|---|---|---|
| intrinsic vs local-PCA (Mabrok) | −0.144 | **+0.000** |
| intrinsic vs Frenet `UᵀU` (Manson) | +0.133 | **−0.003** |
| intrinsic vs Euclidean angle (King et al.) | +0.137 | **+0.112** |
| local-PCA vs Manson | +0.071 | — |
| local-PCA vs King | +0.178 | — |
| Manson vs King | +0.248 | **+0.063** |

> ### **No instrument tracks any other.** Every pooled \|ρ\| ≤ 0.25; every within-layer \|ρ\| ≤ 0.11.
>
> In particular **the three published proxies do not detectably track intrinsic Fisher–Rao curvature** — the finding the experiment was built to test.

**Positive control:** `K` against scalar `R`, the same tensor under a different contraction, gives ρ = **+0.720**. The nulls are informative, not vacuous.

**Layer confound removed:** Manson's quantity is strongly depth-dependent (ρ = −0.724 against layer). The within-layer column removes that and every conclusion is unchanged or weaker.

### 6.3.1 One claim retracted here

An `n` = 40 draft reported ρ = **+0.685** between Manson and King and read it as "both measure path bending". At `n` = 360 it is **+0.248** pooled, **+0.063** within-layer. It was a small-sample artefact; the ±0.32 interval at `n` = 40 admitted almost anything.

In hindsight there was no reason to expect agreement: **Manson's curvature runs along the *layer* axis at a fixed token; King's runs along the *token* axis at a fixed layer.** They are path-bending measures on different paths.

### 6.3.2 The conclusion survives a change of corpus — and the instruments split on a new axis

Same protocol, both corpora, `n` = 360 WikiText / 293 hand-written:

| instrument | hand-written | **WikiText-103** | \|change\| |
|---|---|---|---|
| **intrinsic `K`** | −0.447 | **−0.475** | **0.028** |
| **intrinsic `R`** | −0.579 | **−0.544** | **0.034** |
| local-PCA | +0.175 | +0.050 | 0.126 |
| Manson | −0.113 | −0.217 | 0.104 |
| King | +0.043 | −0.151 | **0.194** |

> ## 🎯 **The intrinsic instrument is corpus-invariant. The proxies are not.**
>
> The two intrinsic quantities move by 0.03 between hand-written sentences and encyclopedia prose. Every proxy moves by 0.10–0.19 — three to seven times as much — and King's angle changes sign.

This is an **instrument-quality** argument independent of which instrument is "right": a measurement that reports a different answer depending on what text you feed it is not measuring a property of the model. E2's own conclusion also gets *cleaner* on real text — the largest cross-instrument correlation falls from 0.295 to **0.108**, while the positive control holds at +0.695.

---

## 6.4 🎯 RQ2b — Mabrok's `10⁻⁵`, reproduced and disqualified

The published claim is that local-PCA curvature is "uniformly small across all layers (order 10⁻⁵)", supporting an essentially-flat manifold. The paper specifies the corpus (WikiText-103 validation) and the cloud size (~1 800 vectors per layer) but **not** the neighbourhood size `k`, and **not** how the dimension of the "dominant principal subspace" is chosen.

Replicating his setup where it *is* specified and sweeping the two choices that are not:

**The number reproduces.** `~10⁻⁵` appears at a **0.9999 variance threshold with `k` ≥ 50** — 6.2e−05 to 9.6e−05 across layers.

**But the magnitude is the threshold.** On GPT-2 activations at `k` = 200:

| threshold | 0.90 | 0.95 | 0.99 | 0.999 |
|---|---|---|---|---|
| `1 − thr` | 1.0e−01 | 5.0e−02 | 1.0e−02 | 1.0e−03 |
| **measured** | 9.89e−02 | 4.94e−02 | 9.77e−03 | 9.31e−04 |
| **ratio** | **0.99** | **0.99** | **0.98** | **0.93** |

**Why**, established against clouds whose answers are known — the statistic is *not* vacuous in general:

| cloud | ratio to `1 − thr` |
|---|---|
| isotropic noise (no manifold) | 0.98, 0.98, 0.97 |
| **GPT-2 activations** | **0.99, 0.99, 0.98** |
| 3-sphere, curvature +1 | 0.22, 0.44, 0.00 |
| flat 3-plane, curvature 0 | 0.00, 0.00, 0.00 |

The proxy works on a clean low-dimensional manifold. **Real hidden states have no spectral gap and are indistinguishable from isotropic noise on this test**, so on them the residual is whatever the threshold leaves over.

### 6.4.1 The decisive demonstration

At the threshold that produces `10⁻⁵`, **the same code reports a unit 3-sphere as flat**:

| `k` | threshold | flat3 | sphere3 | separates? |
|---|---|---|---|---|
| 20 | 0.99 | 2.7e−31 | **4.65e−03** | ✅ |
| 20 | **0.9999** | 2.7e−31 | **2.54e−31** | ❌ |
| 50 | **0.9999** | 4.4e−31 | 3.61e−31 | ❌ |

> **An instrument that cannot see the curvature of a sphere is not evidence that transformer manifolds are flat.** RQ2b's answer is a positive control, not a cross-paper comparison.

### 6.4.2 The other proxy fails too

Mabrok's second proxy, `‖II‖`, has no variance threshold and was expected to be the safe alternative. It is not:

| cloud | q=2 | q=3 | q=5 | q=10 |
|---|---|---|---|---|
| **flat3** (curvature 0) | 1.275 | **3.0e−08** | **2.871** | **4.636** |
| **sphere3** (curvature +1) | 1.722 | 0.579 | **2.701** | 5.205 |

It separates curved from flat at `q` = 3 — the true intrinsic dimension — **and nowhere else**. At `q` = 5 a perfectly flat plane reads as *more* curved than the sphere. On real data the intrinsic dimension is estimated, not known, and the paper's own estimates span 3.4–7.5.

> **Neither of the two published proxies supports a magnitude claim.** The "essentially flat" conclusion is an artefact of extrinsic proxies whose magnitude is set by an unreported free parameter.

---

## 6.5 🎯 The central methodological result: spectrum reads concentration, curvature does not

One control — the paired scramble of Chapter 4 §4.5.1 — applied to three quantities derived from the same metric:

| quantity | reproduced by a structure-free scramble at matched entropy? |
|---|---|
| **sectional `K`** | ❌ **no** — collapses 0.2540 → 0.0139 (113/126, z = +8.91) |
| layer-wise log-volume | ✅ yes, ρ = **+0.967** |
| layer-wise `k_eff` | ✅ yes, ρ = **+0.987** |

> ## **Everything derived from the metric's *spectrum* reads predictive concentration. Only the *curvature* — which needs the second derivatives — carries the learned token→direction assignment.**

This is the sharpest form of the thesis's claim, and it rests on three quantities under one control rather than on an argument.

### 6.5.1 Consequences for two published shapes

**The volume profile.** The U-shape with a minimum at layer 20 is real but **substantially definitional**: a scramble at matched entropy puts the minimum at the same layer and keeps 95% of the range. The genuine learned component is a level shift of 0.11 nats/dimension against a range of 0.78.

**The hourglass.** `k_eff` dips at relative depth **0.40**, inside the 0.3–0.4 band reported for ambient intrinsic dimension by Valeriani et al. (2023) and Mabrok (2026). But a scramble reproduces the entire profile — same minimum layer, 99.5% of the range, ρ = **+0.987**.

> The depth agreement with the published estimates is therefore **not** evidence that this metric recovers their intrinsic dimension. It is evidence that **both are tracking predictive concentration** — a testable prediction about their result.

### 6.5.2 Log-volume is not a cross-model quantity

Cross-model log-volume differs by 0.69 across four models and **four candidate mechanisms have been tested and rejected**: embedding tying, the final-norm gain, predictive entropy, and unembedding row-norm uniformity (the last by intervention, with a passing positive control). Combined with §6.5.1's finding that the *layer profile* is definitional, the supported reading is that log-volume is the sum of many small parameterisation differences.

**Use it within a model, across layers, after the sphere correction. Do not use it between models, and do not use it as evidence about learned structure.**
