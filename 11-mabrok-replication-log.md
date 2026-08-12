# Mabrok's `10⁻⁵` — reproduced, and shown to be a restatement of a free parameter

**Date:** 11 August 2026
**Status:** ✅ **Settled.** The number reproduces — at a ≈0.9999 variance threshold with `k` ≥ 50, on his corpus and cloud size. But on transformer activations the proxy's magnitude is set by that unreported threshold rather than by the geometry, and **at the threshold needed to produce 10⁻⁵ the instrument reads a genuine 3-sphere as exactly flat.** The "four orders of magnitude" comparison stays dropped, for a much stronger reason than "we could not reproduce it".
**Reproduce:** `python run_mabrok_replication.py gpt2 200` → `results/mabrok/mabrok.json`; `python check_pca_tautology.py` for the known-geometry controls

> **§5 was rewritten after its own verification.** The first draft claimed the statistic is a tautology on any data. Running it against a 3-sphere and a flat 3-plane showed that is false in general — and produced a sharper result. **Read §5.1–5.3, not the summary above, before quoting anything.**

---

## 1. Why this was open

Stage 3 claimed this project's intrinsic curvature and Mabrok's proxy differ by **four orders of magnitude**. Stage 4 withdrew it ([07-stage4-log.md](07-stage4-log.md) §1.2): reimplementing the proxy on this project's corpus gave **1.94 × 10⁻²**, not 10⁻⁵, so nothing had been compared like for like. The diagnosis at the time was that a 20-token sentence gives a ~20-point "local neighbourhood", which is not local.

The plan of record was to **email the author**. That turned out to be premature — most of what was missing is in the paper, and the rest can be settled by sweeping it.

## 2. What the paper actually specifies

*Mohamed A. Mabrok, "Latent Semantic Manifolds in Large Language Models", [arXiv:2603.22301](https://arxiv.org/abs/2603.22301), 17 March 2026.*

| | |
|---|---|
| corpus | **"WikiText-103 validation set"** |
| point cloud | **"approximately 1,800 hidden-state vectors per layer"** |
| the proxy | *"for each hidden state `hᵢ` at layer `ℓ`, we compute a local PCA over its **k-nearest neighbors**"*, taking *"the fraction of variance captured by directions **orthogonal to the dominant principal subspace**"* |
| intrinsic dimension | TWO-NN (Facco et al.) and MLE (Levina & Bickel) |
| the result | *"The PCA curvature values are uniformly small across all layers **(order 10⁻⁵)**"* |
| models | six architectures, **124M–1.5B** — so GPT-2 small is inside the range |

**And what it does not specify — the two choices that decide the answer:**

- **the value of `k`**
- **how the dimension of the "dominant principal subspace" is chosen** — estimated intrinsic dimension? a fixed number? a variance threshold?

Both are omitted. That is the reproducibility gap, and it is why the number could not be matched by following the text.

## 3. The design that avoids needing the author

Rather than guess once, replicate his setup exactly where it *is* specified — WikiText-103 validation, GPT-2, **1,800 vectors per layer** — and sweep the two unspecified choices, asking a question that does not depend on the paper being more precise:

> **Is there any `(k, q)` that yields ~10⁻⁵ on his data at his cloud size?**

`k` ∈ {5, 10, 20, 50, 100, 200}; `q` from variance thresholds {0.90, 0.95, 0.99, 0.999, 0.9999, 0.99999} and from fixed values {2, 3, 5, 10}. Layers 1, 6, 12 of GPT-2. TWO-NN intrinsic dimension reported alongside, since the paper cites it: **3.43, 6.72, 7.48** — the hourglass shape it describes.

## 4. ✅ The answer is yes — and that is the least interesting part

`~10⁻⁵` appears at a **0.9999 variance threshold with `k` ≥ 50**:

| | `k`=50 | `k`=100 | `k`=200 |
|---|---|---|---|
| layer 1 | 7.1e−05 | 8.9e−05 | 9.6e−05 |
| layer 6 | — | — | 7.5e−05 |
| layer 12 | — | — | 6.2e−05 |

and at a 0.99999 threshold, layer 1 gives **6.1e−06** and **8.6e−06**. So the published magnitude reproduces on his corpus at his cloud size.

## 5. The measured value tracks the threshold — but *why* took a second experiment

On GPT-2 layer 12, `k` = 200, the residual sits almost exactly at `1 − threshold`:

| threshold | 0.90 | 0.95 | 0.99 | 0.999 | 0.9999 | 0.99999 |
|---|---|---|---|---|---|---|
| `1 − thr` | 1.0e−01 | 5.0e−02 | 1.0e−02 | 1.0e−03 | 1.0e−04 | 1.0e−05 |
| **measured** | 9.89e−02 | 4.94e−02 | 9.77e−03 | 9.31e−04 | 6.20e−05 | 7.9e−33 |
| **ratio** | **0.99** | **0.99** | **0.98** | **0.93** | 0.62 | 0.00 |

> ### 🔴 A first draft of this log stopped here and concluded *"the statistic is a tautology; it returns the threshold on any data."*
>
> **That was too strong, and the verification below caught it.** The mechanism is real but conditional, and the conditional version is the more useful result.

### 5.1 The control: run it on geometries whose answer is known

`check_pca_tautology.py`. Four clouds, same `n` = 1800, same `d` = 768, same sweep. If the statistic really returned the threshold *regardless of data*, all four rows would match `1 − thr`.

| cloud | what it is | ratio to `1 − thr` at `k`=200, thr .90/.95/.99 |
|---|---|---|
| **gaussian** | isotropic noise, no manifold at all | **0.98, 0.98, 0.97** |
| **gpt2** | real activations, WikiText-103 | **0.99, 0.99, 0.98** |
| **sphere3** | 3-sphere in ℝ⁷⁶⁸, curvature +1 | 0.22, 0.44, 0.00 |
| **flat3** | 3-dim linear subspace, curvature 0 | 0.00, 0.00, 0.00 |

**Two rows track the threshold and two do not.** So the statistic is *not* vacuous in general:

- On **flat3** it returns **2.7 × 10⁻³¹** — exact zero in float64. Correct: the patch is flat.
- On **sphere3** it returns **4.65 × 10⁻³** and — the telling part — **the same value at thresholds 0.90, 0.95 and 0.99**. When there is real residual variance the threshold is not binding, and the number is the geometry.

### 5.2 🎯 The corrected claim

> **The proxy is pinned to the threshold exactly when the local spectrum has no gap** — and **real transformer activations have no gap.** GPT-2 (0.99, 0.99, 0.98) is indistinguishable from isotropic noise (0.98, 0.98, 0.97) on this test, while a genuine low-dimensional manifold is not.
>
> So on transformer hidden states, *"PCA curvature ≈ 10⁻⁵"* still reports the analyst's threshold rather than the manifold — **but because of a property of the data, not of the statistic.** That is a claim about what these activations are like, and it is testable, which the tautology version was not.

### 5.3 🔴 And at Mabrok's operating point the instrument stops discriminating

Reaching 10⁻⁵ requires thr ≈ 0.9999. **At that threshold the proxy reads the 3-sphere as exactly flat:**

| `k` | threshold | flat3 | sphere3 | separates? |
|---|---|---|---|---|
| 20 | 0.90 | 2.7e−31 | **4.65e−03** | ✅ yes |
| 20 | 0.99 | 2.7e−31 | **4.65e−03** | ✅ yes |
| 20 | **0.9999** | 2.7e−31 | **2.54e−31** | ❌ **no** |
| 50 | 0.99 | 4.4e−31 | 7.51e−03 | ✅ yes |
| 50 | **0.9999** | 4.4e−31 | 3.61e−31 | ❌ **no** |
| 200 | **0.99** | 4.1e−31 | 3.58e−31 | ❌ **no** |

A threshold that tight demands more principal components than a curved low-dimensional patch possesses, so the tail empties and everything reads as flat. **The operating point that produces 10⁻⁵ is precisely the one at which the instrument cannot tell a sphere from a plane.**

**Under a fixed-`q` rule instead**, the same GPT-2 cloud gives **0.2–0.8** when `q` is below the local rank, and **exactly 0** once `q ≥ k`. Nothing in between, and never 10⁻⁵.

| layer 12, `k`=20 | q=2 | q=3 | q=5 | var0.99 |
|---|---|---|---|---|
| | 0.661 | 0.558 | 0.400 | 0.0065 |

## 6. What this settles

**(a) The magnitude comparison stays dropped — permanently.** Not because the number is irreproducible, but because on transformer activations **its magnitude is set by an unreported free parameter**. Comparing `K ≈ +0.25` against it was never meaningful, and no amount of careful replication would make it so. [06-stage3-log.md](06-stage3-log.md) §6 and the `run_stage3.py` output have already been corrected; this is the evidence behind that correction.

**(b) The correlation result is untouched and is the right claim.** [07-stage4-log.md](07-stage4-log.md) §1 reports Spearman **ρ = −0.14** between the local-PCA proxy and intrinsic curvature at n = 360. Rank correlation is invariant to any monotone reparameterisation, so it does not care what threshold was used. **The instrument-comparison result never depended on the magnitude, which is exactly why it survived.**

**(c) It sharpens RQ2b, and gives it a demonstration rather than an argument.** The question was *"is Mabrok's 'essentially flat, 10⁻⁵' conclusion an artefact of the PCA proxy?"* The answer is now **yes, with a positive control**: at the threshold that yields 10⁻⁵, the same code reports **a unit 3-sphere as flat to 10⁻³¹**. An instrument that cannot see the curvature of a sphere is not evidence that transformer manifolds are flat. This is the cleanest statement of RQ2b in the project and belongs in Chapter 6.

**(d) 🔴 `second_fundamental` is NOT the safe alternative — retracted.** An earlier version of this line said Mabrok's other proxy "has no such degeneracy and is the proxy to use for any magnitude claim". **That was an argument, not a measurement.** Measured the same way (`check_pca_tautology.py` §3, same four clouds):

| cloud | q=2 | q=3 | q=5 | q=10 |
|---|---|---|---|---|
| gaussian noise | 0.345 | 0.419 | 0.534 | 0.735 |
| **flat3** (curvature **0**) | 1.275 | **3.0e−08** | **2.871** | **4.636** |
| **sphere3** (curvature **+1**) | 1.722 | 0.579 | **2.701** | 5.205 |
| gpt2 | 0.122 | 0.143 | 0.177 | 0.244 |

> ## **It separates curved from flat at `q` = 3 and nowhere else — and `q` = 3 is exactly the true intrinsic dimension of both synthetic clouds.**
>
> At `q` = 2 (below the true dimension) a perfectly **flat 3-plane reads as curved** (1.275). At `q` = 5 it reads as **more curved than the sphere** (2.871 vs 2.701). At `q` = 10, more so again.

**So `second_fundamental` has a degeneracy of its own, and a worse one**: `pca_curvature` at a bad threshold reports an uninformative number, whereas `second_fundamental` at a bad `q` reports the **wrong ordering**. Both are pinned to a free parameter; only the failure mode differs.

**And on real data you do not know the right `q`.** The paper's own TWO-NN estimates run 3.4–7.5 across layers ([11-mabrok-replication-log.md](11-mabrok-replication-log.md) §3), and the proxy's value on GPT-2 doubles over that range (0.143 → 0.244).

> **Neither of Mabrok's two proxies supports a magnitude claim.** RQ2b's answer is therefore general rather than about one statistic: *the "essentially flat" conclusion is an artefact of extrinsic proxies whose magnitude is set by an unreported free parameter.* That is the strongest form of the claim in this project and it now rests on positive controls with known answers rather than on a cross-paper comparison.

## 7. Limits

- GPT-2 **small** (124M), not GPT-2 XL. It is inside the paper's stated 124M–1.5B range, but the paper's Figure 5 is XL.
- The tautology argument is exact for **variance-threshold** rules and shown empirically for fixed-`q` rules; if the paper's rule is something else again (say `q` = TWO-NN estimate rounded), the sweep covers that case numerically (`q` = 3, 5, 10 against TWO-NN of 3.4–7.5) and it does not reach 10⁻⁵.
- Preprocessing is unstated in the paper. These clouds are **raw** hidden states. If the paper normalises or PCA-reduces first, the spectrum changes — but the `1 − thr` identity is a property of the statistic, not of the data, so the conclusion in §5 is unaffected.
- Six architectures were used there; one here.

## 8. Next

1. ~~Run the same analysis on `second_fundamental`~~ ✅ **done — §6(d). It is also parameter-pinned, and RQ2b's answer is now general.**
2. Repeat on **GPT-2 XL** to match the paper's headline figure exactly, now that downloads work ([10-architecture-log.md](10-architecture-log.md)).
3. **Re-run E2 on WikiText-103** rather than the hand-written corpus — the proxies were only ever compared on 40 sentences, and this shows the point cloud matters enormously for them.
4. The email to Mabrok is now worth sending for a **different** reason: not to ask how he did it, but to ask whether the threshold reading is right, and to share this. It is a substantive point about his instrument, not a request for help.
