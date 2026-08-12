# Chapter 7 — Behavioural and causal analysis

> **Draft.** Sources: [07-stage4-log.md](../07-stage4-log.md) §3b, [08-rq3a-log.md](../08-rq3a-log.md) §5, [09-e5-log.md](../09-e5-log.md).

---

## 7.1 What this chapter has to establish

Chapters 5 and 6 describe a *metric* and its curvature. Neither shows the metric is about the **model** rather than about a construction laid over it. The fair challenge after Chapter 6 is:

> *You have computed the curvature of something. Why should I believe it is the network's geometry?*

This chapter answers it three ways: the geometry **changes the answer** to a question about the model (§7.2), it **is not a restatement of predictive concentration** (§7.4), and perturbations along directions the metric calls null **provably do nothing to the model's output** (§7.5).

---

## 7.2 🎯 RQ3b — the metric changes where two contextual variants separate, by ~20 layers

The same surface word appears in two contexts. Early in the network the two hidden states are nearly identical; as context is integrated they separate. **Where?**

The Fisher–Rao distance between two categorical distributions is closed form,

```
d_FR(p, q) = 2 · arccos( Σᵢ √(pᵢ qᵢ) )
```

so this needs no `k`, no frame, no cutoff and no autodiff — it is the cheapest genuinely-Fisher measurement in the work and immune to every sensitivity problem in Chapter 6.

**Half-separation depth** — the first layer reaching 50% of final separation, over 64 hand-written minimal pairs across 8 sense domains:

| instrument | mean | median |
|---|---|---|
| **`d_FR` (Fisher–Rao)** | **4.53** | **4** |
| `d_UU` (Manson) | 26.28 | 28 |
| `d_euc` (Euclidean) | 26.56 | 27 |

> **Paired, per-pair: mean gap 21.75 layers, 95% bootstrap CI [20.45, 22.94], and `d_FR` is earlier in 64/64 pairs.**

**No subgroup carries it.** By sense domain the gap runs 20.3–23.6 across eight domains; heteronyms (`bass`, `bow`, `lead`, where the two senses differ in pronunciation and a subword tokeniser cannot see it) give 18.3 against 21.9 for the rest — if anything the weakest cases, as expected.

### 7.2.1 It reproduces on a corpus nobody here wrote

WiC (SuperGLUE validation), restricted to the **339 pairs whose target is the same token in both sentences** (299 were dropped because WiC targets are lemmas and the two sentences may carry different inflections — two different tokens differ at layer 0 with no context at all):

| instrument | half-depth, different-sense (n = 163) |
|---|---|
| **`d_FR`** | **6.75** |
| `d_UU` | 26.57 |
| `d_euc` | 26.61 |

**A ~20-layer gap on a standard dataset**, against 21.8 on the hand-written one.

### 7.2.2 🔴 The mechanism is scale-invariance, not Fisher–Rao specifically

A first draft read this as showing the Fisher–Rao metric uniquely able to localise the computation. **Two controls narrow that.**

**Control A — is the flat metrics' lateness just residual-norm growth?** The residual norm reaches ~3.7 × 10⁴ by layer 29. Dividing each flat distance by the norm at that layer:

| measure | half-depth mean |
|---|---|
| `d_FR` Fisher–Rao | 4.53 |
| `d_euc` raw | 26.56 |
| **`d_euc / ‖h‖`** | **5.16** |
| `d_UU` raw | 26.28 |
| **`d_UU / ‖h‖_G`** | **2.45** |

**Normalising moves Euclidean from layer 27 to layer 5** — essentially the answer Fisher–Rao gives. So the lateness of the raw flat metrics *was* entirely norm growth.

**Control B — is `d_FR`'s early rise an arccos artefact?** `2 arccos(BC)` expands differences near `BC` = 1 and compresses near 0, which could manufacture "fast early rise, flat late". Against measures without that shape: total variation gives 3.72, `1 − BC` gives 6.78, versus `d_FR`'s 4.53. **No artefact.**

> **The real division is scale-invariant versus scale-sensitive measures, not Fisher–Rao versus everything else.** Every scale-invariant measure places the resolution in the early layers; every raw scale-sensitive one places it at the output, and is wrong for a mechanical reason.
>
> **Fisher–Rao's advantage is that it is scale-invariant by construction, for a principled reason** — the normalisation layer makes the radial direction exactly null (Chapter 4 §4.2.2), so no ad-hoc normalisation is needed or possible to get wrong. Manson (2025) and King et al. (2026) apply no such fix, which is why their instruments report the computation happening ~20 layers too late.

### 7.2.3 Does it read *sense*, or only *context*?

Two same-sense controls were run, **confounded in opposite directions**, and the honest answer required building the second.

**WiC's same-sense arm** says sense is barely detectable: once both targets have real preceding context, same-sense pairs separate 1.01–1.14× as much as different-sense ones. But WiC's pairs are arbitrary sentences that differ in every way at once.

**A purpose-built minimal-pair control**, three arms sharing sentence A, with lexical overlap equalised by construction and enforced by the corpus validator:

| | sentence A | sentence B |
|---|---|---|
| **different sense** | *…beside the flooded river, she studied the **bank*** | *…the quiet lobby beside the mortgage adviser, she studied the **bank*** |
| **same sense** | *…beside the flooded river, she studied the **bank*** | *…along the swollen stream, he examined the **bank*** |

Overlap with A: same-sense **0.235**, different-sense **0.250** — matched, if anything biased *against* the effect.

| instrument | different sense | same sense | ratio | different > same | z |
|---|---|---|---|---|---|
| **`d_FR`** | **1.532** | **0.754** | **2.03×** | **30/32** | **+4.95** |
| `d_UU` | 17838 | 10459 | 1.71× | 23/32 | +2.47 |
| `d_euc` | 553 | 260 | 2.13× | 32/32 | +5.66 |

> **With lexical overlap controlled, sense separates by 2.03×.** WiC's null came from its pairs differing in wording as much as in meaning.

**But the two claims are separate, and that is the refinement:**

| | different sense | same sense |
|---|---|---|
| half-separation depth, `d_FR` | 4.69 | **4.72** |
| final separation, `d_FR` | 1.532 | **0.754** |

**Sense changes *how much* two variants diverge. It barely changes *when*.** The ~20-layer metric disagreement is about **context integration in general** — it is there for same-sense pairs too — while semantic content shows up as magnitude.

*"The layer at which ambiguity resolves"* conflates the two and should not be used.

---

## 7.3 🎯 RQ4 — null directions do nothing

The falsification test, and the one result in the thesis that could have failed outright.

Perturb `h` along an exact null direction of `G(h)` at a step size that **doubles the hidden state**, and along a random direction of the same Euclidean size:

| model | exact null | random | ratio |
|---|---|---|---|
| SmolLM2-135M | 6.0e−15 | 2.59 | 1.1e−12 |
| gpt2 | 1.97e−13 | 3.86 | 3.2e−12 |
| pythia-160m | 6.88e−11 | 7.77 | 6.4e−11 |
| llama-160m | — | — | 3.7e−14 |

**Passes on all six models**, across two normalisation kinds and four architecture families.

**The graded middle case is the part worth quoting.** The *numerical* null — the smallest eigendirection of `G` among those the metric can see — sits between: `KL` = 8.5 × 10⁻² against random's 2.59, a factor of 30. **The metric does not merely separate "null" from "not null"; it ranks directions by how much they matter, and the model's behaviour bears the ranking out.**

### 7.3.1 The companion claim, narrowed three times

Gate A says `KL = ½ε²vᵀGv + O(ε³)`, so **three directions scaled to `vᵀGv = 1` are predicted to behave identically** to leading order. Any systematic difference at larger `ε` is beyond what the metric encodes — which is where curvature should live. The prediction: directions of high Ricci curvature depart from the quadratic sooner.

At `n` = 40 on one model this was significant (30/40, z = +3.16). **It did not survive.**

| | what happened |
|---|---|
| `n` = 40 → ~105 | collapsed on two of three models (SmolLM2 +3.16 → **+0.29**; gpt2 → **+0.00**) |
| a 2×2 across six models at `k` = 5 | suggested a clean **tied vs untied** split — 3/3 untied significant, 3/3 tied at chance |
| sweeping `k` ∈ {4, 5, 6} | **broke that too** — a *tied* model is significant at `k` = 4, and llama-160m only at `k` = 5 |

**What survives:** only **pythia-160m** is significant at every `k` tested (+4.75, +4.54, +3.86), with pythia-70m at both `k` tried. The robust statement is about the **GPT-NeoX family**, not about embedding tying.

> **Three successive narrowings of one claim.** The rule this work draws from it is stated in Chapter 4 §4.4.1 and applies generally: *a result at one `n` and one `k` is a hypothesis.*

---

## 7.4 🎯 RQ3a — the curvature–entropy relationship, and whether it is definitional

Intrinsic curvature tracks next-token entropy far more strongly than any published proxy:

| instrument | pooled ρ | within-layer ρ |
|---|---|---|
| **intrinsic scalar `R`** | **−0.578** | **−0.454** |
| **intrinsic `K`** | **−0.485** | **−0.428** |
| local-PCA (Mabrok) | +0.281 | −0.102 |
| Frenet `UᵀU` (Manson) | −0.252 | +0.018 |
| Euclidean angle (King) | −0.204 | −0.131 |

**2–3× stronger than any proxy**, and the gap widens under the within-layer control where the proxies fall to −0.13…+0.02.

The sign is negative: **curvature is higher where the prediction is sharper.**

### 7.4.1 But is that a fact about the model, or about concentration?

Entropy and curvature are both computed from the same `p(h)`. As `p` concentrates, the reachable region shrinks and the induced geometry changes **whether or not the model learned anything**. This is the question the whole control apparatus of Chapter 4 §4.5 exists to answer.

With the clean paired scramble — `p` untouched, direction set identical, only the assignment destroyed:

| `k` | n | `K` real | `K` scrambled | sign test | z |
|---|---|---|---|---|---|
| 4 | 60 | 0.2554 | 0.0097 | 55/60 | **+6.45** |
| 5 | 60 | 0.2555 | 0.0003 | 55/60 | **+6.45** |
| **5** | **221** | **0.2546** | **0.0109** | **201/221** | **+12.18** |

The last row is the full point file; the n = 60 and n = 126 rows quoted elsewhere are earlier checkpoints of the same run, and every shared row agrees to 1e−12 ([12-audit-log.md](../12-audit-log.md) §3).
| 6 | 60 | 0.2537 | 0.0297 | 51/60 | **+5.42** |

And it is not conditioning: matching `cond_eff` between the arms (21.4 vs 25.8) leaves the effect intact.

> ## 🎯 **At exactly matched entropy, matched conditioning, matched direction set, and stably across `k`, destroying the learned token→direction assignment collapses curvature from `K ≈ +0.255` to ≈ 0.**
>
> **Which probability sits on which direction** is what puts the manifold at the simplex value. The geometry is not a function of predictive concentration alone.

### 7.4.2 Where in the network the assignment matters

Layer-stratified, 14 points at each of 9 layers, `n` = 126, 113/126, z = **+8.91**:

| layer | 1 | 5 | 10 | 15 | 20 | 25 | 28 | 29 | 30 |
|---|---|---|---|---|---|---|---|---|---|
| `K` retained after scrambling | 22.6% | **2.3%** | **2.4%** | **1.4%** | −5.3% | −186% | 56.2% | 18.4% | 66.2% |
| sign test | 12/14 | **14/14** | 13/14 | 13/14 | 13/14 | **14/14** | 12/14 | 12/14 | 10/14 |

**Present at every layer, but U-shaped in magnitude.** Layers 5–25 are where the learned assignment does nearly all the work; the two ends are where it matters least — at layer 1 context has barely been integrated, and at layers 28–30 the prediction is nearly committed and `p` is concentrated enough that the geometry is closer to being fixed by concentration alone.

### 7.4.3 And where the model-specific effect is strongest

Entropy-stratified, 12 points per bin:

| entropy (nats) | `K` real | `K` scrambled | retained | sign test |
|---|---|---|---|---|
| **[0, 0.5)** | 0.2833 | **−1.4226** | −502% | **12/12** |
| [0.5, 1) | 0.2555 | −0.0802 | −31% | **12/12** |
| [1, 2) | 0.2554 | +0.0044 | +2% | **12/12** |
| [2, 4) | 0.2545 | +0.0884 | 35% | 10/12 |
| [4, ∞) | 0.2028 | +0.0346 | 17% | 9/12 |

> **`K_real` is nearly constant across the entropy range (0.20–0.28) while `K_scrambled` swings from −1.42 to +0.09.**
>
> The learned assignment is not merely *adding* curvature — it is **stabilising** it at the simplex value across concentration regimes.

### 7.4.4 Two caveats that must travel with these numbers

**The comparison to King et al. is not like-for-like.** They report Pearson; the intrinsic quantities have heavy tails (`K` ranges over [−0.51, 16.7]) that destroy product-moment statistics — intrinsic `K` gives Pearson −0.110 against their published 0.15. Spearman is the appropriate statistic here and must be reported as the primary one **with this stated**, not quietly substituted. Their sign is also opposite, because their quantity is path bending and this one is space bending, and Chapter 6 §6.3 showed the two are uncorrelated.

**The defensible claim is the within-study one:** on identical activations under an identical protocol, intrinsic Fisher–Rao curvature tracks predictive entropy substantially better than the three published proxies. That needs no cross-paper comparison.
