# E5 — Causal intervention (RQ4). The geometry predicts behaviour.

**Date:** 11 August 2026 · **extended to six models across four families, 12 August 2026**
**Status:** ✅ **Prediction 1 passes on all six models** (worst ratio 6.4 × 10⁻¹¹) — this is a machine-precision result and it is solid. ⚠️ **Prediction 2 is robust on the GPT-NeoX family only.** At `k` = 5 it looked like a clean tied/untied split across six models (§2.2a) — but sweeping `k` broke that: a *tied* model is significant at `k` = 4, and only **pythia-160m** is significant at every `k` tested (4, 5, 6), with pythia-70m at both `k` tried. Everything else is `k`-dependent or null (§2.2b). **Three successive narrowings; treat one-`k` results as hypotheses.**
**Reproduce:** `python run_e5.py 120 <model_id> 5 440` — checkpointed, re-run until it reports the full `n` → `results/e5/e5_<model>_k5.json`

> 🔴 **A sign bug was found in the course of extending this to GPT-2, and it had corrupted the ε=1 column of the original single-model table.** See §1.1. The corrected numbers are throughout; nothing else in the project was affected, because the bug lived in a vector whose *sign* only matters when you step along it.

---

## 0. Why this experiment is the one that matters

Everything before this describes a *metric*. None of it shows the metric is about the **model** rather than about an arbitrary construction laid over it. A reviewer's fair question after Stage 4 is:

> You have computed the curvature of something. Why should I believe it is the network's geometry and not an artefact of your chart?

E5 answers it causally rather than by correlation. It is also the experiment Manson 2025 lists in its own future work as the missing piece of the layer-curvature literature ([01-literature-review.md](01-literature-review.md) Stream B).

**Two predictions, which fail in different ways** — so passing both is informative and passing one is not.

---

## 1. ✅ Prediction 1 — null directions do nothing

Perturb `h` along an exact null direction of `G(h)`; the output must not move. At the **same Euclidean step size** a random direction must move it a great deal. Steps are `ε·‖h‖`, so `ε = 1` doubles the hidden state.

**SmolLM2-135M** (`n` = 40):

| direction | ε=0.001 | ε=0.01 | ε=0.1 | ε=0.3 | **ε=1** |
|---|---|---|---|---|---|
| **exact null** (radial) | −1.05e−16 | 1.74e−16 | 5.52e−16 | 1.21e−15 | **6.01e−15** |
| **numerical null** (smallest `G` eigendirection) | 3.13e−13 | 1.82e−09 | 1.66e−05 | 1.28e−03 | 8.48e−02 |
| **random** | 2.90e−06 | 2.88e−04 | 2.81e−02 | 3.03e−01 | **2.593** |
| **\|null\|/random** | 3.6e−11 | 6.1e−13 | 2.0e−14 | 4.0e−15 | 2.3e−15 |

**And on the other architectures**, including the two LayerNorm ones whose norm has a *second* exact null direction:

| model | worst \|null\|/random | exact null at ε=1 | random at ε=1 |
|---|---|---|---|
| SmolLM2-135M (RMSNorm) | **1.1e−12** | 6.0e−15 | 2.59 |
| gpt2 (LayerNorm) | **3.2e−12** | 1.97e−13 | 3.86 |
| pythia-160m (LayerNorm) | **6.4e−11** | 6.88e−11 | 7.77 |
| llama-160m (RMSNorm, untied) | **3.7e−14** | — | — |

> **PASS on all six.** Doubling a hidden state changes the prediction by `KL` ≈ 10⁻¹³–10⁻¹⁴ while a random step of the same size changes it by 2.6–7.8. **The radial-nullity result is a property of scale-invariant normalisation, not of one model** — it holds across two normalisation kinds, four architecture families and both embedding regimes.

### 1.1 🔴 The bug this uncovered — and why the median hid it

Extending E5 to GPT-2 produced a **FAIL**: exact-null `KL` = **4.15** at ε=1 against a random direction's 3.86. The ε ≤ 0.3 columns were clean at 10⁻¹³, so it was not a general failure of scale invariance — and a direct check confirmed `KL(p(h) ‖ p(2h))` = 10⁻¹²–10⁻¹⁵ on all three models.

**The cause was the sign of a basis vector.** `LM.null_directions` builds its basis with `torch.linalg.qr`, and QR is free to return `−q` for any column. Measured, it returned **−ĥ** at **12/12 layers of gpt2**, 10/12 of pythia and 19/30 of SmolLM2 — data- and LAPACK-dependent, with no pattern. The step then becomes

```
h + ε·‖h‖·(−ĥ)  =  (1 − ε)·h
```

so **ε = 1 lands on the origin**, where a LayerNorm model outputs its bias and the KL against `p(h)` is naturally large. It is not a null perturbation at all.

**Why nothing else broke.** Everywhere else this basis is used as a *projector*, `I − N Nᵀ`, which is sign-blind. Verified explicitly: the projector is bit-identical before and after the fix on all 30/12/12 layers, so **no curvature, volume or control result changes.** Only code that *steps along* the vector was affected, and E5 is the only such code.

**Why the original single-model table looked fine.** SmolLM2's signs came out mixed across layers, so a majority of the sampled points were unaffected and the **median** was clean — while individual points were catastrophically wrong. *A median over a mixture of correct and broken points is not a robustness property; it is a way to not notice.*

Fixed by signing the columns canonically (column 0 along `+h`, column 1 along `+1`).

**A failure here would have voided every curvature number in the thesis** — it is the one result that is not a matter of degree. The KLs in the exact-null row are ±10⁻¹⁶, i.e. zero up to float64 rounding, and their sign is noise (which is why the verdict compares magnitudes; a signed maximum would let a large negative excursion read as a pass).

**The graded middle row is the part worth quoting.** The *numerical* null — the smallest eigendirection of `G` among those the metric can actually see — sits between the two, at **8.5 × 10⁻²** against random's **2.59**, a factor of 30. So the metric does not merely separate "null" from "not null"; it **ranks directions by how much they matter**, and that ranking is borne out by the model's own behaviour.

---

## 2. 🎯 Prediction 2 — where first order goes blind, curvature does not

This is the new contribution of E5 and it needs the setup stated carefully.

Gate A established

```
KL( p(h) ‖ p(h + εv) )  =  ½ ε² vᵀG(h)v  +  O(ε³)
```

So **if three directions are scaled to `vᵀGv = 1`, the metric predicts the same KL for all three**, to leading order. It has nothing left to say about which of them is special. Any systematic difference at larger `ε` is therefore, by construction, beyond what the metric alone encodes — and that is exactly where curvature lives.

**Prediction:** directions of high Ricci curvature depart from the quadratic sooner than directions of low Ricci curvature.

`KL / (½ε²)` — 1.00 means the metric fully explains the behaviour:

| direction | ε=0.001 | ε=0.01 | ε=0.1 | ε=0.3 | **ε=1** |
|---|---|---|---|---|---|
| **high Ricci** | 0.9996 | 0.9962 | 1.0359 | 1.2274 | **2.4023** |
| **low Ricci** | 1.0002 | 1.0015 | 0.9979 | 1.0290 | **0.9956** |
| **random** | 1.0000 | 1.0003 | 1.0018 | 1.0022 | **0.9593** |

Median Ricci eigenvalue: high **+1.547**, low **+0.975**.

> **Paired sign test at ε=1: `KL(high Ricci) > KL(low Ricci)` in 30/40 points, z = +3.16, significant.**

> 🔴 **THE TABLE ABOVE IS `n` = 40 AND DOES NOT SURVIVE. Do not quote it — read §2.1.**

### 🔴 2.1 It collapses at higher `n` on two of three models

*(§2.2 then explains which two, and why.)*

The table above is `n` = 40 on one model. Raising `n` was the obvious next step, and it **destroyed the result on two of three architectures**:

| model | 40 | ~52–61 | **~105** |
|---|---|---|---|
| SmolLM2-135M | 30/40, **z = +3.16** ✅ | 35/52, z = +2.50 | **54/105, z = +0.29** ❌ |
| gpt2 | 25/40, z = +1.58 ❌ | 35/61, z = +1.15 | **50/100, z = +0.00** ❌ |
| pythia-160m | 28/40, z = +2.53 ✅ | — | **77/107, z = +4.54** ✅ |

**Monotone decay with `n` on two models is the signature of a small-sample artifact**, and 50/100 on GPT-2 is exactly chance. The final position, `n` ≈ 105 each:

| model | hi | lo | rand | sign test | z | |
|---|---|---|---|---|---|---|
| SmolLM2-135M | 1.526 | 1.159 | 0.952 | 54/105 | +0.29 | ❌ |
| gpt2 | 1.192 | 0.984 | 0.953 | 50/100 | +0.00 | ❌ |
| **pythia-160m** | **2.157** | 0.962 | 1.033 | **77/107** | **+4.54** | ✅ |
| pooled | | | | 181/312 | +2.83 | ⚠️ *carried entirely by Pythia* |

> ## 🔴 **The claim does NOT hold in general.**
>
> It holds on **Pythia-160m** and fails on the other two at comparable `n`. The pooled `z` = +2.83 is real arithmetic and **misleading** — remove Pythia and it is 104/205, z = +0.10.
>
> **This looked like the end of the result. §2.2 shows it is a boundary condition instead** — the split is tied vs untied embeddings, and a fourth model makes it clean.

**Note the discrepancy the sign test exposes.** On all three models the *median* ratios are still ordered as predicted (`hi` > `lo` ≥ `rand`), yet the *per-point paired* comparison is a coin flip on two of them. That means what median differences there are come from a **minority of points with large excursions**, not from a consistent per-point ordering — and the median alone would have hidden that, exactly as it hid the sign bug in §1.1.

**This is the fifth claim in this project overturned by a control or by raising `n`** — after the simplex sign flip, the curvature-deviation "signal", the Manson↔King agreement, and the RQ3b over-reading. The pattern is consistent enough to be a working rule: **on this data, `n` = 40 is not enough to distinguish an effect from noise, whatever the `z` says.**

**What survives:** §1 — null directions do nothing, on all three architectures, at machine precision. That is the falsification test and it is not a statistical claim.

### 🎯 2.2 The surviving effect is about TIED vs UNTIED EMBEDDINGS

§2.1 left Pythia as the one model where the prediction held, and Pythia is also the only **untied-embedding** model of the three. That is one observation and two possible explanations — "it's Pythia" or "it's untied" — and they are separable by adding a model that breaks the pairing.

**`JackFram/llama-160m` breaks it exactly**: a Llama, like SmolLM2, but with **untied** embeddings. It passes all four correctness rungs (jacobian 9.98e−17, logit lens 1.03e−07, RQ4 3.74e−14).

| model | family | **tied** | median `K` | E5 sign test | z | |
|---|---|---|---|---|---|---|
| SmolLM2-135M | Llama | ✅ tied | 0.2547 | 54/105 | +0.29 | ❌ |
| gpt2 | GPT-2 | ✅ tied | 0.2599 | 50/100 | +0.00 | ❌ |
| **llama-160m** | **Llama** | ❌ **untied** | 0.2559 | **72/114** | **+2.81** | ✅ |
| **pythia-160m** | GPT-NeoX | ❌ **untied** | 0.2716 | **77/107** | **+4.54** | ✅ |

> ## 🎯 **At `k` = 5 the split is on tied vs untied, not on architecture family.**
>
> Both **tied** models are at chance. Both **untied** models are significant. And the two sides each contain a Llama, so the family is controlled.

### ✅ 2.2a Extended to SIX models — and the split holds 3/3 against 3/3

Two more models, both passing all four correctness rungs first:

- **`EleutherAI/pythia-70m`** — untied, but `d` = 512 and only **6 layers**, so if the effect is really about Pythia's size or depth it should vanish here.
- **`EleutherAI/gpt-neo-125m`** — tied, and a **fourth architecture family**, so the tied side is no longer two families.

| model | family | tied | `d` | `L` | sign test | z | |
|---|---|---|---|---|---|---|---|
| SmolLM2-135M | Llama | ✅ tied | 576 | 30 | 54/105 | +0.29 | ❌ |
| gpt2 | GPT-2 | ✅ tied | 768 | 12 | 50/100 | +0.00 | ❌ |
| **gpt-neo-125m** | **GPT-Neo** | ✅ tied | 768 | 12 | 69/127 | **+0.98** | ❌ |
| **llama-160m** | Llama | ❌ **untied** | 768 | 12 | 72/114 | **+2.81** | ✅ |
| **pythia-160m** | GPT-NeoX | ❌ **untied** | 768 | 12 | 77/107 | **+4.54** | ✅ |
| **pythia-70m** | GPT-NeoX | ❌ **untied** | **512** | **6** | 80/120 | **+3.65** | ✅ |

| | pooled | z |
|---|---|---|
| **tied** (3 models, 3 families) | 173/332 = **52%** | **+0.77** — chance |
| **untied** (3 models, 2 families) | 229/341 = **67%** | **+6.34** |

> ## 🎯 **Three of three tied models at chance; three of three untied models significant.**
>
> The **Llama family appears on both sides**, so architecture is controlled. **pythia-70m has a third the depth and two thirds the width of pythia-160m and still shows it**, so it is not size. And the tied side now spans four families.

⚠️ **The `k` sweep (§2.2b) was run only on llama-160m and pythia-160m.** The six-model table is `k` = 5 throughout, so it establishes the tied/untied split *at that `k`* — the `k`-stability of the four unswept models is untested.

> 🔴 **They have now been swept, and the split does not survive. Read §2.2b before quoting the 3/3-vs-3/3 table above.**

### 🔴 2.2b The `k` sweep, all six models — and the tied/untied split does NOT survive it

| model | tied | **`k` = 4** | **`k` = 5** | **`k` = 6** |
|---|---|---|---|---|
| SmolLM2-135M | tied | 41/76, z = +0.69 ❌ | 54/105, z = +0.29 ❌ | — |
| gpt2 | tied | 46/86, z = +0.65 ❌ | 50/100, z = +0.00 ❌ | — |
| **gpt-neo-125m** | **tied** | **79/120, z = +3.47** ✅ | 69/127, z = +0.98 ❌ | — |
| llama-160m | UNTIED | 61/114, z = +0.75 ❌ | 72/114, z = **+2.81** ✅ | 67/122, z = +1.09 ❌ |
| **pythia-160m** | UNTIED | 86/120, **+4.75** ✅ | 77/107, **+4.54** ✅ | 77/113, **+3.86** ✅ |
| **pythia-70m** | UNTIED | 80/120, **+3.65** ✅ | 80/120, **+3.65** ✅ | — |

> ## 🔴 **A TIED model — gpt-neo-125m — is significant at `k` = 4 (z = +3.47).**
>
> So the clean 3/3-vs-3/3 split in §2.2a is a **`k` = 5 phenomenon**. At `k` = 4 the tally is two untied plus one tied.

**What is left standing after the sweep:**

| | significant at |
|---|---|
| **pythia-160m** | `k` = **4, 5 and 6** — the only model robust at every `k` tested |
| **pythia-70m** | `k` = 4 and 5 (6 not run) |
| llama-160m | `k` = 5 only |
| gpt-neo-125m | `k` = 4 only |
| SmolLM2-135M, gpt2 | never |

> **The robust statement is about the GPT-NeoX family, not about embedding tying.** Both Pythia models show the effect at every `k` tested; every other model is `k`-dependent or null. Whether that is the family, the training data (the Pile), or something else is untested.

**This is the third narrowing of the same claim** — `n` = 40 → ~105 killed it on two models (§2.1); a 2×2 then suggested tied/untied (§2.2a); the `k` sweep now removes that too. **I flagged this exact risk in §2.2a and it materialised.** The pattern is consistent enough to be a standing rule for this project: *a result at one `k` and one `n` is a hypothesis, not a finding.*

⚠️ The `k` = 4 cells for SmolLM2 (n = 76) and gpt2 (n = 86) are smaller than the rest; both are far from significance, so raising them is unlikely to change the picture, but they are not equal-`n` comparisons.

`k`-fragility is this project's known failure mode — it retracted the Stage 3 deviation statistic ([06-stage3-log.md](06-stage3-log.md) §4.4) — so the 2×2 above is not finished until it is swept. Sweeping it changes the reading:

| model | tied | `k`=4 | **`k`=5** | `k`=6 |
|---|---|---|---|---|
| SmolLM2-135M | tied | — | 54/105, z = +0.29 ❌ | — |
| gpt2 | tied | — | 50/100, z = +0.00 ❌ | — |
| gpt-neo-125m | tied | — | 69/127, z = +0.98 ❌ | — |
| **llama-160m** | UNTIED | 61/114, **z = +0.75** ❌ | 72/114, z = +2.81 ✅ | 67/122, **z = +1.09** ❌ |
| **pythia-160m** | UNTIED | 86/120, **z = +4.75** ✅ | 77/107, z = +4.54 ✅ | 77/113, **z = +3.86** ✅ |
| **pythia-70m** | UNTIED | 80/120, **z = +3.65** ✅ | 80/120, z = +3.65 ✅ | — |

> ## **The Pythia family is `k`-stable; llama-160m is not.** Both Pythias clear significance at every `k` tested (67–72% positive). llama-160m clears it at `k` = 5 and at neither neighbour (54%, 63%, 55%).

**So the six-model split at `k` = 5 is real, and one of its six cells is `k`-fragile.** What the data supports:

| | fraction positive, by `k` = 4/5/6 |
|---|---|
| **pythia-160m** (untied) | **72%, 72%, 68%** — robust |
| **pythia-70m** (untied) | **67%, 67%** — robust (`k`=6 not run) |
| llama-160m (untied) | 54%, **63%**, 55% — significant at `k` = 5 only |
| tied models | 51%, 50%, 54% at `k` = 5 — chance |

**The honest position:**

1. **The tied/untied split is established at `k` = 5** on six models across four families, pooled z = +0.77 (tied) against **+6.34** (untied), with Llama on both sides.
2. **It is confirmed off `k` = 5 for the Pythia family** — both models, every `k` tested.
3. **llama-160m is the weak cell.** It carries the "untied Llama" leg of the family control, and it is the one model whose effect depends on `k`. **Until a second untied Llama is tested, the family control rests on a `k`-fragile cell.**
4. The four unswept models (three tied, one untied) have no `k` evidence at all.
3. **`k` = 4 and `k` = 6 were both taken to n = 114**, where they give the *identical* 61/114. An earlier draft reported `k` = 6 as 48/79 and flagged it as possibly underpowered; at full `n` it is a clean null. So llama-160m's `k` = 5 result is the outlier among its own `k` values, not an `n` artifact.

**Reported this way because the alternative was available and wrong.** The `k` = 5 table alone gives a clean, quotable 2×2. It took one sweep to find that half of it rests on a single `k`, and this project has now been caught by exactly that four times over.

**So §2.1's "retraction" may be a boundary condition rather than a dead end.** The candidate claim is:

> **On models with untied input and output embeddings, Ricci curvature predicts which direction, at matched Fisher norm, the output moves furthest along. On tied models it does not.**

⚠️ **§2.2b sweeps `k` and finds only Pythia supports this robustly** — read it before quoting the 2×2.

A plausible mechanism, offered as a hypothesis and not tested here: with tied embeddings the unembedding *is* the input embedding, so `U` carries a second job and its rows are shaped by a constraint that has nothing to do with the output geometry. That would blunt exactly the structure §2 is probing. **`n` = 4 models is `n` = 4** — this is a hypothesis with a clean 2×2 behind it, not an established mechanism.

### 2.3 And the volume offset is a DIFFERENT fact

[10-architecture-log.md](10-architecture-log.md) §3b flagged Pythia as also the outlier on layer-wise log-volume, and asked whether that and the E5 effect were one fact. **They are not.** Median log-volume per dimension:

| model | tied | log-vol | E5 |
|---|---|---|---|
| gpt2 | tied | −0.397 | ❌ |
| SmolLM2 | tied | −0.121 | ❌ |
| **llama-160m** | **untied** | **−0.067** | ✅ |
| **pythia-160m** | **untied** | **+0.342** | ✅ |

The untied Llama has the E5 effect but sits with the tied models on volume (−0.067, against Pythia's +0.342). **E5 splits on tied/untied; volume does not.**

**And the volume offset remains unexplained** ([10-architecture-log.md](10-architecture-log.md) §3b.1): embedding tying, the final-norm gain and predictive entropy have all been tested and none accounts for it. Two separate phenomena, and the second is still open.

**`K ≈ +1/4` holds on the fourth architecture too** (median 0.2559, 100% of planes positive at all nine depths), which is now **4 models across 3 families and both embedding regimes**.

Three things to read off this table:

1. **At ε=0.001 all three sit at 1.000 to four decimals.** That is Gate A reproduced as a by-product, on three independently constructed directions, and it confirms the matching is real rather than nominal.
2. **The separation is monotone in ε and appears exactly where second order starts to matter.** Nothing separates the rows until ε≈0.1.
3. **The direction of the effect is the predicted one.** High-Ricci departs *upward* (2.40× the quadratic); low-Ricci and random stay at ≈1.0.

**Why Ricci and not sectional curvature.** "The highest sectional-curvature direction" is not well defined — sectional curvature is a function of a 2-*plane*, not a direction. Ricci is the natural per-direction object, and the generalised eigenproblem `Ric v = λ G v` gives exactly the extremal curvature directions at unit Fisher norm, which is the matching the design calls for. It is solved through a Cholesky factor of `G` rather than by inverting it.

---

## 3. What this licenses, and what it does not

**Licensed:**

> Under the Fisher–Rao pullback metric, **directions the metric calls null are directions the model is exactly invariant to** — verified on three architectures at machine precision, at a step size that doubles the hidden state. The metric's null space is the model's null space.
>
> 🔴 **The second half of this claim is withdrawn.** An earlier version continued *"and among directions the metric cannot distinguish at all — matched Fisher norm — the intrinsic Ricci curvature predicts which one the model's output moves furthest along."* At `n` ≈ 105 that holds on Pythia only (§2.1). It cannot be stated as a property of the framework.

**Not licensed:**

- **Three models, ~105 points each, `k`=5.** Prediction 2 is significant on **one** of them (§2.1). The `k`-sensitivity of Riemann-derived quantities ([06-stage3-log.md](06-stage3-log.md) §4.4) has **not** been checked and would now only matter for the surviving Pythia effect.
- `ε = 1` is a large perturbation — it doubles the state. The effect is clear at ε=0.3 (1.23 vs 1.03) but the significance test uses ε=1.
- The high/low Ricci contrast is between +1.55 and +0.98, a ratio of 1.6. This is not a comparison of curved against flat.

---

## 4. Next

1. ~~Raise `n` on GPT-2.~~ ✅ **done, and it settled the question the other way** — §2.1. All three are now at `n` ≈ 105.
2. **Test whether the surviving Pythia effect is about untied embeddings.** Add an untied Llama and a tied GPT-NeoX; Pythia is also the volume outlier in [10-architecture-log.md](10-architecture-log.md) §3b, and one experiment can tell whether that is one fact or two.
3. **Repeat across `k`** — now only for the Pythia effect. Still `k`=5 only, and `k`-fragility is the known failure mode of every Riemann-derived quantity here. Non-negotiable before publication. (`run_e5.py 40 <model> 6` now takes `k` as an argument.)
3. ~~Repeat on GPT-2 and Pythia~~ ✅ **done — §1 and §2.1.**
3. **Push ε lower with more points.** The effect at ε=0.3 is the more defensible one; it needs the `n` to be significant on its own.
4. Add the intervention to the polysemy setting: does perturbing along high-curvature directions at the disambiguation layer flip the resolved sense? That would connect E5 to RQ3b, which is the project's strongest result.
