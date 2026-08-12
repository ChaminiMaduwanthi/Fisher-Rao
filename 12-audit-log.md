# 12 — Full end-to-end audit, 12 August 2026

**Scope:** every saved result file re-read and every headline number recomputed
from the raw points by an independent script, without reusing the analysis code
that produced it. Plus a regression check on the six bugs previously fixed.

**Verdict:** one headline claim is **wrong and is withdrawn**; two are
**imprecise and are restated**; everything else reproduces exactly.

---

## 1. 🔴 The root cause: `spearman` was rank-blind to ties

Every script carried its own copy of

```python
rx = torch.argsort(torch.argsort(x))          # "rank"
```

That is a **permutation, not a rank**. Tied values receive distinct consecutive
integers in the order `argsort` visits them — and for a stable sort that is
**file order**.

For continuous quantities (`K`, entropy, log-volume) float64 ties do not occur
and the two agree to the last digit — verified on 21 of the E2 pairs, max
discrepancy 0.0185. For **discrete** quantities the difference is large, and it
is not noise: the point files are grouped by model and then by layer, and
`k_eff` varies strongly with layer, so tie-breaking piped the layer ordering
straight into the correlation.

```
rho(file index, k_eff_99) = +0.510          <- the channel it leaked through
```

**Fixed** in [fisherrao/stats.py](fisherrao/stats.py) (midrank Spearman, a true
median, and a tie-corrected Mann-Whitney). All six local copies now delegate to
it.

---

## 2. 🔴 WITHDRAWN — "negative planes track effective dimension at ρ = +0.42"

`n_planes_neg` is **zero at 407 of 456 points**. Spearman was the wrong test for
it, and the reported value was almost entirely tie-breaking artefact.

| ρ(`n_planes_neg`, ·) | reported | corrected |
|---|---|---|
| `k_eff_99` | **+0.42** | **+0.118** |
| entropy | +0.212 | **−0.034** |
| `cond_eff` | −0.024 | −0.063 |

**The claim "effective dimension explains roughly twice what entropy does" is
withdrawn.** Entropy explains nothing; that comparison was an artefact on both
sides.

### What actually survives, tested properly

With a binary grouping and a continuous covariate the right test is
Mann-Whitney, which is tie-corrected by construction:

| | with a negative plane (n = 49) | without (n = 407) | AUC | z |
|---|---|---|---|---|
| **`k_eff_99`** | **248.0** | **122.0** | 0.618 | **+2.69** ✅ |
| entropy | 1.91 | 1.95 | 0.480 | −0.45 n.s. |
| `cond_eff` | 70.9 | 80.6 | 0.439 | −1.39 n.s. |

> **The direction survives; the magnitude does not.** Points carrying a
> negatively curved plane have **2.03× the effective dimension at
> indistinguishable entropy** (1.91 vs 1.95 nats), and neither conditioning nor
> entropy separates them. But it is a modest effect — AUC 0.618, z = +2.69 —
> not a ρ = 0.42 relationship.

Two supporting facts are **rank-free and therefore untouched** by the bug, and
both reproduce exactly:

- the binned gradient, entropy-restricted to [1, 3) nats:
  1.6% → 5.1% → 7.1% → 6.1% across `k_eff` bins (n = 61/39/42/49);
- 49/456 = **10.7%** of points carry a negative plane, inside the "6–12%" range
  the write-ups state.

⚠️ But collapsing that gradient to a 2×2 (`k_eff` < 60 vs ≥ 60, entropy
restricted) gives **χ² = 1.37, not significant** — 9 positive cases in 191
points is too thin. **The entropy-restricted evidence is weaker than
"unchanged at +0.397 vs +0.419" suggested**; the pooled Mann-Whitney null on
entropy (z = −0.45) is the better argument that this is not entropy in disguise.

---

## 3. 🟡 RESTATED — the centrepiece scramble has three saved `n`

The same experiment is quoted at three different sample sizes across the
write-ups, and **the largest one was never reported**.

| source | n | `K_real` | `K_within` | sign | z | quoted in |
|---|---|---|---|---|---|---|
| `scramble.json` (global variant) | 60 | 0.2586 | 0.0002 | 56/60 | +6.71 | paper |
| `within.json` (a checkpoint) | 126 | 0.2543 | 0.0141 | 113/126 | +8.91 | README, thesis |
| **`within_points.jsonl` (full)** | **221** | **0.2546** | **0.0109** | **201/221** | **+12.18** | **nowhere** |

The 126 rows are a subset of the 221 (35 `_i` values recur across checkpoint
resumes) and **every shared row agrees to 1e−12** — so this is a stale summary,
not a data conflict. The conclusion strengthens with `n`.

**The number to quote is n = 221: `K` 0.2546 → 0.0109, 201/221, z = +12.18**,
with the `k`-sweep on the 69-point subset where all three `k` were computed
(k = 4: +6.86, k = 6: +5.66).

---

## 4. 🟡 RESTATED — ρ = +0.967 depends on an unintended median convention

`torch.median` returns the **lower** of the two middle values for even `n`; the
profile used it with `n` = 40 per layer. Over only 9 layer medians, that shift
is enough to move two ranks.

| | log-volume | `k_eff` |
|---|---|---|
| `torch.median` (as run) | **+0.967** | +0.987 |
| true median | **+0.867** | +0.987 |
| **per point (n = 360, convention-free)** | **+0.957** | **+0.991** |

**The per-point value is the one to quote** — it has n = 360 instead of 9 and
does not depend on the convention. The conclusion is unchanged in every version:
a structure-free scramble reproduces both profiles. But "+0.967" should not be
quoted to three digits as though it were stable.

*(`k_eff` was reported as +0.983 under the old Spearman; tie-corrected it is
+0.987. Immaterial, but corrected for consistency.)*

`torch.median` is used in ~60 places project-wide. It has **not** been changed —
it is a valid order statistic, re-running everything would be expensive, and
this is the only place it changed a reported figure. It is now documented.

---

## 5. ✅ Everything else reproduced exactly

| claim | check | result |
|---|---|---|
| `K ≈ +1/4`, 4 architectures | 456 usable of 463 rows; medians 0.2548–0.2738 | spread **0.0190** ("within 0.022") ✅ |
| E2 — no instrument tracks any other | all 6 pairs recomputed, pairwise-complete | saved = recomputed = tie-corrected; max \|ρ\| **0.248** ✅ |
| corpus invariance | intrinsic vs 3 proxies, n = 653 | intrinsic **0.028/0.034**, proxies **0.104–0.194** ✅ |
| E5 / RQ4 null test | 14 runs, 6 models | **14/14 pass** ✅ |
| E5 prediction 2 pooled | SmolLM2 k=5 | n = 105, z = **+0.293** — matches 54/105 ✅ |
| E5 NeoX-only robustness | z by family | NeoX **all** > 0 (+1.10…+4.75); others +0.00…+0.75, one outlier (llama k=5, +2.81) ✅ |
| same-sense control | 1.5316 / 0.7541 | **2.031×**, 30/32, z = +4.95 ✅ |
| Mabrok tables | `10-architecture-log` `k_eff` rows | **identical to the saved rows, digit for digit** ✅ |
| validation ladder | re-run | rungs 2,3,6 ✅ · rung 7 PASS ✅ · geomstats 3/5 ✅ · Gate A 8.7e−06 ✅ |

### Regression check on the six previously fixed bugs

| bug | check | gpt2 | pythia-70m |
|---|---|---|---|
| `k` ceiling | `K_RIEMANN_MAX` = 8, `riemann()` raises | ✅ | — |
| scramble confound | `rows=` present on both metric fns | ✅ | — |
| LayerNorm bias placement | `norm_check` vs the model's own module | fwd 5.2e−08, jac 1.0e−16 ✅ | fwd 4.3e−08, jac 1.2e−16 ✅ |
| `null_directions` sign | ⟨n₀, ĥ⟩ must be **positive** | **+1.000000** ✅ | **+1.000000** ✅ |
| null space is exact | \|Δp\| at ε = \|h\| | 8.4e−08 ✅ | 3.2e−07 ✅ |
| projector | \|P²−P\|, \|PN\| | 1.8e−15, 3.3e−16 ✅ | 1.4e−15, 1.1e−16 ✅ |

---

## 6. What this adds to the methodological postscript

The postscript in Chapter 8 lists three recurring traps. This audit adds a
fourth, and it is the one that would have been hardest to catch by reading the
code:

> **A statistic can be wrong because of the data's *storage order*.** Nothing in
> the analysis referenced the file layout. The rank function did, invisibly, and
> only for the variables that had ties — which were exactly the discrete ones
> carrying the newest claims. The tell was that ρ(row index, `k_eff`) = +0.51,
> a quantity nobody would think to compute.

The general rule: **when a variable is mostly one value, do not rank it.**
Test the grouping it actually induces.

---

## 7. Second audit — the paper's own tables, 12 August 2026

The first audit checked the result files. This one checked **the paper against
them**, claim by claim, and found that two tables had been assembled from
numbers that were never measured on the rows they named.

### 7.1 🔴 Table I (null-direction test) — values misattributed

The table named six models. `check_architectures.py` measures three, and
`09-e5-log.md` §1 tabulates four while asserting "PASS on all six". The numeric
column mixed a LayerNorm-bias forward error, a logit-lens error and a `|Δp|`
from the regression check, assigned to the wrong rows.

**Fixed by measuring it.** `check_null_all.py` runs one code path over all six
models and reports `KL(p(h) ‖ p(h + ‖h‖n))` against a random step of identical
size. **6/6 pass**, sign guard ⟨n₀, ĥ⟩ = 1.000000 on every model:

| model | norm | null (exp.) | KL null | KL random |
|---|---|---|---|---|
| SmolLM2-135M | RMSNorm | 1 (1) | 2.78e−15 | 3.76 |
| LLaMA-160M | RMSNorm | 1 (1) | 3.82e−16 | 2.96 |
| Pythia-70M | LayerNorm | 2 (2) | 1.65e−13 | 13.41 |
| Pythia-160M | LayerNorm | 2 (2) | 2.02e−13 | 7.79 |
| GPT-2 | LayerNorm | 2 (2) | 1.16e−15 | 3.70 |
| GPT-Neo-125M | LayerNorm | 2 (2) | 9.66e−16 | 32.31 |

### 7.2 🔴 Table II (validation ladder) — wrong implementation's precision

The error column quoted 1e−14 to 1e−16 for rungs 2, 3 and 6. Those rungs are run
by `validate_curvature.py`, which uses **central finite differences** at
`h` = 1e−5 — an accuracy floor of ~1e−6. The quoted figures were rung 7's
per-dimension errors and a geomstats agreement, not ladder errors at all.

`check_ladder_precision.py` measures the **autodiff implementation that every
transformer number actually uses**, worst-case over 12 base points and 12 random
2-planes per rung:

| rung | manifold | expect | worst \|err\| |
|---|---|---|---|
| 1 | sphere, radius 2 | +1/4 | 1.8e−15 |
| 2 | Poincaré half-plane | −1 | 1.0e−13 |
| 3 | categorical simplex | +1/4 | 6.6e−10 |
| 6 | univariate Gaussian | −1/2 | 2.6e−15 |

Rung 1 was added because the text said "seven manifolds" over a six-row table.

### 7.3 🔴 "88 % of the metric's trace" — never measured

`03-methodology.md` §4 quotes "≥ 92 % of the trace" but for a **`cond_eff`-
selected** `k`; the curvature results hold `k` fixed at 5. Different quantity.
`check_trace_fraction.py` measures the actual one on `PGP` over 60 states:

| k | median | mean | min | max |
|---|---|---|---|---|
| 4 | 92.1 % | 88.7 % | 40.7 % | 99.6 % |
| **5** | **95.6 %** | **91.2 %** | 43.2 % | 99.8 % |
| 6 | 96.8 % | 92.9 % | 45.5 % | 100.0 % |

### 7.4 🟡 Four smaller corrections

- **"on a single CPU core"** — the timing table records no hardware. Removed.
- **"depths from 12 to 30 layers"** — Pythia-70M has 6. Now 6 to 30.
- **geomstats agreement** quoted first-point errors as if worst-case; gamma is
  2.3e−13 and beta 4.9e−14 at worst, not 2.5e−14 and 3.2e−14.
- **E2 "the same 360 states"** — 360 sampled, but pairwise-complete counts are
  320 (Manson) and 338 (King). Now stated as pairwise complete.

### 7.5 The rule this adds

> **A number is not verified by being in your own notes.** Every figure in §7.1
> and §7.2 existed somewhere in the logs — as a measurement of something else.
> Transcription between documents is a step at which provenance is silently
> lost, and the only defence is to re-measure into the table rather than copy
> into it.
