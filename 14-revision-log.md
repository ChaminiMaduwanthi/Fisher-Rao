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
| eigen | 0.121 | 0.111 | 0.092 | 0.047 | 0.026 | 0.008 | **0** |
| random | 0.192 | 0.138 | 0.098 | 0.055 | 0.051 | 0.017 | **0** |

**N = 512, d = 511, k = 5** — the transformer runs' retained-vocabulary size, by predictive entropy:

| entropy (nats) | n | eigen `K_S` | eigen \|ext\| | random `K_S` | random \|ext\| |
|---|---|---|---|---|---|
| [0, 0.5) | 21 | +0.450 | 0.200 | −1.288 | 2.311 |
| [0.5, 1) | 9 | +0.255 | **0.005** | +0.209 | 0.041 |
| [1, 2) | 13 | +0.264 | **0.014** | +0.146 | 0.104 |
| [2, 4) | 20 | +0.278 | **0.028** | +0.086 | 0.155 |
| [4, ∞) | 37 | +0.222 | 0.120 | +0.005 | 0.245 |

Eigen frames, all entropies: median `K_S` = 0.280, 0.272, 0.266, 0.268, 0.260 at `k` = 2–6.

**Reading.**

1. **The extrinsic term is real, and in general not small.** Random 5-dimensional slices of a manifold with `K_M = 1/4` report anything from −1.29 to +0.21 depending on entropy.
2. **The pipeline's eigen-frame slices sit much closer to the manifold** — within 0.005–0.028 of 1/4 between 0.5 and 4 nats. At the extremes the term grows: 0.20 below 0.5 nats and 0.12 above 4.
3. **The check passes where it must.** At `k = d` the slice is the manifold and the term is exactly 0 for both frames.
4. **A random `U` already gives the trained models' value.** With no learning at all, eigen-frame slices report `K` ≈ 0.25–0.28 across 0.5–4 nats, as SmolLM2 and LLaMA do. That agrees with the consistent control: `+1/4` comes from the simplex and the slicing, not from learning. The one exception is below 0.5 nats, where the synthetic gives 0.45 against about 0.28 for the trained models.

**For the paper.** Name the quantity *slice sectional curvature*, state the Gauss equation, report this check, and rest every comparison on like-for-like slices (a model and its controls use the same construction). Absolute values below 0.5 or above 4 nats carry an extrinsic term of order 0.1–0.2 and should not be read as manifold curvature. The paragraph is now in Section III.C of [the paper](paper/fisher-rao-conference-paper.md).

⚠️ Synthetic `U`, no normalisation layer, `d = N − 1`. This sizes the term where the answer is known; it does not measure it on the transformers.
