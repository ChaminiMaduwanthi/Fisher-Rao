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
