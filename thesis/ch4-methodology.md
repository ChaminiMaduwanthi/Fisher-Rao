# Chapter 4 — Methodology

> **Draft.** Sources: [03-methodology.md](../03-methodology.md), [05-stage0-log.md](../05-stage0-log.md), [08-rq3a-log.md](../08-rq3a-log.md) §5.6, [10-architecture-log.md](../10-architecture-log.md).

---

## 4.1 The object

Every hidden state `h` of a decoder-only transformer induces a next-token distribution through the model's own output head:

```
p(h) = softmax( U · norm(h) + b )
```

So hidden states are not merely vectors — they are **points on a statistical manifold**, each one a probability distribution over the vocabulary. The canonical way to measure distance between such points is the **Fisher–Rao metric**, canonical not by preference but by **Čencov's theorem**, which makes it the unique Riemannian metric on a statistical manifold invariant under sufficient statistics.

Pulled back to residual-stream space:

```
G(h) = Jᵀ ( diag(p) − p pᵀ ) J ,       J = ∂ logits / ∂h = U · A(h)
```

where `A(h)` is the Jacobian of the final normalisation layer. This is **closed form**: one forward pass, no estimation, no training. It is `d × d` with `d` = 576–1600 for the models used here.

> **The most likely examiner misunderstanding, stated first.** This is **not** the parameter-space Fisher information of the K-FAC literature, which is billions of dimensions and intractable. It is the pullback of the Fisher–Rao metric of the *output distribution* onto the *hidden state*. Different object, different dimension, closed form.

`Σ_p = diag(p) − p pᵀ` is never formed — it is `N × N` with `N` ≈ 50 000. Instead `G = (Jᵀ diag(p)) J − (Jᵀp)(Jᵀp)ᵀ`, whose largest intermediate is `(n_tok, d)`.

---

## 4.2 The obstacle, stated correctly

Christoffel symbols require `G⁻¹`, and `G(h)` will not cooperate. The intuitive reason given in the literature is **wrong**, and getting it right changes what one does about it.

**The wrong reason.** `Σ_p` annihilates the all-ones direction, which suggests `G` is singular. It is not — not for that reason. `null(G) = {v : Jv ∈ span{1_N}}` requires `1_N ∈ range(J)`, a `d`-dimensional subspace of `ℝᴺ`, and generically it is not.

Two things bite instead, both established by measurement rather than assumed.

### 4.2.1 Conditioning

A sharply peaked softmax collapses the spectrum. Measured across 31 layers: **λ_min underflows to exactly 0 at 22 of them**, so `cond(G) = ∞`; and only **0.5–14.9%** of eigendirections hold 99% of the trace — independently replicating the 2–17% reported by FishBack (2026).

**float64 is necessary but not sufficient. Subspace restriction is unavoidable.**

### 4.2.2 Normalisation scale-invariance — the exact nullity

The predictive map is `h → norm(h) → U → softmax`, and the norm's Jacobian is a **projector**.

For **RMSNorm**, with gain `g` and `r = sqrt(mean(h²) + ε)`:

```
A = diag(g) · (1/r) · ( I − ĥ ĥᵀ )
```

which annihilates `h` itself. **One exact null direction.**

For **LayerNorm**, with `s = sqrt(var + ε)` and `z = (h − μ)/s`:

```
A = diag(g) · (1/s) · ( I − 11ᵀ/d − z zᵀ/d )
```

which annihilates **two**: the radial direction as before, **and the all-ones direction from the mean subtraction**.

Measured consequence: doubling a hidden state changes the model's prediction by `KL` ≈ 10⁻¹³–10⁻¹⁵, against ≈ 2.6–7.8 for a random perturbation of the same size (Chapter 5, §5.5.3).

> **The predictive map depends only on the *direction* of `h`.** The semantic manifold is a space of directions — a sphere, not a vector space — and the quotient formulation is therefore the **correct** formulation, not a workaround for a numerical inconvenience.

---

## 4.3 The method: quotient, then restrict

Every prior author stopped at the wall in §4.2. The combination that gets past it is two steps, in order.

**Step 1 — quotient by the exact null space.** `N` is an orthonormal basis of the norm Jacobian's kernel: one column for RMSNorm, two for LayerNorm. Work in the complement, `P = I − N Nᵀ`.

**Step 2 — restrict to the effective subspace.** Take the top-`k` eigenvectors of `P G P` as an orthonormal frame `F`, and compute curvature of the induced metric on the affine slice `h + span(F)`:

```
g(x) = G( h + F x )   restricted to F-coordinates,   x ∈ ℝᵏ
```

The Riemann tensor is then taken by nested automatic differentiation of `g` at `x = 0`.

> 🟢 **The compensating good news that makes this feasible at all:** *within* the retained subspace `cond_eff` is only 10¹–10², leaving ample float64 headroom for the second derivatives curvature requires. The catastrophic conditioning is entirely in the directions being discarded.

### 4.3.1 Two implementation requirements that are not optional

**Project into the frame *before* forming the metric.** Building the full `(d, d)` metric inside the autodiff graph makes double differentiation carry a `576 × 576` intermediate; that route attempted a **20.6 GB** allocation and died. Projecting first collapses everything to `(n_tok, k)`.

**Freeze the top-`k` token index set at the base point.** If the retained token set is re-selected at every evaluation of `g`, it changes discretely as `x` moves, so `G` is only *piecewise* smooth — it jumps wherever top-`k` membership changes. Differentiating through that gives garbage at the jumps and, worse, **plausible numbers away from them**. Curvature needs second derivatives, so this is not survivable. Freezing the index set costs a local approximation centred on the base point, which is what a local geometric quantity wants anyway.

---

## 4.4 Choosing `k`

`k` is the one free parameter of consequence, and this work's experience with it is the main methodological caution it has to offer.

**The rule that does not work.** The natural criterion is a ceiling on `cond_eff = λ₀/λ_k` — retain as many directions as remain numerically safe. On real activations at `cond_max` = 10², **that rule returns `k` = 11–16**: the retained spectrum is simply flatter than the ceiling is tight.

**And the Riemann tensor cannot be evaluated there.** Measured warm on an idle machine, `d` = 576, `top_k` = 512:

| `k` | 3 | 4 | 5 | 6 | 7 | 8 | 11 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| cost | 0.39 s | 1.00 s | 2.61 s | 8.11 s | 16.9 s | 45.7 s | 13.8 GB | 4.8 GB | **77 GB** |

Every entry of the tensor is a second derivative taken by nested `jacrev`, so the intermediates grow far faster than the `k⁴` of the tensor itself — roughly **×2.7 per unit of `k`**. Three runs of one experiment were killed by this **silently, with no traceback and no output**.

> **The `cond_eff` ceiling is a valid conditioning *diagnostic* and a valid `k` selector for the volume element and spectral quantities, which have no `k` ceiling. It is not usable as a `k` selector for anything derived from the Riemann tensor.**

**What this work does instead.** `k` is held **fixed at 5** for cross-layer and cross-model comparability, `cond_eff` is reported alongside every curvature value, and **every Riemann-derived result is swept over `k` ∈ {4, 5, 6}** before it is believed. `riemann()` now refuses `k` > 8 with an explanatory error rather than attempting the allocation.

### 4.4.1 Why the sweep is mandatory

`k`-fragility is this project's characteristic failure mode, and it has invalidated results twice:

- The **deviation from `+1/4`** looked like the model-specific signal, then failed its sensitivity check — unstable in `k`, with the layer ranking scrambling (Spearman ρ = +0.20 between `k` = 4 and `k` = 7). Reported as not identifiable.
- The **tied/untied split** in the causal experiment was clean at `k` = 5 and did not survive `k` = 4 (Chapter 7, §7.3).

**A result at one `k` is a hypothesis, not a finding.**

---

## 4.5 Controls

Three of this work's conclusions were changed by a control, and one was changed twice. The controls are therefore part of the method rather than an appendix to it.

### 4.5.1 The paired scramble

To ask whether a geometric quantity reflects **learned structure** or merely the **concentration** of `p`, hold `p` exactly as the model produces it and destroy only the association between probabilities and directions:

```
rows = idx[ randperm(len(idx)) ]
```

Entropy is then identical to machine precision, the retained direction **set** is identical, and the only thing destroyed is *which probability sits on which direction*.

**Three earlier versions of this control were wrong, in instructive ways:**

| attempt | what it did | why it failed |
|---|---|---|
| permute `U`'s **rows** | permutes `p` by the same permutation | provably the **identity** on this metric: `(PU)ᵀΣ_{Pp}(PU) = UᵀΣ_pU`. Verified numerically at 1.5e−12 |
| **Gaussian** `U` | replaces the unembedding entirely | changes spectrum, row norms and assignment at once — cannot attribute |
| whole-vocabulary permutation | `rows = perm[idx]` | swaps the direction **set** too: overlap with the real top-512 is **5/512**, median ‖U row‖ 2.44 → 3.10 |

> **A control is only as good as the list of things it holds fixed, and that list has to be written down.** Each of the three looked clean and each needed a specific check — an identity, an inspection, a measurement — to expose.

### 4.5.2 Positive controls

A null result is only informative if the instrument has usable variance. Every null in this work is accompanied by a positive control on the same data:

- `K` against scalar `R` — the same Riemann tensor under a different contraction — gives ρ = **+0.69 to +0.74**, so the near-zero correlations against the published proxies are informative nulls rather than an absence of signal.
- A **random subspace** captures 0.0086 of the metric's trace against an isotropic floor of `k/d` = 0.0087 — dead on, confirming the subspace instrumentation is calibrated.
- In the curved-versus-flat proxy tests, a **flat 3-plane** must read as exactly zero and does (10⁻³¹).

---

## 4.6 Models, corpora and reproducibility

**Six models**, all passing the four correctness rungs of Chapter 5 §5.6:

| model | family | norm | `d` | `L` | tied |
|---|---|---|---|---|---|
| SmolLM2-135M-Instruct | Llama | RMSNorm | 576 | 30 | ✅ |
| gpt2 | GPT-2 | LayerNorm | 768 | 12 | ✅ |
| gpt-neo-125m | GPT-Neo | LayerNorm | 768 | 12 | ✅ |
| llama-160m | Llama | RMSNorm | 768 | 12 | ❌ |
| pythia-160m | GPT-NeoX | LayerNorm | 768 | 12 | ❌ |
| pythia-70m | GPT-NeoX | LayerNorm | 512 | 6 | ❌ |

The design is deliberately **crossed**: the Llama family appears on both sides of the tied/untied divide, and both LayerNorm and RMSNorm appear on each side, so architecture family and embedding regime can be separated.

**Two corpora**, so that corpus-dependence is measurable rather than assumed:

- **hand-written** — 20 general sentences plus 64 polysemy minimal pairs, structurally validated;
- **WikiText-103 validation** — sentence-split and length-bounded, the same split Mabrok (2026) reports on;
- **WiC** (SuperGLUE) for the sense-annotated work, 339 usable same-token pairs.

**Reproducibility.** Every number in Chapters 5–7 is produced by a named script from a fixed seed. Expensive experiments checkpoint per point and resume, because several were killed mid-run. All arithmetic is float64.

---

## 4.7 Limitations of the method

- **`k` ≤ 8 is a hard ceiling** for anything Riemann-derived (§4.4). The retained subspace is a small fraction of `d`, and results are properties of that slice.
- **The frame is an affine slice**, not a geodesic submanifold, so the reported curvature is that of `h + span(F)` under the induced metric — not a sectional curvature of the full manifold restricted to a totally geodesic subspace.
- **The top-`k` truncation** is frozen at the base point, so `g(x)` is exact only near `x = 0`.
- **Models are 70M–160M parameters.** The largest models in the comparison literature (GPT-2 XL, Pythia-2.8B) are not included; the download was throughput-limited and the largest exceeded available memory.
