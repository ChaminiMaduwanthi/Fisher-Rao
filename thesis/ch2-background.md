# Chapter 2 — Background

> **Draft.** Standard material, presented in the order this thesis needs it and with the conventions fixed here once. Nothing in this chapter is original; its job is to make Chapters 4–7 readable and to remove the two ambiguities that cause the most confusion (the two Fishers, and the sign conventions).

---

## 2.1 Statistical manifolds and the Fisher–Rao metric

A **statistical manifold** is a family of probability distributions `{p_θ}` smoothly parameterised by `θ ∈ Θ ⊆ ℝⁿ`. Points are distributions; coordinates are parameters.

The **Fisher information matrix** at `θ`,

```
g_ij(θ) = E_{x∼p_θ} [ ∂_i log p_θ(x) · ∂_j log p_θ(x) ]
```

is symmetric and positive semi-definite, and where it is positive definite it is a Riemannian metric — the **Fisher–Rao metric**.

Two properties make it the right object rather than one choice among many.

**It is the second-order expansion of KL divergence.** For a small displacement `εv`,

```
KL( p_θ ‖ p_{θ+εv} ) = ½ ε² vᵀ g(θ) v + O(ε³)
```

so the metric measures *statistical distinguishability*: the distance between two nearby distributions is how hard they are to tell apart from samples. This identity is what Chapter 5's Gate A verifies numerically.

**Čencov's theorem.** The Fisher–Rao metric is, up to scale, the **unique** Riemannian metric on a statistical manifold invariant under sufficient statistics. Any other choice can be changed by a reparameterisation that provably destroys no information — so if one is going to do Riemannian geometry on distributions, this is not a modelling preference but the canonical structure.

### 2.1.1 The categorical simplex

The case this thesis needs. For a distribution over `N` outcomes with `p_i > 0` and `Σ p_i = 1`, the Fisher–Rao metric in the first `N−1` coordinates is

```
g_ij = δ_ij / p_i + 1 / p_N
```

The substitution `q_i = 2√p_i` maps the simplex isometrically onto the positive orthant of a **sphere of radius 2** in `ℝᴺ`. Hence:

> **The categorical simplex has constant sectional curvature `K = +1/4` everywhere**, and the Fisher–Rao distance is the great-circle distance
>
> ```
> d_FR(p, q) = 2 · arccos( Σᵢ √(pᵢ qᵢ) )
> ```

Both facts are used directly: `+1/4` is the reference value throughout Chapter 6, the closed-form distance is the instrument in Chapter 7 §7.2, and the simplex is rung 3 of Chapter 5's validation ladder.

⚠️ **A normalisation ambiguity worth naming.** The radius-2 convention used here gives `K = +1/4`; a unit-sphere convention with the metric scaled by 4 gives `K = +1`. A disagreement with a reference implementation of exactly `4×` is this, not an error.

---

## 2.2 🔴 Two different Fisher informations

This is the single most likely misreading of the thesis and is therefore stated before anything else that depends on it.

| | **parameter-space Fisher** | **the pullback used here** |
|---|---|---|
| the manifold | model **weights** `θ` | **hidden states** `h` |
| the distribution | `p_θ(y|x)` as `θ` varies | `p(h)` as `h` varies, weights frozen |
| dimension | 10⁸–10¹¹ | `d` = 576–1600 |
| tractability | intractable; K-FAC etc. approximate it | **closed form, one forward pass** |
| used for | optimisation, natural gradient | geometry of the representation |

They share a name and a formula shape and nothing else. **Intractability results from the optimisation literature do not apply to the object studied here.**

---

## 2.3 Riemannian geometry: the minimum needed

Given a metric `g_ij(x)` on a manifold with coordinates `x`:

**Christoffel symbols** — the connection, i.e. how to differentiate vector fields:

```
Γ^k_ij = ½ g^{kl} ( ∂_i g_jl + ∂_j g_il − ∂_l g_ij )
```

Note `g^{kl}`: **the metric must be inverted.** This is where the conditioning obstacle of Chapter 3 §3.3 bites.

**Riemann curvature tensor** — the failure of second covariant derivatives to commute:

```
R^ρ_σij = ∂_i Γ^ρ_jσ − ∂_j Γ^ρ_iσ + Γ^ρ_iλ Γ^λ_jσ − Γ^ρ_jλ Γ^λ_iσ
```

It involves **second derivatives of `g`**, which is why numerical precision matters twice over.

**Sectional curvature** of the 2-plane spanned by `u, v` — the Gaussian curvature of the surface swept by geodesics in that plane:

```
K(u,v) = R(u,v,v,u) / ( |u|²|v|² − ⟨u,v⟩² )
```

**Ricci and scalar curvature** — contractions carrying less information but more stability:

```
Ric_ij = R^k_ikj        R = g^{ij} Ric_ij
```

For a space of constant sectional curvature `K` in `k` dimensions, `R = k(k−1)K` exactly — the identity Chapter 5 uses to validate the contraction separately from `sectional()`.

**Volume element** — `√det g`, the density of the measure the metric induces.

### 2.3.1 Sign conventions, fixed once

Two convention axes cause disagreements with published values and **both look like bugs**. This thesis uses:

```
R(e_i, e_j) e_σ = R^ρ_σij e_ρ          index order of the lowered tensor
K(u,v) = R(u,v,v,u) / (…)              NOT R(u,v,u,v)
```

Contracting `R(u,v,u,v)` instead flips the sign, and did so once in this work (Chapter 5 §5.4).

---

## 2.4 Pullback metrics

If `f : M → S` maps a manifold `M` into a statistical manifold `S` with metric `g_S`, the **pullback** `f*g_S` is the metric on `M` defined by

```
(f*g_S)(u, v) = g_S( df(u), df(v) )
```

i.e. measure lengths in `M` by what they do to the distribution. Concretely, with `J = df` the Jacobian,

```
G = Jᵀ g_S J
```

**The curvature of the pullback is not the curvature of `S`.** `M` sees a submanifold — its intrinsic curvature depends on how `f` embeds it, and by the Gauss equation differs from the ambient curvature by second-fundamental-form terms. This is exactly why `K` measured on transformer states is *informative* rather than trivially `+1/4`: the ambient simplex value is the reference, and departure from it is the model's contribution.

⚠️ **A pullback can be degenerate even when `g_S` is not.** If `df` has a kernel, `G` has a null space regardless of `g_S`. This is precisely what the normalisation layer does in Chapter 4 §4.2.2, and it is a fact about `f`, not about the Fisher metric.

---

## 2.5 Transformers: only the parts that matter here

A decoder-only transformer maps tokens to a **residual stream** `h_ℓ ∈ ℝᵈ` updated additively by each block. The final prediction is

```
p = softmax( U · norm(h_L) + b )
```

Three details carry real weight in this thesis and are usually glossed:

**The logit lens.** Applying the output head to an *intermediate* `h_ℓ` gives a next-token distribution at every depth. This is what makes a layer-wise geometry definable at all: every layer's state is a point on the same statistical manifold.

**Causal attention.** `h` at position `t` depends only on tokens `≤ t`. Any experiment about context must therefore place the disambiguating context **before** the target — a constraint that made the first version of the RQ3b experiment vacuous, returning exactly 0.0 separation at every layer (Chapter 7 §7.2).

**The final normalisation layer is part of the predictive map.** RMSNorm:

```
norm(h) = g ⊙ h / sqrt(mean(h²) + ε)
```

LayerNorm additionally subtracts the mean and adds a bias. Both are **scale-invariant**, and both therefore contribute exact null directions to any pullback through them (Chapter 4 §4.2.2). Omitting the norm's Jacobian from the pullback is a genuine error rather than an approximation, and two of the papers reviewed in Chapter 3 appear to omit it.

**Tied embeddings.** Many models share `U` with the input embedding matrix. This constrains the unembedding geometry and turns out to matter empirically (Chapter 7 §7.3.1), so the model set in Chapter 4 §4.6 is crossed on it deliberately.

---

## 2.6 Notation used throughout

| symbol | meaning |
|---|---|
| `h` | residual-stream hidden state, `∈ ℝᵈ` |
| `U`, `b` | unembedding matrix `(N, d)` and bias |
| `p(h)` | next-token distribution induced by `h` |
| `A(h)` | Jacobian of the final normalisation layer |
| `J = U·A` | Jacobian of logits with respect to `h` |
| `Σ_p` | `diag(p) − p pᵀ`, the Fisher metric on the simplex |
| `G(h)` | `Jᵀ Σ_p J`, the pullback metric on hidden states |
| `N` | orthonormal basis of `G`'s exact null space |
| `P` | `I − N Nᵀ`, projector onto its complement |
| `F` | `(d, k)` orthonormal frame, the retained effective subspace |
| `k` | retained subspace dimension (5 unless stated) |
| `k_eff` | directions holding 99% of the metric's trace |
| `cond_eff` | `λ₀ / λ_k` within the retained subspace |
| `K`, `R` | sectional and scalar curvature |
| `d_FR` | Fisher–Rao geodesic distance on the simplex |
