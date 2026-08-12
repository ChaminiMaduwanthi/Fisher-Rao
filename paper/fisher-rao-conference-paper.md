# Intrinsic Fisher–Rao Curvature of Transformer Representations: Curvature Carries Learned Structure, the Spectrum Carries Entropy

**Chamini Maduwanthi**

---

## Abstract

This article computes the intrinsic Riemannian curvature of the Fisher–Rao metric pulled back onto the hidden states of transformer language models, and reports what that curvature measures. Prior work has derived the pullback metric correctly but has stopped short of the curvature tensor, or has measured curvature under a flat Euclidean metric, or has substituted a local-PCA residual for curvature and concluded that representation manifolds are essentially flat. Christoffel symbols, the Riemann tensor, and sectional and scalar curvature are computed here directly, on a quotient-then-restrict subspace that removes the exact null directions of the normalisation layer. Across four architectures and 456 sampled states the median sectional curvature is +0.2547 to +0.2738, a spread of 0.019 around the ambient simplex value of +1/4. The central finding is a separation. A paired scramble that destroys the learned token-to-direction assignment, while holding the predictive distribution, the conditioning and the retained direction set exactly fixed, collapses sectional curvature from 0.2546 to 0.0109 (201 of 221 points, z = +12.18), yet reproduces the layer-wise log-volume profile at rho = +0.957 and the effective-dimension profile at rho = +0.991 (per state, n = 360), with the same minimum layer in both cases. Quantities derived from the metric's spectrum therefore track predictive concentration; only curvature tracks the learned assignment. A seven-rung validation ladder, a Kullback–Leibler Hessian identity check, and a cross-check against an independent information-geometry library support the measurements.

**Index Terms** — information geometry, Fisher–Rao metric, Riemannian curvature, transformer representations, interpretability, intrinsic dimension.

---

## I. Introduction

The hidden state of a transformer language model [1] is usually treated as a point in a flat Euclidean space. Distances are cosine or L2, directions are compared by inner product, and the geometry of the space is assumed to be the geometry of the coordinates it is written in. That assumption is a choice, and it is not the only available one.

A transformer with hidden state `h` and unembedding `U` defines a probability distribution over the next token, `p(h) = softmax(U · norm(h))`. The map `h → p(h)` sends hidden states into the interior of the vocabulary simplex, a statistical manifold. On such a manifold there is a distinguished Riemannian metric: Čencov's theorem [2] establishes that the Fisher information metric is, up to scale, the only metric on the space of probability distributions invariant under sufficient statistics [3]. Pulling it back through `h → p(h)` gives a metric on hidden-state space that is derived from the model's own predictive behaviour rather than imposed by the coordinate system.

That pullback is not new. It has been derived and used for activation steering [4], [5]. What has not been done is the next step. A metric alone gives lengths, angles and volumes. Curvature requires the second derivatives of the metric — the Christoffel symbols, then the Riemann tensor — and it is curvature, not the metric, that says whether the manifold is intrinsically bent or merely written in curvilinear coordinates. The literature contains curvature measured under a flat metric [6], [7], and it contains a local-PCA residual reported as a curvature of order 10⁻⁵ and interpreted as near-flatness [8]. It does not contain the intrinsic Riemannian curvature of the Fisher–Rao pullback for a transformer.

This article computes it, and then asks the harder question: once computed, what does it measure that cheaper quantities do not? The answer turns out to be sharp, and to run against the intuition that expensive geometric quantities are elaborate restatements of simple ones. Three quantities are put through a single control. Two of them — the log volume element and the effective dimension of the metric — survive the destruction of all learned structure essentially unchanged. Sectional curvature does not.

The contributions are:

1. **The first intrinsic Riemannian curvature of the Fisher–Rao pullback metric on transformer hidden states**, validated on a seven-rung ladder of manifolds with analytically known curvature, and measured at +1/4 across four architectures.
2. **A separation result.** One paired control splits three geometric quantities into those that read predictive concentration and the one that reads learned structure.
3. **A diagnosis of the published near-flat result.** The estimator that produces 10⁻⁵ is shown, at the threshold that produces it, to report a unit 3-sphere as flat to 10⁻³¹.
4. **A reproducible negative record.** Six claims made during this work were withdrawn under controls the work itself constructed; the section describing them is part of the contribution.

---

## II. Related Work and the Gap

**The pullback metric is established.** Mabrok [8] and Wang and Zhao [4] both derive the Fisher–Rao pullback for transformer states. Wang and Zhao use it for steering and do not compute curvature. Mabrok reports a curvature, but the quantity computed is a local-PCA residual on the ambient activation cloud, not a Riemann tensor of the pullback metric, and it is reported as of order 10⁻⁵ and read as near-flatness. Section VII addresses that reading directly.

**Curvature has been measured under the wrong metric.** Manson [6] measures a Frenet-style curvature of residual-stream trajectories, and King et al. [7] measure an angular curvature between successive difference vectors. Both find that curvature varies systematically with semantic content and predictive uncertainty, which is evidence that geometry is doing work. Both compute the quantity in Euclidean coordinates, so what varies is a property of the coordinate chart together with the trajectory, not an invariant of a manifold.

**Intrinsic curvature has been computed, but for other networks.** Zavatone-Veth et al. [9] compute Riemannian geometry of neural-network representations under a data-induced metric, establishing that the machinery is tractable. The metric is not Fisher–Rao and the models are not transformers.

**Intrinsic dimension is a parallel literature.** Valeriani et al. [10] and Modell et al. [11] report a characteristic profile of intrinsic dimension across depth, with a minimum roughly one third of the way through. Section VI.C shows that the analogous profile under this metric is reproduced almost exactly by a control that destroys all learned structure, which places a testable question against that literature.

The gap this article closes is stated precisely: the pullback metric has been derived, and layer-wise curvature has been measured under flat metrics, but the intrinsic Riemannian curvature of the state-dependent Fisher–Rao metric — Christoffel symbols to Riemann tensor to sectional and scalar curvature — has not been computed for a transformer at any scale or any layer.

---

## III. The Metric and Its Curvature

### A. The pullback

Write `norm` for the model's final normalisation layer and `A(h)` for its Jacobian. With `J = U · A(h)` and `p = p(h)`, the Fisher information metric on the simplex pulls back to

> **G(h) = Jᵀ (diag(p) − p pᵀ) J.**  (1)

This is a `d × d` matrix, closed-form and cheap, and it should not be confused with the parameter-space Fisher used in second-order optimisation [12], which is intractable at this scale. Equation (1) is the metric of the model's *output distribution* expressed in the coordinates of its *hidden state*.

`G(h)` is the Hessian of the Kullback–Leibler divergence at zero displacement, which gives a falsification test independent of the implementation of (1):

> **KL(p(h) ‖ p(h + εv)) = ½ ε² vᵀ G(h) v + O(ε³).**  (2)

### B. Exact null directions

Both normalisation layers annihilate directions exactly. RMSNorm [13] has Jacobian `A = diag(g)(1/r)(I − ĥĥᵀ)`, with one null direction, `h` itself. LayerNorm [14] additionally subtracts the mean and so annihilates two: the radial direction and the all-ones direction. `G(h)` is therefore singular by construction, with rank at most `d − 1` or `d − 2`.

Suppressing this with a pseudo-inverse would return a plausible wrong answer. Instead the null space is quotiented explicitly. With `N` an orthonormal basis of the null directions and `P = I − N Nᵀ`, curvature is computed on the span of the top-`k` eigenvectors of `P G P`. Table I reports that the number of null directions found matches theory on every architecture tested, and that the model is exactly invariant along them.

**TABLE I. NULL-DIRECTION FALSIFICATION TEST**

| model | norm | null dirs found (expected) | max output change along null |
|---|---|---|---|
| SmolLM2-135M | RMSNorm | 1 (1) | machine precision |
| LLaMA-160M | RMSNorm | 1 (1) | machine precision |
| Pythia-70M | LayerNorm | 2 (2) | machine precision |
| Pythia-160M | LayerNorm | 2 (2) | machine precision |
| GPT-2 | LayerNorm | 2 (2) | machine precision |
| GPT-Neo-125M | LayerNorm | 2 (2) | machine precision |

The step size used is large enough to double the norm of the state, so the invariance is a property of the architecture and not of a small perturbation.

### C. Curvature

On the retained `k`-dimensional subspace, Christoffel symbols and the Riemann tensor are formed by nested reverse-mode differentiation of (1), and sectional curvature of the plane spanned by `u, v` follows in the standard way [15]. All differentiation is reverse-mode automatic differentiation in double precision [16]. Cost grows by a factor of about 2.7 per unit increase in `k`, and memory makes `k > 8` infeasible on the hardware used; all curvature results below are at `k = 4, 5, 6`, with `k = 5` as the default. The retained subspace holds approximately 88 % of the metric's trace.

This is a real limitation and is stated as such: the reported curvature is that of a `k`-dimensional slice under the induced metric, not a sectional curvature of a totally geodesic submanifold of the full space.

---

## IV. Validation

Intrinsic curvature is expensive, silent when wrong, and plausible either way. Validation therefore precedes results.

### A. The ladder

Seven manifolds with analytically known curvature are run through the same code path used for the transformer. Table II reports the outcome.

**TABLE II. VALIDATION LADDER**

| rung | manifold | expected | measured | worst error |
|---|---|---|---|---|
| 2 | Poincaré half-plane | K = −1 | −1.000000 | 1.3e−14 |
| 3 | Categorical simplex | K = +1/4 | +0.250000 | 2.6e−14 |
| 4 | Gamma family | negative | −0.463903 | — |
| 5 | Beta family | negative | −0.456795 | — |
| 6 | Univariate Gaussian | K = −1/2 | −0.500000 | 1.1e−16 |
| 7 | Synthetic pullback | K = +1/4 | +0.250000 | 9.7e−12 |

Rung 3 is the load-bearing one: the categorical simplex under the full Fisher–Rao metric is isometric to the positive orthant of a radius-2 sphere and therefore has constant `K = +1/4`, which is the same metric family as the object under study.

Rung 7 validates the assembly of (1) specifically. With a synthetic linear model `p(h) = softmax(U h)` and `d = N − 1`, the map `h → p` is a diffeomorphism onto the simplex interior, so the pullback metric is the simplex metric in other coordinates and the curvature must be exactly +1/4 for every valid `U`. Three further checks accompany it: reparameterisation invariance under `U → UM` (worst error 1.9e−11), scalar curvature `R = k(k−1)/4` through the pullback (4.9e−12), and a deliberately broken assembly with the outer-product term of (1) dropped, which must fail — and does, at 0.263. A test that cannot fail proves nothing.

### B. The Hessian identity

Equation (2) was verified on a real model by an independent code path. Sweeping `ε` produces the expected plateau, at relative error 8.7e−06, confirming that the assembled `G(h)` is the Hessian of the model's own divergence and not an unrelated quadratic form.

### C. Independent implementation

Rungs 2, 3, 4, 5 and 6 were re-run against the `information_geometry` module of geomstats [17], an independently implemented, peer-reviewed library, on identical inputs. Three families agree at machine precision across eight comparisons: univariate normal to 1.1e−16, gamma to 2.5e−14, beta to 3.2e−14.

Two families disagree, and the disagreement is reported rather than resolved by assumption. For the Poincaré half-plane and the categorical simplex the two libraries return different curvatures. The diagnostic applied was to compare the *metrics* rather than the curvatures, since a disagreement arising from a parameterisation convention is harmless and one arising from a curvature routine is not. **In both failing cases the metrics agree exactly** — Poincaré to 1e−12, categorical inner products to 7e−15 — while the curvatures do not. Neither disagreement is a convention.

On the evidence available the corroboration is asymmetric: the library's Poincaré curvature is not constant across base points, which is impossible for a hyperbolic space, whereas the +1/4 reported here is independently corroborated by the analytic radius-2 sphere isometry, by rung 3, and by rung 7's entirely separate code path. The discrepancy remains open.

---

## V. The Control

The results in Section VI rest on one control, so its construction is given in full.

Let `idx` be the indices of the vocabulary rows retained at a given state. The **paired within-set scramble** replaces `idx` by `idx[randperm(len(idx))]`: the same rows are used, in a permuted assignment to probabilities. This holds fixed, exactly:

- the predictive distribution `p`, and therefore the entropy, to machine precision;
- the multiset of retained unembedding rows, and therefore their norms and spectrum;
- the conditioning of the retained subspace;
- the dimension `k`.

What it destroys is only which probability sits on which direction — the learned assignment.

An earlier version of this control drew the permutation over the whole vocabulary. That version was found on audit to change the retained row *set* as well as the pairing, overlapping the real set at only 5 of 512 rows and shifting the median row norm from 2.44 to 3.10. It is reported alongside the clean control below, and it moves the results in the same direction, but only the within-set version supports the claim.

---

## VI. Results

### A. Curvature sits at the simplex value across architectures

Fig. 1 shows sectional curvature at 456 sampled states across four architectures — GPT-2 [18], Pythia-160M [19], LLaMA-160M and SmolLM2-135M — spanning both normalisation schemes, tied and untied embeddings, depths from 12 to 30 layers, and widths from 512 to 768.

**[FIGURE 1: fig1_curvature.png]**
*Fig. 1. Sectional curvature at 456 states across four architectures. The dashed line is the ambient simplex value +1/4. Boxes are quartiles; red bars are medians.*

Median values are +0.2599 (GPT-2), +0.2738 (Pythia-160M), +0.2559 (LLaMA-160M) and +0.2547 (SmolLM2-135M) — a spread of 0.019. The representation manifold under this metric is strongly positively curved, at the ambient value, and consistently so across architecture.

The deviation from exactly +1/4 is *not* claimed as a model-specific signal. It is unstable under changes of `k` and its layer ranking does not survive resampling, so any magnitude signal in absolute curvature remains unfound.

### B. The separation

Fig. 2 is the central result. One control, three quantities, two outcomes.

**[FIGURE 2: fig2_split.png]**
*Fig. 2. One paired control applied to three quantities. Sectional curvature collapses; the log volume element and the effective dimension are reproduced almost exactly.*

**TABLE III. ONE CONTROL, THREE QUANTITIES**

| quantity | real | scrambled | retained |
|---|---|---|---|
| sectional curvature `K` (n = 221) | 0.2546 | **0.0109** | **4 %** |
| log volume element, layer profile | rho = 1 by definition | **rho = +0.967** | 97 % |
| effective dimension, layer profile | rho = 1 by definition | **rho = +0.987** | 99 % |

The two rank correlations are taken over nine layer medians, and at that sample size they are sensitive to the median convention: the profile was computed with `torch.median`, which returns the lower of the two middle values for even `n`, and with a true median the log-volume figure is +0.867 rather than +0.967. **The per-point correlations, over all 360 states and free of that convention, are +0.957 and +0.991** — quoted here in preference. Every version supports the same reading; none of them is stable to three digits.

The curvature collapse is 201 of 221 points in the same direction, z = +12.18, and it is stable across the accessible range of `k`: at `k` = 4, 0.2543 to 0.0137 (63/69, z = +6.86); at `k` = 5, as tabulated; at `k` = 6, 0.2532 to 0.0512 (58/69, z = +5.66). This is the one Riemann-derived result in this work that is not `k`-fragile. The whole-vocabulary variant of the control gives 0.2586 to 0.0002 (56/60, z = +6.71).

By contrast, the scrambled log-volume profile has its minimum at the same layer as the real one (layer 20) and the scrambled effective-dimension profile has its minimum at the same layer as the real one (layer 10), retaining 99.5 % of the real profile's range.

**TABLE IV. EFFECTIVE DIMENSION, REAL AND SCRAMBLED**

| layer | 1 | 5 | 10 | 15 | 20 | 25 | 28 | 29 | 30 |
|---|---|---|---|---|---|---|---|---|---|
| real | 14 | 8 | **6** | 7 | 6 | 22 | 77 | 167 | 217 |
| scrambled | 13 | 8 | **5** | 6 | 6 | 24 | 54 | 150 | 215 |

The interpretation is direct. Quantities computed from the *spectrum* of `G(h)` — its determinant, its eigenvalue decay — are reproduced by a structure-free control at matched entropy, so they are largely restatements of the predictive concentration profile. Curvature requires the *second derivatives* of the metric, and it is the quantity that collapses.

This is prescriptive rather than merely descriptive: it says which geometric quantities are worth the cost of computing.

### C. A layer-resolved reading

The collapse is not uniform. Table V gives the fraction of the real curvature retained by the scramble at each layer.

**TABLE V. CURVATURE RETAINED UNDER THE SCRAMBLE, BY LAYER**

| layer | 1 | 5 | 10 | 15 | 20 | 25 | 28 | 29 | 30 |
|---|---|---|---|---|---|---|---|---|---|
| retained | 2.3 % | 2.3 % | 1.5 % | 1.7 % | −5.3 % | −95.5 % | 7.0 % | 25.8 % | 61.4 % |
| sign test | 19/21 | 17/17 | 22/25 | 23/24 | 28/29 | 20/20 | 20/23 | 30/35 | 22/27 |

The learned assignment does nearly all the work through the middle of the network and least at the two ends — where context has not yet been integrated, and where the prediction is already committed and concentration alone largely fixes the geometry.

### D. The instruments do not agree with each other

If the intrinsic curvature and the published proxies were measuring the same underlying property, they would correlate. On 653 states, with all four instruments computed at the same points, they do not. The largest absolute rank correlation between any two different instrument families is 0.295, and most pairs fall below 0.14. The two intrinsic quantities computed here, sectional `K` and scalar `R`, correlate with each other at +0.69 to +0.74, which is what internal consistency looks like and what the cross-family pairs lack.

A second discriminator is corpus stability. Moving from hand-written probe sentences to WikiText-2 [20], the intrinsic instrument's correlation with entropy moves by 0.028 (from −0.475 to −0.447), while the three proxies move by 0.126, 0.104 and 0.194 respectively — one of them changing sign. An instrument whose reading depends this strongly on the corpus is measuring a property of the corpus.

---

## VII. Why a Near-Zero Reading Is Not Evidence of Flatness

The published value of order 10⁻⁵ [8] reproduces here. That is not in dispute, and Fig. 3(a) shows it: with a tight retained-variance threshold the local-PCA residual on GPT-2 falls below 10⁻⁵ and continues to fall until it underflows.

**[FIGURE 3: fig3_threshold.png]**
*Fig. 3. (a) The local-PCA residual proxy on GPT-2 layer 12 as a function of its retained-variance threshold, for several neighbourhood sizes. (b) The same estimator applied to manifolds of known curvature. Bars at the axis floor are numerically zero.*

The question is what the number licenses. Fig. 3(b) applies the same estimator, with the same thresholds, to a unit 3-sphere embedded in 12 dimensions — a manifold whose sectional curvature is exactly +1 everywhere — and to a flat 3-plane, whose curvature is exactly 0. At loose thresholds the estimator separates them. At the threshold that produces the published value it does not: the sphere's residual falls to order 10⁻³¹, numerically indistinguishable from the plane's exact zero.

An estimator that reports a unit sphere as flat to 10⁻³¹ cannot support the inference from "the residual is 10⁻⁵" to "the manifold is flat". The residual is a function of the threshold at least as much as of the geometry. Under the intrinsic metric the same states give `K` ≈ +1/4, which is not a small number.

This is offered as a methodological correction, not as a claim that the underlying observation was fabricated. The measurement replicates; the interpretation does not follow from it.

---

## VIII. A Behavioural Probe, Reported at the Strength the Evidence Supports

If the metric tracks the model's predictive state, distance under it should respond to a change in word sense. On 32 polysemous words drawn in the style of the word-in-context task [21], with frame-matched controls — one context pair holding the sense fixed and varying the surrounding words, one changing the sense while matching lexical overlap (0.235 against 0.250) — the Fisher–Rao distance is 1.88 times larger for a sense change than for a same-sense context change, 31 of 32 pairs in the same direction, z = +5.30.

The effect is real and it is in the expected direction. It does not, however, demonstrate an advantage for this metric: on the same pairs a Euclidean distance gives a ratio of 2.25 and an unembedding-space distance gives 2.15. The honest summary is that the Fisher–Rao metric detects the sense change reliably, and that this particular probe does not distinguish it from cheaper alternatives. A probe that would — perturbation along high-curvature directions at the disambiguation layer, testing whether the resolved sense flips — is left to future work.

---

## IX. What Did Not Survive

Seven claims made during this work were weakened or withdrawn by controls the work constructed to attack itself. They are listed because a paper that reports only what survived gives no information about the filter.

- **A cross-model volume difference** was confidently reported from 24 states chosen for being cheap to compute, and reversed at 220 randomly drawn ones. Four candidate mechanisms were then tested and all rejected.
- **A "Ricci predicts departure" result** held at n = 40 (z = +3.16) and collapsed at n = 105 (z = +0.29). After three narrowings it survives on the GPT-NeoX family only, where z ranges from +3.47 to +4.75, against +0.00 to +0.75 on the other three architectures.
- **A same-sense control** gave a 2.00× effect that turned out to be lexical overlap; the frame-matched replacement in Section VIII is the corrected version.
- **Three successive versions of the scramble control were wrong**: one was provably the identity on this metric, one changed spectrum, row norms and assignment simultaneously, and one swapped the retained direction set as well as the pairing. Each looked clean.
- **A sign error in a basis vector** corrupted individual states catastrophically while leaving the median clean, and was found only when a second architecture made the signs come out uniformly wrong.
- **A timing table** was five times too high, having been measured on a cold and contended machine.
- **A correlation of +0.42** between the count of negatively curved planes and effective dimension was an artefact of the rank function. The count is zero at 407 of 456 states, and ties were being broken by row order in a file grouped by model and then by layer; the row index itself correlates with effective dimension at +0.51. Tie-corrected the value is +0.118, and the accompanying claim that effective dimension explained "twice what entropy does" dissolves entirely, entropy going from +0.212 to −0.034. Tested properly, with a Mann–Whitney statistic on the grouping the count actually induces, a real but modest effect remains: states carrying a negatively curved plane have 2.03 times the median effective dimension at indistinguishable entropy (248 against 122, AUC 0.618, z = +2.69), and neither conditioning nor entropy separates the groups.

The pattern is consistent enough to state as a working rule: on this data, a result at one `n`, one `k`, or one control is a hypothesis, and becomes a finding when it survives the second one.

The last item generalises differently from the rest and is worth stating separately, because no amount of re-running would have caught it. A statistic can be wrong because of the *storage order* of the data. Nothing in the analysis referenced the file layout; the rank function did, invisibly, and only for the variables that had ties — which were exactly the discrete ones carrying the newest claims. The diagnostic that exposed it was to correlate the row index with the covariate, a quantity there is otherwise no reason to compute. Where a variable is mostly a single value, it should not be ranked at all; the grouping it induces should be tested instead.

---

## X. Limitations

**Scale.** Models are 70M to 160M parameters. Nothing here rules out the geometry changing at 1B and above, and the like-for-like comparison to King et al. [7] requires the larger models used there.

**The `k` ≤ 8 ceiling.** Everything Riemann-derived is computed on a slice of dimension 4 to 6 holding about 88 % of the metric's trace. It is a slice, and the reported curvature is that of the slice under the induced metric.

**Cell sizes.** Pooled samples are 221 to 653; per-layer cells are 17 to 35. Given how often this work's own claims died as `n` rose, per-cell numbers should be read as indicative and pooled ones as the result.

**Two corpora, both English edited prose.** No code, no dialogue, no low-resource languages.

**One validation rung open.** The independent-implementation arm agrees on three Fisher–Rao families and disagrees on two, as reported in Section IV.C.

---

## XI. Conclusion

The intrinsic Riemannian curvature of the Fisher–Rao pullback metric on transformer hidden states is computable, is validated against seven manifolds of known curvature, and sits at the ambient simplex value of +1/4 across four architectures.

The result expected to be most useful is the separation. A single control — destroying the learned token-to-direction assignment at matched entropy, matched conditioning and an identical direction set — collapses sectional curvature from 0.2546 to 0.0109 while reproducing the layer-wise log-volume profile at rho = +0.957 and the effective-dimension profile at rho = +0.991. Stability and informativeness turned out to be different axes, and they pointed in opposite directions: the well-conditioned, `k`-robust, cheaply computed spectral quantities are largely restatements of the predictive entropy profile, while the fragile and expensive curvature is the quantity carrying learned structure.

This has an immediate consequence for the intrinsic-dimension literature. The characteristic depth profile reported by Valeriani et al. [10] and others is reproduced here, under this metric, at rho = +0.991 by a control containing no learned structure at all. Whether the published profile has the same character is a testable question: apply a matched-entropy control to an ambient TWO-NN estimate [22] and see whether the dip survives. If it does not, a substantial part of that literature is measuring predictive concentration.

---

## Reproduction

All code, saved result files and figure-generation scripts are available at `https://github.com/ChaminiMaduwanthi/Fisher-Rao`. Every figure in this article is regenerated from the saved result files by `make_figures.py`; no number in a figure is typed by hand. The validation ladder is `validate_curvature.py`, `validate_pullback.py` and `validate_geomstats.py`, and each prints a pass or fail verdict.

---

## References

[1] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2017, pp. 5998–6008.

[2] N. N. Čencov, *Statistical Decision Rules and Optimal Inference*. Providence, RI, USA: American Mathematical Society, 1982.

[3] S. Amari and H. Nagaoka, *Methods of Information Geometry*. Providence, RI, USA: American Mathematical Society, 2000.

[4] Y. Wang and L. Zhao, "FishBack: Pullback Fisher geometry for optimal activation steering in transformers," *arXiv:2605.17231*, 2026.

[5] A. Zaher, M. Trzaskowski, D. Nguyen, and F. Roosta, "The geometric wall: Manifold structure predicts layerwise sparse autoencoder scaling laws," *arXiv:2605.09887*, 2026.

[6] R. Manson, "Curved inference: Concern-sensitive geometry in large language model residual streams," *arXiv:2507.21107*, 2025.

[7] M. King, E. Fedorenko, and E. A. Hosseini, "Representational curvature modulates behavioral uncertainty in large language models," *arXiv:2604.23985*, 2026.

[8] M. A. Mabrok, "Latent semantic manifolds in large language models," *arXiv:2603.22301*, 2026.

[9] J. A. Zavatone-Veth, S. Yang, J. A. Rubinfien, and C. Pehlevan, "How does training shape the Riemannian geometry of neural network representations?," *arXiv:2301.11375*, 2025.

[10] L. Valeriani, D. Doimo, F. Cuturello, A. Laio, A. Ansuini, and A. Cazzaniga, "The geometry of hidden representations of large transformer models," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2023.

[11] A. Modell, P. Rubin-Delanchy, and N. Whiteley, "The origins of representation manifolds in large language models," *arXiv:2505.18235*, 2025.

[12] J. Martens and R. Grosse, "Optimizing neural networks with Kronecker-factored approximate curvature," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2015, pp. 2408–2417.

[13] B. Zhang and R. Sennrich, "Root mean square layer normalization," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2019.

[14] J. L. Ba, J. R. Kiros, and G. E. Hinton, "Layer normalization," *arXiv:1607.06450*, 2016.

[15] M. P. do Carmo, *Riemannian Geometry*. Boston, MA, USA: Birkhäuser, 1992.

[16] A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2019, pp. 8026–8037.

[17] A. Le Brigant, J. Deschamps, A. Collas, and N. Miolane, "Parametric information geometry with the package Geomstats," *ACM Trans. Math. Softw.*, vol. 49, no. 4, pp. 1–26, 2023.

[18] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, "Language models are unsupervised multitask learners," OpenAI Tech. Rep., 2019.

[19] S. Biderman et al., "Pythia: A suite for analyzing large language models across training and scaling," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2023, pp. 2397–2430.

[20] S. Merity, C. Xiong, J. Bradbury, and R. Socher, "Pointer sentinel mixture models," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2017.

[21] M. T. Pilehvar and J. Camacho-Collados, "WiC: The word-in-context dataset for evaluating context-sensitive meaning representations," in *Proc. NAACL-HLT*, 2019, pp. 1267–1273.

[22] E. Facco, M. d'Errico, A. Rodriguez, and A. Laio, "Estimating the intrinsic dimension of datasets by a minimal neighborhood information," *Sci. Rep.*, vol. 7, no. 1, p. 12140, 2017.
