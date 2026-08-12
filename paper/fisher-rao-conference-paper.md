# Intrinsic Fisher–Rao Curvature of Transformer Representations: Curvature Carries Learned Structure, the Spectrum Carries Entropy

**Chamini Maduwanthi**

---

## Abstract

This article computes the intrinsic Riemannian curvature of the Fisher–Rao metric pulled back onto the hidden states of transformer language models, and establishes what that curvature measures. Prior work derives the pullback metric correctly but stops short of the curvature tensor, measures curvature under a flat Euclidean metric, or substitutes a local-PCA residual for curvature. Christoffel symbols, the Riemann tensor, and sectional and scalar curvature are computed here directly, on a quotient-then-restrict subspace that removes the exact null directions of the normalisation layer. Across four architectures and 456 sampled states the median sectional curvature is +0.2547 to +0.2738, a spread of 0.019 around the ambient simplex value of +1/4. The central finding is a separation. A paired scramble that destroys the learned token-to-direction assignment, while holding the predictive distribution, the conditioning and the retained direction set exactly fixed, collapses sectional curvature from 0.2546 to 0.0109 (201 of 221 states, z = +12.18), yet reproduces the layer-wise log-volume profile at rho = +0.957 and the effective-dimension profile at rho = +0.991, with the same minimum layer in both cases. Quantities derived from the metric's spectrum therefore track predictive concentration, and curvature tracks the learned assignment. A seven-rung validation ladder, a Kullback–Leibler Hessian identity check, and a cross-check against an independent information-geometry library support the measurements.

**Index Terms** — information geometry, Fisher–Rao metric, Riemannian curvature, transformer representations, interpretability, intrinsic dimension.

---

## I. Introduction

The hidden state of a transformer language model [1] is usually treated as a point in a flat Euclidean space. Distances are cosine or L2, directions are compared by inner product, and the geometry of the space is taken to be the geometry of the coordinates it is written in. That is a choice, and it is not the only available one.

A transformer with hidden state `h` and unembedding `U` defines a probability distribution over the next token, `p(h) = softmax(U · norm(h))`. The map `h → p(h)` sends hidden states into the interior of the vocabulary simplex, a statistical manifold. On such a manifold there is a distinguished Riemannian metric: Čencov's theorem [2] establishes that the Fisher information metric is, up to scale, the only metric on the space of probability distributions invariant under sufficient statistics [3]. Pulling it back through `h → p(h)` yields a metric on hidden-state space derived from the model's own predictive behaviour rather than imposed by the coordinate system.

That pullback is established. It has been derived and used for activation steering [4], [5]. The next step is the open one. A metric alone gives lengths, angles and volumes. Curvature requires the second derivatives of the metric — the Christoffel symbols, then the Riemann tensor — and it is curvature, not the metric, that determines whether a manifold is intrinsically bent or merely written in curvilinear coordinates. The literature contains curvature measured under a flat metric [6], [7], and a local-PCA residual of order 10⁻⁵ interpreted as near-flatness [8]. It does not contain the intrinsic Riemannian curvature of the Fisher–Rao pullback for a transformer.

This article computes it, and then asks the more useful question: once computed, what does curvature measure that cheaper quantities do not? The answer is sharp, and it runs against the intuition that expensive geometric quantities are elaborate restatements of simple ones. Three quantities are put through a single control. Two of them — the log volume element and the effective dimension of the metric — are reproduced almost exactly when all learned structure is destroyed. Sectional curvature is not.

The contributions are:

1. **The first intrinsic Riemannian curvature of the Fisher–Rao pullback metric on transformer hidden states**, validated on a seven-rung ladder of manifolds with analytically known curvature, and measured at +1/4 across four architectures.
2. **A separation result.** One paired control splits three geometric quantities into those that read predictive concentration and the one that reads learned structure. The result is prescriptive: it identifies which geometric quantities repay the cost of computing.
3. **A calibration of the near-flat reading.** The estimator that produces 10⁻⁵ is shown, at the threshold that produces it, to report a unit 3-sphere as flat to 10⁻³¹.
4. **A fully reproducible record.** Every figure and table is regenerated from saved result files by a single script, and the validation ladder prints a pass or fail verdict.

---

## II. Related Work and the Gap

**The pullback metric is established.** Mabrok [8] and Wang and Zhao [4] both derive the Fisher–Rao pullback for transformer states. Wang and Zhao use it for steering and do not compute curvature. Mabrok reports a curvature, but the quantity computed is a local-PCA residual on the ambient activation cloud rather than a Riemann tensor of the pullback metric. Section VII examines what that measurement supports.

**Curvature has been measured under a flat metric.** Manson [6] measures a Frenet-style curvature of residual-stream trajectories, and King et al. [7] measure an angular curvature between successive difference vectors. Both find that curvature varies systematically with semantic content and predictive uncertainty, which is good evidence that geometry is doing work. Both compute the quantity in Euclidean coordinates, so what varies is a property of the coordinate chart together with the trajectory rather than an invariant of a manifold.

**Intrinsic curvature has been computed for other networks.** Zavatone-Veth et al. [9] compute Riemannian geometry of neural-network representations under a data-induced metric, establishing that the machinery is tractable. The metric is not Fisher–Rao and the models are not transformers.

**Intrinsic dimension is a parallel literature.** Valeriani et al. [10] and Modell et al. [11] report a characteristic profile of intrinsic dimension across depth, with a minimum roughly one third of the way through, typically estimated by a nearest-neighbour method such as TWO-NN [12]. Section VI.C reports the analogous profile under the present metric and the control applied to it.

The gap this article closes is precise. The pullback metric has been derived, and layer-wise curvature has been measured under flat metrics, but the intrinsic Riemannian curvature of the state-dependent Fisher–Rao metric — Christoffel symbols to Riemann tensor to sectional and scalar curvature — has not been computed for a transformer at any scale or any layer.

---

## III. The Metric and Its Curvature

### A. The pullback

Write `norm` for the model's final normalisation layer and `A(h)` for its Jacobian. With `J = U · A(h)` and `p = p(h)`, the Fisher information metric on the simplex pulls back to

> **G(h) = Jᵀ (diag(p) − p pᵀ) J.**  (1)

This is a `d × d` matrix, closed-form and inexpensive. It is distinct from the parameter-space Fisher used in second-order optimisation [13], which is intractable at this scale. Equation (1) is the metric of the model's *output distribution* expressed in the coordinates of its *hidden state*.

`G(h)` is the Hessian of the Kullback–Leibler divergence at zero displacement, which provides a falsification test independent of the implementation of (1):

> **KL(p(h) ‖ p(h + εv)) = ½ ε² vᵀ G(h) v + O(ε³).**  (2)

The ambient geometry that (1) pulls back is fully known, and this is what makes the measurement interpretable rather than merely numerical. Under the full Fisher–Rao metric the interior of the vocabulary simplex is isometric to the positive orthant of a sphere of radius 2, through the map `p ↦ 2√p`. Its sectional curvature is therefore the constant `+1/4`, and the geodesic distance between two predictive distributions has the closed form

> **d(p, q) = 2 · arccos( Σᵢ √(pᵢ qᵢ) ).**  (3)

Equation (3) is the distance used for the behavioural probe in Section VIII. It replaces a cosine or L2 distance between hidden states with the length of the shortest path between the distributions those states induce, and it is bounded above by `π`, unlike the unbounded ambient distances it is compared against.

### B. Exact null directions

Both normalisation layers annihilate directions exactly. RMSNorm [14] has Jacobian `A = diag(g)(1/r)(I − ĥĥᵀ)`, with one null direction, `h` itself. LayerNorm [15] additionally subtracts the mean and so annihilates two: the radial direction and the all-ones direction. `G(h)` is therefore singular by construction, with rank at most `d − 1` or `d − 2`.

Suppressing this with a pseudo-inverse would return a plausible wrong answer. The null space is instead quotiented explicitly. With `N` an orthonormal basis of the null directions and `P = I − N Nᵀ`, curvature is computed on the span of the top-`k` eigenvectors of `P G P`. Table I reports that the number of null directions found matches theory on every architecture tested, and that the model is exactly invariant along them.

**TABLE I. NULL-DIRECTION FALSIFICATION TEST**

| model | norm | null dirs found (expected) | max output change along null |
|---|---|---|---|
| SmolLM2-135M | RMSNorm | 1 (1) | machine precision |
| LLaMA-160M | RMSNorm | 1 (1) | machine precision |
| Pythia-70M | LayerNorm | 2 (2) | machine precision |
| Pythia-160M | LayerNorm | 2 (2) | machine precision |
| GPT-2 | LayerNorm | 2 (2) | machine precision |
| GPT-Neo-125M | LayerNorm | 2 (2) | machine precision |

The step size used is large enough to double the norm of the state, so the invariance is a property of the architecture rather than of a small perturbation.

### C. Curvature

On the retained `k`-dimensional subspace, Christoffel symbols and the Riemann tensor are formed by nested reverse-mode differentiation of (1), and sectional curvature of the plane spanned by `u, v` follows in the standard way [16]. All differentiation is reverse-mode automatic differentiation in double precision [17]. Cost grows by a factor of about 2.7 per unit increase in `k`, and memory makes `k > 8` infeasible on the hardware used; all curvature results below are at `k = 4, 5, 6`, with `k = 5` as the default. The retained subspace holds approximately 88 % of the metric's trace.

The scope of the measurement follows from this: the reported curvature is that of a `k`-dimensional slice under the induced metric, not a sectional curvature of a totally geodesic submanifold of the full space.

### D. Models, corpora and probes

Four architectures carry the main results — GPT-2 [18], Pythia-160M [19], LLaMA-160M and SmolLM2-135M — spanning both normalisation schemes, tied and untied embeddings, depths from 12 to 30 layers and widths from 512 to 768. Pythia-70M and GPT-Neo-125M extend the null-direction test to six models. Two corpora are used: hand-written probe sentences and WikiText-2 [20]. The behavioural probe in Section VIII uses 32 polysemous words with contexts constructed in the style of the word-in-context task [21].

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

Rungs 2, 3, 4, 5 and 6 were re-run against the `information_geometry` module of geomstats [22], an independently implemented and peer-reviewed library, on identical inputs. Three families agree at machine precision across eight comparisons: univariate normal to 1.1e−16, gamma to 2.5e−14, and beta to 3.2e−14. Both new rungs are among them.

For the Poincaré half-plane and the categorical simplex the two libraries return different curvatures, and the disagreement is reported rather than resolved by assumption. The diagnostic applied was to compare the *metrics* rather than the curvatures, since a disagreement arising from a parameterisation convention is harmless while one arising from a curvature routine is not. In both cases the metrics agree exactly — Poincaré to 1e−12, categorical inner products to 7e−15 — which localises the disagreement to a curvature routine. On the evidence available the corroboration is asymmetric: the library's Poincaré curvature is not constant across base points, which a hyperbolic space requires, whereas the +1/4 reported here is independently corroborated by the analytic radius-2 sphere isometry, by rung 3, and by rung 7's entirely separate code path.

---

## V. The Control

The results in Section VI rest on one control, so its construction is given in full.

Let `idx` be the indices of the vocabulary rows retained at a given state. The **paired within-set scramble** replaces `idx` by `idx[randperm(len(idx))]`: the same rows are used, in a permuted assignment to probabilities. This holds fixed, exactly:

- the predictive distribution `p`, and therefore the entropy, to machine precision;
- the multiset of retained unembedding rows, and therefore their norms and spectrum;
- the conditioning of the retained subspace;
- the dimension `k`.

What it destroys is only which probability sits on which direction — the learned assignment. Because everything else is held fixed by construction, any quantity that changes under it depends on that assignment, and any quantity that does not is determined by the predictive distribution alone.

A whole-vocabulary variant of the permutation was also run. It changes the retained row set as well as the pairing, so it answers a broader question; it is reported alongside the clean control and moves the results in the same direction.

---

## VI. Results

### A. Curvature sits at the simplex value across architectures

Fig. 1 shows sectional curvature at 456 sampled states across the four main architectures.

**[FIGURE 1: fig1_curvature.png]**
*Fig. 1. Sectional curvature at 456 states across four architectures. The dashed line is the ambient simplex value +1/4. Boxes are quartiles; red bars are medians.*

Median values are +0.2599 (GPT-2), +0.2738 (Pythia-160M), +0.2559 (LLaMA-160M) and +0.2547 (SmolLM2-135M) — a spread of 0.019. The representation manifold under this metric is strongly positively curved, at the ambient value, and consistently so across architecture, normalisation scheme, depth and width.

The interpretation is specific. Because +1/4 is the ambient simplex value, agreement with it establishes that the pullback is faithful and that the manifold is not flat; the magnitude of the small deviation from +1/4 is not itself treated as a model-specific signal, since it varies with `k`.

### B. The separation

Fig. 2 is the central result: one control, three quantities, two outcomes.

**[FIGURE 2: fig2_split.png]**
*Fig. 2. One paired control applied to three quantities. Sectional curvature collapses; the log volume element and the effective dimension are reproduced almost exactly.*

**TABLE III. ONE CONTROL, THREE QUANTITIES**

| quantity | real | scrambled | retained |
|---|---|---|---|
| sectional curvature `K` (n = 221) | 0.2546 | **0.0109** | **4 %** |
| log volume element, layer profile | rho = 1 by definition | **rho = +0.957** | 96 % |
| effective dimension, layer profile | rho = 1 by definition | **rho = +0.991** | 99 % |

The curvature collapse is 201 of 221 states in the same direction at z = +12.18, and it holds across the whole accessible range of `k`. Table IV records the sweep. Because `k` fixes the dimension of the slice on which the Riemann tensor is formed, agreement across it is the relevant robustness check for any Riemann-derived quantity, and the separation is the one such quantity in this work that passes it. The whole-vocabulary variant of the control, which changes the retained row set as well as the pairing, gives 0.2586 to 0.0002 (56 of 60, z = +6.71) — the same direction from a broader intervention. The two rank correlations in Table III are computed per state over 360 states.

**TABLE IV. STABILITY OF THE SEPARATION IN `k`**

| `k` | n | real | scrambled | same direction | z |
|---|---|---|---|---|---|
| 4 | 69 | 0.2543 | 0.0137 | 63 / 69 | +6.86 |
| **5** | **221** | **0.2546** | **0.0109** | **201 / 221** | **+12.18** |
| 6 | 69 | 0.2532 | 0.0512 | 58 / 69 | +5.66 |

By contrast, the scrambled log-volume profile has its minimum at the same layer as the real one, layer 20, and the scrambled effective-dimension profile has its minimum at the same layer as the real one, layer 10, retaining over 99 % of the real profile's range.

**TABLE V. EFFECTIVE DIMENSION, REAL AND SCRAMBLED**

| layer | 1 | 5 | 10 | 15 | 20 | 25 | 28 | 29 | 30 |
|---|---|---|---|---|---|---|---|---|---|
| real | 14 | 8 | **6** | 7 | 6 | 22 | 77 | 167 | 217 |
| scram. | 13 | 8 | **5** | 6 | 6 | 24 | 54 | 150 | 215 |

The interpretation is direct. Quantities computed from the *spectrum* of `G(h)` — its determinant, its eigenvalue decay — are reproduced by a structure-free control at matched entropy, so they are largely restatements of the predictive concentration profile. Curvature requires the *second derivatives* of the metric, and it is the quantity that responds to the learned assignment. This is prescriptive rather than merely descriptive: it identifies which geometric quantities are worth their cost.

### C. A layer-resolved reading

The collapse is not uniform. Table VI gives the fraction of the real curvature retained by the scramble at each layer, with the paired sign count.

**TABLE VI. CURVATURE RETAINED UNDER THE SCRAMBLE, BY LAYER**

| layer | 1 | 5 | 10 | 15 | 20 | 25 | 28 | 29 | 30 |
|---|---|---|---|---|---|---|---|---|---|
| retained | 2.3% | 2.3% | 1.5% | 1.7% | −5.3% | −95.5% | 7.0% | 25.8% | 61.4% |
| sign | 19/21 | 17/17 | 22/25 | 23/24 | 28/29 | 20/20 | 20/23 | 30/35 | 22/27 |

The learned assignment does nearly all the work through the middle of the network and least at the two ends — where context has not yet been integrated, and where the prediction is already committed and concentration alone largely fixes the geometry. The sign counts are near-unanimous in every layer, so the layer-wise pattern rests on the direction of the paired difference rather than on the magnitude of any single cell.

### D. The instruments do not agree with each other

If the intrinsic curvature and the published proxies measured the same underlying property, they would correlate. Computed at the same 653 states, they do not. The largest absolute rank correlation between any two different instrument families is 0.295, and eleven of the twelve cross-family correlations fall below 0.14. The two intrinsic quantities computed here, sectional `K` and scalar `R`, correlate with each other at +0.69 and +0.74 on the two corpora, which is what internal consistency looks like and what the cross-family pairs lack.

A second discriminator is corpus stability. Each instrument's correlation with predictive entropy is measured on hand-written probe sentences and again on WikiText-2, and the two readings are compared. Table VII gives the result.

**TABLE VII. INSTRUMENT STABILITY ACROSS CORPORA**

| instrument | hand-written | WikiText-2 | shift |
|---|---|---|---|
| **intrinsic sectional `K`** | −0.447 | −0.475 | **0.028** |
| **intrinsic scalar `R`** | −0.579 | −0.544 | **0.034** |
| Frenet under `UᵀU` | −0.113 | −0.217 | 0.104 |
| local-PCA residual | +0.175 | +0.050 | 0.126 |
| Euclidean angle | +0.043 | −0.151 | 0.194 |

The two intrinsic quantities move by 0.028 and 0.034 between corpora; the three proxies move by four to seven times as much, and the Euclidean angle changes sign, reporting the opposite relationship with entropy on the two corpora. Stability of this kind is the behaviour expected of an instrument reading a property of the model rather than of the text it is fed, and it is a practical reason to prefer the intrinsic quantity even where a proxy is cheaper: a reading that must be re-established for every corpus supports no comparison across studies.

---

## VII. What a Near-Zero Reading Establishes

The published value of order 10⁻⁵ reproduces here. That is not in dispute, and Fig. 3(a) shows it: with a tight retained-variance threshold the local-PCA residual on GPT-2 falls below 10⁻⁵ and continues to fall until it underflows.

**[FIGURE 3: fig3_threshold.png]**
*Fig. 3. (a) The local-PCA residual proxy on GPT-2 layer 12 as a function of its retained-variance threshold, for several neighbourhood sizes. (b) The same estimator applied to manifolds of known curvature. Bars at the axis floor are numerically zero.*

The useful question is what the number licenses. Fig. 3(b) applies the same estimator, with the same thresholds, to a unit 3-sphere embedded in 12 dimensions — a manifold whose sectional curvature is exactly +1 everywhere — and to a flat 3-plane, whose curvature is exactly 0. At loose thresholds the estimator separates them cleanly, which is the regime in which it is informative. At the threshold that produces the published value the sphere's residual falls to order 10⁻³¹, numerically indistinguishable from the plane's exact zero.

The residual is therefore a function of the threshold at least as much as of the geometry, and the calibration curve in Fig. 3(b) is what makes the estimator interpretable: it identifies the thresholds at which the quantity discriminates curvature and those at which it does not. Under the intrinsic metric the same states give `K` ≈ +1/4. The measurement replicates; the calibration establishes the range over which it can be read as curvature.

---

## VIII. A Behavioural Probe

If the metric tracks the model's predictive state, distance under it should respond to a change in word sense. The probe uses 32 polysemous words. For each, one context pair holds the sense fixed and varies the surrounding words, and a second pair changes the sense while matching lexical overlap between the two arms (0.235 against 0.250), so that overlap cannot account for a difference.

Under the frame-matched control the Fisher–Rao distance is **2.03 times larger for a sense change than for a same-sense context change**, with 30 of 32 pairs in the same direction, z = +4.95. Against the looser control that does not match overlap the ratio is 2.19, 31 of 32 pairs, z = +5.30. The effect is clear, in the expected direction, and robust to which of the two controls is used.

On the same pairs a Euclidean distance gives 2.13 and an unembedding-space distance gives 1.71. The Fisher–Rao metric therefore detects the sense change reliably and separates the two conditions more sharply than the unembedding-space distance, while performing comparably to the Euclidean distance on this particular probe. A probe designed to discriminate between the metrics — perturbation along high-curvature directions at the disambiguation layer, testing whether the resolved sense flips — is the natural next experiment and is left to future work.

---

## IX. Robustness Practice

Several intermediate results in this work were revised as sample sizes and controls were strengthened, and the practices that caught them are worth stating, since they apply to any measurement of this kind.

**Controls are specified by what they hold fixed.** Three successive versions of the scramble were built before the version in Section V; each earlier one varied more than one property at a time, and the fix in each case was to enumerate the invariants and verify them numerically rather than by inspection. The control reported here has its four invariants checked at run time.

**Effects are confirmed at a second sample size.** A layer-wise result that held at n = 40 was re-run at n = 105 and narrowed to the GPT-NeoX family, where it is strong (z = +3.47 to +4.75). Reporting it at the scope where it holds is more useful than reporting it at the scope where it was first seen.

**Discrete variables are not rank-correlated.** A count that is zero at 407 of 456 states was initially summarised with a Spearman correlation. Rank correlations break ties by row order, and the point files are grouped by model and layer, so the row index itself correlated with the covariate at +0.51. Testing the grouping the count actually induces, with a Mann–Whitney statistic, gives the reliable version of the result: states carrying a negatively curved plane have 2.03 times the median effective dimension at indistinguishable entropy (248 against 122, AUC 0.618, z = +2.69), and neither conditioning nor entropy separates the groups. All rank correlations in this article use midrank tie correction.

The working rule that emerges is simple: a result at one `n`, one `k`, or one control is a hypothesis, and becomes a finding when it survives the second one. The results reported in Section VI are those that did.

---

## X. Limitations

**Scale.** Models are 70M to 160M parameters. Nothing here rules out the geometry changing at 1B and above, and a like-for-like comparison with the Euclidean-curvature literature requires the larger models used there.

**The `k` ≤ 8 ceiling.** Everything Riemann-derived is computed on a slice of dimension 4 to 6 holding about 88 % of the metric's trace, and the reported curvature is that of the slice under the induced metric.

**Cell sizes.** Pooled samples are 221 to 653 states; per-layer cells are 17 to 35. Pooled numbers carry the conclusions and per-cell numbers are indicative.

**Two corpora, both English edited prose.** No code, no dialogue, and no low-resource languages.

**One validation arm is partial.** The independent-implementation check agrees on three Fisher–Rao families and disagrees on two, as reported in Section IV.C.

---

## XI. Conclusion

The intrinsic Riemannian curvature of the Fisher–Rao pullback metric on transformer hidden states is computable, is validated against seven manifolds of known curvature, and sits at the ambient simplex value of +1/4 across four architectures, two normalisation schemes and a factor of two in depth.

The most useful result is the separation. A single control — destroying the learned token-to-direction assignment at matched entropy, matched conditioning and an identical direction set — collapses sectional curvature from 0.2546 to 0.0109 while reproducing the layer-wise log-volume profile at rho = +0.957 and the effective-dimension profile at rho = +0.991. Stability and informativeness are different axes here, and they point in opposite directions: the well-conditioned, `k`-robust, cheaply computed spectral quantities are largely restatements of the predictive entropy profile, while curvature is the quantity carrying the learned structure. For practitioners choosing a geometric summary of a representation, that is an actionable distinction.

This has an immediate and testable consequence for the intrinsic-dimension literature. The characteristic depth profile of effective dimension is reproduced here, under this metric, at rho = +0.991 by a control containing no learned structure at all. Whether published ambient intrinsic-dimension profiles share that character can be settled directly: apply a matched-entropy control to a nearest-neighbour estimate and see whether the dip survives. The prediction is specific, the control is cheap, and either outcome is informative.

---

## XII. Future Work

Four directions follow directly from the results above, in order of expected value.

**Test the separation against the intrinsic-dimension literature.** The control in Section V is inexpensive and applies unchanged to a nearest-neighbour dimension estimate on ambient activations. Running it against a published depth profile settles, in one experiment, whether the characteristic dip reflects learned structure or predictive concentration.

**Scale the measurement.** The pullback in (1) is `d × d` and its cost is dominated by the vocabulary sum, so the metric itself scales comfortably to billion-parameter models. Establishing whether curvature remains at the ambient value at that scale is the single most informative extension, and it also enables direct comparison with the Euclidean-curvature results reported on larger models.

**Raise the `k` ceiling.** The Riemann tensor is presently limited to a slice of dimension six by memory. A matrix-free formulation that never materialises the full tensor would widen the slice and test whether the separation strengthens as more of the metric's trace is retained.

**Close the loop from geometry to behaviour.** Section VIII establishes that distance under the metric responds to sense change. The sharper experiment perturbs a state along a high-curvature direction at the layer where disambiguation occurs and tests whether the resolved sense flips, which would connect the curvature measured here to a causal effect on model output.

---

## Reproduction

All code, saved result files and figure-generation scripts are available at `https://github.com/ChaminiMaduwanthi/Fisher-Rao`. Every figure and table in this article is regenerated from the saved result files; no number is transcribed by hand. The validation ladder is `validate_curvature.py`, `validate_pullback.py` and `validate_geomstats.py`, and each prints a pass or fail verdict.

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

[12] E. Facco, M. d'Errico, A. Rodriguez, and A. Laio, "Estimating the intrinsic dimension of datasets by a minimal neighborhood information," *Sci. Rep.*, vol. 7, no. 1, p. 12140, 2017.

[13] J. Martens and R. Grosse, "Optimizing neural networks with Kronecker-factored approximate curvature," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2015, pp. 2408–2417.

[14] B. Zhang and R. Sennrich, "Root mean square layer normalization," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2019.

[15] J. L. Ba, J. R. Kiros, and G. E. Hinton, "Layer normalization," *arXiv:1607.06450*, 2016.

[16] M. P. do Carmo, *Riemannian Geometry*. Boston, MA, USA: Birkhäuser, 1992.

[17] A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2019, pp. 8026–8037.

[18] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, "Language models are unsupervised multitask learners," OpenAI Tech. Rep., 2019.

[19] S. Biderman et al., "Pythia: A suite for analyzing large language models across training and scaling," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2023, pp. 2397–2430.

[20] S. Merity, C. Xiong, J. Bradbury, and R. Socher, "Pointer sentinel mixture models," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2017.

[21] M. T. Pilehvar and J. Camacho-Collados, "WiC: The word-in-context dataset for evaluating context-sensitive meaning representations," in *Proc. NAACL-HLT*, 2019, pp. 1267–1273.

[22] A. Le Brigant, J. Deschamps, A. Collas, and N. Miolane, "Parametric information geometry with the package Geomstats," *ACM Trans. Math. Softw.*, vol. 49, no. 4, pp. 1–26, 2023.
