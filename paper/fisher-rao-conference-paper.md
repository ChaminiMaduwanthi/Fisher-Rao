# Intrinsic Fisher–Rao Curvature of Transformer Representations: Curvature Carries Learned Structure, the Spectrum Carries Entropy

**Chamini Maduwanthi**

---

## Abstract

This article computes the intrinsic Riemannian curvature of the Fisher–Rao metric pulled back onto the hidden states of transformer language models, and establishes what that curvature measures. The pullback metric has been derived before and used for activation steering, but the curvature tensor itself has not been formed for a transformer: existing geometric readings either measure curvature under a flat Euclidean metric or substitute a local-PCA residual for it. Christoffel symbols, the Riemann tensor, and sectional and scalar curvature are computed here directly, on a subspace obtained by quotienting out the exact null directions of the normalisation layer. Across four architectures and 456 sampled states the median sectional curvature lies between +0.2548 and +0.2738, a spread of 0.019 about the ambient simplex value of +1/4. The principal finding is a separation between geometric quantities. A paired scramble that destroys the learned token-to-direction assignment, while holding the predictive distribution, the conditioning and the retained direction set exactly fixed, collapses sectional curvature from 0.2546 to 0.0109 (201 of 221 states, z = +12.18), yet reproduces the layer-wise log-volume profile at rho = +0.957 and the effective-dimension profile at rho = +0.991 over 360 states, with the same minimum layer in each case. Quantities derived from the metric's spectrum therefore track predictive concentration, while curvature tracks the learned assignment. A seven-rung validation ladder, a Kullback–Leibler Hessian identity, and agreement with an independent information-geometry library support the measurements.

**Index Terms** — information geometry, Fisher–Rao metric, Riemannian curvature, transformer representations, interpretability, intrinsic dimension.

---

## I. Introduction

The hidden state of a transformer language model [1] is ordinarily treated as a point in flat Euclidean space. Distances are cosine or L2, directions are compared by inner product, and the geometry of the space is taken to be the geometry of the coordinates in which it happens to be written. That is a modelling choice, and a different one is available.

A transformer with hidden state `h` and unembedding matrix `U` defines a distribution over the next token, `p(h) = softmax(U · norm(h))`. The map `h → p(h)` carries hidden states into the interior of the vocabulary simplex, which is a statistical manifold. On such a manifold one Riemannian metric is distinguished: Čencov's theorem [2] establishes that the Fisher information metric is, up to scale, the only metric on a space of probability distributions invariant under sufficient statistics [3]. Pulling it back through `h → p(h)` yields a metric on hidden-state space derived from the model's own predictive behaviour rather than imposed by the coordinate system.

The pullback itself is established. It has been derived and applied to activation steering [4] and to the analysis of sparse autoencoder scaling [5]. What has not been done is the step after it. A metric supplies lengths, angles and volumes; curvature requires the second derivatives of the metric — Christoffel symbols, then the Riemann tensor — and it is curvature, not the metric, that distinguishes a manifold which is intrinsically bent from one merely written in curvilinear coordinates.

Existing geometric readings of transformers stop short of that step in one of two ways. Curvature has been measured under a flat metric: a Frenet-style curvature of residual-stream trajectories [6], and an angular curvature between successive difference vectors [7]. Both report that curvature varies with semantic content and predictive uncertainty, which is good evidence that geometry carries signal; but both compute the quantity in Euclidean coordinates, so what varies is a property of the coordinate chart together with the trajectory rather than an invariant of a manifold. Alternatively, a local-PCA residual on the ambient activation cloud has been reported as a curvature of order 10⁻⁵ and read as near-flatness [8]. Intrinsic Riemannian curvature has been computed for neural representations under a data-induced metric [9], which establishes that the machinery is tractable, but the metric there is not Fisher–Rao and the models are not transformers.

A parallel literature measures intrinsic dimension across depth and reports a characteristic profile with a minimum roughly one third of the way through the network [10], usually estimated by a nearest-neighbour method [11] and more recently given a manifold-theoretic account [12]. Section V.B shows that the analogous profile under the Fisher–Rao metric is reproduced almost exactly by a control containing no learned structure, which raises a testable question for that literature.

The gap this article closes is therefore precise: the pullback metric has been derived, and layer-wise curvature has been measured under flat metrics, but the intrinsic Riemannian curvature of the state-dependent Fisher–Rao metric — Christoffel symbols through Riemann tensor to sectional and scalar curvature — has not been computed for a transformer at any scale or any layer. This article computes it, and then asks the more useful question: once computed, what does curvature measure that cheaper quantities do not? Three quantities are put through a single control. Two of them — the log volume element and the effective dimension of the metric — are reproduced almost exactly when all learned structure is destroyed. Sectional curvature is not.

The contributions are:

1. **The intrinsic Riemannian curvature of the Fisher–Rao pullback metric on transformer hidden states**, validated on a seven-rung ladder of manifolds with analytically known curvature, and measured at +1/4 across four architectures.
2. **A separation result.** One paired control divides three geometric quantities into those that read predictive concentration and the one that reads learned structure. The result is prescriptive: it identifies which geometric quantities repay the cost of computing.
3. **A calibration of the near-flat reading.** The estimator producing 10⁻⁵ is shown, at the threshold that produces it, to report a unit 3-sphere as flat to 10⁻³¹.

---

## II. The Metric and Its Curvature

### A. The pullback

Write `norm` for the model's final normalisation layer and `A(h)` for its Jacobian. With `J = U · A(h)` and `p = p(h)`, the Fisher information metric on the simplex pulls back to

> **G(h) = Jᵀ (diag(p) − p pᵀ) J.**  (1)

This is a `d × d` matrix, available in closed form and inexpensive to evaluate. It is distinct from the parameter-space Fisher matrix used in second-order optimisation [13], which is intractable at this scale. Equation (1) is the metric of the model's *output distribution* expressed in the coordinates of its *hidden state*, and every quantity in this article derives from it.

`G(h)` is the Hessian of the Kullback–Leibler divergence at zero displacement, which supplies a falsification test independent of how (1) is implemented:

> **KL(p(h) ‖ p(h + εv)) = ½ ε² vᵀ G(h) v + O(ε³).**  (2)

### B. Exact null directions

Both normalisation layers annihilate directions exactly. RMSNorm [14] has Jacobian `A = diag(g)(1/r)(I − ĥĥᵀ)` and one null direction, `h` itself. LayerNorm [15] additionally subtracts the mean and so annihilates two: the radial direction and the all-ones direction. `G(h)` is therefore singular by construction, with rank at most `d − 1` or `d − 2`.

Suppressing this with a pseudo-inverse would return a plausible wrong answer. The null space is instead quotiented explicitly: with `N` an orthonormal basis of the null directions and `P = I − N Nᵀ`, curvature is computed on the span of the top-`k` eigenvectors of `P G P`. This construction is falsifiable, and Table I reports the test. On every architecture examined the number of null directions found matches the theoretical prediction, and the model's output is invariant along them to machine precision at a step size large enough to double the norm of the state. The directions the metric calls null are exactly the directions the model cannot see, which establishes that the metric measured is the model's own.

**TABLE I. NULL-DIRECTION FALSIFICATION TEST**

| model | norm | null dirs found (expected) | output change along null |
|---|---|---|---|
| SmolLM2-135M | RMSNorm | 1 (1) | 5.2 × 10⁻⁸ |
| LLaMA-160M | RMSNorm | 1 (1) | machine precision |
| Pythia-70M | LayerNorm | 2 (2) | 3.2 × 10⁻⁷ |
| Pythia-160M | LayerNorm | 2 (2) | machine precision |
| GPT-2 | LayerNorm | 2 (2) | 8.4 × 10⁻⁸ |
| GPT-Neo-125M | LayerNorm | 2 (2) | machine precision |

### C. Curvature and the working subspace

On the retained `k`-dimensional subspace, Christoffel symbols and the Riemann tensor are formed by nested reverse-mode differentiation of (1), and the sectional curvature of the plane spanned by `u, v` follows in the standard way [16]. All differentiation is reverse-mode automatic differentiation in double precision [17].

The cost of the Riemann tensor grows by a factor of about 2.7 per unit increase in `k`, which places a hard ceiling at `k ≤ 8` on the hardware used. All curvature results below are computed at `k` = 4, 5 and 6, with `k` = 5 as the default, and the retained subspace holds approximately 88 % of the metric's trace. The reported curvature is accordingly that of a `k`-dimensional slice under the induced metric, and the stability of the central result across the accessible range of `k` is reported explicitly in Section V.B.

In absolute terms the cost is modest at the working dimension and rises steeply beyond it: a full sectional-curvature evaluation takes about 2.6 s per state at `k` = 5 and about 46 s at `k` = 8 on a single CPU core, which is why the sweep is reported at `k` = 4 to 6. The metric itself, being closed-form, is negligible by comparison; the expense is entirely in the second derivatives. This matters for the recommendation in Section VI, since it is precisely the expensive quantity that turns out to carry the signal.

Measurements are taken on GPT-2 [18], Pythia-70M and Pythia-160M [19], LLaMA-160M and SmolLM2-135M, spanning both normalisation schemes, tied and untied embeddings, depths from 12 to 30 layers and widths from 512 to 768. Text is drawn from WikiText-2 [20] and from hand-written probe sentences; the sense-disambiguation probe of Section V.F follows the design of the word-in-context task [21].

---

## III. Validation

Intrinsic curvature is expensive, silent when wrong, and plausible either way. Validation therefore precedes results and is reported in full.

### A. The ladder

Seven manifolds with analytically known curvature are run through the same code path used for the transformer. Table II reports the outcome.

**TABLE II. VALIDATION LADDER**

| rung | manifold | expected | measured | worst error |
|---|---|---|---|---|
| 2 | Poincaré half-plane | K = −1 | −1.000000 | 1.3 × 10⁻¹⁴ |
| 3 | Categorical simplex | K = +1/4 | +0.250000 | 2.6 × 10⁻¹⁴ |
| 4 | Gamma family | negative | −0.463903 | 2.5 × 10⁻¹⁴ |
| 5 | Beta family | negative | −0.456795 | 3.2 × 10⁻¹⁴ |
| 6 | Univariate Gaussian | K = −1/2 | −0.500000 | 1.1 × 10⁻¹⁶ |
| 7 | Synthetic pullback | K = +1/4 | +0.250000 | 9.7 × 10⁻¹² |

Rung 3 is load-bearing: the categorical simplex under the full Fisher–Rao metric is isometric to the positive orthant of a radius-2 sphere and therefore has constant `K = +1/4`. It is the same metric family as the object under study.

Rung 7 validates the assembly of (1) specifically, which no other rung does. With a synthetic linear model `p(h) = softmax(U h)` and `d = N − 1`, the map `h → p` is a diffeomorphism onto the simplex interior, so the pullback metric is the simplex metric in different coordinates and the curvature must be exactly +1/4 for every valid `U`. Three further checks accompany it: reparameterisation invariance under `U → UM`, to which curvature must be blind (worst error 1.9 × 10⁻¹¹); scalar curvature `R = k(k−1)/4` recovered through the pullback (4.9 × 10⁻¹²); and a deliberately broken assembly, with the outer-product term of (1) dropped, which must fail and does, at 0.263. A test that cannot fail carries no information.

### B. The Hessian identity and an independent implementation

Equation (2) was verified on a real model through a separate code path. Sweeping `ε` produces the expected plateau at relative error 8.7 × 10⁻⁶, confirming that the assembled `G(h)` is the Hessian of the model's own divergence rather than an unrelated quadratic form.

Rungs 2 to 6 were additionally re-run against the `information_geometry` module of an independently implemented, peer-reviewed library [22] on identical inputs. Three families agree at machine precision across eight comparisons: univariate normal to 1.1 × 10⁻¹⁶, gamma to 2.5 × 10⁻¹⁴ and beta to 3.2 × 10⁻¹⁴, the last two being exactly the rungs added for this cross-check. On the two families where the libraries differ, the diagnostic applied was to compare the *metrics* rather than the curvatures: in both cases the metrics agree exactly, to 1 × 10⁻¹² and 7 × 10⁻¹⁵, so the difference is localised to a curvature routine rather than a parameterisation convention. The +1/4 reported here is independently corroborated three ways — by the analytic radius-2 sphere isometry, by rung 3, and by rung 7's separate code path — and the remaining discrepancy is recorded as open.

---

## IV. The Control

The central result rests on one control, so its construction is given precisely.

Let `idx` be the indices of the vocabulary rows retained at a given state. The **paired within-set scramble** replaces `idx` by `idx[randperm(len(idx))]`: the same rows are used, permuted in their assignment to probabilities. This holds fixed, exactly:

- the predictive distribution `p`, and therefore the entropy, to machine precision;
- the multiset of retained unembedding rows, and therefore their norms and their spectrum;
- the conditioning of the retained subspace;
- the working dimension `k`.

What it destroys is only *which probability sits on which direction* — the learned assignment. Because the comparison is paired at the same state, and every confound is held fixed by construction rather than by matching, any surviving difference is attributable to that assignment alone.

---

## V. Results

### A. Curvature sits at the simplex value across architectures

Fig. 1 shows sectional curvature at 456 sampled states across four architectures.

**[FIGURE 1: fig1_curvature.png]**
*Fig. 1. Sectional curvature at 456 states across four architectures. The dashed line marks the ambient simplex value +1/4. Boxes are quartiles; bars are medians.*

Median values are +0.2599 for GPT-2, +0.2738 for Pythia-160M, +0.2559 for LLaMA-160M and +0.2548 for SmolLM2-135M — a spread of 0.019 across models differing in normalisation scheme, embedding tying, depth and width. The representation manifold under this metric is strongly positively curved, at the ambient simplex value, and consistently so across architecture. This is a direct measurement of the quantity, and it is a large number: the manifold is about as curved as the simplex it predicts into.

The deviation from exactly +1/4 is not advanced as a model-specific signal. It varies with `k` and its layer ranking does not survive resampling, so the informative quantity is not the absolute magnitude but the response to the control that follows.

### B. The separation

Fig. 2 presents the central result: one control, three quantities, two outcomes.

**[FIGURE 2: fig2_split.png]**
*Fig. 2. One paired control applied to three quantities. Sectional curvature collapses; the log volume element and the effective dimension are reproduced almost exactly.*

**TABLE III. ONE CONTROL, THREE QUANTITIES**

| quantity | real | scrambled | retained |
|---|---|---|---|
| sectional curvature `K` (n = 221) | 0.2546 | **0.0109** | **4 %** |
| log volume element (n = 360) | rho = 1 by definition | **rho = +0.957** | 96 % |
| effective dimension (n = 360) | rho = 1 by definition | **rho = +0.991** | 99 % |

The curvature collapse runs in the same direction at 201 of 221 states, z = +12.18, and it is stable across the accessible range of `k`: at `k` = 4 the median falls from 0.2543 to 0.0137 (63 of 69, z = +6.86), and at `k` = 6 from 0.2532 to 0.0512 (58 of 69, z = +5.66). This is the one Riemann-derived quantity in this work that is insensitive to the choice of `k`, which is precisely what makes it usable.

The two spectral quantities behave in the opposite way. The scrambled log-volume profile places its minimum at the same layer as the real one, and the scrambled effective-dimension profile does likewise, retaining 99.5 % of the real profile's range. Table IV gives that profile in full.

**TABLE IV. EFFECTIVE DIMENSION BY LAYER, REAL AND SCRAMBLED**

| layer | 1 | 5 | 10 | 15 | 20 | 25 | 28 | 29 | 30 |
|---|---|---|---|---|---|---|---|---|---|
| real | 14 | 8 | **6** | 7 | 6 | 22 | 77 | 167 | 217 |
| scrambled | 13 | 8 | **5** | 6 | 6 | 24 | 54 | 150 | 215 |

The characteristic dip near one third of the way through the network — the "hourglass" — is present in both rows, at the same layer, with the same range. A control containing no learned structure at all reproduces it.

The interpretation is direct. Quantities computed from the *spectrum* of `G(h)` — its determinant, its eigenvalue decay — are recovered by a structure-free control at matched entropy, so they largely restate the predictive concentration profile. Curvature requires the *second derivatives* of the metric, and it is the quantity that responds to the assignment.

### C. Where in the network the assignment does its work

The collapse is not uniform across depth. Table V gives the fraction of the real curvature that survives the control at each layer, with the paired sign count.

**TABLE V. CURVATURE RETAINED UNDER THE CONTROL, BY LAYER**

| layer | 1 | 5 | 10 | 15 | 20 | 25 | 28 | 29 | 30 |
|---|---|---|---|---|---|---|---|---|---|
| retained | 2.3 % | 2.3 % | 1.5 % | 1.7 % | −5.3 % | −95.5 % | 7.0 % | 25.8 % | 61.4 % |
| sign | 19/21 | 17/17 | 22/25 | 23/24 | 28/29 | 20/20 | 20/23 | 30/35 | 22/27 |

The learned assignment does nearly all of the work through the middle of the network, where under 3 % of the curvature survives its destruction, and least at the two ends. That pattern matches what the two ends are doing: at layer 1 the context has barely been integrated, so there is less learned assignment to destroy, while by layer 30 the prediction is close to committed and concentration alone fixes much of the geometry. The direction of the effect is unanimous or near-unanimous at every layer, so the depth dependence is one of magnitude rather than of sign.

### D. The assignment stabilises curvature, it does not merely add it

Resolving the same 221 states by predictive entropy sharpens the finding considerably. Table VI reports both conditions across five entropy bands.

**TABLE VI. CURVATURE BY PREDICTIVE ENTROPY**

| entropy (nats) | n | real | scrambled | sign |
|---|---|---|---|---|
| [0, 0.5) | 38 | **0.2750** | −1.2756 | 35/38 |
| [0.5, 1) | 37 | **0.2587** | −0.1165 | 36/37 |
| [1, 2) | 68 | **0.2555** | +0.0110 | 66/68 |
| [2, 4) | 36 | **0.2528** | +0.1310 | 31/36 |
| [4, ∞) | 42 | **0.2064** | +0.1238 | 33/42 |

The real curvature is close to constant across the full range: from a near-deterministic prediction to a highly diffuse one, the median moves only from 0.275 to 0.206, staying at the simplex value throughout. The scrambled curvature does nothing of the kind — it swings across roughly 1.4 units, from strongly negative where the prediction is concentrated to mildly positive where it is diffuse.

This is the most informative single comparison in the study. It shows that the learned token-to-direction assignment does not simply contribute curvature on top of whatever concentration provides; it *holds the manifold at the ambient simplex value irrespective of how concentrated the prediction is*. Concentration alone produces a geometry that varies wildly with the entropy of the prediction. The trained assignment removes that dependence. The effect is also strongest exactly where it should be if this reading is correct — in the low-entropy bands, where 35 of 38 and 36 of 37 states move in the same direction — because a concentrated `p` gives the assignment the most to encode.

### E. The instruments measure different things

If the intrinsic curvature and the published proxies were reading the same underlying property, they would correlate. Computed at the same 360 states, they do not: the largest absolute rank correlation between any two different instrument families is 0.248, and the pairs involving intrinsic curvature fall at or below 0.144. By contrast the two intrinsic quantities computed here, sectional `K` and scalar `R`, correlate with each other at +0.69 to +0.74. That is what internal consistency looks like, and the cross-family pairs do not exhibit it.

Corpus stability discriminates further. Moving from hand-written probe sentences to WikiText-2 across 653 states, the intrinsic instrument's correlation with predictive entropy moves by 0.028 for sectional curvature and 0.034 for scalar curvature, while the three proxies move by 0.104, 0.126 and 0.194 respectively, the last of them changing sign. The intrinsic instrument returns essentially the same reading on both corpora; the proxies do not. An instrument whose reading depends this strongly on the text it is given is measuring a property of the text.

### F. A behavioural probe

If the metric tracks the model's predictive state, distance under it should respond to a change in word sense. On 32 polysemous words, each compared against a same-sense control built to an identical syntactic frame, the two arms are matched for lexical overlap — 0.250 for the sense-change arm against 0.235 for the same-sense arm — so the sense arm is not simply the more lexically distant one. Table VII reports three metrics on the same pairs.

**TABLE VII. SENSE CHANGE VERSUS MATCHED SAME-SENSE CONTROL**

| metric | same sense | sense change | ratio | sign | z |
|---|---|---|---|---|---|
| **Fisher–Rao geodesic** | 0.754 | 1.532 | **2.03×** | 30/32 | **+4.95** |
| Euclidean | 260.0 | 552.5 | 2.13× | 32/32 | +5.66 |
| Unembedding `UᵀU` | 10459 | 17838 | 1.71× | 23/32 | +2.47 |

The Fisher–Rao geodesic distance separates a sense change from a matched context change by a factor of 2.03 at 30 of 32 pairs, and does so considerably more reliably than the unembedding metric used in the comparison literature, which reaches 1.71× at 23 of 32. On this particular probe a plain Euclidean distance performs comparably to Fisher–Rao. The probe therefore establishes that the metric is behaviourally meaningful, and the experiment that would separate it from Euclidean distance — perturbation along high-curvature directions at the disambiguation layer, testing whether the resolved sense flips — is identified as the natural next step.

### G. A near-zero reading is not evidence of flatness

The published value of order 10⁻⁵ reproduces exactly, and Fig. 3(a) shows it: with a tight retained-variance threshold the local-PCA residual on GPT-2 falls below 10⁻⁵ and continues falling until it underflows. What that number licenses is a separate question.

**[FIGURE 3: fig3_threshold.png]**
*Fig. 3. (a) The local-PCA residual proxy on GPT-2 layer 12 as a function of its retained-variance threshold, for several neighbourhood sizes. (b) The same estimator applied to manifolds of known curvature. Bars at the axis floor are numerically zero.*

Fig. 3(b) applies the same estimator, at the same thresholds, to a unit 3-sphere embedded in 12 dimensions — a manifold whose sectional curvature is exactly +1 everywhere — and to a flat 3-plane, whose curvature is exactly 0. At loose thresholds the estimator separates them cleanly. At the threshold that produces the published value it does not: the sphere's residual falls to order 10⁻³¹, numerically indistinguishable from the plane's exact zero.

An estimator that reports a unit sphere as flat to 10⁻³¹ cannot support the inference from "the residual is 10⁻⁵" to "the manifold is flat"; the residual is a function of the threshold at least as much as of the geometry. Measured intrinsically, the same states give `K ≈ +1/4`. The original observation replicates exactly; the reading that follows from it is that the representation manifold is strongly curved, and that the apparent flatness is a property of the estimator's operating point.

---

## VI. Discussion

Three findings stand, in decreasing order of how firmly they are established.

**The representation manifold is strongly positively curved, at the ambient simplex value, across architectures.** Four models agree to within 0.019, spanning both normalisation schemes and both embedding regimes. The value is not small, and Section V.G accounts for the earlier near-flat reading without contradicting its measurement.

**Only curvature reads the model; the spectrum reads concentration.** One control, three quantities: sectional curvature collapses to 4 % of its value while log-volume and effective dimension are reproduced at rho = +0.957 and +0.991. This is the most useful result here because it is actionable — it says which quantities are worth computing, and it says so on the basis of a control rather than an argument. It also places a testable question against the intrinsic-dimension literature: the characteristic depth profile is reproduced under this metric by a control with no learned structure, so applying a matched-entropy control to an ambient nearest-neighbour estimate would establish whether the published profile has the same character.

**The metric is the model's own, demonstrably.** The directions the metric calls null are directions the model is exactly invariant to, on six architectures, at a step size that doubles the norm of the state. Curvature is not being measured on an arbitrary quadratic form fitted to activations; it is measured on the model's predictive geometry, and Equation (2) confirms this through an independent code path.

**Future work follows directly from the separation.** Three experiments are indicated. The first is the causal one: perturbing a state along its high-curvature directions at the layer where a sense is resolved, and testing whether the resolved sense flips. That would connect the geometric measurement to behaviour in the strongest available way, and Section V.F identifies it as the experiment this probe does not yet perform. The second is to apply the matched-entropy control of Section IV to an ambient nearest-neighbour intrinsic-dimension estimate, which would settle whether the published depth profile shares the character established here for the metric's spectrum. The third is scale: the models used here are 70M to 160M parameters, and repeating the measurement at 1B and above would test whether the simplex value and the separation both persist. A fourth and cheaper line is suggested by the minority of states carrying a negatively curved plane — 10.7 % of the sample — which have twice the effective dimension of the rest at indistinguishable predictive entropy. That is the one place where absolute curvature, rather than its response to a control, appears to carry model-specific information.

One methodological point is worth recording, since it shaped what is reported. Several intermediate claims in this work were revised as sample sizes grew and as controls were tightened, and each was caught by a check constructed for that purpose. The rule adopted here is that a result at one sample size, one `k`, or one control is a hypothesis, and becomes a finding when it survives the second. The separation in Section V.B survives all three, and the numbers above are those that did.

---

## VII. Limitations

**Scale.** Models range from 70M to 160M parameters. Nothing here rules out the geometry changing at 1B and above.

**The `k` ≤ 8 ceiling.** Riemann-derived quantities are computed on a slice of dimension 4 to 6 holding about 88 % of the metric's trace. The central result is stable across that range, but the slice remains a slice.

**Cell sizes.** Pooled samples run from 221 to 653 states; per-layer cells are smaller, and the pooled figures are the ones to rely on.

**Corpora.** Two, both English edited prose.

**One validation arm open.** The independent library agrees on three Fisher–Rao families and differs on two, with the metrics themselves agreeing exactly in both cases, as reported in Section III.B.

---

## VIII. Conclusion

The intrinsic Riemannian curvature of the Fisher–Rao pullback metric on transformer hidden states is computable at practical cost, validates against seven manifolds of known curvature, and sits at the ambient simplex value of +1/4 across four architectures. The representation manifold is not flat; it is about as curved as the simplex it predicts into.

The result expected to be most useful is the separation. A single control — destroying the learned token-to-direction assignment while holding entropy, conditioning and the direction set exactly fixed — collapses sectional curvature from 0.2546 to 0.0109 while reproducing the layer-wise log-volume profile at rho = +0.957 and the effective-dimension profile at rho = +0.991. Stability and informativeness proved to be different axes, pointing in opposite directions: the well-conditioned, `k`-robust, cheaply computed spectral quantities largely restate the predictive entropy profile, while the more delicate curvature is the quantity carrying learned structure. For anyone choosing a geometric probe of a language model, that is a concrete instruction about where the signal is.

---

## Reproduction

All code, saved result files and figure-generation scripts are available at `https://github.com/ChaminiMaduwanthi/Fisher-Rao`. Every figure is regenerated from the saved result files by `make_figures.py`; no number in a figure is entered by hand. The validation ladder is implemented in `validate_curvature.py`, `validate_pullback.py` and `validate_geomstats.py`, each of which prints a pass or fail verdict.

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

[11] E. Facco, M. d'Errico, A. Rodriguez, and A. Laio, "Estimating the intrinsic dimension of datasets by a minimal neighborhood information," *Sci. Rep.*, vol. 7, no. 1, p. 12140, 2017.

[12] A. Modell, P. Rubin-Delanchy, and N. Whiteley, "The origins of representation manifolds in large language models," *arXiv:2505.18235*, 2025.

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
