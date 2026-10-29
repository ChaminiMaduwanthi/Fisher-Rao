# Information-Geometric Curvature Analysis of Transformer Representations Using Fisher–Rao Metrics and Genuine-Pullback Controls

**Anonymous Authors**

*Affiliation withheld for double-blind review*

## Abstract

This article computes the intrinsic curvature of the Fisher–Rao metric on the hidden states of transformer language models and tests whether that curvature reflects learned structure. Geometric summaries of hidden states increasingly serve as interpretability evidence, yet the Riemann tensor of this singular metric has not been formed for a transformer, and Fisher–Rao quantities have not been tested against controls that are themselves valid statistical models. The proposed pipeline removes the exact null directions of the normalisation layer, computes sectional curvature on low-dimensional slices by automatic differentiation, and compares each trained model with two genuine-pullback controls: a consistent scramble of the readout rows and an entropy-matched random readout. The pipeline reproduces seven manifolds of known curvature and a Kullback–Leibler Hessian identity. Across four models and 425 hidden states, median curvature lies between 0.255 and 0.272, close to the simplex value of one quarter. Both controls give 0.230 to 0.285, so the value is inherited from the probability simplex rather than learned, although the trained models sit 0.002 to 0.024 below the consistent scramble on every model. A control that is not a genuine pullback instead manufactures a spurious collapse to near zero. The spectral summaries of the metric survive the same scramble, and effective dimension tracks predictive entropy. These findings identify which Fisher–Rao quantities can support interpretability claims.

**Index Terms**— explainable artificial intelligence, information geometry, language models, representation geometry, Riemannian curvature

## I. Introduction

Interpretability research increasingly describes what language models represent through the geometry of their hidden states: the intrinsic dimension of representations across depth [1], [2], the curvature of layer-to-layer trajectories [3]–[5], curved manifolds of features and personas [6], [7], and the Fisher information of the model's prediction [8]–[11]. These quantities are already used as evidence: trajectory curvature has been linked to behavioural uncertainty [4], and Fisher geometry has been used to steer activations [9] and to price interventions [10]. A geometric quantity is informative about a model only if it responds to what the model has learned. The same requirement underlies sanity checks for saliency maps [12] and control tasks for probing classifiers [13], but it has rarely been applied to representation geometry.

Among candidate geometries, the Fisher–Rao metric is distinguished: up to scale, it is the only Riemannian metric on probability distributions that is invariant under sufficient statistics [14], [15]. Pulled back through the model's readout, it measures how a small change of a hidden state changes the next-token distribution [8], [9]. Existing work uses this pullback metric through its spectrum, volume or trace [9]–[11]. Curvature, the quantity that separates an intrinsically curved space from a flat space written in curved coordinates, has been measured only under Euclidean or constant metrics [3]–[5], or replaced by a local principal-component residual whose value of about 10⁻⁵ was read as near-flatness [8].

Two gaps follow. First, the intrinsic Riemannian curvature of the Fisher–Rao pullback metric has not been computed for a transformer: the metric is singular along the directions removed by the normalisation layer, and curvature requires second derivatives of a state-dependent metric, a step described as computationally intractable [8]. Second, Fisher–Rao quantities have not been tested against controls that are themselves genuine pullbacks, that is, controls in which the predictive distribution and the metric derive from one and the same map. Without such controls, a change in a geometric quantity cannot be attributed to the removal of learned structure rather than to an invalid construction.

This article addresses both gaps. It computes the sectional curvature of the Fisher–Rao pullback metric after quotienting the exact null directions of the normalisation layer, validates the computation against manifolds of known curvature, and compares four trained models with two genuine-pullback controls: a consistent scramble of the readout rows and an entropy-matched random readout. Three research questions are addressed. RQ1: Can the intrinsic curvature of the Fisher–Rao pullback metric be computed reliably on transformer hidden states? RQ2: What value does this curvature take across architectures, and is that value produced by learning? RQ3: What do the spectral summaries of the same metric, log-volume and effective dimension, measure?

This work makes three contributions: (1) a validated pipeline for the intrinsic curvature of the Fisher–Rao pullback metric on transformer hidden states (RQ1); (2) evidence on four architectures that the measured curvature, close to one quarter, is inherited from the probability simplex rather than produced by learning, together with a demonstration that a control which is not a pullback manufactures a spurious collapse (RQ2); and (3) evidence that the metric's spectral summaries depend on the predictive distribution rather than on learned structure, with a comparison against three published curvature instruments (RQ3). The results address researchers who use geometric quantities as interpretability evidence. Code, per-state results and a script for every table are available in an anonymised repository [16].

## II. Related Work and Research Gap

*Fisher–Rao pullback metric.* The pullback of the Fisher–Rao metric onto transformer hidden states was derived by Mabrok [8], who defined its Riemann tensor but estimated curvature with local principal-component (PCA) and second-fundamental-form proxies, reporting values near 10⁻⁵. FishBack [9] uses the same metric to select steering directions in GPT-2 and analyses its spectrum without forming Christoffel symbols. Picozzi [10] studies the spectrum and effective dimension of the output Fisher–Rao geometry for models of up to 7B parameters and finds the spectrum predictable from token probabilities; the full text contains no Christoffel symbols, Riemann tensor or sectional curvature. Mudarisov et al. [11] identify the pullback operator with the Hessian of the loss-to-go in models of 1B to 9B parameters, and their curvature score is a trace of that operator. Zaher et al. [17] relate the rank of the metric to sparse-autoencoder scaling.

*Curvature of representations.* Curvature of transformer representations has so far been measured under flat geometry. Manson [3] computes the Frenet curvature of residual-stream trajectories under the constant metric UᵀU, whose intrinsic curvature is identically zero. King et al. [4] use the Euclidean angle between successive difference vectors and report a correlation of about 0.15 with next-token entropy. Pandey et al. [5] measure ambient trajectory curvature and find that its effects vanish under shuffled-layer and random-embedding controls. PersonaManifold [7] estimates Ollivier–Ricci curvature, a discrete notion, on persona activations. Intrinsic curvature of a pullback metric has been computed for convolutional and fully connected networks under a data-induced metric [18], following earlier Riemannian treatments of neural networks [19], but not for transformers and not for the Fisher–Rao metric.

*Representation manifolds and controls.* Intrinsic-dimension studies report a characteristic profile across depth [1], [2], estimated by nearest-neighbour methods [20], and representation manifolds have been given a theoretical account [6]. Controls that remove learned structure are standard for other interpretability methods [12], [13], and a randomly initialised encoder recovers much of the benefit of pullback-metric pooling [21]. Such controls have not been applied to Fisher–Rao curvature. Table I places the closest studies side by side.

**TABLE I. CLOSEST STUDIES AND THE CURVATURE EACH COMPUTES**

| Study | Metric on hidden states | Curvature quantity |
|---|---|---|
| Mabrok [8] | Fisher–Rao pullback | local-PCA proxy; tensor defined only |
| FishBack [9] | Fisher–Rao pullback | none (spectrum) |
| Picozzi [10] | Fisher–Rao pullback | none (spectrum, effective dimension) |
| Mudarisov et al. [11] | Fisher pullback operator | trace score |
| Manson [3] | constant UᵀU | Frenet curvature, flat space |
| King et al. [4] | Euclidean | angle between steps |
| Pandey et al. [5] | Euclidean | trajectory angle |
| Zavatone-Veth et al. [18] | data-induced, not transformers | Ricci scalar |
| This article | Fisher–Rao pullback, null directions quotiented | sectional and scalar, on slices |

The research gap is therefore twofold. No study computes the Riemann tensor of the Fisher–Rao pullback metric on transformer hidden states, and no study tests whether Fisher–Rao quantities survive controls that remove learned structure while remaining genuine pullbacks. The second gap matters as much as the first: Section V shows that a natural but inconsistent control manufactures a collapse of curvature that a consistent control does not reproduce.

## III. Research Methodology

### A. Research Design

The study is a controlled measurement study. Each hidden state is measured under the trained readout and, at the same base point, under each control, so that every comparison is paired within a state. Known-answer validation addresses RQ1, the paired controls address RQ2, and the same controls, applied to the spectral summaries together with three published curvature instruments as external comparators, address RQ3. Fig. 1 shows the workflow.

**[FIGURE 1: fig1_pipeline.png | full]**
*Fig. 1. Workflow of the study. The measurement chain (top) is repeated at the same base point with the readout replaced by each control (bottom left); validation (bottom centre) tests the chain against known answers. Equation numbers refer to Section IV.*

### B. Models and Data

Four pretrained models span both normalisation schemes and both embedding regimes: SmolLM2-135M-Instruct [22] (RMSNorm [23], tied embeddings, 30 layers, width 576, 49 152 tokens), LLaMA-160M [24] (RMSNorm, untied, 12 layers, width 768, 32 000 tokens), GPT-2 [25] (LayerNorm [26], tied, 12 layers, width 768, 50 257 tokens) and Pythia-160M [27] (LayerNorm, untied, 12 layers, width 768, 50 304 tokens). Pythia-70M and GPT-Neo-125M [28] are added for the null-direction test. The inputs are 148 hand-written probe sentences, released with the code, and sentences from the WikiText-103 validation split [29]. Hidden states are taken from the residual stream at nine relative depths (layers 1, 5, 10, 15, 20, 25, 28, 29 and 30 for SmolLM2) and at every token position.

Three samples are used. The profile sample holds 12 states per depth, 425 distinct states over the four models. The control sample holds 186 SmolLM2 states and, for each other model, 120 states drawn as 24 per band of predictive entropy: [0, 0.5), [0.5, 1), [1, 2), [2, 4) and [4, ∞) nats. The spectral sample holds 40 states per depth, 360 per model. LLaMA-160M's start-of-sequence state is exactly zero at every layer, so its prediction is uniform and the metric is undefined there; these 34 states are excluded. Rows duplicated by resumed runs are removed before any statistic is computed.

### C. Evaluation Protocol

*Comparators.* Three controls replace the readout while keeping the hidden state, the final normalisation and the retained-token set fixed (Section IV-D): C0, an inconsistent scramble used in a preliminary version of this analysis; C1, a consistent scramble; and C2, an entropy-matched Gaussian readout. For RQ3, the local-PCA residual [8], Frenet curvature under UᵀU [3] and the Euclidean angle [4] are computed on 360 SmolLM2 states as external comparators.

*Metrics.* The primary metric is slice sectional curvature K on slices of dimension k = 5 (Section IV-B), the median over 16 random planes per state; scalar curvature R is also reported. The spectral summaries are the effective dimension k<sub>eff</sub>, the number of leading eigenvalues of the quotiented metric that hold 99 % of its trace, and the log-volume per dimension of its top-k block, ℓ = (1/2k) Σᵢ log(r² λᵢ), where r, the root-mean-square (RMSNorm) or standard deviation (LayerNorm) of h, removes the metric's 1/r² scaling.

*Statistics.* Medians are reported. Trained and control values are compared by a paired sign test, z = (n₊ − n/2)/(√n/2), where the trained value is higher at n₊ of n states, and monotone agreement by Spearman's ρ with midranks. A decision rule was fixed after a 30-state pilot and before the reported runs: a trained model and a control agree when |z| < 2 and their medians differ by less than 0.01; agreement on at least three of the four models would indicate that learning leaves curvature unchanged, and otherwise the differences are reported.

## IV. Pipeline Design and Implementation

### A. The Pullback Metric

For a hidden state h of width d, the model predicts p(h) = softmax(U n(h)), where n is the final normalisation and U is the readout restricted to the 512 most probable tokens at the base point h₀, with p renormalised over them. With A(h) the Jacobian of n and J = U A(h), the Fisher–Rao metric of the simplex pulls back to

> G(h) = Jᵀ (diag p − p pᵀ) J. (1)

The metric is the Hessian of the Kullback–Leibler (KL) divergence at zero displacement,

> KL(p(h) ‖ p(h + εv)) = ½ ε² vᵀ G(h) v + O(ε³), (2)

which tests an implementation of (1) independently. The retained token set is frozen at the base point before any derivative is taken, because re-selecting it would make G only piecewise smooth; on SmolLM2, truncation to 500 tokens changes the metric by 3.2 × 10⁻³ (relative norm). G is a d × d matrix in closed form, distinct from the parameter-space Fisher matrix of second-order optimisation.

### B. Exact Null Directions and Slices

Both normalisation layers annihilate directions exactly. RMSNorm has one null direction, h itself, and LayerNorm two, the centred state and the all-ones vector, so G(h) is singular by construction. Instead of a pseudo-inverse, the null space is quotiented: with N an orthonormal null basis and P = I − NNᵀ, curvature is computed on the slice S = h₀ + span(F), where F holds the top-k eigenvectors of P G(h₀) P. The default is k = 5, with k = 4 and 6 as checks; at k = 5 the slice holds a median 95.6 % of the quotiented trace (60 states).

### C. Curvature on a Slice

On S, the Christoffel symbols and the Riemann tensor Rm of the induced metric g are formed by nested reverse-mode automatic differentiation of (1) in double precision [30]. The sectional curvature of the plane spanned by u and v is [31]

> K(u, v) = Rm(u, v, v, u) / (g(u, u) g(v, v) − g(u, v)²), (3)

and the value reported for a state is the median over 16 random planes. Every number is therefore a slice sectional curvature K<sub>S</sub>. For orthonormal u and v it differs from the curvature K<sub>M</sub> of the full quotient manifold through the Gauss equation [31],

> K<sub>S</sub> = K<sub>M</sub> + ⟨II(u, u), II(v, v)⟩ − ‖II(u, v)‖², (4)

where II is the second fundamental form of S in the manifold. The extrinsic term was measured where K<sub>M</sub> is known exactly (Table II), and each model and its controls are compared on slices of the same dimension and construction.

### D. Genuine-Pullback Controls

Let Π permute the retained rows of U. C0 takes p from the trained readout but the Jacobian from ΠU. At h₀ it changes only which probability sits on which direction, but away from h₀ its metric is the Hessian of no probability map, so the derivatives from which curvature is built belong to no model. C1 realises the same permutation as a model with logits

> z(h) = U n(h₀) + ΠU (n(h) − n(h₀)) (5)

and p(h) = softmax(z(h)). It coincides with C0 at h₀ (same p, metric, slice and row multiset) and is a genuine pullback everywhere; with Π the identity it reproduces the trained metric to 2.4 × 10⁻¹⁴ on GPT-2 and 1.2 × 10⁻¹³ on Pythia-160M. The permutation does not hold the conditioning of the slice fixed state by state, but it leaves its distribution unchanged (median ratio of first to fifth eigenvalue 15.6 against 15.1, z = −0.4, on 30 SmolLM2 states). C2 removes the trained readout altogether: p(h) = softmax(sW n(h)), with W one fixed Gaussian matrix with entries of variance 1/d and s chosen per state so that the predictive entropy equals the trained model's, to within 10⁻¹³ nats. C2 keeps the trained hidden states and final normalisation, including LayerNorm's learned bias.

### E. Validation and Implementation

Table II lists the validation. Seven manifolds of known curvature are run through the code path used for the transformers, among them the categorical simplex, which under the Fisher–Rao metric is part of a sphere of radius 2 with K = 1/4 [15], and a synthetic pullback softmax(Uh) with d equal to the number of outcomes minus one, whose curvature must be exactly 1/4 and must not change when U is replaced by UM for an invertible M. A deliberately broken assembly of (1) must fail. Identity (2) is evaluated on SmolLM2 at the base point and, for each control, 5 % of ‖h₀‖ away from it, which separates genuine pullbacks from C0. An independent implementation, geomstats [32], provides a second opinion on the parametric families.

The implementation uses Python 3.13, PyTorch 2.12 in float64 on a four-core laptop CPU without a GPU, Transformers [33] with pinned model revisions, and geomstats 2.8. A state costs about 3 s at k = 5. Seeds are fixed: 0 for planes and for W, and 1000 plus the state index for Π. One correction was made during the study: the projected metric had used the RMSNorm Jacobian for LayerNorm models; the exact form changes K by at most 1.1 × 10⁻³ (median 6 × 10⁻⁵), and every control run uses it.

**TABLE II. VALIDATION OF THE PIPELINE: WORST-CASE ERROR UNLESS MARKED AS A MEDIAN**

| Check | Expected | Observed |
|---|---|---|
| Sphere of radius 2 | K = 1/4 | 1.8 × 10⁻¹⁵ |
| Poincaré half-plane | K = −1 | 1.0 × 10⁻¹³ |
| Categorical simplex | K = 1/4 | 6.6 × 10⁻¹⁰ |
| Univariate Gaussian | K = −1/2 | 2.6 × 10⁻¹⁵ |
| Gamma, beta families | geomstats [32] | 2.3 × 10⁻¹³ |
| Synthetic pullback | K = 1/4 | 9.7 × 10⁻¹² |
| Reparameterised, U → UM | invariant | 1.9 × 10⁻¹¹ |
| Outer product of (1) dropped | must fail | fails, 0.263 |
| Identity (2), SmolLM2 | ratio 1 | 4.3 × 10⁻⁵ |
| (2) off h₀: trained, C1 (median) | ratio 1 | 8.5 × 10⁻⁴, 1.1 × 10⁻³ |
| (2) off h₀: C0 (median) | ratio 1 | 0.66–0.93, fails |
| Null directions, six models | KL = 0 | ≤ 2.0 × 10⁻¹³ nats |
| Eigen-frame slice, 0.5–4 nats (median) | K<sub>S</sub> = K<sub>M</sub> = 1/4 | 0.005–0.028 |

## V. Results and Discussion

### A. Validation (RQ1)

Table II answers RQ1. The pipeline reproduces every analytic curvature to 6.6 × 10⁻¹⁰ or better, including the categorical simplex, which carries the metric under study, and the synthetic pullback, which tests the assembly of (1). It is blind to reparameterisation and fails, as it must, when the outer-product term of (1) is dropped. On SmolLM2, (2) holds to 4.3 × 10⁻⁵. On all six models the number of null directions matches the normalisation type, and a step of length ‖h‖ along them changes the prediction by at most 2.0 × 10⁻¹³ nats, against 2.96 to 32.3 nats for a random step of the same length; the quotiented metric is therefore the model's own. The independent library agrees on three families; on the Poincaré half-plane and the categorical simplex it reports different curvatures from identical metrics, whereas this pipeline reproduces the analytic values; the difference therefore arises in the curvature computation rather than in the metric, and its cause is not resolved. Eigen-frame slices deviate from the known manifold curvature by 0.005 to 0.028 between 0.5 and 4 nats, against 0.04 to 0.16 for random slices, and by 0.20 and 0.12 below 0.5 and above 4 nats.

### B. Curvature Across Architectures and Controls (RQ2)

On the profile sample, median slice curvature is 0.255 for SmolLM2 (quartiles 0.251–0.261), 0.256 for LLaMA-160M (0.240–0.286), 0.260 for GPT-2 (0.240–0.294) and 0.272 for Pythia-160M (0.261–0.308): a spread of 0.017 around the simplex value of 1/4 across both normalisation schemes and both embedding regimes. Table III and Fig. 2 compare the trained models with the controls.

**TABLE III. SLICE SECTIONAL CURVATURE ON FIVE-DIMENSIONAL SLICES: MEDIANS, WITH THE SIGN-TEST STATISTIC FOR TRAINED ABOVE CONTROL IN PARENTHESES**

| | SmolLM2 | LLaMA | GPT-2 | Pythia |
|---|---|---|---|---|
| States | 186 | 118 | 120 | 120 |
| Trained | 0.2547 | 0.2565 | 0.2580 | 0.2696 |
| C0, inconsistent | 0.0096 (+11.0) | −0.0056 (+10.5) | 0.0095 (+9.5) | 0.0218 (+10.4) |
| C1, consistent | 0.2565 (−3.1) | 0.2794 (−4.8) | 0.2823 (−5.3) | 0.2827 (−4.0) |
| C2, Gaussian | 0.2730 (−7.8) | 0.2846 (−4.4) | 0.2303 (+7.3) | 0.2816 (−3.8) |

**[FIGURE 2: fig2_controls.png]**
*Fig. 2. Median slice sectional curvature by predictive-entropy band for the trained models and the three controls. Dashed line: K = 1/4. C0 values below the axis are marked by triangles on its lower edge.*

The inconsistent control C0 collapses curvature on all four models (z from +9.5 to +11.0), which reproduces the preliminary result. The two genuine pullbacks do not: C1 and C2 give medians from 0.230 to 0.285, within 0.035 of 1/4, although neither contains the learned pairing between readout rows and probabilities and C2 contains no trained readout at all. The collapse is therefore an artefact of a control whose derivatives belong to no probability map (Table II), and a value near 1/4 does not require learning. It is inherited from the simplex and the slice construction: with no learning at all, the eigen-frame slices of the synthetic pullback give 0.25 to 0.28 between 0.5 and 4 nats.

Learning nevertheless leaves a trace. By the decision rule of Section III-C, the trained models agree with neither control on any model. They lie below C1 on all four models, by 0.002 to 0.024 at the median (z from −3.1 to −5.3); the direction is the same at k = 4 and k = 6 in all eight cases (|z| ≥ 2 in six), and in every entropy band fewer than half of the states have the trained value higher. Against C2 the direction depends on the model: GPT-2's Gaussian readout gives the lower curvature (0.230 against 0.258), the other three the higher. Removing LayerNorm's bias from C2 leaves the GPT-2 result unchanged (0.114 against 0.117 above 4 nats, where the trained model gives 0.209), and its cause remains open. A further hypothesis from the preliminary analysis, that learning moves curvature towards 1/4, is not supported: the trained value is significantly closer to 1/4 in only four of the eight model–control comparisons.

Fig. 2 locates the differences. Between 0.5 and 4 nats the trained medians stay within 0.02 of 1/4 and the genuine controls within 0.06; the larger departures occur below 0.5 and above 4 nats, the bands with the largest extrinsic term in Table II. Because the trained and control slices are chosen from different metrics, their extrinsic terms in (4) differ, and the offset, 0.002 to 0.024, is of the same order as that term. It is therefore a property of slice curvature; whether it reflects the curvature of the full manifold cannot be settled with slices of dimension k ≤ 6.

### C. Spectral Summaries (RQ3)

Both spectral summaries are evaluated at h₀, where C0 coincides with C1, so for them C0 is a valid control. Table IV and Fig. 3 show that the control reproduces both summaries on every model, with the same lowest layer for both. Effective dimension is reproduced closely, with rank correlations across states of 0.97 to 0.99 and 94 % to 99 % of the range of the depth profile; log-volume is reproduced less tightly (0.85 to 0.96). Both summaries therefore depend on the predictive distribution and the multiset of readout rows rather than on their learned pairing. Effective dimension tracks predictive entropy on every model (ρ = 0.71 to 0.90), in line with the finding that the Fisher spectrum is predictable from token probabilities [10]; log-volume does so clearly on SmolLM2 and Pythia-160M and weakly on LLaMA-160M and GPT-2. The depth profile itself is model-specific: SmolLM2's effective dimension is lowest at layer 10 of 30, near the one-third depth reported for intrinsic dimension [2], whereas the 12-layer models are lowest at layer 9 or 11. A control without the learned pairing reproduces each profile, which suggests applying such controls before reading intrinsic-dimension profiles as learned structure.

**TABLE IV. SPECTRAL SUMMARIES UNDER THE CONTROL: RANK CORRELATION WITH THE TRAINED VALUE, AND WITH PREDICTIVE ENTROPY**

| Model (states) | ρ, log-volume | ρ, k<sub>eff</sub> | Lowest layer | ρ with entropy, k<sub>eff</sub> / log-volume |
|---|---|---|---|---|
| SmolLM2 (360) | 0.957 | 0.991 | 10 / 10 | 0.71 / 0.86 |
| LLaMA (335) | 0.847 | 0.988 | 11 / 11 | 0.90 / 0.38 |
| GPT-2 (360) | 0.879 | 0.972 | 9 / 9 | 0.79 / 0.45 |
| Pythia (360) | 0.958 | 0.994 | 11 / 11 | 0.78 / 0.69 |

**[FIGURE 3: fig3_spectral.png]**
*Fig. 3. Median effective dimension by layer for the trained metric and the scrambled readout, which coincides with C1 at the base point; ρ is the rank correlation across states.*

### D. Comparison With Published Instruments

Table V compares intrinsic curvature with the three published instruments on 360 SmolLM2 states, and their stability on 293 hand-written and 360 WikiText-103 states. The two intrinsic quantities agree with each other (ρ = 0.72) but not with the instruments: |ρ| ≤ 0.144 between K and each instrument pooled over layers, and ≤ 0.112 within layers, so the instruments measure different properties. The intrinsic quantities are also the most stable across corpora: their correlation with predictive entropy shifts by 0.028 and 0.034, against 0.104 to 0.194 for the instruments, one of which changes sign. Finally, the published near-flat value is reproduced and calibrated. On GPT-2's last layer the local-PCA residual reaches 6.2 × 10⁻⁵ at a retained-variance threshold of 0.9999 with 200 neighbours and underflows to about 10⁻³² with fewer, while at a threshold of 0.999 the same estimator reports a unit 3-sphere, whose curvature is +1, as 1.5 × 10⁻³¹. A residual near zero is a property of the estimator's operating point, not evidence of flatness.

**TABLE V. INTRINSIC CURVATURE AGAINST PUBLISHED INSTRUMENTS (SMOLLM2)**

| Instrument | ρ with K, pooled / within layer | ρ with entropy, hand-written → WikiText |
|---|---|---|
| Intrinsic K (this article) | — | −0.447 → −0.475 |
| Intrinsic R (this article) | +0.720 / +0.689 | −0.579 → −0.544 |
| Local-PCA residual [8] | −0.144 / −0.001 | +0.175 → +0.050 |
| Frenet curvature, UᵀU [3] | +0.133 / −0.003 | −0.113 → −0.217 |
| Euclidean angle [4] | +0.137 / +0.112 | +0.043 → −0.151 |

### E. Implications for Interpretability

Three practical rules follow for studies that use Fisher–Rao geometry as evidence. First, a curvature near 1/4 is the expected value of a softmax pullback and is not by itself evidence of learned structure; claims should rest on differences from genuine-pullback controls computed on slices of identical construction. Second, every control should be checked with identity (2) away from the base point, since a control that fails it can manufacture a large effect, as C0 does. Third, spectral summaries of the metric should be read as summaries of predictive concentration unless a control shows otherwise.

### F. Threats to Validity

*Construct.* Every reported curvature is a slice curvature (4); the extrinsic term was sized only on a synthetic pullback, and the learned offset is of the same order. *Internal.* C2 keeps the trained hidden states and normalisation, and a randomly initialised network was not tested; C0 shows that a control must be checked with (2) away from the base point; the token truncation changes the metric by about 3 × 10⁻³; states from one sentence and adjacent layers are not independent, so z values are approximate, and only the fixed decision rule guards against multiple comparisons. *External.* The models have at most 160M parameters and the text is English; spectral studies now reach 7B parameters [10], and the curvature results have not been tested at that scale. *Reliability.* The independent library disagrees on two families, and the zero start-of-sequence states of LLaMA-160M are excluded. All tables are regenerated from saved per-state results by one script that checks 69 stored summaries.

## VI. Conclusion and Future Work

This article asked what the intrinsic curvature of the Fisher–Rao metric on transformer hidden states measures. A pipeline that quotients the normalisation layer's null directions and differentiates the metric twice on low-dimensional slices was validated against seven manifolds of known curvature and a Kullback–Leibler identity, which answers RQ1. On four models, slice curvature sits near the simplex value of 1/4 (medians 0.255 to 0.272), and two genuine-pullback controls without learned structure give 0.230 to 0.285, so the value is inherited rather than learned; learning leaves a small offset, 0.002 to 0.024 below a scramble of the same rows on every model, while an inconsistent control manufactures a spurious collapse (RQ2). The metric's spectral summaries are reproduced by the same control, and effective dimension tracks predictive entropy (RQ3). For interpretability, the curvature and spectrum of this metric are dominated by the predictive distribution and the simplex, and geometric controls must themselves be genuine pullbacks. The main limitations are the slice construction and model scale. Future work will test models above 1B parameters, add randomly initialised networks as controls, and estimate the extrinsic term on real models.

## Acknowledgment

An AI assistant (Claude, Anthropic) was used to draft analysis code and text and to check figures and references. The authors directed the work, reviewed every section, and regenerated every reported number from the released results.

## References

[1] A. Ansuini, A. Laio, J. H. Macke, and D. Zoccolan, "Intrinsic dimension of data representations in deep neural networks," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 32, 2019.

[2] L. Valeriani, D. Doimo, F. Cuturello, A. Laio, A. Ansuini, and A. Cazzaniga, "The geometry of hidden representations of large transformer models," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 36, 2023.

[3] R. Manson, "Curved inference: Concern-sensitive geometry in large language model residual streams," arXiv:2507.21107, 2025.

[4] J. King, E. Fedorenko, and E. A. Hosseini, "Representational curvature modulates behavioral uncertainty in large language models," arXiv:2604.23985, 2026.

[5] V. Pandey, G. Singh, and Y. Mahdid, "Trajectory geometry of transformer representations across layers," arXiv:2606.09287, 2026.

[6] A. Modell, P. Rubin-Delanchy, and N. Whiteley, "The origins of representation manifolds in large language models," arXiv:2505.18235, 2025.

[7] R. Xu, Y. Xu, and L. Wu, "PersonaManifold: Revealing and exploiting curved geometry in LLM persona representations," arXiv:2609.34571, 2026 (accepted to NeurIPS 2026).

[8] M. A. Mabrok, "Latent semantic manifolds in large language models," arXiv:2603.22301, 2026.

[9] S. Wang, J. Zhao, Q. Cao, H. Yao, and L. Shu, "FishBack: Pullback Fisher geometry for optimal activation steering in transformers," arXiv:2605.17231, 2026.

[10] D. Picozzi, "The information geometry of large language models is shared, learned, and controllable," arXiv:2609.11063, 2026.

[11] T. Mudarisov, M. Burtsev, T. Petrova, and R. State, "Predictive geometry of hidden trajectories in transformers," arXiv:2609.37717, 2026.

[12] J. Adebayo, J. Gilmer, M. Muelly, I. Goodfellow, M. Hardt, and B. Kim, "Sanity checks for saliency maps," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 31, 2018.

[13] J. Hewitt and P. Liang, "Designing and interpreting probes with control tasks," in *Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP-IJCNLP)*, 2019, pp. 2733–2743.

[14] N. N. Čencov, *Statistical Decision Rules and Optimal Inference* (Transl. Math. Monographs, vol. 53). Providence, RI, USA: Amer. Math. Soc., 1982.

[15] S. Amari, *Information Geometry and Its Applications* (Appl. Math. Sci., vol. 194). Tokyo, Japan: Springer, 2016.

[16] Anonymous, "Code, per-state results and table scripts for this article," 2026. [Online]. Available: https://anonymous.4open.science (link to be inserted before submission).

[17] E. Zaher, M. Trzaskowski, Q. Nguyen, and F. Roosta, "The geometric wall: Manifold structure predicts layerwise sparse autoencoder scaling laws," arXiv:2605.09887, 2026.

[18] J. A. Zavatone-Veth, S. Yang, J. A. Rubinfien, and C. Pehlevan, "How does training shape the Riemannian geometry of neural network representations?" in *Proc. 3rd Workshop Symmetry Geometry Neural Represent. (NeurReps)*, 2025, arXiv:2301.11375.

[19] M. Hauser and A. Ray, "Principles of Riemannian geometry in neural networks," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 30, 2017.

[20] E. Facco, M. d'Errico, A. Rodriguez, and A. Laio, "Estimating the intrinsic dimension of datasets by a minimal neighborhood information," *Sci. Rep.*, vol. 7, Art. no. 12140, 2017.

[21] S. Konior, A. Quemy, P. Klocek, B. Sobieski, and G. Cattan, "Riemannian geometry for pre-trained language model embeddings," arXiv:2607.07047, 2026.

[22] L. Ben Allal *et al.*, "SmolLM2: When Smol goes big—Data-centric training of a small language model," arXiv:2502.02737, 2025.

[23] B. Zhang and R. Sennrich, "Root mean square layer normalization," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 32, 2019.

[24] X. Miao *et al.*, "SpecInfer: Accelerating generative LLM serving with speculative inference and token tree verification," arXiv:2305.09781, 2023.

[25] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, "Language models are unsupervised multitask learners," OpenAI, Tech. Rep., 2019.

[26] J. L. Ba, J. R. Kiros, and G. E. Hinton, "Layer normalization," arXiv:1607.06450, 2016.

[27] S. Biderman *et al.*, "Pythia: A suite for analyzing large language models across training and scaling," in *Proc. 40th Int. Conf. Mach. Learn. (ICML)*, 2023, pp. 2397–2430.

[28] S. Black, L. Gao, P. Wang, C. Leahy, and S. Biderman, "GPT-Neo: Large scale autoregressive language modeling with Mesh-Tensorflow," Zenodo, 2021.

[29] S. Merity, C. Xiong, J. Bradbury, and R. Socher, "Pointer sentinel mixture models," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2017.

[30] A. Paszke *et al.*, "PyTorch: An imperative style, high-performance deep learning library," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 32, 2019.

[31] M. P. do Carmo, *Riemannian Geometry*. Boston, MA, USA: Birkhäuser, 1992.

[32] A. Le Brigant, J. Deschamps, A. Collas, and N. Miolane, "Parametric information geometry with the package Geomstats," *ACM Trans. Math. Softw.*, vol. 49, no. 4, 2023, doi: 10.1145/3627538.

[33] T. Wolf *et al.*, "Transformers: State-of-the-art natural language processing," in *Proc. Conf. Empirical Methods Natural Lang. Process.: Syst. Demonstrations (EMNLP)*, 2020, pp. 38–45.
