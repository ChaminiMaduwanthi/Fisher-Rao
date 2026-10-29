# 13 — Revision plan: ICARC 2027 resubmission

**Created:** 29 October 2026
**Target:** ICARC 2027, Track 01 — Artificial Intelligence and Machine Learning (its topic list includes "Advances in GenAI and XAI")
**Dates:** submission 25 November 2026 · notification 6 January 2027 · camera-ready 21 January 2027 · conference 16–17 February 2027
**Format:** at most six A4 pages in the IEEE conference template, PDF, submitted through CMT, double-blind review

---

## 1. Where things stand

The conference paper ([paper/fisher-rao-conference-paper.md](paper/fisher-rao-conference-paper.md); the six-page version is [fisher-rao.pdf](fisher-rao.pdf)) was rejected at NSBM ICACT 2026 on two reviews.

- **Reviewer 1** scored most criteria 2/5. The paper was judged "mathematically strong", but the research question, the gap, the methodology and the analysis were not clear to the reader, and reproduction details were missing.
- **Reviewer 3** scored most criteria 3/5 and named the validation ladder, the KL–Hessian check, the reparameterisation test and the deliberately broken assembly as strengths. The objections were technical: whether the scrambled metric behind the central claim is a valid and properly controlled geometric object, and the difference between curvature on a slice and curvature of the full manifold.
- Both scored significance 2/5 and reported important literature missing.

A check made while writing this plan (§3) shows that Reviewer 3's first objection is correct and decisive. The revision therefore changes what the paper claims, not only how it is written.

## 2. Reviewer comments and planned responses

| # | Reviewer | Comment (paraphrased) | Planned response | Paper section |
|---|---|---|---|---|
| 1 | R1 | The problem and research question are not clear | Open with the problem; state three research questions; tie each contribution to one | I |
| 2 | R1, R3 | The gap is not established; important literature is missing | Related Work in four groups, each ending with what it leaves open; add the 2026 papers of §6; 35–40 references | II |
| 3 | R3 | The scramble's local probability map, derivative construction and conditioning invariance are not established | Replace it with the consistent control of §3; report both and show why the inconsistent one fails; add an untrained-model control | III, V |
| 4 | R3 | Slice curvature is not distinguished from full-manifold curvature | State the Gauss equation; measure the extrinsic term on a known answer and by nested slices (E4); call the quantity slice sectional curvature throughout | III, V |
| 5 | R1, R3 | Important claims are unsupported | Drop "curvature carries learned structure" unless E1 reverses the pilot; tie every claim to a table or figure | all |
| 6 | R1, R3 | Comparisons with baselines are missing | A controls table (real model, both scrambles, Gaussian `U`, untrained model) and an instruments table (intrinsic `K`, the three published proxies, Euclidean, `UᵀU`) | V |
| 7 | R1, R3 | Limitations and threats to validity are insufficient | A dedicated Threats to Validity section | VI |
| 8 | R1 | Reproduction details are missing | Hyperparameters, seeds, sampling, hardware, runtimes and an anonymised code link | III |
| 9 | R1 | Clarity is weak | Problem-first writing, a pipeline figure, fewer symbols before Section III | all |
| 10 | R3 | Formatting problems | Official IEEE conference template, IEEE reference style, IEEE figure and table conventions | all |
| 11 | R1, R3 | Significance is limited | Say in the first paragraph who needs the result and why: which geometric probes of a language model can be trusted for interpretability | I, VII |

## 3. The finding that changes the central claim

**The problem.** The paper's paired scramble (`fisher_metric_projected(rows=...)` in [fisherrao/metrics.py](fisherrao/metrics.py)) takes the predictive distribution `p(h)` from the real model but the Jacobian from permuted unembedding rows. At the base point that matches the paper's description. Curvature, however, is built from the first and second derivatives of the metric, and away from the base point `p` moves along the real directions while the metric measures along the permuted ones. The result is a valid Riemannian metric, but it is not the Fisher pullback of any statistical model.

**The consistent control.** [run_scramble_consistent.py](run_scramble_consistent.py) realises the same permutation as a genuine model,

    z(h) = U[idx] n(h0) + U[rows] (n(h) − n(h0)),    p(h) = softmax(z(h)),

which agrees with the paper's control exactly at the base point (same `p`, metric, frame, conditioning and row multiset; relative difference 0) and differs only in that `p` moves along the directions the metric uses. With the identity permutation it reproduces the real metric to 1.4 × 10⁻¹⁴.

**Result** (`k` = 5, on the same states; [results/tables/tables.md](results/tables/tables.md) Table 1):

| condition | SmolLM2-135M (186 states) | LLaMA-160M (118) | GPT-2 (120) | Pythia-160M (120) |
|---|---|---|---|---|
| real model | 0.2547 | 0.2565 | 0.2580 | 0.2696 |
| paper's scramble (not a pullback) | 0.0096 (real higher 168/186, z = +11.00) | −0.0056 (116/118, z = +10.49) | 0.0095 (112/120, z = +9.49) | 0.0218 (117/120, z = +10.41) |
| consistent scramble (a pullback) | 0.2565 (real higher 72/186, z = −3.08) | 0.2794 (33/118, z = −4.79) | 0.2823 (31/120, z = −5.29) | 0.2827 (38/120, z = −4.02) |
| Gaussian `U`, entropy-matched (no learning) | 0.2730 (real higher 40/186, z = −7.77) | 0.2846 (35/118, z = −4.42) | 0.2303 (100/120, z = +7.30) | 0.2816 (39/120, z = −3.83) |

The earlier Gaussian control of [08-rq3a-log.md](08-rq3a-log.md) §5.1, matched by entropy band rather than state by state, had already pointed the same way.

**What it implies.** The collapse reported in the paper comes from the scrambled metric not being a pullback, not from destroying the learned assignment. `K ≈ +1/4` is a property of softmax-pullback geometry rather than of learning: every genuine pullback gives 0.230–0.285. The trained models are not identical to the controls, though. They sit below the consistent scramble on all four models (by 0.002–0.024), and differ from the Gaussian readout by 0.012–0.028 in a direction that depends on the model (GPT-2 above, the others below). By the E1 decision rule (§5) they agree on none of the four, so the paper reports that offset ([14-revision-log.md](14-revision-log.md) §14). The earlier reading, that learning pulls `K` towards +1/4, is not supported state by state.

**What it does not affect.** The spectral results (log-volume ρ = +0.957, effective dimension ρ = +0.991 on SmolLM2) are functions of `G(h0)` alone, and at `h0` the two controls are identical. They replicate on all four models: ρ across states 0.85–0.96 and 0.97–0.99, with the same lowest layer in every case ([14-revision-log.md](14-revision-log.md) §12). The validation ladder, the null-direction test, `K ≈ +1/4` as a measurement, the calibration of the 10⁻⁵ reading and the behavioural probe also stand.

**Status.** Done on all four models at `k` = 4–6, with a learning-free Gaussian control in place of E2, and the spectral control on all four ([14-revision-log.md](14-revision-log.md) §4–§6, §12–§14). An untrained-model control remains optional. Full record: [08-rq3a-log.md](08-rq3a-log.md) §5.7, [12-audit-log.md](12-audit-log.md) §8, [14-revision-log.md](14-revision-log.md).

## 4. Proposed framing (to be agreed with the supervisor)

**Title.** The paper keeps the thesis topic and its main title, *Intrinsic Fisher–Rao Curvature of Transformer Representations*; only the subtitle, which states the claim, changes. An alternative if the supervisor prefers: *What Does Fisher–Rao Geometry Measure in Transformer Representations? A Controlled Study of Curvature and Spectrum*

**Problem statement (draft).** Geometric summaries of the Fisher–Rao metric on transformer hidden states — its spectrum, effective dimension, volume and curvature — are increasingly used to describe what language models represent. Whether these quantities reflect learned structure, or are fixed by the predictive distribution and the geometry of the probability simplex, has not been tested with controls that hold the predictive distribution fixed and remain genuine pullbacks.

**Research questions.**

- **RQ1.** Can the intrinsic Riemannian curvature of the Fisher–Rao pullback metric on transformer hidden states be computed reliably? *(method, validation ladder, null-direction test)*
- **RQ2.** What value does that curvature take across architectures, and is the value produced by learning? *(`K ≈ +1/4`; Gaussian, untrained and consistent controls)*
- **RQ3.** What do the spectral summaries — log-volume and effective dimension — read? *(matched-entropy controls; the depth profile)*

**Contributions, if E1–E2 confirm the pilot.**

1. A validated pipeline for the intrinsic curvature of the Fisher–Rao pullback on transformer hidden states.
2. `K ≈ +1/4` across four architectures, shown by genuine-pullback controls to be inherited from the simplex rather than learned. The trained models sit slightly below a structure-free pairing of their own rows on every model (by 0.002–0.024), which the paper reports as a small offset in slice curvature. *(Revised after E1: [14-revision-log.md](14-revision-log.md) §14.)*
3. The spectral profiles reproduced, on four models, by a control that destroys the learned pairing; effective dimension tracks predictive entropy on all four. The depth "hourglass" is SmolLM2's and is not claimed in general ([14-revision-log.md](14-revision-log.md) §12).
4. A methodological result: a control that is not itself a pullback manufactures a spurious collapse (0.255 → 0.000), and a consistency criterion that prevents it.
5. A calibration of the published near-flat reading (10⁻⁵): the same estimator reports a unit 3-sphere as flat.

**Dropped:** "curvature carries learned structure". **Moved to future work:** the causal probe (current §VI.H). **Kept briefly if space allows:** the behavioural probe.

**If E1 differs from the pilot** on other architectures, the framing changes to report where real and consistent curvature diverge. The claim follows the data either way. **Outcome, 29 October 2026:** agreement on none of the four models. The divergence is a consistent offset of 0.002–0.024, so the framing stands with contribution 2 revised as above ([14-revision-log.md](14-revision-log.md) §14).

## 5. Experiments

**Scope agreed on 29 October 2026.** The thesis topic is unchanged. Required: E1 on SmolLM2-135M at the paper's sample with `k` = 4–6, and E4. Added: E1 on LLaMA-160M, which needs no new code. Optional, only if time allows: E1 on the LayerNorm models, E2, E3 and E6–E8. With two models, the E1 decision rule becomes agreement on both. Results are logged in [14-revision-log.md](14-revision-log.md).

| ID | Purpose | Method | Estimated cost | Decision rule |
|---|---|---|---|---|
| **E1** | Answer Reviewer 3 (a) at scale | The consistent control on SmolLM2-135M, LLaMA-160M, GPT-2 and Pythia-160M: the 186 distinct states (221 rows) of the paper's control on SmolLM2, 60–120 on the others, `k` = 4, 5, 6. LayerNorm models need the general norm Jacobian (`LM.norm_jacobian`) in place of the RMSNorm closed form | About 8 s per state for three conditions at `k` = 5 (measured), so about 30 min per model at n = 221; `k` = 6 costs about three times as much | If real and consistent agree (sign test \|z\| < 2, median difference < 0.01) on at least three of four models, adopt §4; otherwise report where they differ |
| **E2** | A learning-free genuine pullback | The same architectures with randomly initialised weights, states stratified by entropy | As E1 | `K` near +1/4 supports contribution 2 |
| **E3** | Complete the paired picture | Re-run the Gaussian-`U` control on the E1 states | Small | Reported alongside E1 |
| **E4** | Answer Reviewer 3 (b) | (i) Known answer: on the synthetic pullback of ladder rung 7 the full manifold has `K` = 1/4 exactly, so the curvature of `k`-dimensional linear slices measures the extrinsic (Gauss-equation) term directly. (ii) Real models: the same 2-planes in nested slices `k` = 2…8; convergence indicates a small extrinsic term in the retained directions | Moderate; `k` = 8 costs about 46 s per Riemann evaluation | Small extrinsic term: report slice `K` as representative. Large: restrict every claim to slices |
| **E5** | Confirm the spectral result is unaffected | Show the two controls give the same `G(h0)` (already measured: relative difference 0) | None | One sentence and one check |
| E6 | Scale (optional) | One model near 1B parameters (Pythia-1B or SmolLM2-1.7B), feasibility first | Unknown | Only if E1–E4 finish by 12 November |
| E7 | Theory (optional) | Conditions under which a slice of a softmax pullback is locally close to a face of the simplex, extending rung 7 | Unknown | Drop if not complete by 12 November |
| E8 | Spectrum from `p` (optional) | Predict the effective dimension from the predictive distribution alone, in the spirit of Picozzi (2026) | Small | Supports contribution 3 |

## 6. Literature to add

Five papers from 2026 are now in [references.bib](references.bib) and summarised in [01-literature-review.md](01-literature-review.md) §9. None computes the Riemann tensor of the Fisher pullback.

| Paper | What it does | Use in the paper |
|---|---|---|
| Picozzi, `arXiv:2609.11063` | Spectrum and effective dimension of the same pullback, Pythia 70M–6.9B; spectrum predictable from token probabilities | Closest work; supports contribution 3 |
| Mudarisov et al., `arXiv:2609.37717` | The pullback operator as the loss-to-go Hessian, 1B–9B models; its "curvature score" is a trace | Scale contrast; distinguish the terminology |
| Xu et al., `arXiv:2609.34571` (listed for NeurIPS 2026) | Ollivier-Ricci curvature of persona activations; geodesic steering | Curvature with behavioural consequences, a different notion |
| Pandey et al., `arXiv:2606.09287` | Ambient-space trajectory curvature with shuffled controls | Flat-metric curvature group |
| Konior et al., `arXiv:2607.07047` | Pullback metrics of encoders; a random encoder recovers much of the gain | Precedent for random-network controls (E2) |

**Still to read:** Zhang, *A Markov Categorical Framework for Language Modeling*, `arXiv:2507.19247`, found by search and not yet read. Repeat the search in the week before submission.

**Related Work structure.** (a) The Fisher pullback metric on hidden states: Mabrok, FishBack, Zaher et al., Picozzi, Mudarisov et al. (b) Curvature under flat metrics: Manson, King et al., Pandey et al. (c) Other curvature notions and other networks: Zavatone-Veth et al., PersonaManifold. (d) Intrinsic dimension and representation manifolds: Valeriani et al., Facco et al., Modell et al. Each group ends with what it leaves open.

**Rules.** 35–40 references in IEEE style, with every preprint's authors and venue re-checked. One known inconsistency to resolve against arXiv: FishBack's authors are "Sihan Wang, Jiayi Zhao" in [references.bib](references.bib) but "Y. Wang and L. Zhao" in the paper's reference [4]. No reused source text — ICARC desk-rejects above 30 % similarity.

## 7. Writing plan (six pages)

| Section | Budget | Content |
|---|---|---|
| Abstract | about 200 words | Problem, method, three findings, implication |
| I. Introduction | 0.6 page | The problem, why it matters for interpretability, RQ1–RQ3, contributions |
| II. Related Work | 0.6 page | The four groups of §6 |
| III. Method | 1.0 page | Pullback, null directions, curvature on slices with the Gauss-equation statement, the controls and the consistency criterion, models, data, reproduction details |
| IV. Validation | 0.5 page | Ladder, KL–Hessian identity, null-direction test (merge Tables I and II if space is short) |
| V. Results | 2.0 pages | `K` across architectures; the controls table; spectral profiles; the 10⁻⁵ calibration |
| VI. Threats to Validity | 0.4 page | Scale, slices, sample sizes, corpora, the open library comparison |
| VII. Discussion and Conclusion | 0.5 page | What the geometry measures, what it does not, future work |
| References | 0.4 page | 35–40 entries |

**Figures.** (1) A pipeline diagram, new. (2) The controls figure: `K` for the real model, both scrambles, Gaussian `U` and the untrained model. (3) The spectral profiles, real against scrambled. The current threshold figure becomes a small panel or is cut.

**Rules.** Problem before method. One claim per paragraph, each tied to a table or figure. Every number re-measured into its table from a results file ([12-audit-log.md](12-audit-log.md) §7.5). Call what is measured "slice sectional curvature".

## 8. ICARC submission checklist

- [ ] At most six A4 pages in the IEEE conference template; PDF only
- [ ] Anonymised: no names, affiliations, e-mail addresses, acknowledgements or grant information; the author's earlier work cited in the third person; any code link anonymised
- [ ] Similarity well under the 30 % desk-reject threshold; no reused source text
- [ ] AI-use disclosure as required by the IEEE PSPB Operations Manual. The blind version must omit acknowledgements, so ask the organisers where the disclosure goes
- [ ] Track 01 selected in CMT
- [ ] References complete, in IEEE style, preprint metadata re-checked
- [ ] Every number in the paper re-measured from a results file
- [ ] Supervisor has read the final PDF
- [ ] Submitted by 24 November 2026, one day before the deadline

## 9. Timeline

| Week | Dates | Work | Gate |
|---|---|---|---|
| 1 | 30 Oct – 5 Nov | Supervisor meeting on §3–§4; E1 on the two RMSNorm models, then LayerNorm support; E2; start E4 | Framing agreed with the supervisor by 2 Nov |
| 2 | 6 – 12 Nov | E1 on the LayerNorm models; E3; E4; literature to 35–40 references; tables and figures | E1–E2 decide the claim by 9 Nov |
| 3 | 13 – 19 Nov | Full six-page draft in the IEEE template | Draft to the supervisor by 19 Nov |
| 4 | 20 – 24 Nov | Revision; §8 checklist; submission | Submitted by 24 Nov |

**Fallback.** If E1–E2 are inconclusive by 12 November, or the supervisor prefers more time, do not submit to ICARC; target a later venue with the complete study. A correct paper later is worth more than an unsupported one now.

## 10. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| E1 differs from the pilot on other architectures | The framing changes | The decision rule in §5 sets the claim either way |
| LayerNorm support takes longer than planned | E1 incomplete | Run the two RMSNorm models first (SmolLM2-135M, LLaMA-160M) |
| `k` = 6 costs about three times `k` = 5 | Compute overrun | Run `k` = 6 on a subset of about 70 states, as before |
| Six pages are not enough | Content cut | Merge the validation tables; move detail to the anonymised code repository |
| A null-type result is judged insignificant | Rejection | State the practical consequence — which geometric probes can be trusted for interpretability — and position against the 2026 papers that rely on them |
| If this repository is public, it shows the paper and the author's name | Reviewers could identify the author | Consider making it private until the decision on 6 January 2027 |
| The deadline slips | No submission | The fallback in §9 |

## 11. Decisions needed from the supervisor

1. Adopt the framing in §4, with contribution 2 revised after E1? The agreement rule was not met ([14-revision-log.md](14-revision-log.md) §14).
2. Submit to ICARC by 25 November, or take more time for a later venue?
3. Authorship and affiliation for the camera-ready version.
4. Wording of the AI-use disclosure.
5. Whether to post a preprint, given double-blind review.

## 12. Progress

**29 October 2026 — step 1 (science) complete** ([14-revision-log.md](14-revision-log.md) §15).

| step | content | status |
|---|---|---|
| 1. Science | consistent and entropy-matched Gaussian controls on four models; spectral control on four models; E1 decision rule applied; slice against manifold (Gauss); control validity; duplicate audit; LayerNorm Jacobian fixed; tables rebuilt and checked (69/69); reproducibility record | ✅ |
| 2. Writing | title, abstract, introduction with RQ1–RQ3, related work, method, validation, results, threats to validity, discussion, figures, six pages | ✅ draft, 29 October 2026: six pages in the IEEE template, 96 number checks ([14-revision-log.md](14-revision-log.md) §16, [16-reviewer-response.md](16-reviewer-response.md)); awaiting the supervisor's reading |
| 3. Compliance | anonymisation (author line, Word metadata), IEEE references, number check, similarity check, AI disclosure, anonymous code link | partly done: author line and document properties anonymous, references in IEEE style and checked, number check passing; to do: similarity check, AI disclosure, anonymous code link |
| 4. Submission | supervisor decisions (§11), final read, CMT Track 01 by 24 November | to do |

The rewrite corrects the eleven errors listed in [12-audit-log.md](12-audit-log.md) §8.6.
