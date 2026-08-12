# Chapter 8 — Discussion, limitations and future work

---

## 8.1 What the results amount to

Three things, in decreasing order of how firmly they are established.

**1. The representation manifold is strongly positively curved, at the ambient simplex value, across architectures.** `K ≈ +1/4` on four models within 0.022 of each other. This is not "essentially flat at 10⁻⁵", and Chapter 6 §6.4 shows why the published flat result arose: the instrument that produced it cannot distinguish a sphere from a plane at the threshold that produces it.

**2. Only curvature reads the model; everything spectral reads concentration.** One control, three quantities: sectional `K` collapses under a structure-free scramble at matched entropy (0.2546 → 0.0109, n = 221, z = +12.18), while log-volume and effective dimension are reproduced at ρ = +0.96 and +0.99. This is the most useful result in the thesis, because it is *prescriptive* — it says which quantities are worth computing and which are elaborate restatements of the entropy profile.

**3. The metric is the model's, causally.** Directions the metric calls null are directions the model is exactly invariant to, on six models at machine precision, at a step size that doubles the state.

### 8.1.1 The interpretive claim, stated at the strength the evidence supports

`+1/4` is the *ambient* value, so the headline number does not by itself distinguish one model from another. What distinguishes them is the scramble control: **which probability sits on which direction is what holds the manifold at the simplex value**, and destroying that assignment collapses curvature to zero at exactly matched entropy, matched conditioning and matched direction set, stably across `k`.

The layer breakdown sharpens this: the assignment does nearly all the work in layers 5–25 and least at the two ends — where context has not yet been integrated, and where the prediction is already committed.

---

## 8.2 What the thesis does not establish

Stated plainly, because several of these were claimed at some point in the work and withdrawn.

- **The deviation from `+1/4` is not identifiable.** It is `k`-unstable and its layer ranking scrambles. Any model-specific *magnitude* signal in absolute curvature remains unfound.
- **Cross-model log-volume comparisons are not supported.** Four candidate mechanisms for the one cross-model difference observed were tested and all rejected.
- **The Ricci-predicts-departure result is not general.** It survives on the GPT-NeoX family only, after three successive narrowings.
- **The comparison to King et al. is not like-for-like** — different statistic, model, corpus and sign convention. Only the within-study comparison is defensible.
- **Sense versus context is bracketed, not settled.** Two same-sense controls confounded in opposite directions give 1.0× and 2.03×; the tighter one is better but rests on 32 pairs.
- **No claim about causality of the learned assignment.** The scramble shows dependence, not mechanism.

---

## 8.3 Limitations

**Scale.** Models are 70M–160M parameters. GPT-2 XL and Pythia-2.8B — the models the comparison literature uses — are absent: the download was throughput-limited and the largest exceeded available memory. Nothing here rules out the geometry changing at 1B+.

**The `k` ≤ 8 ceiling.** Everything Riemann-derived is computed on an affine slice of dimension 5 (swept over 4–6). The retained subspace holds ~88% of the metric's trace, but it is a slice, and the reported curvature is that of the slice under the induced metric — not a sectional curvature of a totally geodesic submanifold.

**Sample sizes.** `n` = 360 for the centrepiece, 456 for the profiles, but per-layer and per-bin cells are 12–14. Given how often this project's own claims died when `n` rose, **the per-cell numbers should be read as indicative and the pooled ones as the result.**

**One rung of the ladder is still open.** Rung 8 — real LM "ripple" manifolds — has not been run, and its published answer is only qualitative. Separately, the independent-implementation arm of Gate B is met on three Fisher–Rao families and fails on two, where the metrics agree exactly and the curvatures do not (Chapter 5 §5.3b). That discrepancy is unresolved.

**Two corpora, both English, both edited prose.** No code, no dialogue, no low-resource languages.

---

## 8.4 Future work

**In order of value per unit effort.**

**1. Test whether the published hourglass is also definitional.** Chapter 6 §6.5.1 shows this work's `k_eff` hourglass is reproduced by a structure-free scramble at ρ = +0.987, and that its minimum sits at the same relative depth Valeriani et al. and Mabrok report for ambient intrinsic dimension. **That is a testable prediction about their results**: apply a matched-entropy control to an ambient TWO-NN estimate and see whether the dip survives. If it does not, a substantial part of the intrinsic-dimension literature is measuring predictive concentration.

**2. Scale up.** GPT-2 XL and Pythia-2.8B, which also enables the like-for-like comparison to King et al. that §8.2 says is currently impossible.

**3. Chase the negative-curvature minority.** 10.7% of points carry a negatively curved plane. They are not conditioning failures (`cond_eff` z = −1.39, n.s.) and not an entropy effect (z = −0.45, n.s.), and they carry **2.03× the median effective dimension at indistinguishable entropy** (248 vs 122; AUC 0.618, z = +2.69). This is the only place absolute curvature carries a model-specific signal that has survived a control — but it is a modest effect, and the ρ = +0.42 originally reported for it was a rank-tie artefact ([12-audit-log.md](../12-audit-log.md) §2).

**4. Connect the causal and behavioural experiments.** Perturb along high-curvature directions *at the disambiguation layer* and test whether the resolved sense flips. This links Chapter 7 §7.2 to §7.3 and would be the strongest single result the framework could produce.

**5. Untied vs tied embeddings, properly.** The clean 2×2 at `k` = 5 did not survive a `k` sweep, but the direction was consistent and the mechanism is plausible: with tied embeddings the unembedding carries a second job. Six more models would settle it.

**6. Resolve the Geomstats discrepancy.** Two families where the metrics agree
to 1e−14 and the curvatures do not (Chapter 5 §5.3b). The evidence points to the
library, but "points to" is not "shows"; a minimal reproducer submitted upstream
would settle it either way, and it is a day's work.

---

## 8.5 A methodological postscript

Six claims in this work were weakened or withdrawn by controls the work built to attack itself. The pattern is consistent enough to state as a rule:

> **On this data, a result at one `n`, one `k`, or one control is a hypothesis. It becomes a finding when it survives the second one.**

Three specific traps recur and are worth naming for anyone extending this work:

**A control is only as good as the list of things it holds fixed.** Three successive versions of the same control were wrong: one was provably the *identity* on this metric; one changed the spectrum, row norms and assignment at once; one swapped the retained direction set as well as the pairing. Each looked clean. Each needed a specific arithmetic or measured check to expose.

**A median over a mixture of correct and broken values is not robustness; it is a way to not notice.** A sign bug in a basis vector corrupted individual points catastrophically while leaving the median clean, and was found only when a second architecture made the signs come out uniformly wrong.

**Convenience samples reverse.** The most confidently wrong statement in this work came from 24 points chosen for being easy to compute, and reversed at 220 randomly drawn ones.

None of these is exotic. All three are the ordinary failure modes of a quantity that is expensive to compute, silent when wrong, and plausible either way — which is exactly what intrinsic curvature is.
