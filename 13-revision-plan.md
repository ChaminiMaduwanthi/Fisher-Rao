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
