# 16 — Reviewer comments and where the revised paper answers them

**Created:** 29 October 2026. **For:** the supervisor, and a response letter if one is wanted. The comments are the two NSBM ICACT 2026 reviews, by question number. The paper is [paper/fisher-rao-conference-paper.md](paper/fisher-rao-conference-paper.md); the PDF is [paper/Fisher-Rao-Curvature-IEEE.pdf](paper/Fisher-Rao-Curvature-IEEE.pdf). Section, table and figure numbers refer to the revised paper.

## Reviewer 1

| Q | Score or comment | What the revision does | Where |
|---|---|---|---|
| 3 | 2, significant weakness in soundness | The unsupported claim, that curvature carries learned structure, is withdrawn, and the control behind it is shown to be invalid. Every claim is tied to a table with its n and z, and a decision rule stated before the runs is applied, including where it fails | IV-D, V-B, Tables II–III |
| 4 | 2, limited novelty | The gap is stated precisely and compared with the eight closest studies; the closest 2026 study was read in full and computes no curvature | I, II, Table I |
| 5 | 2, weak validation | One validation table: seven manifolds of known curvature, the Kullback–Leibler identity on a real model, the identity off the base point for each control, the null test on six models, and the slice-against-manifold check | IV-E, V-A, Table II |
| 6 | 2, limited significance | The first paragraph says who uses geometric quantities as interpretability evidence and why the result matters; Section V-E gives three practical rules | I, V-E |
| 7 | 2, methodological weaknesses | A methodology section: research design, models and data, three samples with their sizes, comparators, metrics, statistics and the decision rule, with a workflow figure | III, Fig. 1 |
| 8 | 4, strong fit | Kept; framed for Track 01 as explainable AI (index terms) | Abstract |
| 9 | 2, literature missing | 33 references, thirteen of them from 2025–2026, grouped in three themes and ending in an explicit gap statement | II |
| 10 | 2, analysis | Negative findings are reported with their tests: the decision rule is not met, the "towards 1/4" hypothesis is rejected, GPT-2's Gaussian control reverses | V-B |
| 11 | 2, clarity | Rewritten problem first; RQ1–RQ3 answered in order; symbols defined before use | All |
| 12 | 5, formatting compliant | Kept: IEEE A4 template, six pages | All |
| 13 | "It is not clear what the authors are going to solve. The research question is not clear." | RQ1–RQ3 are stated in the Introduction and answered in the Conclusion | I, VI |
| 14 | "Mathematically strong, but the gap is not clearly identified." | As Q4: two gaps, each tied to a contribution | I, II |
| 15 | "Research question, methodology and analysis are not clear." | As Q7, Q10 and Q13 | I, III, V |
| 16 | 2, important claims unsupported | As Q3 | V |
| 17 | Important comparisons missing | The controls are the baselines (Table III); the three published curvature instruments are compared on the same states, with their stability across two corpora (Table V); the published near-flat reading is reproduced and calibrated | V-B, V-D |
| 18 | Limitations insufficiently discussed | A threats-to-validity subsection: construct, internal, external and reliability | V-F |
| 19 | 2, poor reproducibility | Models, data, samples, seeds, software versions, hardware and runtime are stated; an anonymised repository holds the code, per-state results and a script for every table | III-B, IV-E, ref. [16] |
| 24–25 | "Explain the problem you are going to solve and how you solve it." | The abstract and Introduction follow problem, gap, approach, evaluation, result and meaning | Abstract, I |

## Reviewer 3

| Q | Score or comment | What the revision does | Where |
|---|---|---|---|
| 3 | 3, acceptable | As Reviewer 1, Q3 | IV-D, V-B |
| 4 | 3, moderate novelty | As Reviewer 1, Q4 | I, II, Table I |
| 5 | 3, adequate validation | Extended as in Reviewer 1, Q5 | Table II |
| 6 | 2, limited significance | As Reviewer 1, Q6 | I, V-E |
| 7 | 3, minor missing details | As Reviewer 1, Q7 and Q19 | III, IV-E |
| 9 | 2, literature missing | As Reviewer 1, Q9 | II |
| 10 | 2, analysis | As Reviewer 1, Q10 | V-B |
| 12 | 2, significant formatting problems | IEEE A4 template with its own styles; numbered sections; tables titled above (TABLE I–V) and figures captioned below; five numbered equations; IEEE references numbered in order of citation; six pages | All |
| 13, 25 (a) | Is the scrambled metric "a valid and appropriately controlled geometric object"? Its "local probability map, derivative construction and conditioning invariance" are not established | It is not valid: away from the base point its metric is the Hessian of no probability map (the identity fails by 0.66–0.93). It is replaced by a consistent scramble built as a genuine pullback (C1) and by an entropy-matched Gaussian readout with no learned readout at all (C2). Conditioning is not fixed state by state, but it is not shifted systematically (15.6 against 15.1) | IV-D, Table II, V-B |
| 25 (b) | "The distinction between slice curvature and full-manifold curvature requires more precise treatment." | The quantity is called slice sectional curvature throughout; the Gauss equation is stated; the extrinsic term is measured where the answer is known (0.005–0.028 between 0.5 and 4 nats); every comparison uses slices of identical construction; the learned offset is interpreted as a slice property | IV-C, eq. (4), V-A, V-B |
| 14 | Strengths: the validation ladder, the KL–Hessian check, the reparameterisation test and the broken assembly | All kept | Table II |
| 16 | 3, partially supported | As Reviewer 1, Q3 | V |
| 17 | Important comparisons missing | As Reviewer 1, Q17 | V-B, V-D |
| 18 | Limitations largely missing | As Reviewer 1, Q18 | V-F |
| 19 | 3, partially reproducible | As Reviewer 1, Q19 | III-B, IV-E |

## Beyond the reviews

The rewrite also corrects the eleven errors in the submitted paper listed in [12-audit-log.md](12-audit-log.md) §8.6, including three wrong reference author lists. [check_paper_numbers.py](check_paper_numbers.py) recomputes all 96 numbers that the paper quotes from the saved results.
