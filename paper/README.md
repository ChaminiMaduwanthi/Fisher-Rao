# Conference paper — ICARC 2027 revision

**[fisher-rao-conference-paper.md](fisher-rao-conference-paper.md)** is the source of the revised paper. It builds to six A4 pages in the IEEE template: [PDF](Fisher-Rao-Curvature-IEEE.pdf) and [Word](Fisher-Rao-Curvature-IEEE.docx). The version rejected at NSBM ICACT 2026 is kept at the repository root as [fisher-rao.pdf](../fisher-rao.pdf).

## The argument

| Research question | Answer | Evidence |
|---|---|---|
| RQ1 — can the intrinsic curvature of the Fisher–Rao pullback metric be computed reliably on transformer hidden states? | Yes | Table II: seven manifolds of known curvature, the Kullback–Leibler identity, null directions on six models, slice against manifold |
| RQ2 — what value does it take, and is that value learned? | Slice curvature sits near 1/4 on four models. Genuine-pullback controls give the same value to within 0.035, so it is inherited from the simplex; learning leaves a small offset, 0.002–0.024 below a scramble of the same rows | Table III, Fig. 2 |
| RQ3 — what do the spectral summaries measure? | The predictive distribution: a scramble of the readout reproduces them, and effective dimension tracks predictive entropy | Table IV, Fig. 3 |

The submitted paper's central claim, that curvature carries learned structure, rested on a control that is not a pullback. The claim is withdrawn, and the paper shows why the control fails (Section IV-D, Table II). [16-reviewer-response.md](../16-reviewer-response.md) maps every reviewer comment to the part of the paper that answers it.

## Structure

The section order and budgets follow the supervisor's six-page guideline.

| Section | Content |
|---|---|
| Abstract | 218 words, beginning "This article"; no first-person pronouns anywhere in the paper |
| I. Introduction | the problem and its significance for interpretability, two gaps, RQ1–RQ3, three contributions |
| II. Related Work and Research Gap | three themes, Table I of the closest studies, the gap |
| III. Research Methodology | design, models and data, comparators, metrics, statistics and the decision rule; Fig. 1 |
| IV. Pipeline Design and Implementation | equations (1)–(5), null directions, slices and the Gauss equation, the three controls, validation, implementation; Table II |
| V. Results and Discussion | RQ1–RQ3, comparison with published instruments, implications, threats to validity; Tables III–V, Figs. 2–3 |
| VI. Conclusion and Future Work | answers to RQ1–RQ3, limitations, next steps |

## Building it

1. `python make_tables.py` rebuilds every table from the saved results and checks 69 stored summaries.
2. `python make_figures.py` writes `figures/fig1_pipeline.png`, `fig2_controls.png` and `fig3_spectral.png`.
3. `python check_paper_numbers.py` recomputes the 96 numbers the paper quotes and checks that each appears in the text.
4. `python build_paper_docx.py` writes the Word manuscript into the IEEE template, which it reads from the repository history; pass a path to use another copy. The document properties are left blank.
5. Export the PDF from Word and confirm that it has six pages.

## Before submission

- [ ] Mirror the code to an anonymous host and put the link in reference [16].
- [ ] Agree the AI-use disclosure (the Acknowledgment) with the supervisor, or place it where the organisers advise.
- [ ] Supervisor reads the PDF; similarity check below 30 %.
- [ ] If the Word file is edited and saved, run *File → Info → Inspect Document* before exporting the PDF, because Word writes the editor's name into the document properties.
- [ ] Check the PDF with IEEE PDF eXpress if ICARC requires it.
