# Conference paper

**[fisher-rao-conference-paper.md](fisher-rao-conference-paper.md)** — full draft, ~5,600 words + tables.

## The argument, in one line

Everything derived from the Fisher metric's **spectrum** reads predictive concentration; only **curvature** reads the model. One control, three quantities, a clean split.

## Why this argument and not the `+1/4` headline

`K ≈ +1/4` versus the published `10⁻⁵` is the eye-catching number, but it is the weaker paper. It invites a magnitude dispute, and §6.4 shows that dispute is void — at its own threshold the published proxy reads a unit 3-sphere as flat to 10⁻³¹, so the two numbers are not measuring one thing and cannot be ranked.

The spectrum/curvature separation is the stronger contribution because it is **prescriptive** (it tells other people which quantities to compute), **survives its own audit** (the corrected control made the split *wider*, §5.3), and **makes a falsifiable prediction about an active literature** (§7.4 — apply a matched-entropy control to ambient TWO-NN intrinsic dimension and see whether the hourglass survives).

`+1/4` is still in the paper, as §7.5. It is a supporting result, not the claim.

## Structure

| § | Content |
|---|---|
| 1–2 | Setting, the four-instrument disagreement, the gap |
| 3 | The metric: pullback, exact null directions, quotient-then-restrict |
| 4 | **Validation** — 6-rung ladder, geomstats cross-check, KL–Hessian, causal null test, 2 silent bugs |
| 5 | **The control** — all three failed attempts, then the clean one |
| 6 | Instruments do not agree (n = 360) + replication and diagnosis of the flat result |
| 7 | ⭐ **The central result** + the prediction + cross-architecture |
| 8 | Behavioural: a 21.75-layer disagreement about when the model decided |
| 9 | Six of our own claims that controls killed |
| 10–11 | Limitations, conclusion |
| — | Reproduction table: every § → script → output |

## Before submission

- [ ] **References** — §12 is a placeholder. Needs Čencov, the simplex–sphere isometry, TWO-NN, geomstats, the two replicated proxies, WiC.
- [ ] **Pick a venue and cut to its page limit.** At full length this is ~9–10 pages of two-column. §4 and §9 are the compressible parts; §5 and §7 are not.
- [ ] **Figures.** None yet — everything is text and tables. Three would earn their space: the layer profile with real vs scrambled overlaid (§7.1), the four-instrument correlation matrix (§6.2), and the `d_FR` vs Euclidean separation curves (§8).
- [ ] **Anonymise** the repository link for double-blind review.
- [ ] Decide whether §9 stays. It is unusual and it is honest; it also hands a reviewer a list of things that went wrong. Recommendation: **keep it** — every entry is paired with the control that caught it, which is an argument for the methodology rather than against the results.
