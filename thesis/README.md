# Thesis drafts

Eight chapter drafts, written from the progress logs in the parent directory. **Every number traces to a named script and a saved result file** — nothing here was written from memory.

| Chapter | File | Status |
|---|---|---|
| 1 | [ch1-introduction.md](ch1-introduction.md) | draft |
| 2 | [ch2-background.md](ch2-background.md) | draft |
| 3 | [ch3-literature-and-gap.md](ch3-literature-and-gap.md) | draft |
| 4 | [ch4-methodology.md](ch4-methodology.md) | draft |
| **5** | **[ch5-validation.md](ch5-validation.md)** | draft — **read this first** |
| 6 | [ch6-results.md](ch6-results.md) | draft |
| 7 | [ch7-behavioural.md](ch7-behavioural.md) | draft |
| 8 | [ch8-discussion.md](ch8-discussion.md) | draft |

## Where the numbers come from

| chapter | primary sources |
|---|---|
| 2 | standard material; conventions fixed to match `validate_curvature.py` |
| 3 | `01-literature-review.md`, `02-research-gap.md`, plus Ch. 6's replication |
| 4 | `03-methodology.md`, `05-stage0-log.md`, `08-rq3a-log.md` §5.6, `10-architecture-log.md` |
| 5 | `validate_curvature.py`, `validate_pullback.py`, `validate_geomstats.py`, `gate_a_kl_test.py`, `check_architectures.py`, `run_frame_resolution.py` |
| 6 | `run_profiles.py`, `run_stage4.py`, `run_corpus_compare.py`, `run_mabrok_replication.py`, `check_pca_tautology.py`, `run_volume_scramble.py` |
| 7 | `run_polysemy.py`, `check_polysemy.py`, `run_wic.py`, `run_samesense.py`, `run_e5.py`, `run_scramble_within.py` |
| 8 | all of the above |

## What still needs doing

**Before submission**

- [ ] Unify notation across all eight chapters in one deliberate pass, against Ch. 2 §2.6
- [ ] Regenerate every figure from a pinned script and seed (no figures are in these drafts yet — they are text and tables)
- [ ] Convert to the university's required format
- [ ] A full read-through for repetition: Chapters 6 and 7 both restate the scramble control, and Chapter 4 §4.5 sets it up

**Known gaps flagged in the text**

- Ch. 5 §5.8 — ladder rung 8 not run (rungs 2, 3, 4, 5, 6, 7 pass). Geomstats cross-check **executed**: agrees on 3 families of 5; 2 disagree with identical metrics (§5.3b, unresolved)
- Ch. 8 §8.3 — models are 70M–160M; no figures for the layer profiles yet
- Ch. 7 §7.2.3 — sense-versus-context is bracketed between two controls, not settled
- Ch. 8 §8.4 — the negative-plane lead is now **2.03× median `k_eff`, AUC 0.618, z = +2.69**, not ρ = +0.42; the original figure was a rank-tie artefact ([../12-audit-log.md](../12-audit-log.md) §2)

## Reading order for a supervisor

**5 → 6 → 7** is the argument. Chapter 5 establishes that the machinery is trustworthy; 6 and 7 are what it found. Chapters 1–4 are the frame around that and can be read after.
