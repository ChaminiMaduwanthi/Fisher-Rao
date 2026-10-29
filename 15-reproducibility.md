# 15 — Reproducibility record

**Created:** 29 October 2026, for the ICARC 2027 resubmission ([13-revision-plan.md](13-revision-plan.md)). Reviewer 1 scored reproducibility 2/5: *"substantial methodological, implementation, dataset, parameter, or evaluation information is missing."* This file is the information, in one place. The paper's reproduction paragraph is written from it.

---

## 1. Environment

| item | value |
|---|---|
| OS | Windows 11 (10.0.26200) |
| CPU | Intel Core i5-1135G7, 4 cores / 8 threads, 2.40 GHz; no GPU is used anywhere |
| RAM | 19.7 GB |
| Python | 3.13.14 |
| PyTorch | 2.12.0, CPU build, 4 intra-op threads by default |
| transformers / huggingface_hub / pyarrow | 5.14.1 / 1.19.0 / 24.0.0 |
| numpy / scipy / matplotlib | 2.3.3 / 1.16.2 / 3.10.6 |
| geomstats | 2.8.0 — the independent implementation for ladder rungs 2–6 |
| precision | float64 for every metric, derivative and curvature |

Pinned in [requirements.txt](requirements.txt).

## 2. Models

| model | Hugging Face id | revision | layers | width | vocabulary | final norm | embeddings |
|---|---|---|---|---|---|---|---|
| SmolLM2-135M | `HuggingFaceTB/SmolLM2-135M-Instruct` | `12fd25f7` | 30 | 576 | 49 152 | RMSNorm | tied |
| LLaMA-160M | `JackFram/llama-160m` | `aca9b687` | 12 | 768 | 32 000 | RMSNorm | untied |
| GPT-2 | `gpt2` | `607a30d7` | 12 | 768 | 50 257 | LayerNorm | tied |
| Pythia-160M | `EleutherAI/pythia-160m` | `50f5173d` | 12 | 768 | 50 304 | LayerNorm | untied |
| Pythia-70M | `EleutherAI/pythia-70m` | `a39f36b1` | 6 | 512 | 50 304 | LayerNorm | untied |
| GPT-Neo-125M | `EleutherAI/gpt-neo-125m` | `21def018` | 12 | 768 | 50 257 | LayerNorm | tied |

Revisions are the first eight characters of the commit hash resolved on 29 October 2026.

## 3. Data

| source | what | where |
|---|---|---|
| hand-written probe corpus | 148 sentences | `fisherrao/corpus.py`, the default source |
| WikiText-103 | validation split (`Salesforce/wikitext`, `wikitext-103-raw-v1`), sentence-split and length-bounded, sampled with a seeded permutation | `corpus.sentences(source="wikitext")` |
| same-sense minimal pairs | 32 words, sense change against a matched same-sense context | `run_samesense.py` |
| different-sense pairs | 64 hand-written pairs over 8 sense domains | `run_polysemy.py` |
| WiC | SuperGLUE WiC validation split, 319 different-sense and 319 same-sense pairs | `run_wic.py` |

⚠️ The submitted paper says WikiText-**2**. The code uses **WikiText-103** validation; the revision must say so.

## 4. Fixed settings

| setting | value |
|---|---|
| retained vocabulary | the 512 most probable tokens at the base point, renormalised, frozen before any derivative is taken |
| null directions | quotiented with `P = I − NNᵀ`: one for RMSNorm, two for LayerNorm |
| frame Jacobian | `metrics.frame_jacobian`: the RMSNorm closed form on RMSNorm models, the exact LayerNorm form on LayerNorm models (fixed 29 October 2026; before that the RMSNorm form was used everywhere, an error of about 6e−5 in `K` — [14-revision-log.md](14-revision-log.md) §10) |
| slice | `h₀ + span(F)`, `F` the top-`k` eigenvectors of `PGP` at the base point; `k` = 5, with 4 and 6 for sweeps; Riemann quantities are capped at `k` ≤ 8 |
| planes | 16 random 2-planes per state from a generator seeded 0 |
| per-state `K` | `torch.median` of the 16 plane values — the lower of the two middle values. Every saved `K` in the project uses this definition |
| summary statistics | true median (`fisherrao.stats.median`), midrank Spearman, sign test with `z = (pos − n/2) / (0.5√n)`, tie-corrected Mann–Whitney |
| layers | SmolLM2: 1, 5, 10, 15, 20, 25, 28, 29, 30. Other models: nine relative depths, `run_profiles.depths_for` |
| scramble | per state, a permutation of the retained rows from a generator seeded `1000 + state index` |
| Gaussian control | one `N × d` Gaussian matrix, seed 0, scaled per state by bisection until the entropy matches to 10⁻¹⁵ |
| entropy bands | [0, 0.5), [0.5, 1), [1, 2), [2, 4), [4, ∞) nats |
| duplicates | raw checkpoints can hold a state twice; every statistic is computed on distinct states ([audit_duplicates.py](audit_duplicates.py)) |

## 5. Every result: script, output, cost

Runtimes are wall-clock on the machine in §1.

| paper element | command | output | runtime |
|---|---|---|---|
| validation ladder | `python check_ladder_precision.py` | `results/ladder/ladder_precision.json` | minutes |
| rung 7, the pullback | `python validate_pullback.py > results/validation/validate_pullback.txt` | `results/validation/validate_pullback.txt` | seconds |
| independent implementation | `python validate_geomstats.py > results/validation/validate_geomstats.txt` | `results/validation/validate_geomstats.txt` | minutes |
| KL–Hessian identity, synthetic | `python gate_a_kl_test.py > results/validation/gate_a_kl_test.txt` | `results/validation/gate_a_kl_test.txt` | seconds |
| KL–Hessian identity, real model | `python run_stage0.py` | `results/stage0/stage0.json` (`gate_a`) | minutes |
| LayerNorm Jacobian check | `python check_layernorm_projection.py gpt2 12` (and Pythia-160M) | `results/layernorm/` | minutes |
| null directions, six models | `python check_null_all.py` | `results/nulltest/nulltest.json` | minutes |
| `K` across architectures | `python run_profiles.py` | `results/profiles/` | about 3 s per state at `k` = 5 |
| paper's scramble (`within`) | `python run_scramble_within.py` | `results/scramble/within_points.jsonl` | about 3 s per state per condition |
| consistent scramble, SmolLM2 | `python run_scramble_consistent.py --points paper --ksweep` | `results/scramble/consistent_SmolLM2-135M-Instruct_paper*.json*` | about 8 s per state for three conditions at `k` = 5, about 30 s with the `k` sweep |
| consistent scramble, LLaMA-160M, GPT-2, Pythia-160M | `python run_scramble_consistent.py --model <id> --per-bin 24 --ksweep`, with `<id>` = `JackFram/llama-160m`, `gpt2`, `EleutherAI/pythia-160m` | `results/scramble/consistent_<model>_draw*.json*` | as above; 36 and 39 min for GPT-2 and Pythia-160M |
| Gaussian control, entropy-matched | `python run_gaussian_matched.py --points paper`, and `--model <id> --points draw` for the other three | `results/scramble/gaussian_*` | 4–5 s per state |
| Gaussian control without the LayerNorm bias | `python check_gaussian_bias.py gpt2` (and `EleutherAI/pythia-160m`) | `results/scramble/gaussian_nobias_*` | about 5 min per model |
| control validity (Gate A off the base point, Jacobian, conditioning) | `python check_control_validity.py` | `results/scramble/control_validity.json` | about 2 min |
| slice against manifold curvature | `python check_slice_gauss.py 20` | `results/slice/slice_gauss.json` | 2–6 min |
| spectral profiles | `python run_volume_scramble.py 40 [<id>]`, SmolLM2 by default | `results/volume_scramble/volume_scramble.json` (SmolLM2), `volume_scramble_<model>.json` | 4–7 min per model |
| instrument agreement (E2) | `python run_stage4.py` | `results/stage4/stage4.json` | about 3 s per state |
| two corpora | `python run_corpus_compare.py` | `results/corpus_compare/` | about 3 s per state |
| behavioural probes | `python run_samesense.py`, `run_polysemy.py`, `run_wic.py` | `results/samesense/`, `results/polysemy/`, `results/wic/` | minutes |
| near-flat calibration | `python run_mabrok_replication.py` | `results/mabrok/mabrok.json` | minutes |
| duplicate audit | `python audit_duplicates.py` | `results/audit/duplicates.json` | seconds |
| every table | `python make_tables.py` | `results/tables/tables.md`, `tables.json` | seconds |
| figures | `python make_figures.py` | `paper/figures/fig1_pipeline.png`, `fig2_controls.png`, `fig3_spectral.png` | seconds |
| every number in the paper | `python check_paper_numbers.py` | console: 96 checks | seconds |
| Word manuscript | `python build_paper_docx.py` (it reads the IEEE template from the repository history), then export the PDF from Word | `paper/Fisher-Rao-Curvature-IEEE.docx`, `.pdf` | seconds |

Checkpointed scripts (`run_*` with a `_points.jsonl`) resume where they stopped. Re-running a finished one only rebuilds its summary.

## 6. Determinism

- Re-running the paired controls reproduces the saved per-state `K` to 3 × 10⁻⁶ for the real model and to 2.5 × 10⁻⁴ for the paper's scramble. The larger figure comes from ill-conditioned low-entropy states, where that scramble's `K` reaches −270.
- The slice is chosen by an eigendecomposition, so where `λ_k ≈ λ_{k+1}` a rerun can select a different `k`-th direction. That is the main source of run-to-run variation, and it is why summary statistics are medians over states.
- LLaMA-160M's `<s>` position has a hidden state that is exactly zero at every layer, so its prediction is uniform (entropy ln 32 000) and the quotient metric is undefined there; `eigh` fails. Such states are recorded as errors and excluded: 2 of 120 in the controls and 25 of 360 in the spectral control ([14-revision-log.md](14-revision-log.md) §12).

## 7. Code for double-blind review

Before submission, mirror the repository to an anonymous host (for example anonymous.4open.science) and cite that link. The public repository names the author. See the anonymity item of the checklist in [13-revision-plan.md](13-revision-plan.md) §8.
