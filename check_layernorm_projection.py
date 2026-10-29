"""
Does the projected metric use the right Jacobian on LayerNorm models?

metrics.fisher_metric_projected builds the frame Jacobian in closed form,

    A F = diag(g) (F - hhat hhat^T F) / r,    r = sqrt(mean(h^2) + eps)

which is the RMSNorm Jacobian.  curvature.metric_in_frame calls it for every
model, so every Riemann-derived number on GPT-2 and Pythia (LayerNorm) goes
through it.  The LayerNorm Jacobian (LM.norm_jacobian, verified against
autograd in norm_check) is

    A F = diag(g) (F - 1 (1^T F)/d - c (c^T F)/(d s^2)) / s,
          c = h - mean(h),  s = sqrt(var(h) + eps)

The probabilities inside fisher_metric_projected come from LM.logits, which
uses the correct LayerNorm, so on a LayerNorm model the metric pairs a correct
probability map with an RMSNorm Jacobian.

At the base point the frame is orthogonal to both null directions, so the two
Jacobians differ only by the factor s/r there; away from it they differ in the
projection too.  This script measures, on real states:

  1. the exact closed form against LM.norm_jacobian (must agree to ~1e-15)
  2. the projected metric against F^T G F from the full, correct fisher_metric,
     at h0 and off it
  3. sectional curvature from the projected metric against the exact one

"projected" below is the RMSNorm form as it was (rms_form_Af).  The package now
uses the exact form for LayerNorm; "package_vs_full" confirms that it matches.

Usage:  python check_layernorm_projection.py [model_id] [n_states]
"""

from __future__ import annotations

import json
import pathlib
import sys

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fisherrao import LM, corpus
from fisherrao.metrics import fisher_metric, fisher_metric_projected, topk_indices
from fisherrao.curvature import null_projector
from fisherrao.stats import median
from run_scramble_consistent import depths_for, median_K, K_DIM, TOP_K

OUT = pathlib.Path("results/layernorm")


def exact_Af(lm, h, F):
    """Exact frame Jacobian for either norm, in closed form."""
    d = h.shape[-1]
    if "RMS" in lm.norm_kind:
        r = torch.sqrt(h.pow(2).mean() + lm.eps)
        Af = (F - torch.outer(h, h @ F) / (d * r ** 2)) / r
    else:
        c = h - h.mean()
        s2 = h.var(unbiased=False) + lm.eps
        Af = (F - torch.outer(torch.ones(d, dtype=h.dtype), F.sum(0)) / d
              - torch.outer(c, c @ F) / (d * s2)) / torch.sqrt(s2)
    return Af if lm.g is None else lm.g[:, None] * Af


def rms_form_Af(lm, h, F):
    """The RMSNorm closed form fisher_metric_projected applied to every model before
    29 October 2026 -- kept here so the size of that error stays measurable."""
    r = torch.sqrt(h.pow(2).mean() + lm.eps)
    hhat = h / torch.linalg.vector_norm(h)
    Af = (F - torch.outer(hhat, hhat @ F)) / r
    return Af if lm.g is None else lm.g[:, None] * Af


def metric_with(Af_fn, lm, h, F, idx):
    M = lm.U[idx] @ Af_fn(lm, h, F)
    p = torch.softmax(lm.logits(h)[idx] - lm.logits(h)[idx].max(), -1)
    Mp = M.T @ p
    return (M.T * p) @ M - torch.outer(Mp, Mp)


def metric_exact(lm, h, F, idx):
    return metric_with(exact_Af, lm, h, F, idx)


def metric_rms_form(lm, h, F, idx):
    return metric_with(rms_form_Af, lm, h, F, idx)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    lm = LM(model)
    print(lm.summary())
    layers = depths_for(lm.n_layers)
    states = []
    for s in corpus.sentences()[:20]:
        H, _ = lm.residual_stream(s)
        for l in layers:
            states.append((l, H[l, H.shape[1] // 2]))
    gen = torch.Generator().manual_seed(0)
    pick = torch.randperm(len(states), generator=gen)[:n].tolist()

    rows = []
    for j, i in enumerate(pick):
        layer, h0 = states[i]
        idx = topk_indices(lm, h0, TOP_K)
        P = null_projector(lm, h0)
        G = fisher_metric(lm, h0, idx=idx)
        ev, evec = torch.linalg.eigh(P @ G @ P)
        F = evec[:, torch.argsort(ev, descending=True)[:K_DIM]].contiguous()

        jac = float((lm.norm_jacobian(h0) @ F - exact_Af(lm, h0, F)).norm()
                    / (lm.norm_jacobian(h0) @ F).norm())
        x = 0.05 * float(h0.norm()) * torch.nn.functional.normalize(
            torch.randn(K_DIM, generator=gen, dtype=torch.float64), dim=0)
        diffs = {}
        for where, xx in (("h0", torch.zeros(K_DIM, dtype=torch.float64)), ("x1", x)):
            full = F.T @ fisher_metric(lm, h0 + F @ xx, idx=idx) @ F
            proj = metric_rms_form(lm, h0 + F @ xx, F, idx)
            pkg = fisher_metric_projected(lm, h0 + F @ xx, F, idx)
            exa = metric_exact(lm, h0 + F @ xx, F, idx)
            diffs[f"proj_vs_full_{where}"] = float((proj - full).norm() / full.norm())
            diffs[f"package_vs_full_{where}"] = float((pkg - full).norm() / full.norm())
            diffs[f"exact_vs_full_{where}"] = float((exa - full).norm() / full.norm())
        r = float(torch.sqrt(h0.pow(2).mean() + lm.eps))
        s = float(torch.sqrt(h0.var(unbiased=False) + lm.eps))
        K_proj = median_K(lambda xx: metric_rms_form(lm, h0 + F @ xx, F, idx), K_DIM)
        K_exact = median_K(lambda xx: metric_exact(lm, h0 + F @ xx, F, idx), K_DIM)
        rec = dict(_i=i, layer=layer, r=r, s=s, jac_closed_vs_lm=jac, K_projected=K_proj,
                   K_exact=K_exact, **diffs)
        rows.append(rec)
        print(f"   {j+1:>2}/{n}  L{layer:<2} r/s={r/s:7.3f}  proj-vs-full h0 {diffs['proj_vs_full_h0']:.1e} "
              f"x1 {diffs['proj_vs_full_x1']:.1e} | exact-vs-full x1 {diffs['exact_vs_full_x1']:.1e}  "
              f"K proj {K_proj:+.4f}  K exact {K_exact:+.4f}", flush=True)

    summ = dict(model=model, n=len(rows),
                jac_closed_vs_lm=max(r["jac_closed_vs_lm"] for r in rows),
                median_r_over_s=median([r["r"] / r["s"] for r in rows]),
                proj_vs_full_h0=median([r["proj_vs_full_h0"] for r in rows]),
                proj_vs_full_x1=median([r["proj_vs_full_x1"] for r in rows]),
                exact_vs_full_x1=max(r["exact_vs_full_x1"] for r in rows),
                package_vs_full_x1=max(r["package_vs_full_x1"] for r in rows),
                K_projected=median([r["K_projected"] for r in rows]),
                K_exact=median([r["K_exact"] for r in rows]),
                median_abs_dK=median([abs(r["K_projected"] - r["K_exact"]) for r in rows]))
    print(f"\n{model}: {json.dumps(summ, indent=1)}")
    (OUT / f"projection_{model.split('/')[-1]}.json").write_text(
        json.dumps(dict(summary=summ, rows=rows), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
