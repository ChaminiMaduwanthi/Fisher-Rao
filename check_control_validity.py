"""
Is each scramble control a valid geometric object?  The three points a reviewer raised.

1. DERIVATIVE CONSTRUCTION / LOCAL PROBABILITY MAP.  A Fisher pullback G of a map
   x -> p(x) is the Hessian of the KL divergence of that map:

       KL(p(x1) || p(x1 + eps v)) = 1/2 eps^2 v^T G(x1) v + O(eps^3)

   Tested at the base point and at a point x1 off it, where `within` and
   `consistent` stop agreeing:
       real        G_real    against KL of p_real           must hold
       consistent  G_cons    against KL of p_cons           must hold
       within      G_within  against KL of p_real, p_cons   holds for neither off h0
                                                            if it is not a pullback

2. JACOBIAN.  fisher_metric_projected uses the RMSNorm Jacobian with the projector
   I - hhat hhat^T, which drops eps.  The exact Jacobian is
       diag(g)/r (I - h h^T / (d r^2)),   d r^2 = |h|^2 + d eps.
   Measured: the size of the difference, and its effect on K (real and consistent).

3. CONDITIONING.  The paper says the scramble holds "the conditioning of the
   retained subspace" fixed.  Measured: lambda_1 / lambda_k of the top-k block of
   P G P, real against scrambled (at h0 `within` and `consistent` coincide).

Same states as the consistent-control pilot (6 per entropy band, SmolLM2-135M).
Usage:  python check_control_validity.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fisherrao import LM, corpus
from fisherrao.metrics import fisher_metric_projected, topk_indices
from fisherrao.curvature import null_projector
from run_scramble_consistent import (EDGES, SMOL_LAYERS, K_DIM, frame_for, median_K,
                                     metric_consistent, med, sign_test)

OUT = pathlib.Path("results/scramble")
N_PER_BIN, N_K_CHECK = 6, 6


def kl(lp, lq):
    """KL(p || q) from log-probabilities."""
    return float((lp.exp() * (lp - lq)).sum())


def gate_ratio(logp_fn, G_fn, x1, v, eps):
    """KL(p(x1) || p(x1 + eps v)) / (1/2 eps^2 v^T G(x1) v); 1 for a genuine pullback."""
    quad = 0.5 * eps ** 2 * float(v @ G_fn(x1) @ v)
    return kl(logp_fn(x1), logp_fn(x1 + eps * v)) / quad


def exact_Af(lm, h, F):
    d = h.shape[-1]
    r = torch.sqrt(h.pow(2).mean() + lm.eps)
    Af = (F - torch.outer(h, h @ F) / (d * r ** 2)) / r
    return lm.g[:, None] * Af if lm.g is not None else Af


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lm = LM()
    print(lm.summary())
    states = []
    for s in corpus.sentences():
        H, _ = lm.residual_stream(s)
        for l in SMOL_LAYERS:
            for t in range(H.shape[1]):
                states.append((l, H[l, t]))
    ents = []
    for _, h in states:
        lp = torch.log_softmax(lm.logits(h), -1)
        ents.append(float(-(lp.exp() * lp).sum()))
    gen = torch.Generator().manual_seed(0)
    pick = []
    for lo, hi in EDGES:
        band = [i for i, e in enumerate(ents) if lo <= e < hi]
        take = torch.randperm(len(band), generator=gen)[:N_PER_BIN].tolist()
        pick += [band[t] for t in take]

    # one autodiff check that exact_Af really is the Jacobian of apply_norm
    h_chk = states[pick[0]][1]
    J_auto = torch.autograd.functional.jacobian(lm.apply_norm, h_chk)
    F_chk = torch.linalg.qr(torch.randn(h_chk.shape[0], K_DIM, dtype=torch.float64,
                                        generator=torch.Generator().manual_seed(7)))[0]
    jac_ok = float((J_auto @ F_chk - exact_Af(lm, h_chk, F_chk)).norm()
                   / (J_auto @ F_chk).norm())
    print(f"exact Jacobian vs autodiff: {jac_ok:.1e}")

    rows, t0 = [], time.time()
    vgen = torch.Generator().manual_seed(1)
    for j, i in enumerate(pick):
        layer, h0 = states[i]
        idx = topk_indices(lm, h0, 512)
        P = null_projector(lm, h0)
        within = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(1000 + i))]
        F_r = frame_for(lm, h0, idx, P, K_DIM)
        F_s = frame_for(lm, h0, idx, P, K_DIM, rows=within)
        n0 = lm.apply_norm(h0)
        z0 = lm.U[idx] @ n0
        scale = float(h0.norm())

        logp_real_r = lambda x: torch.log_softmax(lm.U[idx] @ lm.apply_norm(h0 + F_r @ x), -1)
        logp_real_s = lambda x: torch.log_softmax(lm.U[idx] @ lm.apply_norm(h0 + F_s @ x), -1)
        logp_cons = lambda x: torch.log_softmax(
            z0 + lm.U[within] @ (lm.apply_norm(h0 + F_s @ x) - n0), -1)
        G_real = lambda x: fisher_metric_projected(lm, h0 + F_r @ x, F_r, idx)
        G_within = lambda x: fisher_metric_projected(lm, h0 + F_s @ x, F_s, idx, rows=within)
        G_cons = lambda x: metric_consistent(lm, h0 + F_s @ x, h0, F_s, idx, within)

        w = torch.randn(K_DIM, generator=vgen, dtype=torch.float64)
        v = torch.randn(K_DIM, generator=vgen, dtype=torch.float64)
        w, v = w / w.norm(), v / v.norm()
        x0 = torch.zeros(K_DIM, dtype=torch.float64)
        x1 = 0.05 * scale * w                       # off the base point
        eps = 1e-4 * scale
        rec = dict(_i=i, layer=layer, entropy=ents[i])
        for where, x in (("h0", x0), ("x1", x1)):
            rec[f"{where}_real"] = gate_ratio(logp_real_r, G_real, x, v, eps)
            rec[f"{where}_cons"] = gate_ratio(logp_cons, G_cons, x, v, eps)
            rec[f"{where}_within_vs_real"] = gate_ratio(logp_real_s, G_within, x, v, eps)
            rec[f"{where}_within_vs_cons"] = gate_ratio(logp_cons, G_within, x, v, eps)
        rec["kl_real_vs_cons_at_x1"] = kl(logp_real_s(x1), logp_cons(x1))

        A_fast, A_ex = fisher_metric_projected(lm, h0 + F_r @ x1, F_r, idx), None
        Af_c = (lambda h: (F_r - torch.outer(h / h.norm(), (h / h.norm()) @ F_r))
                / torch.sqrt(h.pow(2).mean() + lm.eps))(h0 + F_r @ x1)
        Af_e = exact_Af(lm, h0 + F_r @ x1, F_r)
        if lm.g is not None:
            Af_c = lm.g[:, None] * Af_c
        rec["jac_rel_diff_x1"] = float((Af_c - Af_e).norm() / Af_e.norm())

        if j < N_K_CHECK:
            def G_real_ex(x):
                h = h0 + F_r @ x
                M = lm.U[idx] @ exact_Af(lm, h, F_r)
                p = torch.softmax(lm.U[idx] @ lm.apply_norm(h), -1)
                Mp = M.T @ p
                return (M.T * p) @ M - torch.outer(Mp, Mp)

            def G_cons_ex(x):
                h = h0 + F_s @ x
                M = lm.U[within] @ exact_Af(lm, h, F_s)
                p = torch.softmax(z0 + lm.U[within] @ (lm.apply_norm(h) - n0), -1)
                Mp = M.T @ p
                return (M.T * p) @ M - torch.outer(Mp, Mp)
            rec["dK_real_exact_jac"] = median_K(G_real_ex, K_DIM) - median_K(G_real, K_DIM)
            rec["dK_cons_exact_jac"] = median_K(G_cons_ex, K_DIM) - median_K(G_cons, K_DIM)

        for name, rows_kw in (("real", {}), ("scr", dict(rows=within))):
            d = h0.shape[-1]
            G = fisher_metric_projected(lm, h0, torch.eye(d, dtype=h0.dtype), idx, **rows_kw)
            ev = torch.linalg.eigvalsh(P @ G @ P).flip(0)[:K_DIM]
            rec[f"cond_{name}"] = float(ev[0] / ev[-1])
        rows.append(rec)
        print(f"   {j+1:>2}/{len(pick)}  H={ents[i]:4.2f}  x1: real {rec['x1_real']:.4f}  "
              f"cons {rec['x1_cons']:.4f}  within/real {rec['x1_within_vs_real']:.3f}  "
              f"within/cons {rec['x1_within_vs_cons']:.3f}   cond {rec['cond_real']:.1f} vs "
              f"{rec['cond_scr']:.1f}   ({time.time()-t0:.0f}s)", flush=True)

    dev = lambda key: med([abs(r[key] - 1) for r in rows])
    summary = dict(n=len(rows), exact_jacobian_vs_autodiff=jac_ok,
                   gate_median_abs_dev={k: dev(k) for k in rows[0] if k.startswith(("h0_", "x1_"))},
                   kl_real_vs_cons_at_x1=med([r["kl_real_vs_cons_at_x1"] for r in rows]),
                   jac_rel_diff_x1=med([r["jac_rel_diff_x1"] for r in rows]),
                   dK_exact_jac=dict(
                       real=med([abs(r["dK_real_exact_jac"]) for r in rows if "dK_real_exact_jac" in r]),
                       cons=med([abs(r["dK_cons_exact_jac"]) for r in rows if "dK_cons_exact_jac" in r])),
                   cond_real=med([r["cond_real"] for r in rows]),
                   cond_scr=med([r["cond_scr"] for r in rows]),
                   cond_scr_gt_real=sign_test([r["cond_scr"] - r["cond_real"] for r in rows]))
    print(f"\n{'='*76}\nCONTROL VALIDITY   n = {len(rows)}\n{'='*76}")
    print("   median |KL / (1/2 eps^2 v'Gv) - 1|      at h0        at x1 (off h0)")
    for k in ("real", "cons", "within_vs_real", "within_vs_cons"):
        print(f"     {k:<18} {summary['gate_median_abs_dev']['h0_'+k]:>14.2e} "
              f"{summary['gate_median_abs_dev']['x1_'+k]:>14.2e}")
    print(f"   KL(p_real || p_cons) at x1: median {summary['kl_real_vs_cons_at_x1']:.3e}")
    print(f"   closed-form vs exact RMSNorm Jacobian: median rel. diff {summary['jac_rel_diff_x1']:.1e};"
          f" |dK| real {summary['dK_exact_jac']['real']:.1e}, consistent {summary['dK_exact_jac']['cons']:.1e}")
    c = summary["cond_scr_gt_real"]
    print(f"   conditioning lambda1/lambda{K_DIM}: real {summary['cond_real']:.1f}, scrambled "
          f"{summary['cond_scr']:.1f}; scrambled > real {c['pos']}/{c['n']}, z = {c['z']:+.2f}")
    (OUT / "control_validity.json").write_text(json.dumps(dict(summary=summary, rows=rows), indent=2),
                                               encoding="utf-8")
    print(f"\nwrote {OUT / 'control_validity.json'}")


if __name__ == "__main__":
    main()
