"""
THE CORRECTED definitional control.

run_definitional.py used a "shuffled U rows" condition that is mathematically the
IDENTITY on the Fisher metric (metrics.fisher_metric_scrambled documents why:
||dG||/||G|| = 1.5e-12, entropies identical to 10 decimals).  It destroyed
nothing, and the apparent "shuffled retains 83%" result was pure subsampling
noise from the generator advancing between conditions.  That conclusion is
retracted; this script replaces it.

The genuine control keeps p EXACTLY as the real model produces it, then pairs
those probabilities with PERMUTED directions.  Entropy is matched to machine
precision, so any curvature difference is attributable to the learned
token-to-direction assignment alone -- which is exactly the definitional
question.

Paired design: the same hidden states under both conditions, so differences are
not subsampling noise.

Usage:  python run_scramble_control.py [n]
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
from fisherrao.metrics import (fisher_metric, fisher_metric_projected,
                               topk_indices)
from fisherrao.curvature import riemann, sectional, scalar_curvature

OUT = pathlib.Path("results/scramble")
K_DIM, TOP_K, N_PLANES = 5, 512, 16
LAYERS = [1, 5, 10, 15, 20, 25, 28, 29, 30]


def curvature_from(lm, h, k, idx, perm=None, seed=0):
    """K median and scalar R, under the real metric (perm=None) or the scramble.

    MUST project into the frame BEFORE forming the metric.  Building the full
    (d, d) metric inside the autodiff makes double differentiation carry a
    576x576 intermediate; that route attempted a 20.6 GB allocation earlier in
    this project (03-methodology.md S6.0c) and simply hangs here.
    """
    # Each condition gets ITS OWN eigenframe.  Representing the scrambled metric
    # in the real metric's frame would put it in a badly-adapted basis and inflate
    # curvature through conditioning rather than geometry.  We are comparing two
    # geometries, each on its own effective subspace.
    d = h.shape[-1]
    Ffull = torch.eye(d, dtype=h.dtype)
    G0 = fisher_metric_projected(lm, h, Ffull, idx, perm=perm)
    hhat = h / torch.linalg.vector_norm(h)
    P = torch.eye(h.shape[-1], dtype=h.dtype) - torch.outer(hhat, hhat)
    ev, evec = torch.linalg.eigh(P @ G0 @ P)
    F = evec[:, torch.argsort(ev, descending=True)[:k]].contiguous()

    def g(x):
        return fisher_metric_projected(lm, h + F @ x, F, idx, perm=perm)

    x0 = torch.zeros(k, dtype=h.dtype)
    R = riemann(g, x0)
    gen = torch.Generator().manual_seed(seed)
    Ks = []
    for _ in range(N_PLANES):
        u = torch.randn(k, generator=gen, dtype=h.dtype)
        v = torch.randn(k, generator=gen, dtype=h.dtype)
        val = sectional(g, x0, u, v, R=R)
        if val == val:
            Ks.append(val)
    return (float(torch.tensor(Ks, dtype=h.dtype).median()) if Ks else float("nan"),
            scalar_curvature(g, x0, R=R))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    gen = torch.Generator().manual_seed(0)
    lm = LM()
    print(lm.summary())

    states = []
    for s in corpus.sentences():
        H, _ = lm.residual_stream(s)
        for l in LAYERS:
            for t in range(H.shape[1]):
                states.append((l, H[l, t]))
    pick = torch.randperm(len(states), generator=gen)[:n].tolist()
    perm = torch.randperm(lm.U.shape[0], generator=gen)
    print(f"\nPAIRED design: {n} states, each measured under BOTH conditions.\n")

    rows, t0 = [], time.time()
    for j, i in enumerate(pick, 1):
        layer, h = states[i]
        lp = torch.log_softmax(lm.logits(h), -1)
        ent = float(-(lp.exp() * lp).sum())
        # FREEZE the top-k index set at the base point in BOTH conditions.
        # Re-selecting it inside the autodiff makes the metric only piecewise
        # smooth (see metrics.topk_indices) -- and it is also far slower.
        idx = topk_indices(lm, h, TOP_K)
        try:
            kr, rr = curvature_from(lm, h, K_DIM, idx)
            ks, rs = curvature_from(lm, h, K_DIM, idx, perm=perm)
        except Exception:
            continue
        rows.append(dict(layer=layer, entropy=ent, K_real=kr, R_real=rr,
                         K_scr=ks, R_scr=rs))
        if j % 15 == 0:
            print(f"   {j}/{len(pick)}  ({time.time()-t0:.0f}s)")

    print(f"\n{'='*72}\nPAIRED COMPARISON at EXACTLY matched entropy\n{'='*72}")
    dK = torch.tensor([r["K_real"] - r["K_scr"] for r in rows], dtype=torch.float64)
    dR = torch.tensor([r["R_real"] - r["R_scr"] for r in rows], dtype=torch.float64)
    Kr = torch.tensor([r["K_real"] for r in rows], dtype=torch.float64)
    Ks_ = torch.tensor([r["K_scr"] for r in rows], dtype=torch.float64)
    print(f"  n = {len(rows)} paired points\n")
    print(f"  {'quantity':<22} {'real median':>13} {'scrambled':>12} {'paired diff':>13}")
    print(f"  {'sectional K':<22} {float(Kr.median()):>13.4f} {float(Ks_.median()):>12.4f} "
          f"{float(dK.median()):>+13.4f}")
    Rr = torch.tensor([r["R_real"] for r in rows], dtype=torch.float64)
    Rs_ = torch.tensor([r["R_scr"] for r in rows], dtype=torch.float64)
    print(f"  {'scalar R':<22} {float(Rr.median()):>13.3f} {float(Rs_.median()):>12.3f} "
          f"{float(dR.median()):>+13.3f}")

    # sign test -- nonparametric, robust to the heavy tail
    pos = int((dK > 0).sum()); tot = int((dK != 0).sum())
    print(f"\n  sign test on paired K differences: {pos}/{tot} positive "
          f"({100*pos/max(tot,1):.0f}%)   [50% = no effect]")
    se = 0.5 * tot ** 0.5
    z = (pos - tot / 2) / se if se else 0.0
    print(f"  z = {z:+.2f}   {'SIGNIFICANT' if abs(z) > 1.96 else 'not significant'}")

    def sp(a, b):
        x = torch.argsort(torch.argsort(torch.tensor(a, dtype=torch.float64))).double()
        y = torch.argsort(torch.argsort(torch.tensor(b, dtype=torch.float64))).double()
        return float(torch.corrcoef(torch.stack([x, y]))[0, 1])
    E = [r["entropy"] for r in rows]
    print(f"\n  rho(K, entropy)  real {sp([r['K_real'] for r in rows], E):+.3f}   "
          f"scrambled {sp([r['K_scr'] for r in rows], E):+.3f}")
    print(f"  rho(R, entropy)  real {sp([r['R_real'] for r in rows], E):+.3f}   "
          f"scrambled {sp([r['R_scr'] for r in rows], E):+.3f}")
    print("\n  Entropy is IDENTICAL in both conditions by construction, so any")
    print("  difference in these correlations is attributable to learned structure.")

    # Magnitude on the RATIO scale.  K spans orders of magnitude, so a sign test
    # alone is both underpowered and blind to effect size -- it reported "no
    # effect" while the medians differed by a factor of 18.
    rel = float(dK.abs().median() / Kr.abs().median())
    ratios = torch.tensor([abs(r["K_scr"]) / abs(r["K_real"])
                           for r in rows if r["K_real"] != 0], dtype=torch.float64)
    print(f"\n  median |paired K difference| / median K: {rel:.1%}")
    print(f"  paired |K_scrambled / K_real|: median {float(ratios.median()):.3f}x   "
          f"IQR [{float(torch.quantile(ratios, 0.25)):.3f}, "
          f"{float(torch.quantile(ratios, 0.75)):.3f}]")
    n_big = int(((ratios > 2) | (ratios < 0.5)).sum())
    print(f"  {n_big}/{len(ratios)} points differ by more than 2x in either direction")

    if n_big / max(len(ratios), 1) > 0.5:
        print("  -> LEARNED STRUCTURE MATTERS: destroying the token-to-direction")
        print("     assignment changes curvature by a large factor at EXACTLY")
        print("     matched entropy, so the geometry is NOT a function of")
        print("     concentration alone.")
    elif abs(z) > 1.96:
        print("  -> a consistent but modest effect of learned structure.")
    else:
        print("  -> no detectable effect at matched entropy.")

    print("\n  !! TWO CAVEATS, BOTH SINCE RESOLVED ELSEWHERE -- see below !!")
    print("  1. SUBSPACE, not frame.  Each condition is measured in ITS OWN")
    print("     eigenframe, and measuring one metric in the other's frame gives")
    print("     wildly different numbers.  That is NOT a frame sensitivity:")
    print("     rotation invariance holds at 1e-14 in both conditions (n=24).")
    print("     The two frames are different SUBSPACES, each holding ~88% of its")
    print("     own metric's trace and ~4% of the other's, and two INDEPENDENT")
    print("     scrambles do the same thing to each other -- so the cross-frame")
    print("     blow-up carries no information.  Own-frame is correct.")
    print("     -> 08-rq3a-log.md S5.4, run_frame_resolution.py, diag_frame.py")
    print("  2. THIS SCRIPT'S SCRAMBLE IS THE COMPOUND ONE.  A whole-vocabulary")
    print("     perm also swaps the retained row SET (overlap 5/512, median")
    print("     ||U row|| 2.44 -> 3.10), so it does not isolate the learned")
    print("     assignment on its own.  The clean within-set control gives a")
    print("     collapse of the same size (K 0.2555 -> -0.0014, n=60), so the")
    print("     conclusion holds -- but quote the within-set numbers, not these.")
    print("     -> run_scramble_within.py, 08-rq3a-log.md S5.4(b')")

    (OUT / "scramble.json").write_text(json.dumps(dict(
        n=len(rows), K_real_median=float(Kr.median()), K_scr_median=float(Ks_.median()),
        R_real_median=float(Rr.median()), R_scr_median=float(Rs_.median()),
        paired_dK_median=float(dK.median()), sign_pos=pos, sign_n=tot, z=z,
        rel_effect=rel, rows=rows), indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'scramble.json'}")


if __name__ == "__main__":
    main()
