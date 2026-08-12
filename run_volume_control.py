"""
Is the layer-wise volume profile a model finding, or another concentration effect?

07-stage4-log.md S2.2 reports a U-shaped sphere-corrected log-volume profile with
a minimum near layer 20.  Two deflationary explanations must be ruled out before
that can be called a finding about the network:

  (a) ENTROPY.  Volume and entropy both derive from p(h).  Entropy varies with
      depth (1.13 -> 3.86 across layers here), so the "layer profile" could be
      the entropy profile in disguise -- the same failure that reduced RQ3a to a
      definitional relationship (08-rq3a-log.md S5).

  (b) LEARNED STRUCTURE.  Shuffling U's rows destroys every token-to-direction
      assignment while preserving the spectrum exactly.  If the shuffled model
      reproduces the same layer profile, the profile is not about what the model
      learned.

Volume needs no Riemann tensor, so this runs on the full corpus cheaply.

Usage:  python run_volume_control.py [n_positions]
"""

from __future__ import annotations

import copy
import json
import pathlib
import sys

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fisherrao import LM, corpus
from fisherrao.curvature import spectral_diagnostics
from fisherrao.stats import spearman as _spearman

OUT = pathlib.Path("results/volume_control")
K_DIM, TOP_K = 5, 512
LAYERS = [1, 5, 10, 15, 20, 25, 28, 29, 30]


def spearman(a, b):
    """Tie-corrected Spearman -- see fisherrao/stats.py for why this is
    not argsort(argsort(x))."""
    return _spearman(a, b)
def sweep(lm, states_by_layer, tag):
    rows = []
    for l in LAYERS:
        for h in states_by_layer[l]:
            sd = spectral_diagnostics(lm, h, K_DIM, top_k=TOP_K)
            lp = torch.log_softmax(lm.logits(h), -1)
            rows.append(dict(layer=l, vol=sd["log_vol_per_dim"],
                             entropy=float(-(lp.exp() * lp).sum())))
    # sphere correction: +log r per dimension (see curvature.volume_element)
    return rows


def profile(rows, key="vol"):
    return {l: float(torch.tensor([r[key] for r in rows if r["layer"] == l],
                                  dtype=torch.float64).median()) for l in LAYERS}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n_pos = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    gen = torch.Generator().manual_seed(0)

    lm = LM()
    print(lm.summary())
    # collect states, subsample positions for speed
    allpos = []
    for s in corpus.sentences():
        H, _ = lm.residual_stream(s)
        for t in range(H.shape[1]):
            allpos.append(H[:, t, :])
    pick = torch.randperm(len(allpos), generator=gen)[:n_pos].tolist()
    states = {l: [allpos[i][l] for i in pick] for l in LAYERS}
    print(f"\n{len(pick)} positions x {len(LAYERS)} layers per condition\n")

    conds = {}
    print("  [real]"); conds["real"] = sweep(lm, states, "real")
    lm_s = copy.copy(lm)
    lm_s.U = lm.U[torch.randperm(lm.U.shape[0], generator=gen)].contiguous()
    print("  [shuffled]"); conds["shuffled"] = sweep(lm_s, states, "shuffled")
    lm_r = copy.copy(lm)
    lm_r.U = torch.randn(lm.N, lm.d, generator=gen, dtype=torch.float64) * 4.0 / lm.d**0.5
    print("  [random]"); conds["random"] = sweep(lm_r, states, "random")

    # ---- (a) is the profile just entropy? ---------------------------------
    print(f"\n{'='*72}\n(a) IS THE LAYER PROFILE JUST ENTROPY?\n{'='*72}")
    R = conds["real"]
    print(f"  Spearman(volume, entropy) pooled: {spearman([r['vol'] for r in R], [r['entropy'] for r in R]):+.3f}")
    print(f"  Spearman(volume, layer)   pooled: {spearman([r['vol'] for r in R], [r['layer'] for r in R]):+.3f}")
    print(f"  Spearman(entropy, layer)  pooled: {spearman([r['entropy'] for r in R], [r['layer'] for r in R]):+.3f}")

    print("\n  Layer profile WITHIN entropy bins (holds entropy roughly constant):")
    bins = [(0.5, 1.0), (1.0, 1.5), (1.5, 2.5)]
    print(f"  {'entropy bin':<14} " + "".join(f"{'L'+str(l):>9}" for l in LAYERS))
    survived = []
    for lo, hi in bins:
        g = [r for r in R if lo <= r["entropy"] < hi]
        cells, vals = "", []
        for l in LAYERS:
            v = [r["vol"] for r in g if r["layer"] == l]
            if len(v) >= 5:
                m = float(torch.tensor(v, dtype=torch.float64).median())
                vals.append((l, m)); cells += f"{m:>9.3f}"
            else:
                cells += f"{'-':>9}"
        print(f"  [{lo:.1f},{hi:.1f})     {cells}   n={len(g)}")
        if len(vals) >= 4:
            lo_l = min(vals, key=lambda x: x[1]); hi_l = max(vals, key=lambda x: x[1])
            survived.append((lo, hi, lo_l, hi_l, hi_l[1] - lo_l[1]))
    for lo, hi, lmin, lmax, rng in survived:
        print(f"    [{lo:.1f},{hi:.1f}) : min at layer {lmin[0]}, max at layer {lmax[0]}, range {rng:.3f}")

    # ---- (b) does structure-free reproduce it? ----------------------------
    print(f"\n{'='*72}\n(b) DOES A STRUCTURE-FREE MODEL REPRODUCE THE PROFILE?\n{'='*72}")
    profs = {k: profile(v) for k, v in conds.items()}
    print(f"  {'layer':>6} " + "".join(f"{k:>12}" for k in profs))
    for l in LAYERS:
        print(f"  {l:>6} " + "".join(f"{profs[k][l]:>12.3f}" for k in profs))
    print(f"\n  {'condition':<12} {'min layer':>10} {'max layer':>10} {'range':>8}")
    shapes = {}
    for k, p in profs.items():
        lmin = min(LAYERS, key=lambda l: p[l]); lmax = max(LAYERS, key=lambda l: p[l])
        shapes[k] = dict(min_layer=lmin, max_layer=lmax, range=p[lmax] - p[lmin])
        print(f"  {k:<12} {lmin:>10} {lmax:>10} {p[lmax]-p[lmin]:>8.3f}")

    rv = [profs["real"][l] for l in LAYERS]
    sv = [profs["shuffled"][l] for l in LAYERS]
    nv = [profs["random"][l] for l in LAYERS]
    print(f"\n  shape agreement (Spearman over the 9 layer medians):")
    print(f"     real vs shuffled : {spearman(rv, sv):+.3f}")
    print(f"     real vs random   : {spearman(rv, nv):+.3f}")

    same = shapes["real"]["min_layer"] == shapes["shuffled"]["min_layer"]
    print(f"\n  VERDICT: shuffled minimum at layer {shapes['shuffled']['min_layer']}, "
          f"real at {shapes['real']['min_layer']} -> {'SAME' if same else 'DIFFERENT'}")
    if spearman(rv, sv) > 0.7:
        print("  The structure-free control reproduces the profile shape.  The layer")
        print("  profile is NOT a finding about learned structure.")
    else:
        print("  The profile shape differs from the structure-free control -- something")
        print("  beyond concentration is present.")

    (OUT / "volume_control.json").write_text(json.dumps(
        dict(profiles=profs, shapes=shapes,
             rho_real_shuffled=spearman(rv, sv), rho_real_random=spearman(rv, nv)),
        indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'volume_control.json'}")


if __name__ == "__main__":
    main()
