"""
Is the layer-wise VOLUME profile definitional too?

08-rq3a-log.md S7.4 flagged this and it stayed open: the same "is it the model
or is it concentration?" question that applies to curvature applies to the
volume element, because both are functions of the same p(h).  The U-shaped
profile with a minimum at layer 20 (07-stage4-log.md S2.2) is the project's
PRIMARY layer-wise quantity, chosen precisely because it is stable where
Riemann-derived quantities are fragile -- so if it turns out to be a restatement
of how concentrated the predictive distribution is at each depth, that matters a
great deal.

The control is the one that settled the curvature question (08-rq3a-log.md
S5.4): keep p EXACTLY as the real model produces it and pair those probabilities
with PERMUTED directions.  Entropy is matched to machine precision.

BOTH scramble conditions are run, because they are not the same control and an
earlier version of this script reported only the second:

    within   rows = idx[randperm(len(idx))].  The retained row multiset is
             IDENTICAL to the real one, so the ONLY thing destroyed is which
             probability sits on which direction.  This is the clean one.
    global   rows = perm[idx] for a whole-vocabulary perm.  Overlap with the
             real top-512 is 5/512 and median ||U row|| shifts 2.44 -> 3.10, so
             this destroys the direction SET as well as the pairing.

This is far cheaper than the curvature version -- the volume element needs one
eigendecomposition, not a Riemann tensor -- so it runs at every layer with
proper n instead of the 17-19 points S5.4 could afford.

    log sqrt(det G) restricted to the top-k subspace, per dimension,
    ON THE SPHERE (the +log r correction; see curvature.volume_element --
    without it the profile is dominated by residual-norm growth).

Usage:  python run_volume_scramble.py [n_per_layer]
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
from fisherrao.metrics import fisher_metric, fisher_metric_scrambled, topk_indices
from fisherrao.curvature import null_projector
from fisherrao.stats import spearman as _spearman

OUT = pathlib.Path("results/volume_scramble")
K_DIM, TOP_K = 5, 512
LAYERS = [1, 5, 10, 15, 20, 25, 28, 29, 30]


def k_eff_99(lm, h, G):
    """Directions holding 99% of the metric's trace -- the "intrinsic dimension"
    the hourglass overlay uses (10-architecture-log.md S3b.2).

    That overlay found k_eff dipping at relative depth 0.40, matching the
    published ambient estimates -- but also found rho(k_eff, entropy) = +0.82,
    which is the same confound that turned out to dominate the volume element
    (08-rq3a-log.md S5.5).  Running k_eff through the SAME scramble control
    settles whether the hourglass is geometry or concentration.
    """
    P = null_projector(lm, h)
    ev = torch.linalg.eigvalsh(P @ G @ P).clamp_min(0).flip(0)
    tot = float(ev.sum())
    if tot <= 0:
        return float("nan")
    c = torch.cumsum(ev, 0) / tot
    return float(int(torch.searchsorted(c, torch.tensor(0.99, dtype=c.dtype)) + 1))


def log_vol_per_dim(lm, h, G, k):
    """(log sqrt det of the top-k block) / k, on the sphere of directions."""
    P = null_projector(lm, h)
    ev = torch.linalg.eigvalsh(P @ G @ P).clamp_min(0).flip(0)[:k]
    ev = ev[ev > 0]
    if len(ev) == 0:
        return float("nan")
    lv = float(0.5 * torch.log(ev).sum())
    lv += len(ev) * float(torch.log(torch.sqrt(h.pow(2).mean() + lm.eps)))
    return lv / len(ev)


def median(xs):
    t = torch.tensor([x for x in xs if x == x], dtype=torch.float64)
    return float(t.median()) if len(t) else float("nan")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per_layer = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    gen = torch.Generator().manual_seed(0)
    lm = LM()
    print(lm.summary())

    by_layer: dict[int, list] = {l: [] for l in LAYERS}
    for s in corpus.sentences():
        H, _ = lm.residual_stream(s)
        for l in LAYERS:
            for t in range(H.shape[1]):
                by_layer[l].append(H[l, t])
    perm = torch.randperm(lm.U.shape[0], generator=gen)

    print(f"\nPAIRED volume control: {per_layer} points/layer x {len(LAYERS)} layers,"
          f" k={K_DIM}\n")
    rows, t0 = [], time.time()
    for l in LAYERS:
        pool = by_layer[l]
        idxs = torch.randperm(len(pool), generator=gen)[:per_layer].tolist()
        for i in idxs:
            h = pool[i]
            idx = topk_indices(lm, h, TOP_K)
            lp = torch.log_softmax(lm.logits(h), -1)
            within = idx[torch.randperm(len(idx), generator=gen)]
            G_r = fisher_metric(lm, h, idx=idx)
            G_w = fisher_metric_scrambled(lm, h, idx=idx, rows=within)
            G_g = fisher_metric_scrambled(lm, h, perm, idx=idx)
            rows.append(dict(
                layer=l,
                entropy=float(-(lp.exp() * lp).sum()),
                lv_real=log_vol_per_dim(lm, h, G_r, K_DIM),
                lv_within=log_vol_per_dim(lm, h, G_w, K_DIM),
                lv_scr=log_vol_per_dim(lm, h, G_g, K_DIM),
                ke_real=k_eff_99(lm, h, G_r),
                ke_within=k_eff_99(lm, h, G_w),
                ke_scr=k_eff_99(lm, h, G_g)))
        print(f"   layer {l:>2}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'='*74}\nLAYER-WISE log-volume per dimension, on the sphere"
          f"   n={per_layer}/layer\n{'='*74}")
    print(f"   {'layer':>6} {'real':>10} {'within':>10} {'global':>10}"
          f" {'entropy':>9}")
    prof = []
    for l in LAYERS:
        sub = [r for r in rows if r["layer"] == l]
        mr = median([r["lv_real"] for r in sub])
        mw = median([r["lv_within"] for r in sub])
        ms = median([r["lv_scr"] for r in sub])
        me = median([r["entropy"] for r in sub])
        prof.append(dict(layer=l, real=mr, within=mw, scr=ms, entropy=me))
        print(f"   {l:>6} {mr:>10.4f} {mw:>10.4f} {ms:>10.4f} {me:>9.3f}")

    # ---- does the SHAPE survive, or only the level? -----------------------
    # The published claim is not "log-volume is X" but "the profile is U-shaped
    # with a minimum at layer 20".  A control can leave the level alone and
    # destroy the shape, or vice versa, so test the shape directly.
    def spearman(a, b):
        """Tie-corrected Spearman -- see fisherrao/stats.py for why this is
        not argsort(argsort(x))."""
        return _spearman(a, b)
    lr = [p["real"] for p in prof]
    print(f"\n   {'condition':<10} {'min at layer':>13} {'range':>8} "
          f"{'rho vs real profile':>21}")
    shape = {}
    for c, key in (("real", "real"), ("within", "within"), ("global", "scr")):
        v = [p[key] for p in prof]
        shape[c] = spearman(lr, v)
        print(f"   {c:<10} {LAYERS[v.index(min(v))]:>13} "
              f"{max(v)-min(v):>8.4f} {shape[c]:>21.3f}")

    # ---- k_eff under the SAME control -------------------------------------
    # 10-architecture-log.md S3b.2 found k_eff dipping at relative depth 0.40,
    # matching the published ambient estimates -- and S3b.2a found
    # rho(k_eff, entropy) = +0.82, the same confound that dominated the volume
    # element.  The scramble settles it: entropy is identical by construction,
    # so a scrambled profile that reproduces the real one is reading
    # concentration rather than learned geometry.
    print(f"\n{'='*78}\nIS THE HOURGLASS IN k_eff DEFINITIONAL TOO?\n{'='*78}")
    print(f"   {'layer':>6} {'real':>9} {'within':>9} {'global':>9} {'entropy':>9}")
    kprof = []
    for l in LAYERS:
        sub = [r for r in rows if r["layer"] == l]
        cell = dict(layer=l, **{k: median([r[f"ke_{k}"] for r in sub])
                                for k in ("real", "within", "scr")})
        cell["entropy"] = median([r["entropy"] for r in sub])
        kprof.append(cell)
        print(f"   {l:>6} {cell['real']:>9.1f} {cell['within']:>9.1f} "
              f"{cell['scr']:>9.1f} {cell['entropy']:>9.3f}")
    kr = [c["real"] for c in kprof]
    print(f"\n   {'condition':<10} {'min at layer':>13} {'range':>9} "
          f"{'rho vs real profile':>21}")
    kshape = {}
    for c, key in (("real", "real"), ("within", "within"), ("global", "scr")):
        v = [x[key] for x in kprof]
        kshape[c] = spearman(kr, v)
        print(f"   {c:<10} {LAYERS[v.index(min(v))]:>13} "
              f"{max(v)-min(v):>9.1f} {kshape[c]:>21.3f}")
    print(f"\n   rho(k_eff, entropy):  real "
          f"{spearman([r['entropy'] for r in rows], [r['ke_real'] for r in rows]):+.3f}"
          f"   within "
          f"{spearman([r['entropy'] for r in rows], [r['ke_within'] for r in rows]):+.3f}")

    # ---- paired, per point -------------------------------------------------
    print(f"\n   paired per point, and rho against entropy:")
    stats = {}
    for c, key in (("within", "lv_within"), ("global", "lv_scr")):
        d = [r["lv_real"] - r[key] for r in rows
             if r["lv_real"] == r["lv_real"] and r[key] == r[key]]
        pos, tot = sum(1 for x in d if x > 0), len(d)
        z = (pos - tot / 2) / (0.5 * tot ** 0.5) if tot else 0.0
        rho_e = spearman([r["entropy"] for r in rows], [r[key] for r in rows])
        stats[c] = dict(pos=pos, n=tot, z=z, med=median(d), rho_entropy=rho_e)
        print(f"     real > {c:<8} in {pos:>3}/{tot:<4} z = {z:>+6.2f}   "
              f"median diff {median(d):>+8.4f}   rho(vol, entropy) {rho_e:>+.3f}")
    rho_re = spearman([r["entropy"] for r in rows], [r["lv_real"] for r in rows])
    print(f"     {'real itself':<17}{'':<14}"
          f"{'':<24}rho(vol, entropy) {rho_re:>+.3f}")
    rho_se = stats["global"]["rho_entropy"]
    rho_shape = shape["global"]
    z = stats["global"]["z"]
    pos, tot = stats["global"]["pos"], stats["global"]["n"]
    print("   Entropy is IDENTICAL across conditions by construction, so a")
    print("   scrambled correlation as strong as the real one means the")
    print("   quantity is reading concentration rather than learned structure.")

    (OUT / "volume_scramble.json").write_text(json.dumps(dict(
        n_per_layer=per_layer, k=K_DIM, profile=prof, shape_rho=shape,
        stats=stats, rho_real_entropy=rho_re, rows=rows), indent=2),
        encoding="utf-8")
    print(f"\nwrote {OUT / 'volume_scramble.json'}")


if __name__ == "__main__":
    main()
