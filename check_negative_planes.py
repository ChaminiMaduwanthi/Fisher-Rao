"""
Where does NEGATIVE sectional curvature live, and is it geometry or arithmetic?

07-stage4-log.md S1.1c found a minority tail in `K` and argued on three grounds
that it is real geometry rather than numerical failure.  10-architecture-log.md
S3b.3 then found the same minority in the SIGN distribution across four models:
6-12% of points carry at least one negatively curved 2-plane, reaching -80.

S3b.3a checked those points against the deflationary explanation and they passed
-- conditioning is not elevated, and the two contractions of the Riemann tensor
still agree.  It also turned up ONE distinguishing feature:

    k_eff_99   248 at points with a negative plane, 122 at points without
               -- at IDENTICAL entropy (1.909 vs 1.952)

That is the first mechanism this project has for the tail, and it came from 49
points found incidentally in the E1 profile.  This script tests it properly:

  1. SAMPLE ON PURPOSE.  Stratify by k_eff so both arms are well populated,
     rather than taking whatever the profile run happened to produce.
  2. RULE OUT THE ENTROPY CHANNEL.  k_eff is rho ~ +0.8 with entropy
     (10-architecture-log.md S3b.2a), so the whole effect could be entropy in
     disguise.  Matching entropy between the arms decides it.
  3. CHECK THE DIRECTION.  If high k_eff CAUSES negative planes, the negative
     fraction should rise monotonically with k_eff, not just differ between two
     buckets.

Usage:  python check_negative_planes.py [n_per_bin] [budget_seconds]
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
from fisherrao.curvature import curvature_at, spectral_diagnostics

OUT = pathlib.Path("results/negplanes")
K_DIM, TOP_K, N_PLANES = 5, 512, 32      # more planes: the statistic IS the tail
LAYERS = [1, 5, 10, 15, 20, 25, 28, 29, 30]
BINS = [(0, 8), (8, 20), (20, 60), (60, 200), (200, 10**9)]


def med(xs):
    t = torch.tensor([x for x in xs if x is not None and x == x], dtype=torch.float64)
    return float(t.median()) if len(t) else float("nan")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 460.0
    ckpt = OUT / "points.jsonl"

    done = {}
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["_i"]] = r
    lm = LM()
    print(lm.summary())
    print(f"checkpoint: {len(done)} points   budget {budget:.0f}s")

    states = []
    for s in corpus.sentences():
        H, _ = lm.residual_stream(s)
        for l in LAYERS:
            for t in range(H.shape[1]):
                states.append((l, H[l, t]))

    # ---- stratify by k_eff -------------------------------------------------
    # One eigendecomposition per state -- no curvature, but a (d,d) eigh each,
    # so ~0.1 s.  The corpus has >30k states and screening all of them costs ~50
    # minutes BEFORE any curvature is computed; a first version did exactly that
    # and produced no checkpoint at all inside the budget.  Screen a bounded
    # random subsample instead -- it only has to be large enough to fill the
    # k_eff bins.
    gen = torch.Generator().manual_seed(0)
    SCREEN = min(len(states), 2500)
    screen_idx = torch.randperm(len(states), generator=gen)[:SCREEN].tolist()
    print(f"\nscreening k_eff on {SCREEN} of {len(states)} states "
          f"(cheap; no curvature)...", flush=True)
    keff = [None] * len(states)
    ent = [None] * len(states)
    t_scr = time.time()
    for n_done, i in enumerate(screen_idx, 1):
        l, h = states[i]
        try:
            sd = spectral_diagnostics(lm, h, K_DIM, top_k=TOP_K)
            keff[i] = sd["k_eff_99"]
            lp = torch.log_softmax(lm.logits(h), -1)
            ent[i] = float(-(lp.exp() * lp).sum())
        except Exception:                                     # noqa: BLE001
            pass
        if n_done % 500 == 0:
            print(f"   screened {n_done}/{SCREEN}  ({time.time()-t_scr:.0f}s)",
                  flush=True)
    pick = []
    print(f"   {'k_eff bin':<14} {'available':>10} {'sampled':>8}")
    for lo, hi in BINS:
        band = [i for i, v in enumerate(keff) if v is not None and lo <= v < hi]
        take = torch.randperm(len(band), generator=gen)[:n_per].tolist()
        pick += [band[t] for t in take]
        tag = f"[{lo},{hi})" if hi < 10**9 else f"[{lo},inf)"
        print(f"   {tag:<14} {len(band):>10} {min(len(band), n_per):>8}")

    rows, t0, n_new = [], time.time(), 0
    for j, i in enumerate(pick, 1):
        if i in done:
            rows.append(done[i])
            continue
        if time.time() - t0 > budget:
            print(f"   budget reached (+{n_new}); rerun to continue", flush=True)
            break
        l, h = states[i]
        rec = dict(_i=i, layer=l, k_eff=keff[i], entropy=ent[i])
        try:
            c = curvature_at(lm, h, K_DIM, top_k=TOP_K, n_planes=N_PLANES)
            Ks = torch.tensor([x for x in c["K"] if x == x], dtype=torch.float64)
            rec.update(K=c["K_median"], R=c["scalar_R"],
                       n_planes=int(len(Ks)), n_neg=int((Ks < 0).sum()),
                       K_min=float(Ks.min()))
            sd = spectral_diagnostics(lm, h, K_DIM, top_k=TOP_K)
            rec["cond_eff"] = sd["cond_eff"]
        except Exception as e:                                # noqa: BLE001
            rec["error"] = repr(e)
        rows.append(rec)
        done[i] = rec
        n_new += 1
        with ckpt.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        if n_new % 10 == 0:
            print(f"   {j}/{len(pick)}  (+{n_new}, {time.time()-t0:.0f}s)", flush=True)

    ok = [r for r in rows if "error" not in r and "n_planes" in r]
    print(f"\n{'='*76}\nNEGATIVE PLANES vs EFFECTIVE DIMENSION   n={len(ok)}"
          f"   ({N_PLANES} planes/point)\n{'='*76}")
    print(f"   {'k_eff bin':<14} {'n':>4} {'k_eff':>8} {'entropy':>9} "
          f"{'% pts w/ neg':>13} {'% planes neg':>13} {'min K':>10}")
    prof = []
    for lo, hi in BINS:
        sub = [r for r in ok if lo <= r["k_eff"] < hi]
        if not sub:
            continue
        pts = sum(1 for r in sub if r["n_neg"] > 0) / len(sub)
        pl = sum(r["n_neg"] for r in sub) / sum(r["n_planes"] for r in sub)
        tag = f"[{lo},{hi})" if hi < 10**9 else f"[{lo},inf)"
        prof.append(dict(bin=tag, n=len(sub), k_eff=med([r["k_eff"] for r in sub]),
                         entropy=med([r["entropy"] for r in sub]),
                         frac_pts=pts, frac_planes=pl))
        print(f"   {tag:<14} {len(sub):>4} {med([r['k_eff'] for r in sub]):>8.0f} "
              f"{med([r['entropy'] for r in sub]):>9.3f} {pts:>12.1%} "
              f"{pl:>12.2%} {min(r['K_min'] for r in sub):>10.2f}")

    # ---- is it entropy in disguise? ---------------------------------------
    print(f"\n{'='*76}\nIS IT THE ENTROPY CHANNEL?\n{'='*76}")
    print("   k_eff is rho ~ +0.8 with entropy, so the gradient above could be")
    print("   entropy.  Restricting to a narrow entropy band breaks that link.")
    band = [r for r in ok if 1.0 <= r["entropy"] < 3.0]
    print(f"   entropy restricted to [1,3) nats: n = {len(band)}")
    print(f"   {'k_eff bin':<14} {'n':>4} {'entropy':>9} {'% pts w/ neg':>13}")
    for lo, hi in BINS:
        sub = [r for r in band if lo <= r["k_eff"] < hi]
        if len(sub) < 5:
            continue
        pts = sum(1 for r in sub if r["n_neg"] > 0) / len(sub)
        tag = f"[{lo},{hi})" if hi < 10**9 else f"[{lo},inf)"
        print(f"   {tag:<14} {len(sub):>4} {med([r['entropy'] for r in sub]):>9.3f} "
              f"{pts:>12.1%}")

    def sp(a, b):
        x = torch.argsort(torch.argsort(torch.tensor(a, dtype=torch.float64))).double()
        y = torch.argsort(torch.argsort(torch.tensor(b, dtype=torch.float64))).double()
        return float(torch.corrcoef(torch.stack([x, y]))[0, 1])

    if len(ok) > 10:
        print(f"\n   rho(k_eff,   n_neg) = {sp([r['k_eff'] for r in ok], [r['n_neg'] for r in ok]):+.3f}   all points")
        print(f"   rho(entropy, n_neg) = {sp([r['entropy'] for r in ok], [r['n_neg'] for r in ok]):+.3f}")
        print(f"   rho(cond_eff,n_neg) = {sp([r['cond_eff'] for r in ok], [r['n_neg'] for r in ok]):+.3f}"
              f"   <- the deflationary channel; should be near zero")
    if len(band) > 10:
        print(f"   rho(k_eff,   n_neg) = {sp([r['k_eff'] for r in band], [r['n_neg'] for r in band]):+.3f}"
              f"   entropy-restricted (n={len(band)})")

    (OUT / "negplanes.json").write_text(json.dumps(
        dict(n=len(ok), n_planes=N_PLANES, profile=prof, rows=rows), indent=2),
        encoding="utf-8")
    print(f"\nwrote {OUT / 'negplanes.json'}")


if __name__ == "__main__":
    main()
