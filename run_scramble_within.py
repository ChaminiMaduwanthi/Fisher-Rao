"""
THE CORRECTED-AGAIN definitional control, and a check on 08-rq3a-log.md S5.4.

S5.4 concluded that destroying the learned token->direction assignment collapses
K from 0.259 to 0.005 at matched entropy, and read that as "the geometry is not
a function of concentration alone".  Auditing the control shows it does not
isolate what that sentence claims.

    fisher_metric_scrambled(perm=<whole-vocabulary permutation>)
        retained rows become perm[idx].  Measured: overlap with the real
        top-512 is 5/512, and median ||U row|| is 3.10 against the real 2.44.

So the "scrambled" condition swapped the direction SET as well as the pairing,
and the 0.259 -> 0.005 collapse could be caused by either.  This is the third
time a control in this project has turned out to change more than one thing at
once (S5.0's row-shuffle changed nothing; S5.2's Gaussian U changed everything).

THE CLEAN CONTROL, and the point of this script:

    WITHIN   rows = idx[randperm(len(idx))]
             The retained row multiset is IDENTICAL to the real one.  Entropy is
             identical.  The ONLY thing destroyed is which probability sits on
             which direction.

Three conditions on the same points, paired:

    real     the trained model
    within   clean -- isolates the learned assignment                <- the test
    global   the S5.4 condition -- assignment AND direction set

Prediction if S5.4's conclusion is right: `within` collapses like `global` did.
Prediction if S5.4 was measuring the row swap: `within` stays near the real
value and S5.4's headline needs rewriting.

STRATIFICATION.  Points are drawn stratified by ENTROPY, not uniformly, which
answers two further open items in the same run:

  * S5.3 -- "power the low-entropy bin".  The residual worry was that any
    model-specific geometry lives in the near-deterministic regime, where a
    uniform sample puts almost no points.  Stratifying guarantees coverage.
  * S6   -- "the paired result has not been checked layer by layer".  Layer is
    recorded per point, so that breakdown comes free.

If the collapse is uniform across entropy, the effect is a general property of
the learned assignment.  If it is concentrated at low entropy, then the
near-deterministic regime carries the structure -- which is what S5.3 suspected
on much weaker evidence.

Usage:  python run_scramble_within.py [n_per_bin] [budget_seconds]
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
from fisherrao.curvature import null_projector, riemann, sectional, scalar_curvature

OUT = pathlib.Path("results/scramble")
K_DIM, TOP_K, N_PLANES = 5, 512, 16
LAYERS = [1, 5, 10, 15, 20, 25, 28, 29, 30]


def frame_for(lm, h, idx, P, k, perm=None, rows=None):
    """Top-k eigenvectors of the null-projected metric for a given condition.

    Built through fisher_metric_projected on the identity frame so that the
    real, within and global conditions go through EXACTLY the same code path --
    a separate builder per condition is how a control silently stops matching.
    """
    d = h.shape[-1]
    G = fisher_metric_projected(lm, h, torch.eye(d, dtype=h.dtype), idx,
                                perm=perm, rows=rows)
    ev, evec = torch.linalg.eigh(P @ G @ P)
    return evec[:, torch.argsort(ev, descending=True)[:k]].contiguous()


def curv(lm, h, F, idx, perm=None, rows=None, seed=0):
    def g(x):
        return fisher_metric_projected(lm, h + F @ x, F, idx, perm=perm, rows=rows)

    x0 = torch.zeros(F.shape[1], dtype=h.dtype)
    R = riemann(g, x0)
    gen = torch.Generator().manual_seed(seed)
    Ks = []
    for _ in range(N_PLANES):
        u = torch.randn(F.shape[1], generator=gen, dtype=h.dtype)
        v = torch.randn(F.shape[1], generator=gen, dtype=h.dtype)
        val = sectional(g, x0, u, v, R=R)
        if val == val:
            Ks.append(val)
    return (float(torch.tensor(Ks, dtype=h.dtype).median()) if Ks else float("nan"),
            scalar_curvature(g, x0, R=R))


def median(xs):
    t = torch.tensor([x for x in xs if x == x], dtype=torch.float64)
    return float(t.median()) if len(t) else float("nan")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 430.0
    k_sweep = "--ksweep" in sys.argv
    by_layer = "--bylayer" in sys.argv
    ckpt = OUT / "within_points.jsonl"
    gen = torch.Generator().manual_seed(0)
    lm = LM()
    print(lm.summary())

    done = {}
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["_i"]] = r
    print(f"\ncheckpoint: {len(done)} points   budget {budget:.0f}s")

    states = []
    for s in corpus.sentences():
        H, _ = lm.residual_stream(s)
        for l in LAYERS:
            for t in range(H.shape[1]):
                states.append((l, H[l, t]))

    # ---- stratify by entropy ---------------------------------------------
    # A uniform draw puts almost nothing below 0.5 nats, which is exactly the
    # regime S5.3 flagged.  Entropy is one forward pass per state, so building
    # the strata is cheap next to the curvature that follows.
    ents = []
    for l, h in states:
        lp = torch.log_softmax(lm.logits(h), -1)
        ents.append(float(-(lp.exp() * lp).sum()))
    EDGES = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 1e9)]
    pick, strata = [], {}
    if by_layer:
        # S6 asks for the paired result broken down BY LAYER.  Stratifying by
        # entropy (the default) leaves 3-12 points in each layer cell, which is
        # not enough to say anything per layer.  This mode fills layer cells
        # instead, at the cost of the entropy coverage S5.3 needs -- the two
        # questions want different samples, so they get different runs and share
        # the checkpoint.
        for l in LAYERS:
            band = [i for i, (ll, _) in enumerate(states) if ll == l]
            take = torch.randperm(len(band), generator=gen)[:n].tolist()
            chosen = [band[t] for t in take]
            strata[f"layer {l}"] = (len(band), len(chosen))
            pick += chosen
        print("\nlayer strata (available -> sampled):")
    else:
        for lo, hi in EDGES:
            band = [i for i, e in enumerate(ents) if lo <= e < hi]
            take = torch.randperm(len(band), generator=gen)[:n].tolist()
            chosen = [band[t] for t in take]
            strata[f"[{lo},{hi})" if hi < 1e9 else f"[{lo},inf)"] = (len(band), len(chosen))
            pick += chosen
        print("\nentropy strata (available -> sampled):")
    for kk, (avail, got) in strata.items():
        print(f"   {kk:<12} {avail:>6} -> {got}")
    perm = torch.randperm(lm.U.shape[0], generator=gen)

    rows_out, t0, n_new = [], time.time(), 0
    need = ["K_real", "K_within", "K_global"]
    if k_sweep:
        need += ["K_real_k4", "K_within_k4", "K_real_k6", "K_within_k6"]
    for j, i in enumerate(pick, 1):
        have = done.get(i, {})
        if all(key in have for key in need) or "error" in have:
            rows_out.append(have)
            continue
        if time.time() - t0 > budget:
            print(f"   budget reached, +{n_new} new; rerun to continue", flush=True)
            break
        layer, h = states[i]
        idx = topk_indices(lm, h, TOP_K)
        P = null_projector(lm, h)
        lp = torch.log_softmax(lm.logits(h), -1)
        # The within-set scramble, frozen for this point exactly as idx is.
        # Seeded from the point index, NOT from the running generator, so a
        # resumed run reuses the SAME scramble for a point whose k=5 pass is
        # already on file -- otherwise the k sweep would compare k=4 under one
        # scramble against k=5 under another.
        pgen = torch.Generator().manual_seed(1000 + i)
        within = idx[torch.randperm(len(idx), generator=pgen)]
        rec = dict(have)
        rec.update(layer=layer, entropy=float(-(lp.exp() * lp).sum()), _i=i)
        try:
            for name, kw in (("real", {}), ("within", dict(rows=within)),
                             ("global", dict(perm=perm))):
                F = frame_for(lm, h, idx, P, K_DIM, **kw)
                rec[f"K_{name}"], rec[f"R_{name}"] = curv(lm, h, F, idx, **kw)
            # k-stability of the CLEAN control.  The k=4/5/6 sweep in
            # run_frame_resolution.py was run on the compound `global`
            # condition; 08-rq3a-log.md S6 lists repeating it under `within` as
            # outstanding, and k-fragility is this project's known failure mode.
            if k_sweep:
                for k in (4, 6):
                    for name, kw in (("real", {}), ("within", dict(rows=within))):
                        F = frame_for(lm, h, idx, P, k, **kw)
                        rec[f"K_{name}_k{k}"], _ = curv(lm, h, F, idx, **kw)
        except Exception as e:                                   # noqa: BLE001
            rec["error"] = repr(e)
        rows_out.append(rec)
        n_new += 1
        with ckpt.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"   {j}/{len(pick)}  (+{n_new}, {time.time()-t0:.0f}s)", flush=True)

    ok = [r for r in rows_out if "error" not in r and "K_real" in r]
    print(f"\n{'='*76}\nTHREE CONDITIONS, SAME POINTS, ENTROPY IDENTICAL THROUGHOUT"
          f"   n={len(ok)}\n{'='*76}")
    print(f"   {'condition':<10} {'what it destroys':<34} {'median K':>10} "
          f"{'median R':>10}")
    labels = {"real": "nothing (the trained model)",
              "within": "the pairing ONLY (same rows)",
              "global": "the pairing AND the row set"}
    for c in ("real", "within", "global"):
        print(f"   {c:<10} {labels[c]:<34} "
              f"{median([r[f'K_{c}'] for r in ok]):>10.4f} "
              f"{median([r[f'R_{c}'] for r in ok]):>10.3f}")

    print(f"\n   paired sign tests against `real`:")
    for c in ("within", "global"):
        d = [r["K_real"] - r[f"K_{c}"] for r in ok
             if r["K_real"] == r["K_real"] and r[f"K_{c}"] == r[f"K_{c}"]]
        pos, tot = sum(1 for x in d if x > 0), len(d)
        z = (pos - tot / 2) / (0.5 * tot ** 0.5) if tot else 0.0
        print(f"     real > {c:<8} in {pos:>3}/{tot:<3}   z = {z:>+6.2f}   "
              f"{'SIGNIFICANT' if abs(z) > 1.96 else 'not significant'}")

    # ---- S5.3: is the effect uniform across entropy, or concentrated? -----
    def block(title, keyfn, groups):
        print(f"\n   {title}")
        print(f"     {'group':<14} {'n':>4} {'K_real':>9} {'K_within':>10}"
              f" {'retained':>9} {'sign':>9}")
        out = {}
        for g in groups:
            sub = [r for r in ok if keyfn(r) == g]
            if not sub:
                continue
            kr, kw = median([r["K_real"] for r in sub]), median([r["K_within"] for r in sub])
            d = [r["K_real"] - r["K_within"] for r in sub]
            pos = sum(1 for x in d if x > 0)
            out[str(g)] = dict(n=len(sub), K_real=kr, K_within=kw, pos=pos)
            print(f"     {str(g):<14} {len(sub):>4} {kr:>9.4f} {kw:>10.4f}"
                  f" {kw/kr if kr else float('nan'):>8.1%} {pos:>5}/{len(sub):<3}")
        return out

    def ebin(r):
        e = r["entropy"]
        for lo, hi in EDGES:
            if lo <= e < hi:
                return f"[{lo},{hi})" if hi < 1e9 else f"[{lo},inf)"
        return "?"

    by_ent = block("BY ENTROPY -- does the collapse depend on concentration? (S5.3)",
                   ebin, [f"[{lo},{hi})" if hi < 1e9 else f"[{lo},inf)"
                          for lo, hi in EDGES])
    by_layer = block("BY LAYER (S6)", lambda r: r["layer"], LAYERS)

    # ---- k-stability of the CLEAN control ---------------------------------
    ks = {}
    print(f"\n   K-STABILITY OF THE `within` CONTROL (S6)")
    print(f"     {'k':>3} {'n':>4} {'K_real':>9} {'K_within':>10} {'sign':>9}")
    for k in (4, 5, 6):
        rk, wk = (("K_real", "K_within") if k == 5
                  else (f"K_real_k{k}", f"K_within_k{k}"))
        sub = [r for r in ok if rk in r and wk in r
               and r[rk] == r[rk] and r[wk] == r[wk]]
        if not sub:
            continue
        d = [r[rk] - r[wk] for r in sub]
        pos = sum(1 for x in d if x > 0)
        z = (pos - len(d) / 2) / (0.5 * len(d) ** 0.5)
        ks[k] = dict(n=len(sub), K_real=median([r[rk] for r in sub]),
                     K_within=median([r[wk] for r in sub]), pos=pos, z=z)
        print(f"     {k:>3} {len(sub):>4} {ks[k]['K_real']:>9.4f} "
              f"{ks[k]['K_within']:>10.4f} {pos:>5}/{len(sub):<3} z={z:+.2f}")

    kr, kw = median([r["K_real"] for r in ok]), median([r["K_within"] for r in ok])
    kg = median([r["K_global"] for r in ok])
    print(f"\n   VERDICT")
    if abs(kw) < 0.5 * abs(kr):
        print(f"     `within` collapses too ({kr:.4f} -> {kw:.4f}).")
        print(f"     S5.4's conclusion SURVIVES the corrected control: the learned")
        print(f"     ASSIGNMENT is what puts the manifold at the simplex value,")
        print(f"     independently of which directions are retained.")
    else:
        print(f"     `within` does NOT collapse ({kr:.4f} -> {kw:.4f}) while")
        print(f"     `global` does ({kg:.4f}).  S5.4 was substantially measuring")
        print(f"     the ROW-SET swap, not the assignment, and its headline must")
        print(f"     be rewritten.")

    (OUT / "within.json").write_text(json.dumps(dict(
        n=len(ok), K_real=kr, K_within=kw, K_global=kg,
        by_entropy=by_ent, by_layer=by_layer, k_stability=ks, rows=rows_out),
        indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'within.json'}")


if __name__ == "__main__":
    main()
