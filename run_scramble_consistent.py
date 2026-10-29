"""
Is the paired-scramble collapse (08-rq3a-log.md S5.4) caused by destroying the
learned assignment, or by the scrambled metric not being a Fisher pullback?

The `within` condition of run_scramble_within.py builds its metric with
fisher_metric_projected(rows=...): p(h) is taken from the REAL logits, while
the Jacobian uses PERMUTED rows.  As h moves along the slice, p therefore moves
along the real directions while the metric measures along other ones.  That is
a valid Riemannian metric, but it is not the pullback of any statistical model.

THE CONSISTENT CONTROL destroys the same assignment as a genuine model:

    z(h) = U[idx] n(h0) + U[rows] (n(h) - n(h0))          p(h) = softmax(z(h))

  * p(h0) is the real distribution, so entropy is matched exactly;
  * G(h0) equals the `within` metric exactly -- same frame, same conditioning,
    same row multiset (checked below, relative difference 0);
  * the ONLY difference from `within` is that p now moves along the same
    permuted directions the metric uses, so its curvature is a pullback's.

Prediction if S5.4's reading is right:  K_consistent ~ K_within (~0).
Prediction if the collapse is a consistency artefact:  K_consistent ~ K_real.

Three conditions on the same points, the same entropy-stratified draw as
run_scramble_within.py (a prefix of each band), k = 5, 16 planes per point.

RMSNorm only: the frame Jacobian below is the closed form fisher_metric_projected
uses.  LayerNorm models need the general norm Jacobian -- 13-revision-plan.md E1.

Usage:  python run_scramble_consistent.py [n_per_bin] [budget_seconds]
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
from fisherrao.curvature import null_projector, riemann, sectional

OUT = pathlib.Path("results/scramble")
K_DIM, TOP_K, N_PLANES = 5, 512, 16
LAYERS = [1, 5, 10, 15, 20, 25, 28, 29, 30]
EDGES = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 1e9)]


def metric_consistent(lm, h, h0, F, idx, rows):
    """(k, k) Fisher pullback of h -> softmax(U[idx] n(h0) + U[rows](n(h) - n(h0)))."""
    r = torch.sqrt(h.pow(2).mean() + lm.eps)
    hhat = h / torch.linalg.vector_norm(h)
    Af = (F - torch.outer(hhat, hhat @ F)) / r       # same Jacobian as fisher_metric_projected
    if lm.g is not None:
        Af = lm.g[:, None] * Af
    M = lm.U[rows] @ Af
    n0 = lm.apply_norm(h0).detach()
    z = lm.U[idx] @ n0 + lm.U[rows] @ (lm.apply_norm(h) - n0)
    p = torch.softmax(z - z.max(), -1)
    Mp = M.T @ p
    return (M.T * p) @ M - torch.outer(Mp, Mp)


def frame_for(lm, h, idx, P, k, rows=None):
    d = h.shape[-1]
    G = fisher_metric_projected(lm, h, torch.eye(d, dtype=h.dtype), idx, rows=rows)
    ev, evec = torch.linalg.eigh(P @ G @ P)
    return evec[:, torch.argsort(ev, descending=True)[:k]].contiguous()


def median_K(g, k, seed=0):
    x0 = torch.zeros(k, dtype=torch.float64)
    R = riemann(g, x0)
    gen = torch.Generator().manual_seed(seed)
    Ks = []
    for _ in range(N_PLANES):
        u = torch.randn(k, generator=gen, dtype=torch.float64)
        v = torch.randn(k, generator=gen, dtype=torch.float64)
        val = sectional(g, x0, u, v, R=R)
        if val == val:
            Ks.append(val)
    return float(torch.tensor(Ks).median()) if Ks else float("nan")


def med(xs):
    t = torch.tensor([x for x in xs if x == x], dtype=torch.float64)
    return float(t.median()) if len(t) else float("nan")


def sign_test(d):
    pos = sum(1 for x in d if x > 0)
    z = (pos - len(d) / 2) / (0.5 * len(d) ** 0.5) if d else 0.0
    return pos, len(d), z


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n_per_bin = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 480.0
    gen = torch.Generator().manual_seed(0)
    lm = LM()
    print(lm.summary())
    if "RMS" not in lm.norm_kind:
        raise SystemExit("consistent control is implemented for RMSNorm only")

    states = []
    for s in corpus.sentences():
        H, _ = lm.residual_stream(s)
        for l in LAYERS:
            for t in range(H.shape[1]):
                states.append((l, H[l, t]))
    ents = []
    for _, h in states:
        lp = torch.log_softmax(lm.logits(h), -1)
        ents.append(float(-(lp.exp() * lp).sum()))
    pick = []
    for lo, hi in EDGES:
        band = [i for i, e in enumerate(ents) if lo <= e < hi]
        take = torch.randperm(len(band), generator=gen)[:n_per_bin].tolist()
        pick += [band[t] for t in take]

    rows_out, checks, t0 = [], [], time.time()
    for j, i in enumerate(pick):
        if time.time() - t0 > budget:
            print(f"   budget reached after {len(rows_out)} points", flush=True)
            break
        layer, h0 = states[i]
        idx = topk_indices(lm, h0, TOP_K)
        P = null_projector(lm, h0)
        # the same per-point scramble as run_scramble_within.py
        within = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(1000 + i))]
        F_real = frame_for(lm, h0, idx, P, K_DIM)
        F_scr = frame_for(lm, h0, idx, P, K_DIM, rows=within)

        if j < 3:   # implementation checks
            x = 0.3 * torch.randn(K_DIM, dtype=torch.float64)
            a = metric_consistent(lm, h0 + F_real @ x, h0, F_real, idx, idx)
            b = fisher_metric_projected(lm, h0 + F_real @ x, F_real, idx)
            c = metric_consistent(lm, h0, h0, F_scr, idx, within)
            e = fisher_metric_projected(lm, h0, F_scr, idx, rows=within)
            chk = dict(identity_vs_real=float((a - b).norm() / b.norm()),
                       consistent_vs_within_at_h0=float((c - e).norm() / e.norm()))
            checks.append(chk)
            print(f"   check: rows=idx vs real {chk['identity_vs_real']:.1e}   "
                  f"consistent vs within at h0 {chk['consistent_vs_within_at_h0']:.1e}",
                  flush=True)

        K_real = median_K(lambda x: fisher_metric_projected(lm, h0 + F_real @ x, F_real, idx), K_DIM)
        K_within = median_K(lambda x: fisher_metric_projected(
            lm, h0 + F_scr @ x, F_scr, idx, rows=within), K_DIM)
        K_cons = median_K(lambda x: metric_consistent(lm, h0 + F_scr @ x, h0, F_scr, idx, within), K_DIM)
        rows_out.append(dict(_i=i, layer=layer, entropy=ents[i],
                             K_real=K_real, K_within=K_within, K_consistent=K_cons))
        print(f"   {j+1:>2}/{len(pick)}  L{layer:<2} H={ents[i]:5.2f}  real {K_real:+.4f}  "
              f"within {K_within:+.4f}  consistent {K_cons:+.4f}   ({time.time()-t0:.0f}s)",
              flush=True)

    summary = dict(n=len(rows_out), k=K_DIM, top_k=TOP_K, n_planes=N_PLANES,
                   model=lm.model_id if hasattr(lm, "model_id") else None,
                   checks=checks)
    for c in ("real", "within", "consistent"):
        summary[f"K_{c}"] = med([r[f"K_{c}"] for r in rows_out])
    for c in ("within", "consistent"):
        pos, tot, z = sign_test([r["K_real"] - r[f"K_{c}"] for r in rows_out
                                 if r["K_real"] == r["K_real"] and r[f"K_{c}"] == r[f"K_{c}"]])
        summary[f"real_gt_{c}"] = dict(pos=pos, n=tot, z=z)
    by_ent = {}
    for lo, hi in EDGES:
        sub = [r for r in rows_out if lo <= r["entropy"] < hi]
        if sub:
            by_ent[f"[{lo},{hi})" if hi < 1e9 else f"[{lo},inf)"] = {
                c: med([r[f"K_{c}"] for r in sub]) for c in ("real", "within", "consistent")
            } | dict(n=len(sub))
    summary["by_entropy"] = by_ent

    print(f"\n{'='*72}\nTHREE CONDITIONS, SAME POINTS   n={summary['n']}\n{'='*72}")
    for c in ("real", "within", "consistent"):
        print(f"   {c:<11} median K = {summary[f'K_{c}']:+.4f}")
    for c in ("within", "consistent"):
        s = summary[f"real_gt_{c}"]
        print(f"   real > {c:<11} {s['pos']}/{s['n']}   z = {s['z']:+.2f}")

    (OUT / "consistent_pilot.json").write_text(
        json.dumps(dict(summary=summary, rows=rows_out), indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'consistent_pilot.json'}")


if __name__ == "__main__":
    main()
