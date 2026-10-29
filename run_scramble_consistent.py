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

Three conditions per point -- real, within, consistent -- with the same frames,
planes and per-point scramble seeds as run_scramble_within.py, so `real` and
`within` reproduce that script's saved values exactly.

POINT SETS
  draw   entropy-stratified draw, a prefix of run_scramble_within.py's draw
  paper  the states saved in results/scramble/within_points.jsonl -- the sample
         behind the paper's S5.4 numbers.  That file holds 221 rows but only
         186 distinct states: 35 rows recur across checkpoint resumes with
         identical values, so the unique states are used and n = 186.

RMSNorm only: the frame Jacobian below is the closed form fisher_metric_projected
uses.  LayerNorm models need the general norm Jacobian (13-revision-plan.md E1).

Usage:
  python run_scramble_consistent.py                                  # pilot, 6 per band
  python run_scramble_consistent.py --points paper --ksweep          # the paper's states
  python run_scramble_consistent.py --model JackFram/llama-160m --per-bin 24 --ksweep
Runs checkpoint to results/scramble/consistent_<tag>_points.jsonl and resume.
"""

from __future__ import annotations

import argparse
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
from fisherrao.model import DEFAULT_MODEL

OUT = pathlib.Path("results/scramble")
WITHIN_POINTS = OUT / "within_points.jsonl"
K_DIM, TOP_K, N_PLANES = 5, 512, 16
SMOL_LAYERS = [1, 5, 10, 15, 20, 25, 28, 29, 30]      # the paper's SmolLM2 depths
EDGES = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 1e9)]
CONDS = ("real", "within", "consistent")


def depths_for(n_layers: int, n_depths: int = 9) -> list[int]:
    """Same rule as run_profiles.depths_for: spread over the network, 1 and last included."""
    if n_layers <= n_depths:
        return list(range(1, n_layers + 1))
    step = (n_layers - 1) / (n_depths - 1)
    return sorted({max(1, round(1 + i * step)) for i in range(n_depths)})


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


def three_conditions(lm, h0, idx, P, within, k):
    F_real = frame_for(lm, h0, idx, P, k)
    F_scr = frame_for(lm, h0, idx, P, k, rows=within)
    return dict(
        real=median_K(lambda x: fisher_metric_projected(lm, h0 + F_real @ x, F_real, idx), k),
        within=median_K(lambda x: fisher_metric_projected(
            lm, h0 + F_scr @ x, F_scr, idx, rows=within), k),
        consistent=median_K(lambda x: metric_consistent(
            lm, h0 + F_scr @ x, h0, F_scr, idx, within), k))


def med(xs):
    t = torch.tensor([x for x in xs if x == x], dtype=torch.float64)
    return float(t.median()) if len(t) else float("nan")


def sign_test(d):
    d = [x for x in d if x == x]
    pos = sum(1 for x in d if x > 0)
    z = (pos - len(d) / 2) / (0.5 * len(d) ** 0.5) if d else 0.0
    return dict(pos=pos, n=len(d), z=z)


def ebin(e):
    for lo, hi in EDGES:
        if lo <= e < hi:
            return f"[{lo},{hi})" if hi < 1e9 else f"[{lo},inf)"


def summarise(rows, key="K"):
    out = {c: med([r[f"{key}_{c}"] for r in rows if f"{key}_{c}" in r]) for c in CONDS}
    for c in ("within", "consistent"):
        out[f"real_gt_{c}"] = sign_test([r[f"{key}_real"] - r[f"{key}_{c}"]
                                         for r in rows if f"{key}_{c}" in r])
    out["n"] = sum(1 for r in rows if f"{key}_real" in r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--points", choices=("draw", "paper"), default="draw")
    ap.add_argument("--per-bin", type=int, default=6)
    ap.add_argument("--ksweep", action="store_true")
    ap.add_argument("--ksweep-n", type=int, default=30, help="draw mode: points swept")
    ap.add_argument("--budget", type=float, default=3000.0)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    short = a.model.split("/")[-1]
    default_run = (a.model == DEFAULT_MODEL and a.points == "draw"
                   and a.per_bin == 6 and not a.ksweep)
    tag = "pilot" if default_run else f"{short}_{a.points}"
    ckpt = OUT / f"consistent_{tag}_points.jsonl"

    lm = LM(a.model)
    print(lm.summary())
    if "RMS" not in lm.norm_kind:
        raise SystemExit("consistent control is implemented for RMSNorm only")
    layers = SMOL_LAYERS if "SmolLM2" in a.model else depths_for(lm.n_layers)

    states = []
    for s in corpus.sentences():
        H, _ = lm.residual_stream(s)
        for l in layers:
            for t in range(H.shape[1]):
                states.append((l, H[l, t]))
    ents = []
    for _, h in states:
        lp = torch.log_softmax(lm.logits(h), -1)
        ents.append(float(-(lp.exp() * lp).sum()))

    stored, sweep = {}, set()
    if a.points == "paper":
        if "SmolLM2" not in a.model:
            raise SystemExit("--points paper is the SmolLM2 sample")
        for line in WITHIN_POINTS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                stored.setdefault(r["_i"], r)          # 221 rows, 186 distinct states
        pick = list(stored)
        sweep = {i for i, r in stored.items() if "K_real_k4" in r} if a.ksweep else set()
        bad = [i for i in pick if abs(stored[i]["entropy"] - ents[i]) > 1e-9]
        if bad:
            raise SystemExit(f"state indexing does not match the saved points: {bad[:5]}")
    else:
        gen = torch.Generator().manual_seed(0)
        pick = []
        for lo, hi in EDGES:
            band = [i for i, e in enumerate(ents) if lo <= e < hi]
            take = torch.randperm(len(band), generator=gen)[:a.per_bin].tolist()
            pick += [band[t] for t in take]
        if a.ksweep:
            sweep = set(pick[:a.ksweep_n])
    print(f"\n{len(pick)} points ({a.points}), k-sweep on {len(sweep)}, layers {layers}")

    done = {}
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["_i"]] = r
    print(f"checkpoint: {len(done)} points   budget {a.budget:.0f}s")

    rows, checks, t0, n_new = [], [], time.time(), 0
    for j, i in enumerate(pick):
        if i in done:
            rows.append(done[i])
            continue
        if time.time() - t0 > a.budget:
            print(f"   budget reached, +{n_new} new; rerun to continue", flush=True)
            break
        layer, h0 = states[i]
        idx = topk_indices(lm, h0, TOP_K)
        P = null_projector(lm, h0)
        # the same per-point scramble as run_scramble_within.py
        within = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(1000 + i))]

        if not checks:   # implementation checks, once per run
            F_real = frame_for(lm, h0, idx, P, K_DIM)
            F_scr = frame_for(lm, h0, idx, P, K_DIM, rows=within)
            x = 0.3 * torch.randn(K_DIM, dtype=torch.float64)
            p_ = metric_consistent(lm, h0 + F_real @ x, h0, F_real, idx, idx)
            q_ = fisher_metric_projected(lm, h0 + F_real @ x, F_real, idx)
            c_ = metric_consistent(lm, h0, h0, F_scr, idx, within)
            e_ = fisher_metric_projected(lm, h0, F_scr, idx, rows=within)
            checks.append(dict(identity_vs_real=float((p_ - q_).norm() / q_.norm()),
                               consistent_vs_within_at_h0=float((c_ - e_).norm() / e_.norm())))
            print(f"   check: rows=idx vs real {checks[0]['identity_vs_real']:.1e}   "
                  f"consistent vs within at h0 {checks[0]['consistent_vs_within_at_h0']:.1e}",
                  flush=True)

        rec = dict(_i=i, layer=layer, entropy=ents[i])
        for c, v in three_conditions(lm, h0, idx, P, within, K_DIM).items():
            rec[f"K_{c}"] = v
        if i in sweep:
            for k in (4, 6):
                for c, v in three_conditions(lm, h0, idx, P, within, k).items():
                    rec[f"K{k}_{c}"] = v
        rows.append(rec)
        n_new += 1
        with ckpt.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"   {j+1:>3}/{len(pick)}  L{layer:<2} H={ents[i]:5.2f}  real {rec['K_real']:+.4f}  "
              f"within {rec['K_within']:+.4f}  consistent {rec['K_consistent']:+.4f}"
              f"   ({time.time()-t0:.0f}s)", flush=True)

    summary = dict(model=a.model, points=a.points, k=K_DIM, top_k=TOP_K, n_planes=N_PLANES,
                   layers=layers, checks=checks, complete=len(rows) == len(pick),
                   k5=summarise(rows, "K"))
    if any("K4_real" in r for r in rows):
        summary["k4"] = summarise(rows, "K4")
        summary["k6"] = summarise(rows, "K6")
        summary["k5_on_sweep_points"] = summarise([r for r in rows if "K4_real" in r], "K")
    summary["by_entropy"] = {}
    for lo, hi in EDGES:
        sub = [r for r in rows if ebin(r["entropy"]) == ebin(lo)]
        if sub:
            summary["by_entropy"][ebin(lo)] = summarise(sub, "K")
    summary["by_layer"] = {str(l): summarise([r for r in rows if r["layer"] == l], "K")
                           for l in layers if any(r["layer"] == l for r in rows)}
    if stored:
        summary["reproduces_saved"] = dict(
            max_abs_diff_real=max(abs(r["K_real"] - stored[r["_i"]]["K_real"]) for r in rows),
            max_abs_diff_within=max(abs(r["K_within"] - stored[r["_i"]]["K_within"]) for r in rows))

    s5 = summary["k5"]
    print(f"\n{'='*72}\n{short}  {a.points}   n = {s5['n']}   k = 5\n{'='*72}")
    for c in CONDS:
        print(f"   {c:<11} median K = {s5[c]:+.4f}")
    for c in ("within", "consistent"):
        t = s5[f"real_gt_{c}"]
        print(f"   real > {c:<11} {t['pos']}/{t['n']}   z = {t['z']:+.2f}")
    for kk in ("k4", "k6"):
        if kk in summary:
            s = summary[kk]
            print(f"   {kk}: real {s['real']:+.4f}  within {s['within']:+.4f}  consistent "
                  f"{s['consistent']:+.4f}   real>consistent {s['real_gt_consistent']['pos']}/"
                  f"{s['real_gt_consistent']['n']} z={s['real_gt_consistent']['z']:+.2f}")
    if stored:
        print(f"   reproduces saved: {summary['reproduces_saved']}")

    out = OUT / f"consistent_{tag}.json"
    out.write_text(json.dumps(dict(summary=summary, rows=rows), indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
