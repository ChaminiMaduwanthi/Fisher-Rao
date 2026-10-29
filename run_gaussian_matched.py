"""
A learning-free genuine pullback, entropy-matched state by state.

The consistent control (run_scramble_consistent.py) keeps the trained rows and
destroys only their pairing with probabilities.  This control removes the
trained unembedding altogether: U is replaced by one fixed Gaussian matrix U_g,
and at each state its scale s is set so that the predictive entropy equals the
real model's entropy at that state.

    p(h) = softmax(s U_g n(h)),      H(p(h0)) = H_real(h0)

That map is a genuine pullback (p and its Jacobian come from the same U), it has
no learned structure at all, and it is matched to the real model on the one
quantity the spectral results show the geometry tracks.  The earlier Gaussian
control (08-rq3a-log.md S5.1) matched entropy only by bins across different
states; this one is paired state by state.

The real K at each state is read from the consistent-control run on the same
states, so the comparison is paired.  Frames, planes and k are the pipeline's.

Usage:
  python run_gaussian_matched.py --points paper                       # SmolLM2, 186 states
  python run_gaussian_matched.py --model JackFram/llama-160m --points draw
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fisherrao import LM, corpus
from fisherrao.metrics import fisher_metric_projected, topk_indices
from fisherrao.curvature import null_projector
from fisherrao.model import DEFAULT_MODEL
from run_scramble_consistent import (EDGES, SMOL_LAYERS, K_DIM, TOP_K, depths_for, frame_for,
                                     median_K, med, sign_test, ebin)

OUT = pathlib.Path("results/scramble")


def entropy_of(z):
    lp = torch.log_softmax(z, -1)
    return float(-(lp.exp() * lp).sum())


def matched_scale(z, target, iters=80):
    """s with H(softmax(s z)) = target; H falls monotonically as s grows."""
    lo, hi = -12.0, 12.0                                  # log s
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if entropy_of(math.exp(mid) * z) > target:
            lo = mid
        else:
            hi = mid
    return math.exp(0.5 * (lo + hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--points", choices=("draw", "paper"), default="paper")
    ap.add_argument("--budget", type=float, default=20000.0)
    a = ap.parse_args()

    short = a.model.split("/")[-1]
    tag = f"{short}_{a.points}"
    src = OUT / f"consistent_{tag}_points.jsonl"
    ckpt = OUT / f"gaussian_{tag}_points.jsonl"
    real = {}
    for line in src.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if "error" not in r:
                real.setdefault(r["_i"], r)
    print(f"{len(real)} states with a real K from {src}")

    lm = LM(a.model)
    print(lm.summary())
    layers = SMOL_LAYERS if "SmolLM2" in a.model else depths_for(lm.n_layers)
    states = []
    for s in corpus.sentences():
        H, _ = lm.residual_stream(s)
        for l in layers:
            for t in range(H.shape[1]):
                states.append((l, H[l, t]))

    U_real = lm.U
    U_g = torch.randn(U_real.shape, generator=torch.Generator().manual_seed(0),
                      dtype=torch.float64) / U_real.shape[1] ** 0.5

    done = {}
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["_i"]] = r
    print(f"checkpoint: {len(done)} states   budget {a.budget:.0f}s")

    rows, t0, n_new = [], time.time(), 0
    for j, (i, rr) in enumerate(real.items()):
        if i in done:
            rows.append(done[i])
            continue
        if time.time() - t0 > a.budget:
            print(f"   budget reached, +{n_new} new; rerun to continue", flush=True)
            break
        layer, h0 = states[i]
        if abs(layer - rr["layer"]) > 0:
            raise SystemExit(f"state {i}: layer {layer} does not match the saved {rr['layer']}")
        n0 = lm.apply_norm(h0)
        zg = U_g @ n0
        s = matched_scale(zg, rr["entropy"])
        lm.U = s * U_g
        rec = dict(_i=i, layer=layer, entropy=rr["entropy"], entropy_gauss=entropy_of(s * zg),
                   scale=s, K_real=rr["K_real"])
        try:
            idx = topk_indices(lm, h0, TOP_K)
            P = null_projector(lm, h0)
            F = frame_for(lm, h0, idx, P, K_DIM)
            rec["K_gauss"] = median_K(lambda x: fisher_metric_projected(lm, h0 + F @ x, F, idx), K_DIM)
        except Exception as e:                                   # noqa: BLE001
            rec["error"] = repr(e)
        lm.U = U_real
        rows.append(rec)
        n_new += 1
        with ckpt.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"   {j+1:>3}/{len(real)}  L{layer:<2} H={rr['entropy']:5.2f} (gauss {rec['entropy_gauss']:5.2f})"
              f"  real {rr['K_real']:+.4f}  gauss {rec.get('K_gauss', float('nan')):+.4f}"
              f"   ({time.time()-t0:.0f}s)", flush=True)

    ok = [r for r in rows if "error" not in r]
    summary = dict(model=a.model, points=a.points, n=len(ok), n_errors=len(rows) - len(ok),
                   complete=len(rows) == len(real),
                   max_entropy_mismatch=max(abs(r["entropy"] - r["entropy_gauss"]) for r in ok) if ok else None,
                   K_real=med([r["K_real"] for r in ok]), K_gauss=med([r["K_gauss"] for r in ok]),
                   real_gt_gauss=sign_test([r["K_real"] - r["K_gauss"] for r in ok]),
                   by_entropy={})
    for lo, hi in EDGES:
        sub = [r for r in ok if ebin(r["entropy"]) == ebin(lo)]
        if sub:
            summary["by_entropy"][ebin(lo)] = dict(
                n=len(sub), K_real=med([r["K_real"] for r in sub]), K_gauss=med([r["K_gauss"] for r in sub]),
                real_gt_gauss=sign_test([r["K_real"] - r["K_gauss"] for r in sub]))
    t = summary["real_gt_gauss"]
    print(f"\n{'='*72}\n{short} {a.points}: Gaussian U, entropy matched per state   n = {summary['n']}"
          f"\n{'='*72}\n   real {summary['K_real']:+.4f}   gauss {summary['K_gauss']:+.4f}   "
          f"real > gauss {t['pos']}/{t['n']}  z = {t['z']:+.2f}   "
          f"(max entropy mismatch {summary['max_entropy_mismatch']:.1e})")
    for b, v in summary["by_entropy"].items():
        print(f"   {b:<11} n {v['n']:>3}  real {v['K_real']:+.4f}  gauss {v['K_gauss']:+.4f}  "
              f"{v['real_gt_gauss']['pos']}/{v['real_gt_gauss']['n']}  z {v['real_gt_gauss']['z']:+.2f}")
    out = OUT / f"gaussian_{tag}.json"
    out.write_text(json.dumps(dict(summary=summary, rows=rows), indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
