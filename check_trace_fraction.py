"""
What fraction of the metric's trace does the retained k-dimensional slice hold?

The paper states this number, and it had never been measured: 03-methodology.md
quotes ">= 92% of the trace" but for a `cond_eff`-SELECTED `k`, which is not what
the curvature results use.  Those hold `k` fixed at 5.  Those are different
quantities and the second one is the one that needs reporting.

Measured here on the quotiented metric `P G P` -- the trace of the retained
top-k eigenvalues over the trace of the whole quotiented metric -- at the same
layers the E1 profile samples, so the number describes the actual working
subspace.

Usage:  python check_trace_fraction.py [model_id]
"""

from __future__ import annotations

import json
import pathlib
import sys
import warnings

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

from fisherrao import LM, corpus
from fisherrao.metrics import fisher_metric
from fisherrao.curvature import null_projector

KS = (4, 5, 6)
N_STATES = 60


def main() -> int:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-135M-Instruct"
    lm = LM(model_id)
    layers = [1, 5, 10, 15, 20, 25, lm.n_layers - 2, lm.n_layers - 1, lm.n_layers]
    layers = sorted({l for l in layers if 0 < l <= lm.n_layers})

    sents = corpus.sentences()[:40]
    gen = torch.Generator().manual_seed(0)

    frac = {k: [] for k in KS}
    n = 0
    for s in sents:
        if n >= N_STATES:
            break
        H, _ = lm.residual_stream(s)
        T = H.shape[1]
        for L in layers:
            if n >= N_STATES:
                break
            t = int(torch.randint(max(1, T // 3), T, (1,), generator=gen))
            h = H[L, t].double()
            G = fisher_metric(lm, h, top_k=512)
            P = null_projector(lm, h)
            ev = torch.linalg.eigvalsh(P @ G @ P).flip(0).clamp(min=0)
            tot = float(ev.sum())
            if tot <= 0:
                continue
            for k in KS:
                frac[k].append(float(ev[:k].sum()) / tot)
            n += 1

    print(f"\n{'='*70}\nFRACTION OF THE QUOTIENTED METRIC'S TRACE HELD BY THE TOP k\n{'='*70}")
    print(f"model: {model_id}   states: {n}   layers: {layers}\n")
    print(f"  {'k':>3} {'median':>9} {'mean':>9} {'min':>9} {'max':>9}")
    out = {}
    for k in KS:
        v = torch.tensor(frac[k])
        out[k] = dict(median=float(v.median()), mean=float(v.mean()),
                      min=float(v.min()), max=float(v.max()), n=len(v))
        print(f"  {k:>3} {float(v.median()):>9.1%} {float(v.mean()):>9.1%} "
              f"{float(v.min()):>9.1%} {float(v.max()):>9.1%}")

    p = pathlib.Path("results/tracefrac")
    p.mkdir(parents=True, exist_ok=True)
    (p / "tracefrac.json").write_text(
        json.dumps(dict(model=model_id, n=n, layers=layers, frac=out), indent=2),
        encoding="utf-8")
    print(f"\nsaved -> {p/'tracefrac.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
