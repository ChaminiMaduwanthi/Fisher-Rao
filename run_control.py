"""
CONTROL for the Stage 3 result.

Stage 3 measured intrinsic Fisher-Rao sectional curvature ~ +0.25 at every
layer, 100% of sampled 2-planes positive.  +1/4 is exactly the constant
curvature of the categorical simplex under Fisher-Rao, so the first duty is to
rule out that the CONSTRUCTION produces it regardless of the model.

Test: replace the real unembedding U with random matrices at several scales,
which changes how concentrated p is, and re-measure.

Result (2026-08-10, SmolLM2-135M, layer 20, k=6, top_k=512):

    unembedding            entropy   K median    scalar R
    real U                   0.81     +0.2506      11.08
    random, randn*3          3.47     +0.2690      11.22
    random, randn/sqrt(d)    9.53     +0.1335       2.26
    random, randn*0.3       10.68     -0.0207      -0.67

    exact k-simplex reference (n=4, n=7)         +0.250000

CONCLUSION: +1/4 is NOT automatic.  Diffuse predictive distributions give
curvature far from +1/4 and even NEGATIVE.  The value tracks the
CONCENTRATION of p: a trained LM's next-token distribution is extremely peaked
(0.8-1.8 nats, versus ~10.7 for a random unembedding), so its representations
occupy a near-low-dimensional face of the simplex, where the induced geometry
is the ambient simplex geometry -- constant +1/4.

This is the control that licenses the Stage 3 claim.  Re-run it whenever the
metric, the frame construction, or the model changes.
"""

from __future__ import annotations

import copy
import json
import pathlib
import sys

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fisherrao import LM
from fisherrao.curvature import curvature_at, sectional

OUT = pathlib.Path("results/stage3")
PROMPT = ("The bank was steep and muddy after the heavy rain, so the hikers "
          "climbed down to the river")
K_DIM, TOP_K, LAYER = 6, 512, 20


def g_simplex(x):
    d = x.shape[-1]
    return torch.diag(1.0 / x) + torch.ones(d, d, dtype=x.dtype) / (1.0 - x.sum())


def entropy_of(lm, h):
    logp = torch.log_softmax(lm.logits(h), -1)
    return float(-(logp.exp() * logp).sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lm = LM()
    H, _ = lm.residual_stream(PROMPT)
    h = H[LAYER, -1]
    rows = []

    print("Is K = +0.25 model-specific, or produced by the construction?\n")
    print(f"{'unembedding':<24} {'entropy':>8} {'K median':>11} {'scalar R':>10}")

    c = curvature_at(lm, h, K_DIM, top_k=TOP_K, n_planes=32)
    print(f"{'real U':<24} {entropy_of(lm, h):>8.2f} {c['K_median']:>11.4f} "
          f"{c['scalar_R']:>10.2f}")
    rows.append(dict(kind="real", entropy=entropy_of(lm, h),
                     K_median=c["K_median"], scalar_R=c["scalar_R"]))

    torch.manual_seed(0)
    for scale, lbl in ((3.0, "random, randn*3"),
                       (1.0, "random, randn/sqrt(d)"),
                       (0.3, "random, randn*0.3")):
        lm2 = copy.copy(lm)
        lm2.U = torch.randn(lm.N, lm.d, dtype=torch.float64) * scale / lm.d**0.5
        c2 = curvature_at(lm2, h, K_DIM, top_k=TOP_K, n_planes=32)
        e = entropy_of(lm2, h)
        print(f"{lbl:<24} {e:>8.2f} {c2['K_median']:>11.4f} {c2['scalar_R']:>10.2f}")
        rows.append(dict(kind=lbl, entropy=e, K_median=c2["K_median"],
                         scalar_R=c2["scalar_R"]))

    print()
    gen = torch.Generator().manual_seed(3)
    for n in (4, 7):
        x = torch.distributions.Dirichlet(torch.ones(n, dtype=torch.float64)).sample()[:n - 1]
        u = torch.randn(n - 1, generator=gen, dtype=torch.float64)
        v = torch.randn(n - 1, generator=gen, dtype=torch.float64)
        K = sectional(g_simplex, x, u, v)
        print(f"{'exact simplex n=' + str(n):<24} {'-':>8} {K:>11.6f} {'-':>10}")
        rows.append(dict(kind=f"exact_simplex_n{n}", K_median=K))

    diffuse = [r for r in rows if r.get("entropy", 0) > 8]
    ok = all(abs(r["K_median"] - 0.25) > 0.05 for r in diffuse)
    print(f"\nCONTROL {'PASS' if ok else 'FAIL'}: diffuse-p cases "
          f"{'do' if ok else 'do NOT'} depart from +1/4, so +1/4 is not an")
    print("artifact of the construction.  Curvature tracks the concentration of p.")

    (OUT / "control.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'control.json'}")


if __name__ == "__main__":
    main()
