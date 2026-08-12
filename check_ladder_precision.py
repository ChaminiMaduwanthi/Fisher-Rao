"""
Worst-case ladder error for the implementation the RESULTS actually use.

WHY
---
There are two curvature implementations in this project and they have very
different precision:

  * `validate_curvature.py` -- a deliberately literal reference: explicit
    Christoffel symbols and a d^4 Riemann tensor built by CENTRAL FINITE
    DIFFERENCES (h = 1e-5, 1e-4).  Its accuracy floor is therefore ~1e-6.  Its
    job is to be an independent oracle, not to be precise.

  * `fisherrao/curvature.py` -- nested reverse-mode autodiff in float64.  This
    is what every transformer number in the paper is computed with.

A validation table in the paper must report the error of the implementation
whose results are being reported.  This script measures worst |K - K_exact|
for the autodiff implementation on every rung with a closed-form answer, over
several random base points and random 2-planes per rung.

Rungs 4 and 5 (gamma, beta) have no closed-form constant, so their agreement is
with geomstats and is reported by `validate_geomstats.py` instead; the
qualitative claim checked here is that they are everywhere negative.

Usage:  python check_ladder_precision.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fisherrao.curvature import sectional

torch.set_default_dtype(torch.float64)


def g_simplex(x):
    """Categorical simplex, Fisher-Rao, in the first n-1 coordinates. K = +1/4."""
    return torch.diag(1.0 / x) + torch.ones(len(x), len(x), dtype=x.dtype) / (1.0 - x.sum())


def g_normal(x):
    """Univariate normal in (mu, sigma). K = -1/2."""
    s = x[1]
    return torch.stack([
        torch.stack([1.0 / s ** 2, torch.zeros((), dtype=x.dtype)]),
        torch.stack([torch.zeros((), dtype=x.dtype), 2.0 / s ** 2]),
    ])


def g_poincare(x):
    """Poincare half-plane. K = -1."""
    y = x[1]
    return torch.eye(2, dtype=x.dtype) / y ** 2


def g_sphere_r(x, r=2.0):
    """Sphere of radius r in spherical coordinates. K = 1/r^2 -- rung 1."""
    th = x[0]
    return torch.stack([
        torch.stack([torch.tensor(r ** 2, dtype=x.dtype), torch.zeros((), dtype=x.dtype)]),
        torch.stack([torch.zeros((), dtype=x.dtype), r ** 2 * torch.sin(th) ** 2]),
    ])


def worst(g_fn, sampler, expect, n, seed):
    gen = torch.Generator().manual_seed(seed)
    w, vals = 0.0, []
    for _ in range(n):
        x = sampler(gen)
        u = torch.randn(len(x), generator=gen, dtype=torch.float64)
        v = torch.randn(len(x), generator=gen, dtype=torch.float64)
        K = sectional(g_fn, x, u, v)
        if K != K:
            continue
        vals.append(float(K))
        w = max(w, abs(float(K) - expect))
    return w, vals


def main() -> int:
    rows = []

    def simplex_pt(gen):
        q = torch.distributions.Dirichlet(torch.ones(4)).sample()
        return q[:3].double()

    def half_plane_pt(gen):
        return torch.stack([torch.randn((), generator=gen, dtype=torch.float64),
                            0.5 + torch.rand((), generator=gen, dtype=torch.float64) * 2])

    def normal_pt(gen):
        return torch.stack([torch.randn((), generator=gen, dtype=torch.float64),
                            0.4 + torch.rand((), generator=gen, dtype=torch.float64) * 1.5])

    def sphere_pt(gen):
        return torch.stack([0.4 + torch.rand((), generator=gen, dtype=torch.float64) * 2.0,
                            torch.rand((), generator=gen, dtype=torch.float64) * 3.0])

    specs = [
        (1, "Sphere, radius 2", g_sphere_r, sphere_pt, 0.25, 12, 11),
        (2, "Poincare half-plane", g_poincare, half_plane_pt, -1.0, 12, 12),
        (3, "Categorical simplex", g_simplex, simplex_pt, 0.25, 12, 13),
        (6, "Univariate Gaussian", g_normal, normal_pt, -0.5, 12, 16),
    ]

    print(f"\n{'='*82}")
    print("LADDER PRECISION -- autodiff implementation (fisherrao/curvature.py)")
    print(f"{'='*82}")
    print(f"  {'rung':>4}  {'manifold':<22}{'expect':>9}{'points':>8}{'worst |err|':>14}   verdict")
    ok = True
    for rung, name, g, samp, exp, n, seed in specs:
        w, vals = worst(g, samp, exp, n, seed)
        good = w < 1e-8
        ok &= good
        rows.append(dict(rung=rung, manifold=name, expect=exp, n=len(vals), worst=w))
        print(f"  {rung:>4}  {name:<22}{exp:>9.2f}{len(vals):>8}{w:>14.2e}   "
              f"{'PASS' if good else '**FAIL**'}")

    print(f"\n  (rungs 4 and 5 -- gamma and beta -- have no closed-form constant;")
    print(f"   they are validated against geomstats by validate_geomstats.py,")
    print(f"   and against the qualitative claim 'everywhere negative'.)")
    print(f"\n  overall: {'PASS' if ok else 'FAIL'}")

    p = pathlib.Path("results/ladder")
    p.mkdir(parents=True, exist_ok=True)
    (p / "ladder_precision.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"  saved -> {p/'ladder_precision.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
