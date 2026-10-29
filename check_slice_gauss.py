"""
Slice curvature against full-manifold curvature, on a case where the answer is known.

Every Riemann-derived number in this project is the sectional curvature of a
k-dimensional linear slice  S = h0 + span(F)  under the induced metric, not of
the full (quotient) manifold M.  By the Gauss equation, for an orthonormal pair
u, v spanning a plane in S,

    K_S(u, v) = K_M(u, v) + <II(u,u), II(v,v)> - |II(u,v)|^2

where II is the second fundamental form of S in M.  The question a reviewer
asked is how large that extrinsic term is.  On a real model K_M is unknown, so
it is measured here on ladder rung 7, where it is known exactly:

    p(h) = softmax(U h),   U : (N, N-1),   d = N - 1

h -> p is a diffeomorphism onto the simplex interior, so M is the simplex under
the Fisher metric and K_M = +1/4 for every plane at every point.  Whatever a
slice reports beyond +1/4 is therefore the extrinsic term, exactly.

Two frames, as in the transformer pipeline and against a null:
  eigen   top-k eigenvectors of G(h0)  (what every transformer number uses)
  random  a random orthonormal k-frame

Two sizes:
  N = 9     d = 8, so k runs up to d, where the slice IS the manifold -- the
            extrinsic term must vanish there (a check that cannot pass by luck)
  N = 512   the same retained-vocabulary size as the transformer runs, with the
            base point scaled to sweep predictive entropy

Curvature goes through fisherrao.curvature, the code path every transformer
number uses.

Usage:  python check_slice_gauss.py [points_per_cell]
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fisherrao.curvature import riemann, sectional

OUT = pathlib.Path("results/slice")
N_PLANES = 16
EDGES = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 1e9)]


def make_U(N: int, gen) -> torch.Tensor:
    """Random (N, N-1) unembedding whose columns + 1_N span R^N (as validate_pullback.py)."""
    while True:
        U = torch.randn(N, N - 1, generator=gen, dtype=torch.float64)
        M = torch.cat([U, torch.ones(N, 1, dtype=torch.float64)], dim=1)
        if torch.linalg.matrix_rank(M) == N:
            return U


def slice_metric(U, h0, F):
    """Induced metric on h0 + span(F): g(x) = M^T Sigma_p M, M = U F, p = softmax(U h0 + M x)."""
    M = U @ F
    z0 = U @ h0

    def g(x):
        p = torch.softmax(z0 + M @ x, -1)
        Mp = M.T @ p
        return (M.T * p) @ M - torch.outer(Mp, Mp)
    return g


def frame(U, h0, k, kind, gen):
    if kind == "random":
        Q, _ = torch.linalg.qr(torch.randn(U.shape[1], k, generator=gen, dtype=torch.float64))
        return Q
    p = torch.softmax(U @ h0, -1)
    G = U.T @ (torch.diag(p) - torch.outer(p, p)) @ U
    ev, evec = torch.linalg.eigh(G)
    return evec[:, torch.argsort(ev, descending=True)[:k]].contiguous()


def median_K(g, k, seed=0):
    x0 = torch.zeros(k, dtype=torch.float64)
    R = riemann(g, x0)
    pg = torch.Generator().manual_seed(seed)
    Ks = []
    for _ in range(N_PLANES):
        u = torch.randn(k, generator=pg, dtype=torch.float64)
        v = torch.randn(k, generator=pg, dtype=torch.float64)
        val = sectional(g, x0, u, v, R=R)
        if val == val:
            Ks.append(val)
    return float(torch.tensor(Ks).median()) if Ks else float("nan")


def med(xs):
    t = torch.tensor([x for x in xs if x == x], dtype=torch.float64)
    return float(t.median()) if len(t) else float("nan")


def ebin(e):
    for lo, hi in EDGES:
        if lo <= e < hi:
            return f"[{lo},{hi})" if hi < 1e9 else f"[{lo},inf)"


def run(N, ks, scales, n_pts, gen):
    U = make_U(N, gen)
    d = N - 1
    rows = []
    for s in scales:
        for _ in range(n_pts):
            h0 = s * torch.randn(d, generator=gen, dtype=torch.float64) / d ** 0.5
            p = torch.softmax(U @ h0, -1)
            H = float(-(p * p.log()).sum())
            for kind in ("eigen", "random"):
                for k in ks:
                    F = frame(U, h0, k, kind, gen)
                    K = median_K(slice_metric(U, h0, F), k)
                    rows.append(dict(N=N, k=k, frame=kind, scale=s, entropy=H, K=K,
                                     extrinsic=K - 0.25))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n_pts = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    gen = torch.Generator().manual_seed(0)
    t0 = time.time()

    small = run(9, list(range(2, 9)), [0.5, 2.0, 6.0], n_pts, gen)
    print(f"N = 9 done ({time.time()-t0:.0f}s)", flush=True)
    large = run(512, [2, 3, 4, 5, 6], [0.5, 2.0, 4.0, 8.0, 16.0], n_pts, gen)
    print(f"N = 512 done ({time.time()-t0:.0f}s)", flush=True)

    summary = {}
    print(f"\n{'='*76}\nN = 9 (d = 8): median |K_slice - 1/4| by k -- must vanish at k = d\n{'='*76}")
    print(f"   {'k':>3} {'eigen':>12} {'random':>12}")
    for k in range(2, 9):
        e = med([abs(r["extrinsic"]) for r in small if r["k"] == k and r["frame"] == "eigen"])
        q = med([abs(r["extrinsic"]) for r in small if r["k"] == k and r["frame"] == "random"])
        summary[f"N9_k{k}"] = dict(eigen=e, random=q)
        print(f"   {k:>3} {e:>12.2e} {q:>12.2e}")

    print(f"\n{'='*76}\nN = 512 (d = 511): median K_slice by entropy band, k = 5\n{'='*76}")
    print(f"   {'entropy':<11} {'n':>3} {'eigen K':>9} {'|ext|':>9} {'random K':>10} {'|ext|':>9}")
    for lo, hi in EDGES:
        lab = ebin(lo)
        sub = [r for r in large if r["k"] == 5 and ebin(r["entropy"]) == lab]
        if not sub:
            continue
        E = [r for r in sub if r["frame"] == "eigen"]
        Rr = [r for r in sub if r["frame"] == "random"]
        summary[f"N512_k5_{lab}"] = dict(
            n=len(E), eigen_K=med([r["K"] for r in E]), eigen_abs_ext=med([abs(r["extrinsic"]) for r in E]),
            random_K=med([r["K"] for r in Rr]), random_abs_ext=med([abs(r["extrinsic"]) for r in Rr]))
        s = summary[f"N512_k5_{lab}"]
        print(f"   {lab:<11} {s['n']:>3} {s['eigen_K']:>+9.4f} {s['eigen_abs_ext']:>9.4f} "
              f"{s['random_K']:>+10.4f} {s['random_abs_ext']:>9.4f}")

    print(f"\n   N = 512, eigen frames, median K by k (all entropies):")
    for k in [2, 3, 4, 5, 6]:
        v = med([r["K"] for r in large if r["k"] == k and r["frame"] == "eigen"])
        summary[f"N512_eigen_k{k}"] = v
        print(f"     k = {k}: {v:+.4f}")

    (OUT / "slice_gauss.json").write_text(
        json.dumps(dict(summary=summary, rows=small + large), indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'slice_gauss.json'}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
