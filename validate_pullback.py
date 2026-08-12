"""
RUNG 7 of the validation ladder -- the pullback step, against a known answer.

WHY THIS IS THE ONE THAT WAS MISSING
------------------------------------
Rungs 2, 3 and 6 validate the CURVATURE machinery (Christoffel symbols, the
Riemann tensor, the contraction) against manifolds whose metrics are written
down directly.  Gate A and the null-space test validate the PULLBACK, but only
INDIRECTLY -- Gate A checks that G is the Hessian of KL, and the null test
checks that G's kernel is the model's kernel.  Neither checks that

    G(h) = J^T (diag(p) - p p^T) J ,     J = U * A(h)

is ASSEMBLED correctly, and a transpose error, an index swap, or a missing
outer-product term would survive both.  03-methodology.md S5 lists this as rung
7 and it had not been run.

THE CONSTRUCTION, and why the answer is known
---------------------------------------------
Take a synthetic linear model with NO normalisation layer, N outcomes and
d = N-1 hidden dimensions:

    p(h) = softmax(U h) ,    U : (N, d)

If U's columns together with 1_N span R^N, then h -> p is a DIFFEOMORPHISM onto
the interior of the (N-1)-simplex: the logits determine p up to the all-ones
direction, which softmax quotients out, and d = N-1 is exactly the remaining
freedom.

Curvature is invariant under reparameterisation.  So the pullback metric on
h-space is the Fisher-Rao metric of the simplex written in different
coordinates, and therefore

        K = +1/4  EXACTLY, at every h, for every valid U.

That is a known answer that depends on the pullback being right.  Get the
transpose wrong and the metric is no longer the simplex metric in disguise.

FOUR CHECKS, each isolating something different
-----------------------------------------------
    1  N=3,4,5 -- K = +1/4 through the pullback, several dimensions
    2  reparameterisation h = M x  ->  U becomes U M.  Same manifold, different
       coordinates, so K must be UNCHANGED.  This is what catches index-order
       and transpose errors, because it compares the code against itself under
       a transformation it must be blind to.
    3  scalar R = k(k-1)/4 -- validates the Ricci contraction through the
       pullback, not just `sectional`.
    4  a DELIBERATELY BROKEN metric (the outer-product term dropped) must FAIL.
       A test that cannot fail proves nothing.

Usage:  python validate_pullback.py
"""

from __future__ import annotations

import sys

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fisherrao.metrics import fisher_metric
from fisherrao.curvature import sectional, scalar_curvature


class LinearLM:
    """A synthetic model with an identity normalisation layer.

    Exposes exactly the surface `fisher_metric` uses, so the test exercises the
    PROJECT'S assembly code rather than a re-implementation of it -- which is
    the whole point: a bespoke reference metric here would test nothing.
    """

    def __init__(self, U: torch.Tensor):
        self.U = U
        self.N, self.d = U.shape
        self.g = None
        self.bias = None
        self.eps = 0.0
        self.norm_kind = "Identity"

    def apply_norm(self, h):
        return h

    def norm_jacobian(self, h):
        return torch.eye(self.d, dtype=h.dtype)

    def logits(self, h):
        return self.apply_norm(h) @ self.U.T

    def next_token_probs(self, h):
        return torch.softmax(self.logits(h), dim=-1)


def make_U(N: int, gen) -> torch.Tensor:
    """A random (N, N-1) unembedding whose columns + 1_N span R^N.

    Drawn at random and checked, rather than constructed to be nice: a U that
    happened to be special could hide an assembly error.
    """
    while True:
        U = torch.randn(N, N - 1, generator=gen, dtype=torch.float64)
        M = torch.cat([U, torch.ones(N, 1, dtype=torch.float64)], dim=1)
        if torch.linalg.matrix_rank(M) == N:
            return U


def metric_fn(lm, broken: bool = False):
    """g(h) through the project's own fisher_metric.

    `broken=True` drops the -(J^T p)(J^T p)^T term, i.e. uses J^T diag(p) J
    instead of J^T Sigma_p J.  That is the single most plausible assembly
    error, and check 4 requires it to be caught.
    """
    if not broken:
        return lambda h: fisher_metric(lm, h, top_k=None)

    def g_broken(h):
        p = lm.next_token_probs(h)
        J = lm.U @ lm.norm_jacobian(h)
        return (J.T * p) @ J                       # outer-product term dropped
    return g_broken


def worst_K(g_fn, d, n_points, gen, expect):
    """Largest |K - expect| over random base points and random 2-planes."""
    worst = 0.0
    for _ in range(n_points):
        h = torch.randn(d, generator=gen, dtype=torch.float64) * 0.7
        u = torch.randn(d, generator=gen, dtype=torch.float64)
        v = torch.randn(d, generator=gen, dtype=torch.float64)
        K = sectional(g_fn, h, u, v)
        if K == K:
            worst = max(worst, abs(K - expect))
    return worst


def main() -> int:
    gen = torch.Generator().manual_seed(7)
    ok = True

    print(f"{'='*74}\nRUNG 7 -- THE PULLBACK STEP, AGAINST A KNOWN ANSWER\n{'='*74}")
    print("p(h) = softmax(U h), d = N-1, no normalisation layer.")
    print("h -> p is then a diffeomorphism onto the simplex interior, so the")
    print("pullback metric IS the simplex metric in other coordinates and")
    print("K = +1/4 exactly, for every valid U.\n")

    # ---- 1. K = +1/4 through the pullback --------------------------------
    print("1. SECTIONAL CURVATURE THROUGH THE PULLBACK")
    print(f"   {'N':>3} {'d':>3} {'points':>7} {'expect':>8} {'worst |err|':>13}   verdict")
    Us = {}
    for N in (3, 4, 5):
        U = make_U(N, gen)
        Us[N] = U
        lm = LinearLM(U)
        w = worst_K(metric_fn(lm), N - 1, 4, gen, 0.25)
        good = w < 1e-6
        ok &= good
        print(f"   {N:>3} {N-1:>3} {4:>7} {0.25:>8.2f} {w:>13.2e}   "
              f"{'PASS' if good else '**FAIL**'}")

    # ---- 2. reparameterisation invariance --------------------------------
    print("\n2. REPARAMETERISATION INVARIANCE  (h = M x, so U -> U M)")
    print("   Same manifold, different coordinates: K must be UNCHANGED.")
    print("   This is what catches transpose and index-order errors.")
    print(f"   {'N':>3} {'K(U)':>12} {'K(U M)':>12} {'|diff|':>12}   verdict")
    for N in (3, 4, 5):
        d = N - 1
        U = Us[N]
        M = torch.randn(d, d, generator=gen, dtype=torch.float64)
        while abs(float(torch.linalg.det(M))) < 0.1:
            M = torch.randn(d, d, generator=gen, dtype=torch.float64)
        h = torch.randn(d, generator=gen, dtype=torch.float64) * 0.7
        u = torch.randn(d, generator=gen, dtype=torch.float64)
        v = torch.randn(d, generator=gen, dtype=torch.float64)

        K1 = sectional(metric_fn(LinearLM(U)), h, u, v)
        # the same point and the same 2-plane, expressed in x-coordinates
        Minv = torch.linalg.inv(M)
        K2 = sectional(metric_fn(LinearLM(U @ M)), Minv @ h, Minv @ u, Minv @ v)
        diff = abs(K1 - K2)
        good = diff < 1e-6
        ok &= good
        print(f"   {N:>3} {K1:>12.8f} {K2:>12.8f} {diff:>12.2e}   "
              f"{'PASS' if good else '**FAIL**'}")

    # ---- 3. scalar curvature through the pullback ------------------------
    print("\n3. SCALAR CURVATURE THROUGH THE PULLBACK   (R = k(k-1)/4)")
    print(f"   {'N':>3} {'k':>3} {'expect':>9} {'measured':>12} {'rel.err':>11}   verdict")
    for N in (3, 4, 5):
        d = N - 1
        lm = LinearLM(Us[N])
        h = torch.randn(d, generator=gen, dtype=torch.float64) * 0.7
        exp = d * (d - 1) * 0.25
        got = scalar_curvature(metric_fn(lm), h)
        if exp == 0:
            rel, good = abs(got), abs(got) < 1e-6
        else:
            rel = abs(got - exp) / abs(exp)
            good = rel < 1e-6
        ok &= good
        print(f"   {N:>3} {d:>3} {exp:>9.3f} {got:>12.8f} {rel:>11.2e}   "
              f"{'PASS' if good else '**FAIL**'}")

    # ---- 4. the negative control -----------------------------------------
    print("\n4. NEGATIVE CONTROL -- a deliberately broken assembly MUST fail")
    print("   Drops the -(J^T p)(J^T p)^T term, i.e. uses J^T diag(p) J.")
    print("   A test that cannot fail proves nothing.")
    N = 4
    lm = LinearLM(Us[N])
    w_broken = worst_K(metric_fn(lm, broken=True), N - 1, 4, gen, 0.25)
    caught = w_broken > 1e-3
    ok &= caught
    print(f"   worst |K - 1/4| with the term dropped: {w_broken:.4f}   "
          f"{'CAUGHT' if caught else '**NOT CAUGHT -- the test is vacuous**'}")

    print(f"\n{'='*74}")
    print(f"RUNG 7: {'PASS' if ok else 'FAIL'}"
          f"   -- the pullback assembly is now validated DIRECTLY,")
    print("   not only through Gate A and the null-space test.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
