"""
Validation-ladder reference implementation -- rungs 2, 3 and 6.

Verified working 2026-08-10.  Output:

    RUNG 3  categorical simplex (n=4)      K = +0.250001   (expect +1/4)
    RUNG 6  univariate Gaussian            K = -0.500000   (expect -1/2)
    RUNG 2  Poincare half-plane            K = -1.000000   (expect -1)

This is deliberately the slowest, most literal implementation possible:
explicit Christoffel symbols, explicit d^4 Riemann tensor, central finite
differences.  It does NOT scale -- that is fine.  Its only job is to be an
oracle you can trust while you build the fast version.  Keep it in the test
suite forever and check the fast implementation against it.

    !!  SIGN CONVENTION  !!
The first version of this script returned K = -0.250000 for the simplex --
correct magnitude, wrong sign -- purely from contracting the lowered Riemann
tensor in the order R(u,v,u,v) instead of R(u,v,v,u).  Nothing warned about
it; the number just looked plausible.  This is the single most likely silent
bug in the whole project.  Fix the convention once, here, against a known
answer, and never change it.

Convention used throughout:
    R^r_sij  such that  R(e_i, e_j) e_s = R^r_sij e_r
    R_msij   = g_mr R^r_sij = g( R(e_i,e_j) e_s , e_m )
    K(u,v)   = R(u,v,v,u) / (|u|^2 |v|^2 - <u,v>^2)
             = R_msij u^m v^s u^i v^j / (...)
"""

import numpy as np


def curvature_tools(g_fn, d, h1=1e-5, h2=1e-4):
    """Build sectional-curvature machinery for a metric g_fn: R^d -> R^{dxd}."""

    def dg(th):
        out = np.zeros((d, d, d))
        for k in range(d):
            e = np.zeros(d); e[k] = h1
            out[:, :, k] = (g_fn(th + e) - g_fn(th - e)) / (2 * h1)
        return out

    def christoffel(th):
        # Gamma^k_ij = 1/2 g^kl ( d_i g_jl + d_j g_il - d_l g_ij )
        Gi = np.linalg.inv(g_fn(th))
        dG = dg(th)
        T = np.einsum('jli->ijl', dG) + np.einsum('ilj->ijl', dG) - dG
        return 0.5 * np.einsum('kl,ijl->kij', Gi, T)

    def riemann(th):
        dGam = np.zeros((d, d, d, d))
        for m in range(d):
            e = np.zeros(d); e[m] = h2
            dGam[:, :, :, m] = (christoffel(th + e) - christoffel(th - e)) / (2 * h2)
        Gam = christoffel(th)
        R = np.zeros((d, d, d, d))
        for r in range(d):
            for s in range(d):
                for i in range(d):
                    for j in range(d):
                        R[r, s, i, j] = (
                            dGam[r, j, s, i] - dGam[r, i, s, j]
                            + sum(Gam[r, i, l] * Gam[l, j, s]
                                  - Gam[r, j, l] * Gam[l, i, s] for l in range(d))
                        )
        return R

    def sectional(th, u, v):
        G = g_fn(th)
        Rlow = np.einsum('mr,rsij->msij', G, riemann(th))
        numer = np.einsum('msij,m,s,i,j->', Rlow, u, v, u, v)   # R(u,v,v,u)
        uu, vv, uv = u @ G @ u, v @ G @ v, u @ G @ v
        return numer / (uu * vv - uv ** 2)

    return sectional


# ----------------------------------------------------------------------
# RUNG 3 -- categorical simplex under Fisher-Rao.  Expect K = +1/4.
# THE key rung: same metric family as the thesis, analytically known.
# Isometric to the positive orthant of a sphere of RADIUS 2, so K = 1/2^2.
# Beware: some sources normalise to the unit sphere and quote K = 1 with the
# metric scaled by 4.  A factor-of-4 disagreement with a library is a
# convention mismatch, not necessarily a bug -- check before "fixing" it.
# ----------------------------------------------------------------------
def g_simplex(th):
    d = len(th)
    return np.diag(1.0 / th) + np.ones((d, d)) / (1.0 - th.sum())


# ----------------------------------------------------------------------
# RUNG 6 -- univariate Gaussian N(mu, sigma^2).  Expect K = -1/2.
# Fisher metric in (mu, sigma):  ds^2 = (dmu^2 + 2 dsigma^2) / sigma^2
# ----------------------------------------------------------------------
def g_gauss(th):
    s = th[1]
    return np.array([[1.0 / s ** 2, 0.0], [0.0, 2.0 / s ** 2]])


# ----------------------------------------------------------------------
# RUNG 2 -- Poincare half-plane.  Expect K = -1.
# ----------------------------------------------------------------------
def g_poincare(th):
    y = th[1]
    return np.eye(2) / y ** 2


def main():
    rng = np.random.default_rng(1)

    n = 4
    sec = curvature_tools(g_simplex, n - 1)
    print(f"RUNG 3  categorical simplex (n={n})      expect K = +0.25")
    for _ in range(4):
        q = rng.dirichlet(np.ones(n))
        u, v = rng.standard_normal(n - 1), rng.standard_normal(n - 1)
        print(f"   p={np.round(q, 3)}  K = {sec(q[:n - 1], u, v):+.6f}")

    sec = curvature_tools(g_gauss, 2)
    print("\nRUNG 6  univariate Gaussian            expect K = -0.50")
    for _ in range(4):
        th = np.array([rng.standard_normal(), 0.4 + abs(rng.standard_normal())])
        u, v = rng.standard_normal(2), rng.standard_normal(2)
        print(f"   (mu,sigma)={np.round(th, 3)}  K = {sec(th, u, v):+.6f}")

    sec = curvature_tools(g_poincare, 2)
    print("\nRUNG 2  Poincare half-plane            expect K = -1.00")
    for _ in range(3):
        th = np.array([rng.standard_normal(), 0.5 + abs(rng.standard_normal())])
        u, v = rng.standard_normal(2), rng.standard_normal(2)
        print(f"   (x,y)={np.round(th, 3)}  K = {sec(th, u, v):+.6f}")


if __name__ == "__main__":
    main()
