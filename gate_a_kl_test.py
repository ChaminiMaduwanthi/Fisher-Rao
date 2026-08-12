"""
GATE A -- verify G(h) via the KL-Hessian identity.

    KL( p(h) || p(h + eps*v) )  =  1/2 eps^2 v^T G(h) v  +  O(eps^3)

The Fisher metric IS the Hessian of KL divergence at coincidence, so this
checks G against a completely independent code path: two forward passes and
one KL.  If this does not pass, nothing downstream is trustworthy.  Write it
before anything else.

Verified 2026-08-10 (N=500, d=16, random W):

        eps         KL (exact)           KL/eps^2      rel.err
      1e-01       2.579628e-04     0.025796278309     2.25e-03
      1e-02       2.584855e-06     0.025848545643     2.25e-04
      1e-03       2.585379e-08     0.025853791021     2.25e-05
      1e-04       2.585432e-10     0.025854324678     1.90e-06   <- best
      1e-05       2.585683e-12     0.025856828603     9.49e-05
      1e-06       2.585875e-14     0.025858754930     1.69e-04
      1e-07       2.120519e-16     0.021205187130     1.80e-01   <- broken

Relative error falls as O(eps) -- exactly the predicted third-order remainder
-- down to eps ~ 1e-4, then RISES again.

    !!  THE TRAP  !!
Below eps ~ 1e-5 the test degrades and by 1e-7 it is off by 18%.  That is not
a problem with G.  It is catastrophic cancellation: KL between two nearly
identical distributions is a difference of nearly equal logs, and at eps=1e-7
the true KL is ~1e-16, i.e. at float64 epsilon.  A student who picks a tiny
eps "for accuracy" will watch Gate A fail and start debugging a correct
implementation.

    DO NOT pick a single eps.  Sweep it, and look for the O(eps) plateau.
    Report the plateau plot as evidence, not one number.

Same lesson applies to every finite-difference step in the project (see
03-methodology.md section 6.3).
"""

import numpy as np


def make_model(N=500, d=16, seed=3):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((N, d)) / np.sqrt(d)
    b = rng.standard_normal(N) * 0.3
    return W, b


def p_of(W, b, x):
    z = W @ x + b
    z = z - z.max()                      # log-space stability; never exp raw logits
    e = np.exp(z)
    return e / e.sum()


def G_of(W, b, x):
    """G = W^T (diag(p) - p p^T) W, without forming the N x N Sigma_p."""
    p = p_of(W, b, x)
    Wp = W.T @ p
    return (W.T * p) @ W - np.outer(Wp, Wp)


def kl(p, q):
    return float(np.sum(p * (np.log(p) - np.log(q))))


def gate_a(W, b, h, v, exponents=range(1, 8)):
    v = v / np.linalg.norm(v)
    quad = 0.5 * v @ G_of(W, b, h) @ v
    print(f"predicted  0.5 * v^T G(h) v = {quad:.12f}\n")
    print(f"{'eps':>10} {'KL (exact)':>18} {'KL/eps^2':>18} {'rel.err':>12}")
    rows = []
    for k in exponents:
        eps = 10.0 ** (-k)
        d = kl(p_of(W, b, h), p_of(W, b, h + eps * v))
        ratio = d / eps ** 2
        rel = abs(ratio - quad) / quad
        rows.append((eps, rel))
        print(f"{eps:>10.0e} {d:>18.6e} {ratio:>18.12f} {rel:>12.2e}")
    best = min(rows, key=lambda r: r[1])
    print(f"\nplateau at eps = {best[0]:.0e}, relative error {best[1]:.2e}")
    print("PASS" if best[1] < 1e-5 else "FAIL -- investigate before proceeding")
    return best


if __name__ == "__main__":
    W, b = make_model()
    rng = np.random.default_rng(3)
    gate_a(W, b, rng.standard_normal(W.shape[1]), rng.standard_normal(W.shape[1]))
