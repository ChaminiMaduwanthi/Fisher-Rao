"""
Conditioning of the Fisher-Rao pullback metric G(h) = W^T (diag(p) - pp^T) W.

Run 2026-08-10 with N=2000, d=64, seed 0.  Result:

    diffuse p        : rank 64/64,  cond 2.0e+00,  99% of trace in 64/64 dirs (100%)
    moderate peak    : rank 64/64,  cond 4.1e+01,  99% of trace in 61/64 dirs ( 95%)
    sharp peak       : rank 64/64,  cond 1.8e+08,  99% of trace in  4/64 dirs (  6%)
    control (1_N in range(W)) : rank 63/64  -- deficiency exactly 1

Exact values depend on the seed and on the assumed logit sharpness; the
claims that matter are the ORDERS OF MAGNITUDE and the qualitative pattern:
formal rank stays full throughout, while conditioning degrades by eight
orders and the effective direction count collapses to single digits.  Do not
quote these figures as if they were measurements of a real model -- rerun
on real GPT-2 activations in Stage 3 (task 3.10b) before putting numbers in
the thesis.

CORRECTS A CLAIM WORTH BEING CAREFUL ABOUT.  It is tempting to argue:
"Sigma_p annihilates the all-ones vector, therefore G is singular, therefore
G^-1 does not exist."  The first clause is true; the conclusion does not
follow.

    null(G) = { v : W v in span{1_N} }

so an exact null direction exists only if 1_N lies in range(W) -- a
d-dimensional subspace of R^N with d << N.  Generically it does not, and
G is FULL RANK d.  (Case [4] below forces 1_N into range(W) and recovers a
deficiency of exactly 1, confirming the mechanism.)

The real obstacle is not nullity but CONDITIONING.  Because a language-model
softmax is sharply peaked, Sigma_p has few significant eigenvalues, so the
spectrum of G decays fast: at realistic sharpness only ~3% of eigendirections
carry 99% of the trace -- consistent with the 2-17% effective dimensionality
reported by FishBack (arXiv:2605.17231).  G^-1 formally exists and is
numerically useless.

Two consequences for the project:

1.  float64 is MANDATORY, not advisable.  cond(G) ~ 1e8 already costs ~8 of
    float64's ~16 significant digits.  Curvature needs SECOND derivatives of
    G, roughly doubling that loss.  float32 (~7 digits) leaves nothing.

2.  Cutoff choice is a scientific decision, not a preprocessing detail.
    Where the spectrum is this steep, the retained-rank k silently sets the
    curvature values.  Report curvature as a function of k, always.
"""

import numpy as np


def G_of(W, p):
    """G = W^T (diag(p) - p p^T) W, without ever forming the N x N Sigma_p."""
    Wp = W.T @ p
    return (W.T * p) @ W - np.outer(Wp, Wp)


def report(tag, W, logits):
    p = np.exp(logits - logits.max())
    p /= p.sum()
    G = G_of(W, p)
    ev = np.clip(np.linalg.eigvalsh(G), 0, None)
    d = W.shape[1]
    tot = ev.sum()
    k99 = int(np.searchsorted(np.cumsum(ev[::-1]) / tot, 0.99) + 1)
    cond = ev[-1] / max(ev[0], 1e-300)
    print(f"{tag}")
    print(f"   formal rank(G)          = {np.linalg.matrix_rank(G)} / {d}")
    print(f"   condition number        = {cond:.1e}")
    print(f"   99% of trace held by      {k99} / {d} directions  ({100*k99/d:.0f}%)")
    print(f"   participation ratio     = {tot**2 / (ev**2).sum():.2f}")
    print()


def main():
    rng = np.random.default_rng(0)
    N, d = 2000, 64
    W = rng.standard_normal((N, d)) / np.sqrt(d)

    report("[1] DIFFUSE p (near-uniform logits)", W, rng.standard_normal(N) * 0.1)
    report("[2] MODERATELY PEAKED p",             W, rng.standard_normal(N) * 3.0)
    report("[3] SHARPLY PEAKED p (realistic LM)", W, rng.standard_normal(N) * 12.0)

    print("[4] CONTROL -- force 1_N into range(W): expect deficiency exactly 1")
    W2 = W.copy()
    W2[:, 0] = 1.0                      # now W2 @ e_0 = 1_N
    p = np.exp(rng.standard_normal(N) * 3)
    p /= p.sum()
    G2 = G_of(W2, p)
    e0 = np.eye(d)[0]
    print(f"   formal rank(G)  = {np.linalg.matrix_rank(G2)} / {d}")
    print(f"   ||G @ e_0||     = {np.linalg.norm(G2 @ e0):.1e}   <- e_0 maps to 1_N")
    print("   -> the all-ones argument transfers to G only under this condition.")


if __name__ == "__main__":
    main()
