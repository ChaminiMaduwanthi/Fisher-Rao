"""
The three metrics on residual-stream space.

    euclidean(h)  ->  I            no metric at all       (King et al. 2026)
    manson(h)     ->  U^T U        constant, state-free   (Manson 2025)
    fisher(h)     ->  J^T Sigma_p J   state-dependent     (this project)

The whole thesis is the difference between the third and the first two.  A
constant metric has identically zero Riemann curvature, so `manson` and
`euclidean` can only ever measure how a PATH bends, never how the SPACE
bends.  See 03-methodology.md section 3.

    !!  G(h) HAS AN EXACT NULL DIRECTION: h ITSELF  !!

RMSNorm is scale-invariant.  Since mean(h^2) = |h|^2/d, the norm Jacobian is

    A = diag(g) (1/r) ( I - h h^T/(r^2 d) ) = diag(g) (1/r) ( I - hhat hhat^T )

a PROJECTOR that annihilates the radial direction.  So G(h) = A^T (...) A has
rank <= d-1 with null direction h, and doubling a hidden state changes the
model's prediction by nothing:

    KL( p(h) || p(2h) )      = 1.6e-16      (measured, layer 20)
    KL( p(h) || p(h+v) )     = 1.15         (random v, same magnitude)

Sixteen orders of magnitude apart.  Consequences:

1.  This is NOT the Sigma_p all-ones direction.  That one does not transfer to
    G (see conditioning_check.py).  This is a separate, exact nullity coming
    from the NORMALISATION LAYER, and it is the one that actually bites.
2.  The predictive map depends only on the DIRECTION of h.  The semantic
    manifold is therefore a space of directions -- a sphere, not a vector
    space -- and that is where the Fisher geometry properly lives.
3.  It doubles as the RQ4 falsification test, and the framework passes it.

Use radial_null_check() to re-verify on any model; a normalisation layer that
is not scale-invariant (plain LayerNorm has a mean-subtraction too) will have
a different null structure, so do not assume this transfers.
"""

from __future__ import annotations

import torch


# ----------------------------------------------------------------------
# Constant metrics
# ----------------------------------------------------------------------
def manson_metric(U: torch.Tensor) -> torch.Tensor:
    """G = U^T U, the unembedding Gram matrix.  (d, d), constant in h."""
    return U.T @ U


def euclidean_metric(d: int, dtype=torch.float64) -> torch.Tensor:
    return torch.eye(d, dtype=dtype)


# ----------------------------------------------------------------------
# The Fisher-Rao pullback
# ----------------------------------------------------------------------
def topk_indices(lm, h: torch.Tensor, top_k: int = 2000) -> torch.Tensor:
    """The top-k token indices at h, to be FROZEN before differentiating G.

    Why this must be a separate step
    -------------------------------
    If `fisher_metric` re-selects its own top-k at every evaluation, the
    retained token set changes discretely as h moves, so G(h) is only
    PIECEWISE smooth -- it jumps wherever the top-k membership changes.
    Differentiating through that gives garbage at the jumps and, worse, plausible
    numbers away from them.  Curvature needs second derivatives, so this is not
    survivable.

    Freeze the index set once at the base point and hold it fixed across the
    whole derivative computation; the resulting metric is smooth.  The cost is
    that the approximation is centred on the base point, which is exactly what a
    local geometric quantity wants anyway.
    """
    return torch.topk(lm.next_token_probs(h), top_k).indices


def fisher_metric(
    lm,
    h: torch.Tensor,
    top_k: int | None = 2000,
    include_norm_jacobian: bool = True,
    idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fisher-Rao metric pulled back to residual-stream space at h.

        G(h) = J^T ( diag(p) - p p^T ) J ,     J = d logits / d h

    Sigma_p is never formed (it is N x N).  Instead

        G = J^T diag(p) J - (J^T p)(J^T p)^T

    Parameters
    ----------
    top_k
        Restrict to the top-k tokens by probability mass, renormalising.
        None uses the full vocabulary.  Truncation error must be reported,
        not assumed -- see truncation_error() below.
    include_norm_jacobian
        Whether to include the final norm in J.  Setting this False gives the
        naive G = U~^T Sigma_p U~ that ignores the norm, which is WRONG but is
        useful for quantifying how much the norm actually matters.
    """
    p_full = lm.next_token_probs(h)

    if idx is not None:                   # frozen index set -- smooth in h
        p = p_full[idx]
        p = p / p.sum()
        U = lm.U[idx]
    elif top_k is not None and top_k < lm.N:
        p, sel = torch.topk(p_full, top_k)
        p = p / p.sum()
        U = lm.U[sel]                     # (k, d)
    else:
        p, U = p_full, lm.U

    if include_norm_jacobian:
        A = lm.norm_jacobian(h)           # (d, d)
        J = U @ A                         # (k, d)
    else:
        J = U if lm.g is None else U * lm.g

    Jp = J.T @ p                          # (d,)
    return (J.T * p) @ J - torch.outer(Jp, Jp)


def fisher_metric_projected(
    lm,
    h: torch.Tensor,
    frame: torch.Tensor,
    idx: torch.Tensor,
    perm: torch.Tensor | None = None,
    rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """The (k, k) Fisher metric in `frame` coordinates, WITHOUT ever forming
    the (d, d) metric.

    Why this exists
    ---------------
    Computing the full G(h) and then projecting costs O(N d^2) and, worse,
    materialises a (d, d) intermediate that double autodiff must carry.  With
    N=2000 and d=576 that route tried to allocate 20.6 GB and died.

    Projecting first collapses everything to k dimensions:

        M     = U_idx @ (A @ frame)          (n_tok, k)
        G_k   = (M^T diag(p) M) - (M^T p)(M^T p)^T

    A @ frame is cheap in closed form, because A is a scaled projector:

        A @ frame = diag(g) (1/r) ( frame - hhat (hhat^T frame) )

    Largest intermediate is (n_tok, k).  This is the routine curvature should
    always call; `fisher_metric` remains for diagnostics on the full space.

    THE TWO SCRAMBLE CONTROLS -- pass exactly one
    --------------------------------------------
    `perm`  a permutation of the WHOLE vocabulary.  rows become perm[idx], which
            are ~512 essentially random tokens: measured overlap with idx is
            5/512.  So this changes the row SET, not only the pairing, and the
            row norms shift with it (median ||U row|| 2.44 over the top-512 vs
            3.10 over 512 random, a factor of 0.79).  It answers "what if these
            probabilities sat on arbitrary vocabulary directions" -- a real
            question, but a compound one.

    `rows`  an explicit index array, used verbatim.  Pass idx[randperm(len(idx))]
            to get the CLEAN paired control: the retained row multiset is
            IDENTICAL to the real one, entropy is identical, and the ONLY thing
            destroyed is which probability sits on which direction.

    `rows` is the one that isolates learned assignment, and it is the one the
    definitional question in 08-rq3a-log.md S5 actually needs.  `perm` is kept
    because it is the weaker, more aggressive control and the contrast between
    the two is itself informative.
    """
    d = h.shape[-1]
    r = torch.sqrt(h.pow(2).mean() + lm.eps)
    hhat = h / torch.linalg.vector_norm(h)
    Af = (frame - torch.outer(hhat, hhat @ frame)) / r          # (d, k)
    if lm.g is not None:
        Af = lm.g[:, None] * Af

    if rows is None:
        rows = idx if perm is None else perm[idx]
    M = lm.U[rows] @ Af                                          # (n_tok, k)
    logits = lm.logits(h)                                        # p ALWAYS from the real U
    p = torch.softmax(logits[idx] - logits[idx].max(), dim=-1)   # renormalised
    Mp = M.T @ p
    return (M.T * p) @ M - torch.outer(Mp, Mp)


def fisher_metric_scrambled(lm, h: torch.Tensor, perm: torch.Tensor | None = None,
                            top_k: int | None = 2000,
                            idx: torch.Tensor | None = None,
                            rows: torch.Tensor | None = None) -> torch.Tensor:
    """G(h) with the token->direction pairing BROKEN but p left untouched.

    THE CONTROL THAT ACTUALLY WORKS
    -------------------------------
    The obvious structure-destroying control -- permute U's rows -- is
    mathematically the IDENTITY on this metric, and using it produced a false
    result once already.  Permuting rows permutes the logits, hence permutes p by
    the SAME permutation, so each token's probability travels with its own
    direction:

        (PU)^T (diag(Pp) - (Pp)(Pp)^T) (PU)
            = U^T P^T P (diag(p) - p p^T) P^T P U
            = U^T (diag(p) - p p^T) U                      since P^T P = I

    Verified numerically: ||G_real - G_rowshuffled|| / ||G_real|| = 1.5e-12 and
    the entropies agree to 10 decimals.  It destroys nothing.

    This function instead computes p from the REAL unembedding and then pairs
    those probabilities with PERMUTED directions.  So

        entropy is matched EXACTLY (p is untouched), and
        the learned assignment of probability mass to directions is destroyed.

    Any curvature difference between this and the real metric at the same point
    is therefore attributable to the learned structure, with concentration held
    exactly fixed -- which is what the definitional question needs.

        !!  perm ALONE DOES NOT ISOLATE THE ASSIGNMENT  !!

    With a whole-vocabulary `perm`, the retained rows become perm[idx] -- 512
    essentially random tokens, overlapping the real top-512 by 5/512, with
    median row norm 3.10 against the real 2.44.  So it swaps the direction SET
    as well as the pairing, and a difference in curvature could come from
    either.

    Pass `rows` = idx[randperm(len(idx))] instead for the clean control: the
    retained row multiset is then IDENTICAL to the real one and only the pairing
    of probability to direction is destroyed.  See fisher_metric_projected.
    """
    logits = lm.logits(h)
    if idx is None and top_k is not None and top_k < lm.N:
        idx = torch.topk(logits, top_k).indices     # caller should freeze this
    if rows is not None:
        p = torch.softmax(logits[idx] - logits[idx].max(), dim=-1)
        U = lm.U[rows]
    elif idx is not None:
        p = torch.softmax(logits[idx] - logits[idx].max(), dim=-1)
        U = lm.U[perm[idx]]
    else:
        p = torch.softmax(logits - logits.max(), dim=-1)
        U = lm.U[perm]

    A = lm.norm_jacobian(h)
    J = U @ A
    Jp = J.T @ p
    return (J.T * p) @ J - torch.outer(Jp, Jp)


def radial_null_check(lm, h: torch.Tensor, scale: float = 2.0, seed: int = 0) -> dict:
    """Verify that h is a null direction of G(h), and that a random one is not.

    This is simultaneously (i) a correctness test of the pullback -- if the
    radial direction is NOT null, the norm Jacobian is missing or wrong -- and
    (ii) the RQ4 null-space falsification test from 03-methodology.md E5.

    Returns the two KLs and the relative size of G(h) h.  Expect
    kl_radial / kl_random ~ 1e-16.
    """
    G = fisher_metric(lm, h, top_k=None)
    logp = torch.log_softmax(lm.logits(h), -1)
    p = logp.exp()

    def kl_to(hp):
        logq = torch.log_softmax(lm.logits(hp), -1)
        return float((p * (logp - logq)).sum())

    g = torch.Generator().manual_seed(seed)
    v = torch.randn(lm.d, generator=g, dtype=torch.float64)
    v = v * (torch.linalg.vector_norm(h) * (scale - 1.0) / torch.linalg.vector_norm(v))

    return dict(
        kl_radial=kl_to(scale * h),
        kl_random=kl_to(h + v),
        Gh_rel=float(torch.linalg.vector_norm(G @ h)
                     / (torch.linalg.matrix_norm(G) * torch.linalg.vector_norm(h))),
        Ah_rel=float(torch.linalg.vector_norm(lm.norm_jacobian(h) @ h)
                     / torch.linalg.vector_norm(h)),
    )


def truncation_error(lm, h: torch.Tensor, ks=(100, 500, 1000, 2000, 5000, 10000)):
    """||G_full - G_topk|| / ||G_full|| as a function of k.  Stage 2 task 2.4."""
    G_full = fisher_metric(lm, h, top_k=None)
    nf = torch.linalg.matrix_norm(G_full)
    return {k: float(torch.linalg.matrix_norm(fisher_metric(lm, h, top_k=k) - G_full) / nf)
            for k in ks}


# ----------------------------------------------------------------------
# Conditioning -- the real obstacle (see 03-methodology.md section 4)
# ----------------------------------------------------------------------
def spectrum(G: torch.Tensor, trace_frac: float = 0.99, rel_tol: float = 1e-30) -> dict:
    """Eigen-diagnostics of a PSD metric.

    Reports THREE different notions of rank, because on real activations they
    disagree by hundreds of dimensions and conflating them produces nonsense:

    n_pos     eigenvalues above `rel_tol` * lam_max.  The closest thing to the
              ALGEBRAIC rank.  On real models this is ~d, confirming that G is
              generically full rank -- the Sigma_p null direction does not
              transfer to G unless 1_N lies in range(W).
    rank_num  torch.linalg.matrix_rank at its default tolerance.  The rank at
              float64 WORKING precision.  Much smaller (as low as ~d/2),
              because the spectrum spans ~16 orders of magnitude.
    k_eff     directions holding `trace_frac` of the trace.  The rank that
              actually matters for curvature -- single digits to a few dozen.

    cond is lam_max/lam_min and is routinely `inf` on real data: lam_min
    underflows to exactly 0.0 in float64.  `cond_eff` = lam_max/lam_k_eff is
    the conditioning of the subspace one would actually retain, and is the
    number to report.
    """
    ev = torch.linalg.eigvalsh(G).clamp_min(0.0)
    d = G.shape[0]
    tot = float(ev.sum())
    if tot <= 0:
        return dict(d=d, n_pos=0, rank_num=0, k_eff=0, frac_eff=0.0,
                    cond=float("inf"), cond_eff=float("inf"), pr=0.0,
                    lam_max=0.0, lam_min=0.0)
    desc = ev.flip(0)
    lam_max, lam_min = float(desc[0]), float(ev[0])
    csum = torch.cumsum(desc, 0) / tot
    k_eff = int(torch.searchsorted(csum, torch.tensor(trace_frac, dtype=csum.dtype)) + 1)
    lam_k = float(desc[k_eff - 1])
    return dict(
        d=d,
        n_pos=int((desc > rel_tol * lam_max).sum()),
        rank_num=int(torch.linalg.matrix_rank(G)),
        k_eff=k_eff,
        frac_eff=k_eff / d,
        cond=lam_max / lam_min if lam_min > 0 else float("inf"),
        cond_eff=lam_max / lam_k if lam_k > 0 else float("inf"),
        pr=tot**2 / float((ev**2).sum()),      # participation ratio
        lam_max=lam_max,
        lam_min=lam_min,
    )


# ----------------------------------------------------------------------
# Gate A -- the KL-Hessian identity, on a real model
# ----------------------------------------------------------------------
def gate_a(lm, h: torch.Tensor, v: torch.Tensor, exponents=range(1, 8),
           top_k: int | None = None) -> list[tuple[float, float, float]]:
    """Verify KL(p(h) || p(h+eps v)) ~= 1/2 eps^2 v^T G(h) v + O(eps^3).

    Returns [(eps, ratio, rel_err)].  Sweep eps and look for the O(eps)
    plateau -- do not trust a single eps.  Below ~1e-5 catastrophic
    cancellation in the KL destroys the test even when G is correct.
    """
    v = v / torch.linalg.vector_norm(v)
    quad = 0.5 * float(v @ fisher_metric(lm, h, top_k=top_k) @ v)
    logp = torch.log_softmax(lm.logits(h), dim=-1)
    p = logp.exp()
    out = []
    for k in exponents:
        eps = 10.0 ** (-k)
        logq = torch.log_softmax(lm.logits(h + eps * v), dim=-1)
        kl = float((p * (logp - logq)).sum())
        ratio = kl / eps**2
        out.append((eps, ratio, abs(ratio - quad) / quad if quad > 0 else float("nan")))
    return quad, out
