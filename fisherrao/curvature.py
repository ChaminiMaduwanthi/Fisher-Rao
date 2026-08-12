"""
Stage 3 -- intrinsic Riemannian curvature of the Fisher-Rao pullback.

This is the thesis contribution.  Everything upstream (model.py, metrics.py)
exists to hand this module a trustworthy state-dependent metric.

Pipeline, in the order forced by Stage 0's findings:

    1.  QUOTIENT by the radial direction.        (Option B -- exact, structural)
        RMSNorm is scale-invariant, so h is an exact null direction of G(h).
        Working on the sphere of directions is not a workaround; it is the
        correct manifold.  See 05-stage0-log.md section 4.2.

    2.  RESTRICT to the top-k eigendirections.   (Option A -- numerical)
        lam_min underflows to exactly 0 at 22/31 layers, so float64 alone is
        not enough.  k_eff was measured at 3-86 of 576.  Within the retained
        subspace cond_eff is only 1e1-1e3, leaving ample precision for the
        second derivatives curvature needs.

    3.  Compute sectional curvature in that subspace by autodiff.

Sign and index conventions are inherited verbatim from validate_curvature.py,
which is checked against three analytically known answers.  DO NOT change them
here without re-running that script -- a sign slip in the Riemann contraction
produced a plausible-looking, exactly-constant, wrong answer once already.

    R^r_sij  such that  R(e_i, e_j) e_s = R^r_sij e_r
    R_msij   = g_mr R^r_sij
    K(u,v)   = R(u,v,v,u) / (|u|^2|v|^2 - <u,v>^2)
             = R_msij u^m v^s u^i v^j / (...)
"""

from __future__ import annotations

import torch

from .metrics import fisher_metric, fisher_metric_projected, topk_indices


# ----------------------------------------------------------------------
# Step 1 + 2: the chart
# ----------------------------------------------------------------------
def null_projector(lm, h: torch.Tensor) -> torch.Tensor:
    """I - N N^T, the projector onto the complement of EVERY exact null direction.

        RMSNorm    N = [hhat]                 1 direction  (radial)
        LayerNorm  N = orth([hhat, 1_d])      2 directions (radial + all-ones)

    Was `I - hhat hhat^T` inline in four places, which is correct only for
    RMSNorm.  On a LayerNorm model -- GPT-2 and Pythia, i.e. both models the
    cross-architecture comparison needs -- the mean-subtraction contributes a
    SECOND exact null direction (see LM.norm_jacobian for the algebra).

    HOW MUCH THIS ACTUALLY MATTERS, measured rather than assumed: at k=5 on
    gpt2 and pythia-160m the all-ones leakage into the retained frame is 0.0000
    under the OLD radial-only projector.  Selecting the top-k eigenvectors by
    DESCENDING eigenvalue already excludes an exactly-null direction, because
    its eigenvalue is exactly 0 and k << d.  So this is a guard, not a bug fix,
    and no published number changes because of it.

    It is still the right default for the same reason the radial projection was:
    the guard binds when an exact null direction is only NUMERICALLY null --
    which is precisely the regime k selection pushes into as k grows, and
    exactly where a silently pinv-absorbed singular metric would return a
    plausible wrong answer instead of failing.
    """
    if hasattr(lm, "null_directions"):
        N = lm.null_directions(h)
    else:                                   # controls that wrap a bare object
        N = (h / torch.linalg.vector_norm(h)).unsqueeze(1)
    return torch.eye(h.shape[-1], dtype=h.dtype) - N @ N.T


def effective_frame(lm, h: torch.Tensor, k: int, top_k: int | None = 2000,
                    drop_radial: bool = True) -> torch.Tensor:
    """An orthonormal (d, k) frame spanning the predictively meaningful subspace.

    Exact null directions removed first, then the top-k eigenvectors of G(h)
    within the orthogonal complement.  Columns are ambient vectors.
    """
    G = fisher_metric(lm, h, top_k=top_k)

    if drop_radial:
        # Project G onto the complement of the null space so those directions
        # cannot be selected by the eigendecomposition through numerical noise.
        P = null_projector(lm, h)
        G = P @ G @ P

    evals, evecs = torch.linalg.eigh(G)
    return evecs[:, torch.argsort(evals, descending=True)[:k]].contiguous()


def metric_in_frame(lm, h0: torch.Tensor, frame: torch.Tensor,
                    top_k: int | None = 2000):
    """Return g(x): R^k -> R^{k x k}, the Fisher metric in frame coordinates.

    x are coordinates in the *fixed* frame anchored at h0:  h = h0 + frame @ x.

    Two things are held fixed so that g is genuinely smooth in x:

    the FRAME  -- an eigenbasis recomputed at each point would rotate, and
                  differentiating a rotating basis silently corrupts the
                  derivatives.  Frame-independence of the final answer is
                  verified separately by frame_invariance(); that check is not
                  optional, it is what licenses holding the frame fixed.
    the TOP-K  -- re-selecting the retained tokens at each point makes g only
                  piecewise smooth (see metrics.topk_indices).  The index set is
                  frozen at h0.
    """
    idx = topk_indices(lm, h0, top_k if top_k is not None else lm.N)

    def g(x: torch.Tensor) -> torch.Tensor:
        return fisher_metric_projected(lm, h0 + frame @ x, frame, idx)
    return g


# ----------------------------------------------------------------------
# Step 3: curvature by autodiff
# ----------------------------------------------------------------------
#   !!  USE torch.func, NOT torch.autograd.functional  !!
# torch.autograd.functional.jacobian defaults to create_graph=False, so a
# NESTED call is detached from the graph: the outer derivative then misses the
# d(dG) term entirely.  That bug passed the simplex rung (where the missing term
# happens to vanish) while returning exactly -2x the true curvature for the
# Gaussian and Poincare rungs.  Two of three rungs failing is what caught it;
# with only the simplex in the ladder it would have shipped.
#   torch.func.jacrev composes cleanly for higher derivatives.  Do not
# "simplify" this back.

def christoffel(g_fn, x: torch.Tensor) -> torch.Tensor:
    """Gamma^k_ij at x.  (k, k, k), last index of dG is the derivative index."""
    G = g_fn(x)
    dG = torch.func.jacrev(g_fn)(x)                                   # (k,k,k)
    Gi = torch.linalg.pinv(G, hermitian=True)
    T = torch.einsum("jli->ijl", dG) + torch.einsum("ilj->ijl", dG) - dG
    return 0.5 * torch.einsum("kl,ijl->kij", Gi, T)


#   !!  k IS HARD-CAPPED, AND THE CAP IS LOW  !!
# The tensor itself is k^4 floats, which is nothing.  The cost is that every
# entry is a second derivative taken by NESTED jacrev, so the intermediates the
# graph must carry grow far faster.  Measured warm, on an otherwise idle
# machine, d=576, top_k=512:
#
#     k=3  0.39s   k=4  1.00s   k=5  2.61s   k=6  8.11s   k=7  16.9s   k=8  45.7s
#     k=11        13.8 GB allocation attempt
#     k=15/16     4.8 / 77 GB allocation attempt  ->  process killed
#
# Roughly x2.7 per unit of k, so k=8 at ~46s is already the practical ceiling on
# time as well as memory.  (An earlier version of this comment quoted 1.6 / 4.6 /
# 17.2 / 27.0s -- those were measured cold and under contention and were ~5x too
# high.  Warm up before timing anything here.)
#
# This is not a tuning knob.  Three runs of run_frame_resolution.py were killed
# by exactly this, silently and with no traceback, because select_k's cond_eff
# ceiling happily returns k = 11..16 on real activations.  Anything that chooses
# k automatically MUST cap it before calling this.
K_RIEMANN_MAX = 8


def riemann(g_fn, x: torch.Tensor, allow_large: bool = False) -> torch.Tensor:
    """R^r_sij at x.  (k, k, k, k).  Only use for small k -- see K_RIEMANN_MAX."""
    k = int(x.shape[-1])
    if k > K_RIEMANN_MAX and not allow_large:
        raise ValueError(
            f"riemann() called with k={k}; the nested-jacrev intermediates make "
            f"this a multi-GB allocation above k={K_RIEMANN_MAX} (k=16 attempts "
            f"77 GB and the process dies without a traceback). Cap k before "
            f"calling, or pass allow_large=True if you know the machine can take "
            f"it. Volume element and spectral diagnostics have no such limit and "
            f"are the right quantities at large k.")
    Gam = christoffel(g_fn, x)
    dGam = torch.func.jacrev(lambda y: christoffel(g_fn, y))(x)        # (k,k,k,k)
    #  R^r_sij = d_i Gam^r_js - d_j Gam^r_is + Gam^r_il Gam^l_js - Gam^r_jl Gam^l_is
    return (torch.einsum("rjsi->rsij", dGam)
            - torch.einsum("risj->rsij", dGam)
            + torch.einsum("ril,ljs->rsij", Gam, Gam)
            - torch.einsum("rjl,lis->rsij", Gam, Gam))


def sectional(g_fn, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
              R: torch.Tensor | None = None) -> float:
    """K(u,v) at x.  Pass a precomputed R to sample many planes cheaply."""
    G = g_fn(x)
    if R is None:
        R = riemann(g_fn, x)
    Rlow = torch.einsum("mr,rsij->msij", G, R)
    numer = torch.einsum("msij,m,s,i,j->", Rlow, u, v, u, v)
    uu, vv, uv = u @ G @ u, v @ G @ v, u @ G @ v
    denom = uu * vv - uv**2
    return float(numer / denom) if abs(float(denom)) > 0 else float("nan")


def scalar_curvature(g_fn, x: torch.Tensor, R: torch.Tensor | None = None) -> float:
    """R = g^{ij} R^k_ikj.  The full trace, not a sample."""
    G = g_fn(x)
    if R is None:
        R = riemann(g_fn, x)
    ricci = torch.einsum("kikj->ij", R)
    return float((torch.linalg.pinv(G, hermitian=True) * ricci).sum())


def sectional_sample(g_fn, x: torch.Tensor, n: int = 64, seed: int = 0) -> torch.Tensor:
    """Sectional curvature over n random 2-planes.  Reuses one Riemann tensor."""
    k = x.shape[-1]
    R = riemann(g_fn, x)
    gen = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        u = torch.randn(k, generator=gen, dtype=x.dtype)
        v = torch.randn(k, generator=gen, dtype=x.dtype)
        val = sectional(g_fn, x, u, v, R=R)
        if val == val:                      # drop NaN from degenerate planes
            out.append(val)
    return torch.tensor(out, dtype=x.dtype)


def volume_element(lm, h: torch.Tensor, k: int, top_k: int | None = 2000,
                   drop_radial: bool = True, per_dim: bool = False,
                   on_sphere: bool = True) -> float:
    """log sqrt(det) of the Fisher metric on the retained k-subspace.

    Zavatone-Veth et al. found the volume element both more informative and far
    better conditioned than the Ricci scalar, and Stage 3 confirmed the Riemann-
    derived quantities are k-fragile while this one is not (06-stage3-log.md
    S4.4).  It is therefore the PRIMARY layer-wise quantity, not a fallback.

    per_dim=True returns (log sqrt det)/k -- the log geometric mean of the
    retained eigenvalues.  Use it whenever comparing across different k, since
    the raw sum grows with k by construction.

    drop_radial removes the exact null direction first, matching the convention
    used for curvature so the two are measured on the same manifold.

        !!  on_sphere -- DEFAULT True, and it changes the answer  !!

    G(h) = A^T (...) A with A proportional to 1/r, r = sqrt(mean(h^2)+eps), so
    lambda_i ~ 1/r^2 and log_vol_per_dim carries a  -log(r)  term BY
    CONSTRUCTION.  The residual stream norm grows enormously with depth
    (salience -> 3.7e4 by layer 29), so an UNCORRECTED layer profile is
    dominated by that trivial term: measured slope of log_vol_per_dim on log(r)
    is -0.937 against the -1 the construction mandates.

    The fix is not cosmetic.  Stage 0 showed the semantic manifold is the sphere
    of DIRECTIONS (RMSNorm makes the radial direction exactly null), and the
    metric on the unit sphere is r^2 * G_ambient -- which adds exactly +log(r)
    per dimension.  So on_sphere=True measures the volume on the correct
    manifold; on_sphere=False measures it in ambient coordinates contaminated by
    the residual norm.

    This mattered: the uncorrected profile looked like a monotone contraction
    through the network with a late recovery, and that shape was entirely the
    scale term.  Corrected, the profile is U-shaped with a minimum near layer 20
    (06/07 logs).  See check_volume_confound.py.
    """
    G = fisher_metric(lm, h, top_k=top_k)
    if drop_radial:
        P = null_projector(lm, h)
        G = P @ G @ P
    ev = torch.linalg.eigvalsh(G).clamp_min(0).flip(0)[:k]
    ev = ev[ev > 0]
    if len(ev) == 0:
        return float("nan")
    lv = float(0.5 * torch.log(ev).sum())
    if on_sphere:
        lv = lv + len(ev) * float(torch.log(torch.sqrt(h.pow(2).mean() + lm.eps)))
    return lv / len(ev) if per_dim else lv


def spectral_diagnostics(lm, h: torch.Tensor, k: int, cond_max: float = 1e2,
                         k_max: int = 32, top_k: int | None = 2000) -> dict:
    """volume_element + select_k from ONE eigendecomposition.

    Both of those functions independently build G(h) and eigendecompose a
    (d, d) matrix.  Calling them together per point doubles the cost for no
    reason -- on a 40-sentence corpus across 9 layers that is ~7000 redundant
    576x576 eigendecompositions.  Prefer this in any loop over many points.
    """
    P = null_projector(lm, h)
    G = P @ fisher_metric(lm, h, top_k=top_k) @ P
    ev = torch.linalg.eigvalsh(G).clamp_min(0).flip(0)

    pos = ev[:k]
    pos = pos[pos > 0]
    lv = float(0.5 * torch.log(pos).sum()) if len(pos) else float("nan")

    lam0 = float(ev[0])
    k_sel = 1
    for j in range(1, min(k_max, len(ev))):
        lj = float(ev[j])
        if lj <= 0 or lam0 / lj > cond_max:
            break
        k_sel = j + 1
    tot = float(ev.sum())
    csum = torch.cumsum(ev, 0) / tot if tot > 0 else ev
    return dict(
        log_vol=lv,
        log_vol_per_dim=lv / len(pos) if len(pos) else float("nan"),
        k_sel=k_sel,
        cond_eff=lam0 / float(ev[k_sel - 1]) if float(ev[k_sel - 1]) > 0 else float("inf"),
        cond_eff_at_k=lam0 / float(ev[k - 1]) if float(ev[k - 1]) > 0 else float("inf"),
        trace_frac=float(csum[k_sel - 1]) if tot > 0 else float("nan"),
        k_eff_99=int(torch.searchsorted(csum, torch.tensor(0.99, dtype=ev.dtype)) + 1),
    )


def select_k(lm, h: torch.Tensor, cond_max: float = 1e2, k_max: int = 32,
             top_k: int | None = 2000) -> dict:
    """Choose the retained rank k by a CONDITIONING ceiling.

    Stage 3 established that selecting k by trace fraction (k_eff, 99% of trace)
    admits directions whose eigenvalues are small enough to destroy second
    derivatives: dR/R_ref then swings from -8% to +583% and the layer ordering
    scrambles (Spearman rho = 0.20 between k=4 and k=7).  See
    06-stage3-log.md S4.3.

    The informative diagnostic was cond_eff = lam_max/lam_k of the retained
    subspace, which jumped an order of magnitude (to ~5e2) exactly where the
    instability set in.  So: take the largest k whose cond_eff stays under
    cond_max.  This SUPERSEDES the trace-threshold rule in 03-methodology.md S4.

    cond_max = 1e2 is provisional, chosen from where instability appeared on one
    model.  Report results as a function of it.

        !!  THE RESULT IS NOT DIRECTLY USABLE FOR CURVATURE  !!

    Measured on real activations at cond_max=1e2, this rule returns k = 11..16
    at most points -- the retained spectrum is simply flatter than the ceiling
    is tight.  riemann() cannot be evaluated there at all (77 GB at k=16; see
    K_RIEMANN_MAX), so any caller computing a Riemann-derived quantity must take

        k = min(select_k(...)["k"], K_RIEMANN_MAX)

    and report BOTH numbers.  The rule remains the right CONDITIONING
    diagnostic, and it is directly usable for the volume element and the
    spectral diagnostics, which have no k ceiling.  k_max defaults to 32 here
    for that diagnostic use -- it is not a safe k for curvature.
    """
    P = null_projector(lm, h)
    G = P @ fisher_metric(lm, h, top_k=top_k) @ P
    ev = torch.linalg.eigvalsh(G).clamp_min(0).flip(0)
    lam0 = float(ev[0])
    k = 1
    for j in range(1, min(k_max, len(ev))):
        lj = float(ev[j])
        if lj <= 0 or lam0 / lj > cond_max:
            break
        k = j + 1
    tot = float(ev.sum())
    csum = torch.cumsum(ev, 0) / tot if tot > 0 else ev
    return dict(
        k=k,
        cond_eff=lam0 / float(ev[k - 1]) if float(ev[k - 1]) > 0 else float("inf"),
        trace_frac=float(csum[k - 1]) if tot > 0 else float("nan"),
        k_eff_99=int(torch.searchsorted(csum, torch.tensor(0.99, dtype=ev.dtype)) + 1),
    )


# ----------------------------------------------------------------------
# Mandatory correctness checks
# ----------------------------------------------------------------------
def curvature_at(lm, h: torch.Tensor, k: int, top_k: int = 512,
                 n_planes: int = 48, seed: int = 0) -> dict:
    """All curvature quantities at h, computing the Riemann tensor ONCE.

    The Riemann tensor dominates the cost (~9 s at k=6 on CPU) and both the
    sectional sample and the scalar curvature need it, so never let them each
    build their own.

    CAVEAT on the plane sample.  u, v are drawn iid standard normal in FRAME
    coordinates.  Because the frame is the G-eigenbasis, that weights every
    retained eigendirection equally -- a defensible choice, but it is NOT a
    metric-uniform sample over 2-planes (which would draw u, v ~ N(0, G^-1)).
    With cond_eff of 1e1-1e3 the two differ.  The reported quantiles are
    therefore quantiles over *coordinate-uniform* planes; scalar_R, being a
    full trace, carries no such dependence and is the sampling-free summary.
    State this whenever the K quantiles are reported.
    """
    return _curvature_in_frame(lm, h, effective_frame(lm, h, k, top_k=top_k),
                               k, top_k, n_planes, seed)


def _curvature_in_frame(lm, h, F, k, top_k, n_planes, seed) -> dict:
    """The body of curvature_at, with the frame supplied.

    Split out so curvature_and_volume can share one eigendecomposition without
    duplicating any of the conventions or the baseline arithmetic below -- a
    second copy of this is exactly how a sign convention drifts apart.
    """
    g_fn = metric_in_frame(lm, h, F, top_k=top_k)
    x0 = torch.zeros(k, dtype=h.dtype)
    R = riemann(g_fn, x0)

    gen = torch.Generator().manual_seed(seed)
    Ks = []
    for _ in range(n_planes):
        u = torch.randn(k, generator=gen, dtype=h.dtype)
        v = torch.randn(k, generator=gen, dtype=h.dtype)
        val = sectional(g_fn, x0, u, v, R=R)
        if val == val:
            Ks.append(val)
    Ks = torch.tensor(Ks, dtype=h.dtype)
    q = torch.quantile(Ks, torch.tensor([0.1, 0.5, 0.9], dtype=h.dtype))
    scalar_R = scalar_curvature(g_fn, x0, R=R)

    # ---- baseline-subtracted quantities ------------------------------------
    # K is pinned near the ambient simplex value +1/4 because a trained LM's
    # predictive distribution is sharply concentrated (verified by run_control.py:
    # diffuse p gives K = -0.02 .. +0.14).  Absolute K is therefore a WEAK
    # discriminator by construction.  The model-specific geometry lives in the
    # deviation from constant curvature +1/4, for which the exact references are
    #     K_simplex = 1/4        R_simplex = k(k-1)/4
    # Both verified to machine precision in run_stage3.validate().
    R_ref = k * (k - 1) * 0.25          # scalar curvature of a constant-K=1/4 space
    dK = Ks - 0.25
    return dict(
        K=Ks, K_median=float(q[1]), K_p10=float(q[0]), K_p90=float(q[2]),
        frac_positive=float((Ks > 0).to(h.dtype).mean()),
        scalar_R=scalar_R,
        # deviation from the constant-curvature simplex baseline
        dK_median=float(dK.median()),
        dK_iqr=float(torch.quantile(dK, torch.tensor(0.75, dtype=h.dtype))
                     - torch.quantile(dK, torch.tensor(0.25, dtype=h.dtype))),
        dK_absmax=float(dK.abs().max()),
        scalar_R_ref=R_ref,
        dR=scalar_R - R_ref,
        dR_rel=(scalar_R - R_ref) / R_ref if R_ref else float("nan"),
        n_planes=len(Ks),
    )


def curvature_and_volume(lm, h: torch.Tensor, k: int, top_k: int = 512,
                         n_planes: int = 48, seed: int = 0) -> dict:
    """curvature_at + volume_element from ONE eigendecomposition of G(h).

    Called separately, those two each build the full (d, d) metric and
    eigendecompose it -- the same 576x576 work twice per point, for nothing.
    `spectral_diagnostics` already merged the volume/select_k pair; this closes
    the remaining case, which is the Stage 3 measurement loop.

    HOW MUCH FASTER, measured rather than assumed: 2.79s vs 2.80s at k=5.
    Essentially nothing -- the Riemann tensor dominates so completely that the
    duplicated eigendecomposition is noise beside it.  The ~2x that
    spectral_diagnostics quotes is real for the VOLUME PROFILE loop, which
    computes no curvature at all; it does not carry over to here.  Use this for
    the single code path and the guaranteed-identical frame, not for speed.

    Returns curvature_at's dict plus log_vol and log_vol_per_dim.  The frame is
    the top-k eigenvectors of the null-projected metric -- identical to what
    effective_frame would return, and verified to return bit-identical K, R and
    log-volume.
    """
    P = null_projector(lm, h)
    G = P @ fisher_metric(lm, h, top_k=top_k) @ P
    evals, evecs = torch.linalg.eigh(G)
    order = torch.argsort(evals, descending=True)
    F = evecs[:, order[:k]].contiguous()

    ev = evals[order][:k].clamp_min(0)
    ev = ev[ev > 0]
    if len(ev):
        lv = float(0.5 * torch.log(ev).sum())
        lv += len(ev) * float(torch.log(torch.sqrt(h.pow(2).mean() + lm.eps)))
        lvpd = lv / len(ev)
    else:
        lv = lvpd = float("nan")

    out = _curvature_in_frame(lm, h, F, k, top_k, n_planes, seed)
    out.update(log_vol=lv, log_vol_per_dim=lvpd)
    return out


def frame_invariance(lm, h: torch.Tensor, k: int, seed: int = 0,
                     top_k: int = 512) -> dict:
    """Curvature must not depend on an arbitrary rotation of the retained basis.

    Task 3.8.  If this fails, the moving-frame handling is wrong and every
    curvature number is meaningless.
    """
    F = effective_frame(lm, h, k, top_k=top_k)
    gen = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(k, k, generator=gen, dtype=h.dtype))

    x = torch.zeros(k, dtype=h.dtype)
    s1 = scalar_curvature(metric_in_frame(lm, h, F, top_k), x)
    s2 = scalar_curvature(metric_in_frame(lm, h, F @ Q, top_k), x)
    denom = max(abs(s1), abs(s2), 1e-300)
    return dict(scalar=s1, scalar_rotated=s2, rel_diff=abs(s1 - s2) / denom)
