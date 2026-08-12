"""
The curvature proxies used by prior work, reimplemented so that all four
instruments can be run on IDENTICAL activations.

This module is experiment E2 from 03-methodology.md -- the adjudication
experiment, and the paper's centrepiece figure.  Either the proxies track the
intrinsic quantity (vindicating them and making this thesis a validation study)
or they diverge (justifying it).  Both outcomes are results.

    pca_curvature      Mabrok 2026  -- local-PCA residual variance, EXTRINSIC,
                                       in ambient Euclidean coordinates.
                                       Reported there as ~1e-5, "flat".
    second_fundamental Mabrok 2026  -- ||II||, tangent-plane rotation between
                                       neighbours.
    frenet_curvature   Manson 2025  -- see trajectory.py; metric G = U^T U is
                                       CONSTANT, so intrinsic curvature is
                                       identically zero.
    contextual_angle   King et al.  -- Euclidean angle between adjacent
                                2026   first-order difference vectors, averaged
                                       over a backward window.  No metric.

None of these is wrong as a definition; they simply measure different things
from the intrinsic Riemann curvature of the Fisher metric.  See
03-methodology.md S3 for the three-way distinction.
"""

from __future__ import annotations

import torch


def pca_curvature(points: torch.Tensor, i: int, n_neighbors: int = 20,
                  q: int | None = None) -> float:
    """Mabrok's local-PCA curvature proxy at points[i].

    "the fraction of variance captured by directions orthogonal to the dominant
    principal subspace"

    Parameters
    ----------
    points : (n, d) ambient hidden states forming the point cloud
    i      : index of the base point
    q      : dimension of the "dominant" subspace.  If None, chosen as the
             number of components needed for 90% of local variance, which is the
             usual reading of "dominant".

    Returns the residual variance fraction -- a small number when the cloud is
    locally well approximated by its tangent plane.  This is EXTRINSIC and
    coordinate-dependent: it says nothing directly about the intrinsic curvature
    of a metric defined on the same space.

        !!  ON TRANSFORMER ACTIVATIONS ITS MAGNITUDE IS SET BY `q`  !!

    When q comes from a variance threshold `thr`, the residual is bounded by
    1 - thr, and it ATTAINS that bound whenever the local spectrum has no gap.
    Measured on WikiText-103 with GPT-2, 1800 vectors/layer, k=200
    (run_mabrok_replication.py):

        threshold   0.90    0.95    0.99    0.999   0.9999
        measured    0.0989  0.0494  0.0098  0.00093 6.2e-05
        ratio       0.99    0.99    0.98    0.93    0.62

    This is NOT true of all data, and the difference is the point
    (check_pca_tautology.py, ratio to 1-thr at k=200, thr .90/.95/.99):

        gaussian noise   0.98, 0.98, 0.97      <- no gap
        gpt2             0.99, 0.99, 0.98      <- no gap
        3-sphere         0.22, 0.44, 0.00      <- has a gap; reports geometry
        flat 3-plane     0.00, 0.00, 0.00      <- correctly exactly zero

    So the proxy DOES work on a clean low-dimensional manifold; real hidden
    states simply are not one, and on them it degenerates into reporting the
    threshold.  Under a FIXED q instead, the same cloud gives 0.2-0.8 (q below
    the local rank) or EXACTLY 0 (q >= k, where k+1 neighbours span at most k
    dimensions).  Nothing in between.

    Worst of all, at the threshold needed to reproduce Mabrok's 1e-5 (~0.9999)
    the statistic reports a UNIT 3-SPHERE as flat to 1e-31.  Its discriminating
    range and its published operating point do not overlap.

    Consequence for E2: use this proxy for its RANK correlation against other
    instruments, which is what 07-stage4-log.md S1 reports, and never for its
    magnitude.  An earlier version of this line recommended `second_fundamental`
    below as the safe alternative -- RETRACTED, it has a worse degeneracy of its
    own (it inverts the curved/flat ordering at the wrong q).  NEITHER of
    Mabrok's proxies supports a magnitude claim.
    """
    d2 = torch.cdist(points[i:i + 1], points).squeeze(0)
    idx = torch.argsort(d2)[:n_neighbors + 1]
    nbr = points[idx]
    nbr = nbr - nbr.mean(0, keepdim=True)
    ev = torch.linalg.svdvals(nbr) ** 2
    tot = float(ev.sum())
    if tot <= 0:
        return float("nan")
    if q is None:
        frac = torch.cumsum(ev, 0) / tot
        q = int(torch.searchsorted(frac, torch.tensor(0.90, dtype=frac.dtype)) + 1)
        q = max(1, min(q, len(ev) - 1))
    return float(ev[q:].sum() / tot)


def second_fundamental(points: torch.Tensor, i: int, n_neighbors: int = 20,
                       q: int = 3) -> float:
    """Mabrok's ||II|| proxy: how much the local tangent plane rotates.

    Estimated as the mean principal angle between the q-dimensional tangent
    subspace at points[i] and that at each of its neighbours, divided by the
    distance -- a finite-difference stand-in for the derivative of the tangent
    plane.

        !!  ONLY MEANINGFUL WHEN q IS THE TRUE INTRINSIC DIMENSION  !!

    Measured against clouds with known answers (check_pca_tautology.py S3), a
    flat 3-plane and a unit 3-sphere in R^768:

        q            2        3        5       10
        flat3    1.275  3.0e-08    2.871    4.636
        sphere3  1.722    0.579    2.701    5.205

    It separates them at q=3 -- the true dimension -- and NOWHERE ELSE.  At q=2
    a perfectly FLAT plane reads as curved; at q=5 it reads as MORE curved than
    the sphere.  So this is not the safe alternative to pca_curvature that
    11-mabrok-replication-log.md once called it: pca_curvature at a bad
    threshold returns an uninformative number, while this returns the WRONG
    ORDERING.  On real activations the intrinsic dimension is estimated, not
    known (TWO-NN gives 3.4-7.5 across GPT-2's layers), and this proxy's value
    doubles over that range.
    """
    def tangent(j):
        d2 = torch.cdist(points[j:j + 1], points).squeeze(0)
        idx = torch.argsort(d2)[:n_neighbors + 1]
        nbr = points[idx]
        nbr = nbr - nbr.mean(0, keepdim=True)
        _, _, Vh = torch.linalg.svd(nbr, full_matrices=False)
        return Vh[:q].T                                   # (d, q) orthonormal

    Ti = tangent(i)
    d2 = torch.cdist(points[i:i + 1], points).squeeze(0)
    order = torch.argsort(d2)[1:n_neighbors + 1]
    vals = []
    for j in order.tolist():
        s = torch.linalg.svdvals(Ti.T @ tangent(j)).clamp(-1.0, 1.0)
        angle = float(torch.arccos(s).norm())             # principal angles
        dist = float(d2[j].sqrt()) if float(d2[j]) > 0 else 0.0
        if dist > 0:
            vals.append(angle / dist)
    return float(torch.tensor(vals).mean()) if vals else float("nan")


def contextual_angle(H: torch.Tensor, layer: int, pos: int, window: int = 3) -> float:
    """King et al.'s contextual curvature: Euclidean angle between adjacent
    first-order difference vectors of consecutive-TOKEN activations, averaged
    over a backward window.

    Note this runs along the token axis at a fixed layer, not along layers --
    that is their construction, and it differs from Manson's layer-wise
    trajectory.  Plain Euclidean; no metric enters.

    H : (L+1, T, d) residual stream
    """
    x = H[layer]                                          # (T, d)
    diffs = x[1:] - x[:-1]                                # (T-1, d)
    angles = []
    for t in range(max(1, pos - window + 1), min(pos + 1, len(diffs))):
        a, b = diffs[t - 1], diffs[t]
        na, nb = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
        if na > 0 and nb > 0:
            cos = (a @ b) / (na * nb)
            angles.append(float(torch.arccos(cos.clamp(-1.0, 1.0))))
    return float(torch.tensor(angles).mean()) if angles else float("nan")
