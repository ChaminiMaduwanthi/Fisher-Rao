"""
Is the scramble control's "frame sensitivity" actually a FRAME problem?

run_scramble_control.py reports K_scr = 0.014 in the scrambled metric's own
eigenframe versus 5.21 in the real metric's frame, on the same points, and
warns that no conclusion can be drawn until that is understood.

Task 3.8 already established that curvature is invariant under an arbitrary
ROTATION of the retained basis (rel.err 2.6e-11).  If that invariance also
holds in the scrambled condition, then "frame" is the wrong word for what is
happening: F_real and F_scr are not two bases for one subspace, they are two
DIFFERENT k-dimensional subspaces, and

    metric_in_frame(h, F) = the induced metric on the affine slice h + span(F)

so the two numbers are the curvatures of two different submanifolds.  That is
not an inconsistency to be resolved; it is two different measurements, and the
question becomes which one answers the definitional question.

This script separates the two possibilities WITHOUT computing any curvature,
so it runs in seconds rather than minutes:

    1.  principal angles between F_real and F_scr  -- how different are the
        subspaces?  (If they nearly coincide, subspace choice cannot explain a
        370x difference and something else is wrong.)
    2.  the eigenvalue spectra of both metrics, and the conditioning each
        metric has when restricted to the OTHER condition's subspace.  A metric
        evaluated on a subspace it has no mass in is near-degenerate, and
        near-degenerate metrics inflate curvature through 1/det, not geometry.

TWO CONTROLS, without which (1) proves nothing.  In d = 576 any two random
5-dimensional subspaces are already near-orthogonal, so a large principal angle
between F_real and F_scr is only meaningful against:

    RANDOM   a uniformly random 5-dim subspace -- the floor.  If F_scr is no
             further from F_real than a random subspace is, scrambling has
             merely randomised the frame and the angle carries no information.
    PERM2    a second, independent permutation.  If F_scr1 and F_scr2 are as far
             apart from each other as each is from F_real, then all scrambles
             are mutually unrelated and "distance from real" is not special.

Usage:  python diag_frame.py [n_points]
"""

from __future__ import annotations

import sys

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fisherrao import LM, corpus
from fisherrao.metrics import (fisher_metric, fisher_metric_projected,
                               fisher_metric_scrambled, topk_indices)

K_DIM, TOP_K = 5, 512
LAYERS = [1, 5, 10, 15, 20, 25, 28, 29, 30]


def radial_proj(h):
    hhat = h / torch.linalg.vector_norm(h)
    return torch.eye(h.shape[-1], dtype=h.dtype) - torch.outer(hhat, hhat)


def top_frame(G, P, k):
    """Top-k eigenvectors of P G P -- the same selection curvature_from uses."""
    ev, evec = torch.linalg.eigh(P @ G @ P)
    order = torch.argsort(ev, descending=True)
    return evec[:, order[:k]].contiguous(), ev[order].clamp_min(0.0)


def principal_angles(A, B):
    """Principal angles (degrees) between two orthonormal column spans."""
    s = torch.linalg.svdvals(A.T @ B).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(s))


def cond_of(M):
    ev = torch.linalg.eigvalsh(0.5 * (M + M.T)).clamp_min(0.0)
    lo, hi = float(ev[0]), float(ev[-1])
    return hi / lo if lo > 0 else float("inf")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    gen = torch.Generator().manual_seed(0)
    lm = LM()
    print(lm.summary())

    states = []
    for s in corpus.sentences():
        H, _ = lm.residual_stream(s)
        for l in LAYERS:
            for t in range(H.shape[1]):
                states.append((l, H[l, t]))
    pick = torch.randperm(len(states), generator=gen)[:n].tolist()
    perm = torch.randperm(lm.U.shape[0], generator=gen)
    perm2 = torch.randperm(lm.U.shape[0], generator=gen)      # control PERM2

    print(f"\n{'='*74}")
    print("SUBSPACE DIAGNOSIS -- no curvature computed, this is all linear algebra")
    print(f"{'='*74}\n")

    ang_min, ang_max, rows = [], [], []
    for i in pick:
        layer, h = states[i]
        idx = topk_indices(lm, h, TOP_K)
        P = radial_proj(h)

        G_real = fisher_metric(lm, h, idx=idx)
        G_scr = fisher_metric_scrambled(lm, h, perm, idx=idx)

        F_real, ev_real = top_frame(G_real, P, K_DIM)
        F_scr, ev_scr = top_frame(G_scr, P, K_DIM)

        ang = principal_angles(F_real, F_scr)
        ang_min.append(float(ang.min()))
        ang_max.append(float(ang.max()))

        # ---- controls -----------------------------------------------------
        # RANDOM: a uniformly random k-frame inside the radial complement.
        Qr, _ = torch.linalg.qr(P @ torch.randn(h.shape[-1], K_DIM,
                                                generator=gen, dtype=h.dtype))
        ang_rand = principal_angles(F_real, Qr)
        # PERM2: an independent scramble, compared to the first scramble.
        G_scr2 = fisher_metric_scrambled(lm, h, perm2, idx=idx)
        F_scr2, _ = top_frame(G_scr2, P, K_DIM)
        ang_ss = principal_angles(F_scr, F_scr2)
        gRand = fisher_metric_projected(lm, h, Qr, idx)
        # scramble 1's metric restricted to scramble 2's frame: the paired null
        # for "how much of a metric survives in a DIFFERENT condition's frame".
        gS1S2 = fisher_metric_projected(lm, h, F_scr2, idx, perm=perm)

        # each metric restricted to each subspace, as curvature_from would see it
        gRR = fisher_metric_projected(lm, h, F_real, idx)
        gSS = fisher_metric_projected(lm, h, F_scr, idx, perm=perm)
        gSR = fisher_metric_projected(lm, h, F_real, idx, perm=perm)   # scrambled metric, real frame
        gRS = fisher_metric_projected(lm, h, F_scr, idx)               # real metric, scrambled frame

        lp = torch.log_softmax(lm.logits(h), -1)
        rows.append(dict(
            layer=layer,
            entropy=float(-(lp.exp() * lp).sum()),
            # fraction of each metric's total trace captured by each subspace
            cap_RR=float(torch.diagonal(gRR).sum() / ev_real.sum()),
            cap_SS=float(torch.diagonal(gSS).sum() / ev_scr.sum()),
            cap_SR=float(torch.diagonal(gSR).sum() / ev_scr.sum()),
            cap_RS=float(torch.diagonal(gRS).sum() / ev_real.sum()),
            cond_RR=cond_of(gRR), cond_SS=cond_of(gSS),
            cond_SR=cond_of(gSR), cond_RS=cond_of(gRS),
            # controls
            ang_rand_min=float(ang_rand.min()), ang_rand_max=float(ang_rand.max()),
            ang_ss_min=float(ang_ss.min()), ang_ss_max=float(ang_ss.max()),
            cap_rand=float(torch.diagonal(gRand).sum() / ev_real.sum()),
            cond_rand=cond_of(gRand),
            cap_S1S2=float(torch.diagonal(gS1S2).sum() / ev_scr.sum()),
            cond_S1S2=cond_of(gS1S2),
            # how concentrated is each FULL spectrum
            k99_real=int(torch.searchsorted(
                torch.cumsum(ev_real, 0) / ev_real.sum(),
                torch.tensor(0.99, dtype=ev_real.dtype)) + 1),
            k99_scr=int(torch.searchsorted(
                torch.cumsum(ev_scr, 0) / ev_scr.sum(),
                torch.tensor(0.99, dtype=ev_scr.dtype)) + 1),
        ))

    def med(key):
        return float(torch.tensor([r[key] for r in rows], dtype=torch.float64).median())

    a_min = torch.tensor(ang_min, dtype=torch.float64)
    a_max = torch.tensor(ang_max, dtype=torch.float64)
    print("1. ARE THE TWO SUBSPACES THE SAME?")
    print(f"   principal angles between F_real and F_scr, over n={len(rows)} points")
    print(f"     smallest angle : median {float(a_min.median()):6.2f} deg   "
          f"min {float(a_min.min()):6.2f}   max {float(a_min.max()):6.2f}")
    print(f"     largest  angle : median {float(a_max.median()):6.2f} deg   "
          f"min {float(a_max.min()):6.2f}   max {float(a_max.max()):6.2f}")
    print("   (90 deg = completely orthogonal subspaces, 0 = identical)")
    print("   CONTROLS -- the same statistic against the two null baselines:")
    print(f"     F_real vs RANDOM subspace : smallest {med('ang_rand_min'):6.2f}   "
          f"largest {med('ang_rand_max'):6.2f}")
    print(f"     F_scr1 vs F_scr2 (indep.) : smallest {med('ang_ss_min'):6.2f}   "
          f"largest {med('ang_ss_max'):6.2f}\n")

    print("2. HOW MUCH OF EACH METRIC LIVES IN EACH SUBSPACE?")
    print("   trace fraction captured by the retained k=5 subspace:")
    print(f"     real metric   in its own frame : {med('cap_RR'):.4f}")
    print(f"     real metric   in scrambled frame: {med('cap_RS'):.2e}")
    print(f"     scrambled met in its own frame : {med('cap_SS'):.4f}")
    print(f"     scrambled met in real frame    : {med('cap_SR'):.2e}")
    print(f"     scr1 metric   in scr2 frame    : {med('cap_S1S2'):.2e}   [paired null]")
    print(f"     real metric   in RANDOM frame  : {med('cap_rand'):.2e}   [control]")
    print(f"     (isotropic floor would be k/d  = {K_DIM/lm.d:.2e})\n")

    print("3. CONDITIONING OF THE RESTRICTED (k x k) METRIC")
    print("   curvature needs to invert this; a huge cond means 1/det blowup:")
    print(f"     real       in own frame  : {med('cond_RR'):.3e}")
    print(f"     scrambled  in own frame  : {med('cond_SS'):.3e}")
    print(f"X")
    print(f"     real       in SCR  frame : {med('cond_RS'):.3e}")
    print(f"     scr1       in SCR2 frame : {med('cond_S1S2'):.3e}   [paired null]")
    print(f"     real       in RANDOM frm : {med('cond_rand'):.3e}   [control]\n")

    print("4. SPECTRAL CONCENTRATION OF THE TWO METRICS (dims for 99% of trace)")
    print(f"     real      : median k99 = {med('k99_real'):.0f}")
    print(f"     scrambled : median k99 = {med('k99_scr'):.0f}")


if __name__ == "__main__":
    main()
