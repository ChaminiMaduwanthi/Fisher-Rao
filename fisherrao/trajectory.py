"""
Trajectory geometry across layers -- the Manson (2025) baseline.

IMPORTANT, and the whole point of reproducing this: the quantities here are
the FRENET curvature of a curve, measured under a metric supplied by the
caller.  When that metric is constant (U^T U or I), the space is flat and its
Riemann curvature is identically zero.  These numbers describe how the PATH
bends, not how the SPACE bends.

This module is the baseline the thesis argues against, and the end-to-end
pipeline everything else swaps components into.  See 01-literature-review.md
entry B1.
"""

from __future__ import annotations

import torch


def _gnorm(x: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """sqrt(x^T G x), elementwise over a batch of row vectors."""
    return torch.sqrt(torch.einsum("...i,ij,...j->...", x, G, x).clamp_min(0.0))


def salience(traj: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """S(l) = ||x_{l+1} - x_l||_G.   Returns (L,) for an (L+1, d) trajectory."""
    return _gnorm(traj[1:] - traj[:-1], G)


def curvature(traj: torch.Tensor, G: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Frenet curvature via 3-point central differences, Manson (2025) eq.

        v_l = (x_{l+1} - x_{l-1}) / 2
        a_l =  x_{l+1} - 2 x_l + x_{l-1}
        k_l = sqrt( |a|^2 |v|^2 - <a,v>^2 )_G / |v|_G^3

    Returns (L-1,) for an (L+1, d) trajectory: interior points only.
    """
    v = 0.5 * (traj[2:] - traj[:-2])
    a = traj[2:] - 2 * traj[1:-1] + traj[:-2]
    vv = torch.einsum("li,ij,lj->l", v, G, v).clamp_min(0.0)
    aa = torch.einsum("li,ij,lj->l", a, G, a).clamp_min(0.0)
    av = torch.einsum("li,ij,lj->l", a, G, v)
    return torch.sqrt((aa * vv - av**2).clamp_min(0.0)) / (vv.pow(1.5) + eps)


def layer_trajectory(H: torch.Tensor, token: int = -1) -> torch.Tensor:
    """Extract one token's path through the layers.  (L+1, T, d) -> (L+1, d)."""
    return H[:, token, :]
