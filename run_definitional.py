"""
How much of the curvature-entropy relationship is DEFINITIONAL?

08-rq3a-log.md reports Spearman rho = -0.58 between intrinsic Fisher-Rao scalar
curvature and next-token entropy.  Both are computed from the same p(h), so part
of that is forced by construction: as p concentrates, the reachable region of the
simplex shrinks toward a low-dimensional face, and the induced geometry changes
whether or not the model learned anything.

    Until that component is quantified, rho = -0.58 cannot be called a finding
    ABOUT THE MODEL.

The control: replace the learned unembedding U with a RANDOM matrix, and sweep
its scale to sweep entropy.  Random U has no learned structure, so whatever
K-vs-entropy curve it produces is the purely definitional part.  Comparing the
two curves at MATCHED ENTROPY isolates what the model contributes.

Three conditions, all with the model's real hidden states:
    real     the trained unembedding
    random   Gaussian U, scale swept to cover the same entropy range
    shuffled the real U with its rows permuted -- destroys the token-to-direction
             assignment while preserving the exact spectrum and row-norm
             distribution of the real matrix.  Tighter control than Gaussian.

Usage:  python run_definitional.py [n_per_condition]
"""

from __future__ import annotations

import copy
import json
import pathlib
import sys
import time

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fisherrao import LM, corpus
from fisherrao.curvature import curvature_at
from fisherrao.stats import spearman as _spearman

OUT = pathlib.Path("results/definitional")
K_DIM, TOP_K, N_PLANES = 5, 512, 16
LAYERS = [1, 5, 10, 15, 20, 25, 28, 29, 30]


def spearman(a, b):
    """Tie-corrected Spearman -- see fisherrao/stats.py for why this is
    not argsort(argsort(x))."""
    return _spearman(a, b), len([1 for x, y in zip(a, b) if x == x and y == y])
def entropy_of(lm, h):
    lp = torch.log_softmax(lm.logits(h), -1)
    return float(-(lp.exp() * lp).sum())


def measure(lm, states, tag, n, gen):
    """Compute (entropy, K, R) at n randomly chosen states."""
    idx = torch.randperm(len(states), generator=gen)[:n].tolist()
    rows, t0 = [], time.time()
    for j, i in enumerate(idx, 1):
        h = states[i]
        try:
            c = curvature_at(lm, h, K_DIM, top_k=TOP_K, n_planes=N_PLANES)
        except Exception:
            continue
        rows.append(dict(entropy=entropy_of(lm, h), K=c["K_median"], R=c["scalar_R"]))
        if j % 25 == 0:
            print(f"     {tag}: {j}/{len(idx)}  ({time.time()-t0:.0f}s)")
    return rows


def profile(rows, bins):
    """median K and R per entropy bin."""
    out = []
    for lo, hi in bins:
        g = [r for r in rows if lo <= r["entropy"] < hi]
        if len(g) < 3:
            out.append(None); continue
        out.append(dict(
            n=len(g),
            K=float(torch.tensor([r["K"] for r in g], dtype=torch.float64).median()),
            R=float(torch.tensor([r["R"] for r in g], dtype=torch.float64).median()),
        ))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    gen = torch.Generator().manual_seed(0)

    lm = LM()
    print(lm.summary())
    states = []
    for s in corpus.sentences():
        H, _ = lm.residual_stream(s)
        for l in LAYERS:
            for t in range(H.shape[1]):
                states.append(H[l, t])
    print(f"\n{len(states)} hidden states available; sampling {n_per} per condition\n")

    conditions = {}

    # ---- real ------------------------------------------------------------
    print("  [real] trained unembedding")
    conditions["real"] = measure(lm, states, "real", n_per, gen)

    # ---- shuffled rows (preserves spectrum + row norms exactly) -----------
    print("  [shuffled] real U with rows permuted")
    lm_s = copy.copy(lm)
    perm = torch.randperm(lm.U.shape[0], generator=gen)
    lm_s.U = lm.U[perm].contiguous()
    conditions["shuffled"] = measure(lm_s, states, "shuffled", n_per, gen)

    # ---- random Gaussian, scale swept to span the entropy range ----------
    print("  [random] Gaussian U, scale swept")
    rnd = []
    per_scale = max(6, n_per // 5)
    for scale in (0.5, 1.0, 2.0, 4.0, 8.0):
        lm_r = copy.copy(lm)
        lm_r.U = torch.randn(lm.N, lm.d, generator=gen, dtype=torch.float64) * scale / lm.d**0.5
        rnd.extend(measure(lm_r, states, f"random x{scale}", per_scale, gen))
    conditions["random"] = rnd

    # ---- compare ---------------------------------------------------------
    print(f"\n{'='*72}\nOVERALL Spearman(curvature, entropy) per condition\n{'='*72}")
    print(f"{'condition':<12} {'n':>5} {'rho(K,H)':>10} {'rho(R,H)':>10} "
          f"{'entropy range':>22}")
    res = {}
    for name, rows in conditions.items():
        if not rows:
            continue
        rk, n = spearman([r["K"] for r in rows], [r["entropy"] for r in rows])
        rr, _ = spearman([r["R"] for r in rows], [r["entropy"] for r in rows])
        es = torch.tensor([r["entropy"] for r in rows], dtype=torch.float64)
        print(f"{name:<12} {n:>5} {rk:>+10.3f} {rr:>+10.3f} "
              f"{'[%.2f, %.2f]' % (float(es.min()), float(es.max())):>22}")
        res[name] = dict(n=n, rho_K=rk, rho_R=rr,
                         entropy_min=float(es.min()), entropy_max=float(es.max()))

    # ---- the decisive comparison: matched entropy ------------------------
    bins = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 1e9)]
    print(f"\n{'='*72}\nMATCHED-ENTROPY COMPARISON -- median K per entropy bin\n{'='*72}")
    print("If the curves coincide, the relationship is DEFINITIONAL.")
    print("Where they separate, that gap is what the trained model contributes.\n")
    labels = [f"[{lo:g},{hi:g})" if hi < 1e8 else f"[{lo:g},inf)" for lo, hi in bins]
    profs = {k: profile(v, bins) for k, v in conditions.items()}
    print(f"{'entropy bin':<12} " + "".join(f"{k:>22}" for k in profs))
    print(f"{'':<12} " + "".join(f"{'median K (n)':>22}" for _ in profs))
    for i, lab in enumerate(labels):
        cells = ""
        for k in profs:
            p = profs[k][i]
            cells += f"{('%.4f (%d)' % (p['K'], p['n'])) if p else '-':>22}"
        print(f"{lab:<12} {cells}")

    print(f"\n{'entropy bin':<12} " + "".join(f"{k:>22}" for k in profs))
    print(f"{'':<12} " + "".join(f"{'median scalar R':>22}" for _ in profs))
    for i, lab in enumerate(labels):
        cells = ""
        for k in profs:
            p = profs[k][i]
            cells += f"{('%.3f' % p['R']) if p else '-':>22}"
        print(f"{lab:<12} {cells}")

    # gap where all three overlap
    gaps = []
    for i in range(len(bins)):
        pr, ps = profs.get("real", [None]*len(bins))[i], profs.get("shuffled", [None]*len(bins))[i]
        if pr and ps:
            gaps.append((labels[i], pr["K"] - ps["K"], pr["R"] - ps["R"]))
    if gaps:
        print(f"\n{'bin':<12} {'K(real)-K(shuffled)':>22} {'R(real)-R(shuffled)':>22}")
        for lab, gk, gr in gaps:
            print(f"{lab:<12} {gk:>+22.4f} {gr:>+22.3f}")
        mg = max(abs(g[1]) for g in gaps)
        print(f"\nlargest |K gap| at matched entropy: {mg:.4f}")
        print(f"(compare: the full K range across entropy is ~0.18, from 0.24 to 0.42)")
        if mg < 0.02:
            print("-> THE RELATIONSHIP IS ESSENTIALLY DEFINITIONAL.  rho = -0.58 reflects")
            print("   the geometry of concentration, not learned structure.  Report it")
            print("   as such; it is not a finding about the model.")
        else:
            print("-> a real gap remains at matched entropy: the trained model's geometry")
            print("   differs from the structure-free control beyond concentration alone.")

    res["profiles"] = {k: [p for p in v] for k, v in profs.items()}
    res["bins"] = labels
    (OUT / "definitional.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'definitional.json'}")


if __name__ == "__main__":
    main()
