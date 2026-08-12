"""
E5 -- causal intervention (RQ4).  Does the geometry predict the model's behaviour?

Every number in this project so far describes a metric.  None of it shows the
metric is about the MODEL rather than about an arbitrary construction laid over
it.  E5 is the test that closes that gap, and it is the experiment Manson's
paper explicitly lists as missing from the layer-curvature literature.

TWO PREDICTIONS, and they fail in different ways
------------------------------------------------

    (1)  NULL DIRECTIONS DO NOTHING.
         Perturb h along an exact null direction of G and the output must not
         move.  At EQUAL EUCLIDEAN SIZE a random direction must move it a lot.
         This is the falsification test: if a null perturbation changes the
         prediction, the metric is not the model's geometry and every curvature
         number in the thesis is void.

    (2)  AT MATCHED FISHER NORM, FIRST ORDER IS BLIND AND CURVATURE IS NOT.
         Gate A established KL(p(h) || p(h + eps v)) = 1/2 eps^2 v'Gv + O(eps^3).
         So if three directions are scaled to v'Gv = 1, the metric predicts the
         SAME KL for all three, to leading order.  Any systematic difference at
         larger eps is by construction beyond what the metric alone says -- it
         is where curvature lives.

         Prediction: directions of high Ricci curvature depart from the
         quadratic sooner than directions of low Ricci curvature.

WHY RICCI AND NOT SECTIONAL
---------------------------
"The highest sectional-curvature direction" is not well defined -- sectional
curvature is a function of a 2-PLANE, not a direction.  Ricci curvature is the
natural per-direction object (Ric(u,u) averages the sectional curvatures of the
planes containing u), and the generalised eigenproblem

    Ric v = lambda G v

gives exactly the extremal curvature directions at unit Fisher norm, which is
the matching E5 asks for.  Solved through a Cholesky factor of G rather than by
inverting it.

CHECKPOINTED per (model, k, point index), appended as each point finishes, so a
run that is killed loses at most one point.  Re-run the same command until it
reports the full n.

Usage:  python run_e5.py [n_points] [model_id] [k] [budget_seconds]
        e.g.  python run_e5.py 120 gpt2 5 440
              python run_e5.py 40 HuggingFaceTB/SmolLM2-135M-Instruct 6
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fisherrao import LM, DEFAULT_MODEL, corpus
from fisherrao.metrics import fisher_metric, fisher_metric_projected, topk_indices
from fisherrao.curvature import effective_frame, metric_in_frame, riemann, null_projector

OUT = pathlib.Path("results/e5")
K_DIM, TOP_K = 5, 512
LAYERS = [1, 5, 10, 15, 20, 25, 28, 29, 30]
EPS = (1e-3, 1e-2, 1e-1, 3e-1, 1.0)


def kl(lm, h: torch.Tensor, hp: torch.Tensor) -> float:
    """KL( p(h) || p(hp) ), in float64, without forming either full metric."""
    logp = torch.log_softmax(lm.logits(h), -1)
    logq = torch.log_softmax(lm.logits(hp), -1)
    return float((logp.exp() * (logp - logq)).sum())


def ricci_directions(G: torch.Tensor, R: torch.Tensor):
    """Extremal directions of Ric(u,u) subject to u'Gu = 1.

    Solves Ric v = lambda G v via a Cholesky factor of G:  with G = L L',
    y = L' v turns it into the ordinary symmetric problem (L^-1 Ric L^-T) y.
    Returns (v_max, v_min, lam_max, lam_min), each normalised to v'Gv = 1.
    """
    ric = torch.einsum("kikj->ij", R)
    ric = 0.5 * (ric + ric.T)
    L = torch.linalg.cholesky(G)
    M = torch.linalg.solve_triangular(
        L, torch.linalg.solve_triangular(L, ric, upper=False).T, upper=False)
    M = 0.5 * (M + M.T)
    ev, evec = torch.linalg.eigh(M)
    V = torch.linalg.solve_triangular(L.T, evec, upper=True)     # back to v
    V = V / torch.sqrt(torch.einsum("ik,ij,jk->k", V, G, V)).clamp_min(1e-300)
    return V[:, -1], V[:, 0], float(ev[-1]), float(ev[0])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    model_id = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    k_dim = int(sys.argv[3]) if len(sys.argv) > 3 else K_DIM
    budget = float(sys.argv[4]) if len(sys.argv) > 4 else 1e9
    tag = f"{model_id.split('/')[-1]}_k{k_dim}"
    ckpt = OUT / f"e5_points_{tag}.jsonl"
    gen = torch.Generator().manual_seed(0)
    lm = LM(model_id)
    print(lm.summary())
    print(f"\nE5: {n} points, k={k_dim}, eps sweep {EPS}\n")

    # LAYERS is written for a 30-layer model; on a 12-layer one, sample the
    # same RELATIVE depths instead of clamping everything onto the last block.
    layers = ([l for l in LAYERS if l <= lm.n_layers] if lm.n_layers >= 30
              else sorted({max(1, round(l * lm.n_layers / 30)) for l in LAYERS}))

    states = []
    for s in corpus.sentences():
        H, _ = lm.residual_stream(s)
        for l in layers:
            for t in range(H.shape[1]):
                states.append((l, H[l, t]))
    pick = torch.randperm(len(states), generator=gen)[:n].tolist()

    done = {}
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["_i"]] = r
    print(f"checkpoint: {len(done)} points on file\n")

    rows, t0, n_new = [], time.time(), 0
    for j, i in enumerate(pick, 1):
        if i in done:
            rows.append(done[i])
            continue
        if time.time() - t0 > budget:
            print(f"   budget reached (+{n_new} new); rerun to continue",
                  flush=True)
            break
        layer, h = states[i]
        try:
            F = effective_frame(lm, h, k_dim, top_k=TOP_K)
            g_fn = metric_in_frame(lm, h, F, top_k=TOP_K)
            x0 = torch.zeros(k_dim, dtype=h.dtype)
            G = g_fn(x0)
            R = riemann(g_fn, x0)
            v_hi, v_lo, ric_hi, ric_lo = ricci_directions(G, R)

            u = torch.randn(k_dim, generator=gen, dtype=h.dtype)
            u = u / torch.sqrt(u @ G @ u)                      # matched Fisher norm

            # ---- (2) matched FISHER norm: high / low Ricci / random --------
            fisher = {}
            for name, v in (("hi", v_hi), ("lo", v_lo), ("rand", u)):
                amb = F @ v
                fisher[name] = dict(
                    gnorm=float(v @ G @ v),
                    enorm=float(torch.linalg.vector_norm(amb)),
                    kl=[kl(lm, h, h + e * amb) for e in EPS])

            # ---- (1) matched EUCLIDEAN norm: null vs random ---------------
            # exact null: the norm Jacobian's kernel (radial, +all-ones on LN)
            N = lm.null_directions(h)
            v_null = N[:, 0]
            # numerical null: the SMALLEST eigendirection of G, after removing
            # the exact nulls -- the direction the metric says matters least
            # among those it can actually see.
            P = null_projector(lm, h)
            Gd = P @ fisher_metric(lm, h, top_k=TOP_K) @ P
            ev, evec = torch.linalg.eigh(Gd)
            v_numnull = evec[:, 0]
            w = torch.randn(lm.d, generator=gen, dtype=h.dtype)
            w = P @ w
            w = w / torch.linalg.vector_norm(w)

            scale = float(torch.linalg.vector_norm(h))          # relative steps
            euclid = {}
            for name, v in (("exact_null", v_null), ("num_null", v_numnull),
                            ("random", w)):
                v = v / torch.linalg.vector_norm(v)
                euclid[name] = dict(
                    kl=[kl(lm, h, h + e * scale * v) for e in EPS],
                    gnorm=float(v @ Gd @ v))

            rec = dict(layer=layer, ric_hi=ric_hi, ric_lo=ric_lo,
                       hnorm=scale, fisher=fisher, euclid=euclid, _i=i)
        except Exception as exc:                                # noqa: BLE001
            rec = dict(layer=layer, error=repr(exc), _i=i)
        rows.append(rec)
        done[i] = rec
        n_new += 1
        with ckpt.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        if n_new % 10 == 0:
            print(f"   {j}/{len(pick)}  (+{n_new}, {time.time()-t0:.0f}s)",
                  flush=True)

    ok = [r for r in rows if "error" not in r]
    print(f"\n{'='*76}\nE5 -- CAUSAL INTERVENTION   n = {len(ok)} points"
          f" ({len(rows)-len(ok)} failed)\n{'='*76}")

    def med(vals):
        t = torch.tensor([v for v in vals if v == v], dtype=torch.float64)
        return float(t.median()) if len(t) else float("nan")

    # ---- (1) the falsification test ---------------------------------------
    print("\n(1) NULL-SPACE FALSIFICATION -- matched EUCLIDEAN step, "
          "size = eps * ||h||")
    print(f"    {'direction':<12}" + "".join(f"{f'eps={e:g}':>13}" for e in EPS))
    for name in ("exact_null", "num_null", "random"):
        print(f"    {name:<12}" + "".join(
            f"{med([r['euclid'][name]['kl'][i] for r in ok]):>13.3e}"
            for i in range(len(EPS))))
    # KL at an exact null direction is zero up to rounding, so it comes out at
    # +-1e-16 and the RATIO can be negative.  Compare magnitudes, not signs --
    # a signed max would let a large negative excursion read as a pass.
    ratio = [abs(med([r["euclid"]["exact_null"]["kl"][i] for r in ok]))
             / max(med([r["euclid"]["random"]["kl"][i] for r in ok]), 1e-300)
             for i in range(len(EPS))]
    print(f"    {'|null|/random':<12}" + "".join(f"{x:>13.2e}" for x in ratio))
    verdict1 = max(ratio) < 1e-8
    print(f"    -> {'PASS' if verdict1 else '**FAIL**'}: exact-null perturbations "
          f"leave the output unchanged (worst ratio {max(ratio):.1e})")
    print("       A FAIL here would void every curvature number in the thesis.")

    # ---- (2) does curvature predict departure from the quadratic? ---------
    print("\n(2) MATCHED FISHER NORM (v'Gv = 1 for all three)")
    print("    Gate A says KL = 1/2 eps^2 to leading order, IDENTICALLY for all")
    print("    three.  Ratio KL/(eps^2/2) is 1.00 where the metric fully explains")
    print("    the behaviour and departs where curvature takes over.")
    print(f"    {'direction':<12}" + "".join(f"{f'eps={e:g}':>13}" for e in EPS))
    for name in ("hi", "lo", "rand"):
        print(f"    {name:<12}" + "".join(
            f"{med([r['fisher'][name]['kl'][i] / (0.5*e*e) for r in ok]):>13.4f}"
            for i, e in enumerate(EPS)))
    print(f"    median Ricci eigenvalue   high {med([r['ric_hi'] for r in ok]):+.3f}"
          f"   low {med([r['ric_lo'] for r in ok]):+.3f}")

    # paired, per-point: does the high-Ricci direction depart further?
    last = len(EPS) - 1
    d = [r["fisher"]["hi"]["kl"][last] - r["fisher"]["lo"]["kl"][last] for r in ok]
    pos, tot = sum(1 for x in d if x > 0), len(d)
    z = (pos - tot / 2) / (0.5 * tot ** 0.5) if tot else 0.0
    print(f"\n    paired at eps={EPS[last]:g}: KL(high Ricci) > KL(low Ricci) in "
          f"{pos}/{tot} points   z = {z:+.2f}   "
          f"{'SIGNIFICANT' if abs(z) > 1.96 else 'not significant'}")

    (OUT / f"e5_{model_id.split(chr(47))[-1]}_k{k_dim}.json").write_text(json.dumps(dict(
        n=len(ok), model=model_id, eps=list(EPS), k=k_dim, null_ratio=ratio,
        verdict_null=bool(verdict1), z_ricci=z, rows=rows), indent=2),
        encoding="utf-8")
    print(f"\nwrote {OUT / f'e5_{model_id.split(chr(47))[-1]}_k{k_dim}.json'}")


if __name__ == "__main__":
    main()
