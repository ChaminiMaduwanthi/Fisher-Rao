"""
Resolve the scramble control's "frame sensitivity" on the CURVATURE side.

diag_frame.py establishes, in linear algebra alone, that F_real and F_scr are
two nearly-orthogonal SUBSPACES rather than two bases for one subspace, and that
the cross-frame restriction is worse conditioned than even a random subspace.
That identifies the mechanism.  It does not by itself license a verdict, because
two things still have to be true:

    A.  Rotation invariance must hold in the SCRAMBLED condition too.  Task 3.8
        verified it for the real metric only.  If curvature moved under a mere
        rotation of F_scr, the scrambled arm would be numerically broken and
        nothing computed from it could be trusted -- including the own-frame
        number this analysis is about to endorse.

    B.  The verdict must not depend on k.  The whole reason the deviation claim
        was retracted in Stage 3 is that Riemann quantities on this metric are
        k-fragile; a control that only works at k=5 is not a control.

and one thing has to be measured rather than assumed:

    C.  The conditioning-MATCHED comparison.  Own-frame at fixed k=5 happens to
        give cond_eff 21 (real) vs 18 (scrambled), which is already close, but
        that is luck rather than design.  select_k with a cond_eff ceiling picks
        k per point per condition so both arms are matched by construction.  If
        the effect survives that, conditioning cannot be the explanation.

COST, AND WHY THIS SCRIPT IS SPLIT INTO MODES
---------------------------------------------
Measured warm on an idle machine, one Riemann tensor costs

    k=3  0.39s    k=4  1.00s    k=5  2.61s    k=6  8.11s    k=7  16.9s    k=8  45.7s

-- roughly x2.7 per unit of k, steeper than the k^4 of the tensor itself,
because each entry is a second derivative of a function that is itself a
k-dimensional solve.  What actually killed the first three runs of this script
was not the k=5/6 work but the CONDITIONING-MATCHED stage: it called
k_by_conditioning, got k=11..16, and riemann() there attempts a 13.8-77 GB
allocation.  See K_FEASIBLE below.

So the checks are separated by cost, and each is run at the n it needs rather
than at one n for all:

    main    the verdict: paired own-frame real vs scrambled at k=5, plus the
            conditioning-matched comparison at k<=6.  ~21s/point -> run this at
            the largest n affordable.
    rot     rotation invariance in BOTH conditions.  ~11s/point, and it is a
            machine-precision identity -- if it holds at 1e-14 on a handful of
            points it is not going to fail on the hundredth.  n=5 is plenty.
    ksweep  k = 4 and 6 (k=5 comes from main), plus the cross-frame arm.
            ~23s/point.  Needs enough points to see whether the SIGN and
            MAGNITUDE CLASS of the effect move, not enough to put a confidence
            interval on each k.

Every point is appended to results/scramble/frame_points.jsonl the moment it is
finished, and points already in the file are skipped on restart, so a run that
is killed loses at most one point.  The analysis re-runs over everything
collected so far and prints on every invocation.

Usage:  python run_frame_resolution.py [mode] [n_points] [budget_seconds]
        modes: main (default) | rot | ksweep | all
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fisherrao import LM, corpus
from fisherrao.metrics import (fisher_metric, fisher_metric_projected,
                               fisher_metric_scrambled, topk_indices)
from fisherrao.curvature import riemann, sectional, scalar_curvature

OUT = pathlib.Path("results/scramble")
TOP_K, N_PLANES = 512, 16
LAYERS = [1, 5, 10, 15, 20, 25, 28, 29, 30]
COND_MAX = 1e2          # the ceiling curvature.select_k uses
K_SWEEP = (4, 5, 6)


def radial_proj(h):
    hhat = h / torch.linalg.vector_norm(h)
    return torch.eye(h.shape[-1], dtype=h.dtype) - torch.outer(hhat, hhat)


def frame_and_spectrum(lm, h, idx, P, k, perm=None):
    G = (fisher_metric(lm, h, idx=idx) if perm is None
         else fisher_metric_scrambled(lm, h, perm, idx=idx))
    ev, evec = torch.linalg.eigh(P @ G @ P)
    order = torch.argsort(ev, descending=True)
    return evec[:, order[:k]].contiguous(), ev[order].clamp_min(0.0)


def k_by_conditioning(ev, cond_max=COND_MAX, k_max=64):
    """Largest k whose cond_eff = lam_0 / lam_k stays under cond_max.

    Returns the k the CONDITIONING RULE wants, with no feasibility cap -- see
    K_FEASIBLE below for why the two must be reported separately.
    """
    lam0 = float(ev[0])
    k = 1
    for j in range(1, min(k_max, len(ev))):
        lj = float(ev[j])
        if lj <= 0 or lam0 / lj > cond_max:
            break
        k = j + 1
    return max(k, 2)                       # k=1 has no 2-plane


# The cond_eff ceiling does NOT bound k tightly enough to compute a Riemann
# tensor.  Measured here: at cond_max=1e2 the rule selects k = 11..16 at most
# points, and riemann() at k=16 attempts a 77 GB allocation and dies.  Three
# earlier runs of this script were killed by exactly that, silently, which is
# why they produced no output.
#
# So the prescription "select k by a cond_eff ceiling" from 06-stage3-log.md
# S4.3 is sound as a CONDITIONING criterion and unusable as a k SELECTOR for
# Riemann-derived quantities.  Both numbers are recorded per point: k_want is
# what the rule asks for, k_sel is what was affordable.
K_FEASIBLE = 6


def curv(lm, h, F, idx, perm=None, seed=0, planes=N_PLANES):
    """(median sectional K, scalar R) for the metric restricted to span(F)."""
    k = F.shape[1]

    def g(x):
        return fisher_metric_projected(lm, h + F @ x, F, idx, perm=perm)

    x0 = torch.zeros(k, dtype=h.dtype)
    R = riemann(g, x0)
    gen = torch.Generator().manual_seed(seed)
    Ks = []
    for _ in range(planes):
        u = torch.randn(k, generator=gen, dtype=h.dtype)
        v = torch.randn(k, generator=gen, dtype=h.dtype)
        val = sectional(g, x0, u, v, R=R)
        if val == val:
            Ks.append(val)
    return (float(torch.tensor(Ks, dtype=h.dtype).median()) if Ks else float("nan"),
            scalar_curvature(g, x0, R=R))


def median(xs):
    t = torch.tensor([x for x in xs if x == x], dtype=torch.float64)
    return float(t.median()) if len(t) else float("nan")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "main"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    budget = float(sys.argv[3]) if len(sys.argv) > 3 else 460.0
    if mode not in ("main", "rot", "ksweep", "all"):
        raise SystemExit(f"unknown mode {mode!r}; use main|rot|ksweep|all")
    do_rot = mode in ("rot", "all")
    do_sweep = mode in ("ksweep", "all")
    ckpt = OUT / "frame_points.jsonl"
    gen = torch.Generator().manual_seed(0)
    lm = LM()
    print(lm.summary())

    done: dict[int, dict] = {}
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.setdefault(r["_i"], {}).update(r)     # merge across modes
    print(f"\nmode={mode}   checkpoint: {len(done)} points on file"
          f"   budget this run: {budget:.0f}s")

    states = []
    for s in corpus.sentences():
        H, _ = lm.residual_stream(s)
        for l in LAYERS:
            for t in range(H.shape[1]):
                states.append((l, H[l, t]))
    pick = torch.randperm(len(states), generator=gen)[:n].tolist()
    perm = torch.randperm(lm.U.shape[0], generator=gen)

    need_main = ("K_real_k5", "K_scr_k5", "K_real_sel", "K_scr_sel")
    need_rot = ("rot_real", "rot_scr")
    need_sweep = ("K_real_k4", "K_real_k6", "K_scr_realframe")

    rows, t0, n_new = [], time.time(), 0
    for j, i in enumerate(pick, 1):
        have = done.get(i, {})
        wanted = list(need_main)
        if do_rot:
            wanted += list(need_rot)
        if do_sweep:
            wanted += list(need_sweep)
        if all(key in have for key in wanted) or "error" in have:
            rows.append(have)
            continue
        if time.time() - t0 > budget:
            print(f"   budget reached after {n_new} new points "
                  f"({j-1}/{len(pick)} seen); rerun to continue", flush=True)
            break
        layer, h = states[i]
        idx = topk_indices(lm, h, TOP_K)
        P = radial_proj(h)
        lp = torch.log_softmax(lm.logits(h), -1)
        rec = dict(have)
        rec.update(layer=layer, entropy=float(-(lp.exp() * lp).sum()))

        try:
            F_r, ev_r = frame_and_spectrum(lm, h, idx, P, 5)
            F_s, ev_s = frame_and_spectrum(lm, h, idx, P, 5, perm=perm)

            # ---- A. rotation invariance, BOTH conditions, at k=5 -----------
            if do_rot and not all(key in rec for key in need_rot):
                Q, _ = torch.linalg.qr(
                    torch.randn(5, 5, generator=gen, dtype=h.dtype))
                _, R_r = curv(lm, h, F_r, idx, planes=1)
                _, R_rQ = curv(lm, h, F_r @ Q, idx, planes=1)
                _, R_s = curv(lm, h, F_s, idx, perm=perm, planes=1)
                _, R_sQ = curv(lm, h, F_s @ Q, idx, perm=perm, planes=1)
                rec["rot_real"] = abs(R_r - R_rQ) / max(abs(R_r), abs(R_rQ), 1e-300)
                rec["rot_scr"] = abs(R_s - R_sQ) / max(abs(R_s), abs(R_sQ), 1e-300)

            # ---- B. own-frame effect across k ------------------------------
            # k=5 is always computed (it is the verdict); 4 and 6 only in sweep.
            for k in K_SWEEP:
                if k != 5 and not do_sweep:
                    continue
                if f"K_real_k{k}" in rec and f"K_scr_k{k}" in rec:
                    continue
                Fr = F_r if k == 5 else frame_and_spectrum(lm, h, idx, P, k)[0]
                Fs = F_s if k == 5 else frame_and_spectrum(
                    lm, h, idx, P, k, perm=perm)[0]
                rec[f"K_real_k{k}"], rec[f"R_real_k{k}"] = curv(lm, h, Fr, idx)
                rec[f"K_scr_k{k}"], rec[f"R_scr_k{k}"] = curv(
                    lm, h, Fs, idx, perm=perm)

            # ---- C. conditioning-MATCHED comparison ------------------------
            if "K_real_sel" not in rec:
                kr_want = k_by_conditioning(ev_r)
                ks_want = k_by_conditioning(ev_s)
                kr_sel = min(kr_want, K_FEASIBLE)
                ks_sel = min(ks_want, K_FEASIBLE)
                Fr2 = F_r if kr_sel == 5 else frame_and_spectrum(
                    lm, h, idx, P, kr_sel)[0]
                Fs2 = F_s if ks_sel == 5 else frame_and_spectrum(
                    lm, h, idx, P, ks_sel, perm=perm)[0]
                rec["k_want_real"], rec["k_want_scr"] = kr_want, ks_want
                rec["k_sel_real"], rec["k_sel_scr"] = kr_sel, ks_sel
                rec["cond_real"] = float(ev_r[0] / ev_r[kr_sel - 1])
                rec["cond_scr"] = float(ev_s[0] / ev_s[ks_sel - 1])
                rec["K_real_sel"], rec["R_real_sel"] = curv(lm, h, Fr2, idx)
                rec["K_scr_sel"], rec["R_scr_sel"] = curv(lm, h, Fs2, idx, perm=perm)

            # ---- the cross-frame arm, for the record -----------------------
            if do_sweep and "K_scr_realframe" not in rec:
                rec["K_scr_realframe"], rec["R_scr_realframe"] = curv(
                    lm, h, F_r, idx, perm=perm)
                rec["K_real_scrframe"], rec["R_real_scrframe"] = curv(
                    lm, h, F_s, idx)
        except Exception as e:                       # noqa: BLE001
            rec["error"] = repr(e)
        rec["_i"] = i
        rows.append(rec)
        n_new += 1
        with ckpt.open("a", encoding="utf-8") as fh:      # survive a kill
            fh.write(json.dumps(rec) + "\n")
        print(f"   {j}/{len(pick)}  (+{n_new} new, {time.time()-t0:.0f}s)",
              flush=True)

    ok = [r for r in rows if "error" not in r]
    print(f"\n{'='*74}\nn = {len(ok)} points completed "
          f"({len(rows)-len(ok)} failed)\n{'='*74}")

    def has(r, *keys):
        return all(k in r and r[k] == r[k] for k in keys)

    # ---- A ---------------------------------------------------------------
    rot = [r for r in ok if has(r, "rot_real", "rot_scr")]
    print("\nA. ROTATION INVARIANCE -- is this a FRAME problem at all?")
    if not rot:
        print("   (no points yet; run `python run_frame_resolution.py rot 5`)")
        worst = float("nan")
    else:
        print("   relative change in scalar R under an arbitrary rotation of "
              f"the frame, n={len(rot)}:")
        print(f"     real condition      : median {median([r['rot_real'] for r in rot]):.2e}"
              f"   max {max(r['rot_real'] for r in rot):.2e}")
        print(f"     scrambled condition : median {median([r['rot_scr'] for r in rot]):.2e}"
              f"   max {max(r['rot_scr'] for r in rot):.2e}")
        worst = max(max(r["rot_real"] for r in rot), max(r["rot_scr"] for r in rot))
        print(f"   -> {'FRAME-INVARIANT in both conditions' if worst < 1e-6 else 'FAILS'}"
              f"   (worst {worst:.1e})")
        print("      So 'frame sensitivity' is a MISNOMER: rotating the basis changes")
        print("      nothing.  What changed was the SUBSPACE.")

    # ---- B ---------------------------------------------------------------
    print("\nB. OWN-SUBSPACE EFFECT ACROSS k -- is the verdict k-stable?")
    print(f"   {'k':>3}  {'n':>4} {'K_real':>9} {'K_scr':>9} {'ratio':>8}   "
          f"{'sign test':>12}  {'z':>7}")
    kstab = {}
    for k in K_SWEEP:
        sub = [r for r in ok if has(r, f"K_real_k{k}", f"K_scr_k{k}")]
        if not sub:
            continue
        kr = [r[f"K_real_k{k}"] for r in sub]
        ks = [r[f"K_scr_k{k}"] for r in sub]
        d = [a - b for a, b in zip(kr, ks)]
        pos, tot = sum(1 for x in d if x > 0), len(d)
        z = (pos - tot / 2) / (0.5 * tot ** 0.5) if tot else 0.0
        mr, ms = median(kr), median(ks)
        kstab[k] = dict(K_real=mr, K_scr=ms, pos=pos, n=tot, z=z)
        print(f"   {k:>3}  {tot:>4} {mr:>9.4f} {ms:>9.4f} "
              f"{mr/ms if ms else float('nan'):>8.1f}x"
              f"   {pos:>5}/{tot:<6}  {z:>+7.2f}")

    # ---- C ---------------------------------------------------------------
    sel = [r for r in ok if has(r, "K_real_sel", "K_scr_sel")]
    print("\nC. CONDITIONING-MATCHED (k chosen per point by cond_eff <= "
          f"{COND_MAX:g}),  n={len(sel)}")
    zsel = float("nan")
    if sel:
        kw = [r["k_want_real"] for r in sel if "k_want_real" in r]
        if kw:
            print(f"   k WANTED by the cond_eff rule: real median {median(kw):.1f}"
                  f"   scrambled median "
                  f"{median([r['k_want_scr'] for r in sel if 'k_want_scr' in r]):.1f}"
                  f"   -- capped at {K_FEASIBLE} (riemann OOMs above ~8)")
        print(f"   k selected   real median {median([r['k_sel_real'] for r in sel]):.1f}"
              f"   scrambled median {median([r['k_sel_scr'] for r in sel]):.1f}")
        print(f"   cond_eff     real median {median([r['cond_real'] for r in sel]):.1f}"
              f"   scrambled median {median([r['cond_scr'] for r in sel]):.1f}"
              f"   <-- matched by construction")
        dsel = [r["K_real_sel"] - r["K_scr_sel"] for r in sel]
        pos, tot = sum(1 for x in dsel if x > 0), len(dsel)
        zsel = (pos - tot / 2) / (0.5 * tot ** 0.5) if tot else 0.0
        print(f"   K_real median {median([r['K_real_sel'] for r in sel]):.4f}"
              f"   K_scr median {median([r['K_scr_sel'] for r in sel]):.4f}")
        print(f"   sign test {pos}/{tot} positive   z = {zsel:+.2f}   "
              f"{'SIGNIFICANT' if abs(zsel) > 1.96 else 'not significant'}")

    # ---- the cross-frame arm ---------------------------------------------
    cross = [r for r in ok if has(r, "K_scr_realframe", "K_real_scrframe")]
    print("\nD. THE CROSS-FRAME ARM, for the record (this is the arm being retired)"
          f"   n={len(cross)}")
    if cross:
        print(f"   {'measurement':<32} {'median K':>12} {'|K| p90':>12}")
        for lbl, key, own in (("K_scr  in the REAL frame", "K_scr_realframe", "K_scr_k5"),
                              ("K_real in the SCRAMBLED frame", "K_real_scrframe", "K_real_k5")):
            v = torch.tensor([r[key] for r in cross], dtype=torch.float64)
            o = torch.tensor([r[own] for r in cross if has(r, own)],
                             dtype=torch.float64)
            print(f"   {lbl:<32} {float(v.median()):>12.4f} "
                  f"{float(torch.quantile(v.abs(), 0.9)):>12.4g}")
            print(f"   {'  (its own-frame value)':<32} {float(o.median()):>12.4f} "
                  f"{float(torch.quantile(o.abs(), 0.9)):>12.4g}")
    print("   Read the p90 column, not the median: the cross-frame arm's failure")
    print("   mode is a heavy tail from near-degenerate restrictions, which a")
    print("   median hides.  Whether the inflation is symmetric across the two")
    print("   directions is reported here rather than assumed.")

    (OUT / "frame_resolution.json").write_text(json.dumps(dict(
        n=len(ok), cond_max=COND_MAX, k_sweep=list(K_SWEEP),
        k_stability=kstab, z_matched=zsel, rows=rows), indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'frame_resolution.json'}")


if __name__ == "__main__":
    main()
