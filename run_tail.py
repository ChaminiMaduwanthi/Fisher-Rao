"""
What are the high-curvature points?

Stage 4 found that 93% of points sit tightly at K ~ +1/4 (IQR 0.018) while 7.2%
depart by more than 0.1, and that those departures are REAL geometry rather than
numerical failure (07-stage4-log.md S1.1c):

    outliers track scalar R as well as the core   (rho +0.675 vs +0.695)
    their scalar R is elevated                    (median 24.06 vs 5.42)
    they cluster by layer                         (9/40 at layer 1, 0/40 at layer 10)

Since absolute K is pinned to the ambient simplex value by construction and the
deviation statistic was retracted as k-unidentifiable, this tail is the best
remaining candidate for model-specific geometry.  This script asks what
distinguishes those points, using only quantities already validated.

It deliberately tests CHEAP explanations first.  If the tail is simply "low
entropy" or "poorly conditioned", that is the answer and it is not interesting;
only if it survives those does it deserve more work.

Usage:  python run_tail.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fisherrao import LM, corpus
from fisherrao.curvature import spectral_diagnostics
from fisherrao.stats import spearman as _spearman

OUT = pathlib.Path("results/tail")
SRC = pathlib.Path("results/stage4/stage4.json")


def spearman(a, b):
    """Tie-corrected Spearman -- see fisherrao/stats.py for why this is
    not argsort(argsort(x))."""
    return _spearman(a, b)
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not SRC.exists():
        print(f"need {SRC} -- run run_stage4.py first"); return
    sub = json.load(SRC.open(encoding="utf-8"))["e2_sample"]
    lm = LM()
    sents = corpus.sentences()
    streams = {}

    K_DIM, TOP_K = 5, 512
    print(f"n = {len(sub)} points from Stage 4\n")

    # enrich each point with cheap covariates
    for c in sub:
        si = c["sent"]
        if si not in streams:
            streams[si] = lm.residual_stream(sents[si])
        H, toks = streams[si]
        h = H[c["layer"], c["pos"]]
        sd = spectral_diagnostics(lm, h, K_DIM, top_k=TOP_K)
        logp = torch.log_softmax(lm.logits(h), -1)
        p = logp.exp()
        c["entropy"] = float(-(p * logp).sum())
        c["top1"] = float(p.max())
        c["cond_eff_at_k"] = sd["cond_eff_at_k"]
        c["k_eff_99"] = sd["k_eff_99"]
        c["log_vol"] = sd["log_vol_per_dim"]
        c["tok"] = lm.tok.convert_tokens_to_string([toks[c["pos"]]]).strip()
        c["pos_frac"] = c["pos"] / max(1, H.shape[1] - 1)

    dev = [abs(c["intrinsic_K"] - 0.25) for c in sub]
    tail = [c for c, d in zip(sub, dev) if d > 0.1]
    core = [c for c, d in zip(sub, dev) if d <= 0.1]
    print(f"tail (|K-0.25|>0.1): {len(tail)}    core: {len(core)}\n")

    # ---- cheap explanations first ----------------------------------------
    print("Does |K-0.25| track any cheap covariate?  (Spearman, all points)")
    print(f"{'covariate':<16} {'rho':>8}   interpretation if large")
    checks = [("entropy", "tail is just low/high entropy"),
              ("top1", "tail is just confident predictions"),
              ("cond_eff_at_k", "tail is a conditioning failure"),
              ("k_eff_99", "tail is just effective dimension"),
              ("log_vol", "tail is just small volume"),
              ("pos_frac", "tail is just sentence position"),
              ("layer", "tail is just depth")]
    rows = {}
    for key, note in checks:
        r = spearman(dev, [c[key] for c in sub])
        rows[key] = r
        flag = "  <-- LARGE" if abs(r) > 0.4 else ""
        print(f"{key:<16} {r:>+8.3f}   {note}{flag}")

    print(f"\n{'covariate':<16} {'core median':>13} {'tail median':>13} {'ratio':>8}")
    summary = {}
    for key, _ in checks:
        cm = float(torch.tensor([c[key] for c in core], dtype=torch.float64).median())
        tm = float(torch.tensor([c[key] for c in tail], dtype=torch.float64).median())
        summary[key] = dict(core=cm, tail=tm)
        ratio = tm / cm if cm else float("nan")
        print(f"{key:<16} {cm:>13.4f} {tm:>13.4f} {ratio:>8.2f}")

    # ---- what tokens are they? -------------------------------------------
    from collections import Counter
    print("\ntail tokens (most common):")
    for t, n in Counter(c["tok"] for c in tail).most_common(12):
        print(f"    {n:>2}x  {t!r}")
    print("\ncore tokens for comparison:")
    for t, n in Counter(c["tok"] for c in core).most_common(8):
        print(f"    {n:>2}x  {t!r}")

    # ---- is the tail sign-split? -----------------------------------------
    hi = [c for c in tail if c["intrinsic_K"] > 0.35]
    lo = [c for c in tail if c["intrinsic_K"] < 0.15]
    print(f"\ntail splits: {len(hi)} high-K (>0.35), {len(lo)} low-K (<0.15)")
    if hi:
        print(f"   high-K entropy median {float(torch.tensor([c['entropy'] for c in hi]).median()):.3f}")
    if lo:
        print(f"   low-K  entropy median {float(torch.tensor([c['entropy'] for c in lo]).median()):.3f}")
    print(f"   core   entropy median {float(torch.tensor([c['entropy'] for c in core]).median()):.3f}")

    # ---- THRESHOLD test.  A global rank correlation is the WRONG instrument
    # here and nearly produced a false conclusion: entropy scores only
    # rho = -0.117 against |K-0.25| over all 360 points, which reads as "no cheap
    # explanation".  But the tail is a THRESHOLD effect concentrated at one end
    # of the entropy range, and rank correlation over the whole range dilutes it
    # to nothing.  Binning exposes it immediately.
    print("\nTHRESHOLD test -- tail rate by entropy bin (what global Spearman misses)")
    print(f"{'entropy bin':<16} {'n':>5} {'tail':>5} {'tail %':>8} {'median K':>10}")
    bins = [(0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 1e9)]
    rates = []
    for lo, hi in bins:
        g = [c for c in sub if lo <= c["entropy"] < hi]
        if not g:
            continue
        t = [c for c in g if abs(c["intrinsic_K"] - 0.25) > 0.1]
        mk = float(torch.tensor([c["intrinsic_K"] for c in g], dtype=torch.float64).median())
        rate = len(t) / len(g)
        rates.append(rate)
        print(f"[{lo:>5.2f},{hi:>6.2f}) {len(g):>5} {len(t):>5} {100*rate:>7.1f}% {mk:>10.4f}")

    spread = max(rates) - min(rates) if rates else 0.0
    print(f"\ntail rate spread across entropy bins: {100*spread:.1f} percentage points")
    if spread > 0.3:
        print("-> THE TAIL IS AN ENTROPY THRESHOLD EFFECT, not idiosyncratic geometry.")
        print("   Do not report it as independent model-specific structure.  The right")
        print("   framing is the monotone curvature-entropy relationship (RQ3a), which")
        print("   is stronger and survives excluding the tail entirely.")
    else:
        print("-> no entropy threshold; the tail may be independent structure.")

    verdict = max(rows.items(), key=lambda kv: abs(kv[1]))
    print(f"\nstrongest global rank correlate: {verdict[0]} (rho = {verdict[1]:+.3f})")
    print("NOTE: judge by the binned table above, not this number.")

    (OUT / "tail.json").write_text(json.dumps(
        dict(n=len(sub), n_tail=len(tail), correlations=rows, medians=summary),
        indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'tail.json'}")


if __name__ == "__main__":
    main()
