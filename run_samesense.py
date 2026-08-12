"""
The tight control for RQ3b: same-sense MINIMAL pairs against different-sense ones.

07-stage4-log.md S3b.5 ran WiC's same-sense arm and found that, once both
targets have real preceding context, same-sense pairs separate 1.01-1.14x as
much as different-sense ones -- so the instruments were reading DIFFERENT
CONTEXT, not DIFFERENT SENSE.  But WiC's pairs are arbitrary sentences that
differ in every way at once, so that control cannot distinguish

    "the instruments cannot see sense"
from
    "WiC's same-sense pairs differ so much in wording that sense is swamped".

This script settles it with the tight version.  corpus.POLYSEMY_SAME is built to
the SAME recipe as corpus.POLYSEMY -- same target word, same final position,
matched syntactic frame -- and crucially SHARES SENTENCE A with it:

    different-sense    A = "...beside the flooded river, she studied the bank"
                       B = "...reviewing the mortgage paperwork ... called the bank"
    same-sense         A = "...beside the flooded river, she studied the bank"
                       B = "...along the swollen stream, he examined the bank"

Both arms measure the separation between A and *some other sentence ending in
the same word*.  The ONLY difference between the arms is whether that other
sentence carries the same sense.  Any residual gap is sense; anything shared is
context.

    d_FR    2 arccos(sum sqrt(p q))  -- exact geodesic distance on the simplex
    d_UU    ||h_A - h_B|| under Manson's constant G = U^T U
    d_euc   plain Euclidean

Usage:  python run_samesense.py [model_id]
"""

from __future__ import annotations

import json
import pathlib
import sys

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fisherrao import LM, DEFAULT_MODEL, corpus, manson_metric
from run_polysemy import find_token, fisher_rao_distance

OUT = pathlib.Path("results/samesense")


def half_depth(vals: list[float]) -> int:
    final = vals[-1]
    if final <= 0:
        return len(vals) - 1
    for i, v in enumerate(vals):
        if v >= 0.5 * final:
            return i
    return len(vals) - 1


def measure(lm, G_man, word, sa, sb):
    Ha, ta = lm.residual_stream(sa)
    Hb, tb = lm.residual_stream(sb)
    ia, ib = find_token(lm, ta, word), find_token(lm, tb, word)
    if ia is None or ib is None:
        return None
    per = []
    for l in range(lm.n_layers + 1):
        ha, hb = Ha[l, ia], Hb[l, ib]
        d = ha - hb
        per.append(dict(
            d_fr=fisher_rao_distance(lm.next_token_probs(ha),
                                     lm.next_token_probs(hb)),
            d_uu=float(torch.sqrt((d @ G_man @ d).clamp_min(0))),
            d_euc=float(torch.linalg.vector_norm(d))))
    return per


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    lm = LM(model_id)
    G_man = manson_metric(lm.U)
    print(lm.summary())

    problems = corpus.validate()
    if problems:
        print("\nCORPUS PROBLEMS -- refusing to run:")
        for p in problems:
            print("   -", p)
        return 1

    # THREE arms, same sentence A throughout, paired by word:
    #   same        same sense,      A's frame           (the control)
    #   diff        different sense, A's frame           (frame-matched; the test)
    #   diff_free   different sense, free frame          (the original, confounded
    #                                                     arm -- kept so the
    #                                                     confound's size is visible)
    same = {w: (a, b) for w, a, b in corpus.POLYSEMY_SAME}
    diff = {w: (a, b) for w, a, b in corpus.POLYSEMY_DIFF}
    free = {w: (a, b) for w, a, b in corpus.POLYSEMY}
    words = [w for w, _, _ in corpus.POLYSEMY_SAME if w in diff and w in free]
    print(f"\n{len(words)} words present in all three arms -- paired by word.\n")

    rows = []
    for w in words:
        a, b_same = same[w]
        ps = measure(lm, G_man, w, a, b_same)
        pd = measure(lm, G_man, w, a, diff[w][1])
        pf = measure(lm, G_man, w, a, free[w][1])
        if ps is None or pd is None or pf is None:
            print(f"   skipped {w}")
            continue
        rows.append(dict(word=w, same=ps, diff=pd, diff_free=pf))
    print(f"measured {len(rows)} paired words")

    print(f"\n{'='*76}\nHALF-SEPARATION DEPTH   n={len(rows)} words, paired"
          f"\n{'='*76}")
    print(f"   {'instrument':<10} {'SAME':>10} {'DIFF matched':>14} "
          f"{'DIFF free frame':>17}")
    summary = {}
    for key in ("d_fr", "d_uu", "d_euc"):
        cell = {}
        for arm in ("same", "diff", "diff_free"):
            hd = torch.tensor([half_depth([p[key] for p in r[arm]]) for r in rows],
                              dtype=torch.float64)
            cell[arm] = float(hd.mean())
        summary[key] = dict(half=cell)
        print(f"   {key:<10} {cell['same']:>10.2f} {cell['diff']:>14.2f} "
              f"{cell['diff_free']:>17.2f}")

    print(f"\n{'='*76}\n🎯 THE CONTROL -- final-layer separation, PAIRED BY WORD"
          f"\n{'='*76}")
    print(f"   {'instrument':<10} {'arm':<12} {'median':>10} {'ratio vs same':>14}"
          f" {'>same':>9} {'z':>7}")
    n = len(rows)
    for key in ("d_fr", "d_uu", "d_euc"):
        vs = torch.tensor([r["same"][-1][key] for r in rows], dtype=torch.float64)
        summary[key]["final"] = {"same": float(vs.median())}
        for arm, lbl in (("diff", "DIFF matched"), ("diff_free", "DIFF free")):
            va = torch.tensor([r[arm][-1][key] for r in rows], dtype=torch.float64)
            pos = int((va > vs).sum())
            z = (pos - n / 2) / (0.5 * n ** 0.5)
            summary[key]["final"][arm] = dict(median=float(va.median()), pos=pos,
                                              n=n, z=z)
            print(f"   {key if arm=='diff' else '':<10} {lbl:<12} "
                  f"{float(va.median()):>10.4f} "
                  f"{float(va.median()/vs.median()):>13.2f}x {pos:>5}/{n:<3} "
                  f"{z:>+7.2f}")
        print(f"   {'':<10} {'(same sense)':<12} {float(vs.median()):>10.4f}")

    fr = summary["d_fr"]["final"]["diff"]
    print(f"\n{'='*76}\nVERDICT\n{'='*76}")
    base = summary["d_fr"]["final"]["same"]
    print(f"   Frame-MATCHED different-sense pairs separate "
          f"{fr['median']/base:.2f}x more than same-sense pairs,")
    print(f"   in {fr['pos']}/{fr['n']} words (z = {fr['z']:+.2f}).  Lexical overlap")
    print(f"   with A is equalised across the arms (corpus.validate enforces it).")
    if abs(fr["z"]) > 1.96 and fr["median"] / base > 1.3:
        print("   -> THE INSTRUMENTS READ SENSE, with lexical overlap controlled.")
        print("      This is the claim S3b.6 tried to make and could not, because")
        print("      its different-sense arm had 2.00x LESS overlap with A.")
    elif abs(fr["z"]) > 1.96:
        print("   -> a consistent but SMALL sense effect: the sign test passes")
        print("      while the ratio stays near 1, so most of the separation is")
        print("      still context, not sense.")
    else:
        print("   -> NO detectable sense effect even with wording matched.")
        print("      S3b.5's conclusion stands in its strongest form: these")
        print("      instruments measure different CONTEXT, and RQ3b must not be")
        print("      described as locating where ambiguity resolves.")

    (OUT / f"samesense_{model_id.split('/')[-1]}.json").write_text(json.dumps(
        dict(model=model_id, n=len(rows), summary=summary, rows=rows), indent=2),
        encoding="utf-8")
    print(f"\nwrote {OUT}/samesense_{model_id.split('/')[-1]}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
