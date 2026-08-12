"""
Do E2 and RQ3a survive on a REAL corpus?

Every result in this project was measured on 40 hand-written sentences.  The
corpus module says so plainly -- "the author chose them, so they are not a
random sample of anything" -- and that caveat sits on the centrepiece figure.
It existed because HuggingFace was unreachable; that is fixed
(10-architecture-log.md S1), so the caveat can be retired by MEASURING rather
than by arguing.

11-mabrok-replication-log.md gives a concrete reason to expect the corpus to
matter: the local-PCA proxy's value depends on how dense the point cloud is,
and 40 short hand-written sentences give a very different cloud from encyclopedia
prose.  If any conclusion is corpus-dependent, the proxies are where it will
show.

DESIGN: the same measurement on both corpora, side by side, so the comparison
is paired at the level of the protocol rather than being two separate runs
reported in two separate places.

    handwritten   corpus.GENERAL + the polysemy sentences (the status quo)
    wikitext      WikiText-103 validation, sentence-split, length-bounded

Four instruments per point, exactly as defined in run_stage4.py (E2):

    intrinsic K, R    this work
    pca               local-PCA residual        (Mabrok)
    manson            Frenet under G = U^T U    (Manson)
    king              Euclidean angle           (King et al.)

and the two claims re-tested:

    E2    "no instrument tracks any other"      -- pairwise Spearman
    RQ3a  "intrinsic tracks entropy 2-3x better
           than any proxy"                      -- each instrument vs entropy

Checkpointed per (corpus, sentence, layer, position).

Usage:  python run_corpus_compare.py [n_per_layer] [budget_seconds]
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fisherrao import LM, corpus, manson_metric
from fisherrao.curvature import curvature_at
from fisherrao.proxies import pca_curvature, contextual_angle
from fisherrao.trajectory import curvature as frenet_curvature, layer_trajectory
from fisherrao.stats import spearman as _spearman

OUT = pathlib.Path("results/corpus_compare")
K_DIM, TOP_K = 5, 512
LAYERS = [1, 5, 10, 15, 20, 25, 28, 29, 30]
# wikitext first: the loop fills arms in order and the budget usually runs out
# part way, so whichever arm is listed first is the one that reaches the target
# n.  The wikitext arm is the one that needs raising -- the hand-written numbers
# are already published at n=360 from run_stage4.py.
SOURCES = ("wikitext", "handwritten")


def spearman(a, b):
    """Tie-corrected Spearman -- see fisherrao/stats.py for why this is
    not argsort(argsort(x))."""
    return _spearman(a, b)
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per_layer = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 440.0

    ckpt = OUT / "points.jsonl"
    done = {}
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["source"], r["sent"], r["layer"], r["pos"])] = r
    print(f"checkpoint: {len(done)} points   budget {budget:.0f}s")

    lm = LM()
    G_man = manson_metric(lm.U)
    t0, n_new, rows = time.time(), 0, list(done.values())

    for source in SOURCES:
        # 200 wikitext sentences is far more text than 148 hand-written ones;
        # cap it so the two arms cost the same and the comparison is about the
        # TEXT, not about one arm having more data.
        os.environ["FISHERRAO_CORPUS_N"] = "60"
        sents = (corpus.sentences() if source == "handwritten"
                 else corpus.sentences(source="wikitext"))[:60]
        print(f"\n{source}: {len(sents)} sentences")
        streams = [lm.residual_stream(s) for s in sents]

        gen = torch.Generator().manual_seed(0)
        for l in LAYERS:
            cand = [(si, t) for si, (H, _) in enumerate(streams)
                    for t in range(H.shape[1])]
            pick = torch.randperm(len(cand), generator=gen)[:per_layer].tolist()
            for j in pick:
                si, t = cand[j]
                key = (source, si, l, t)
                if key in done:
                    continue
                if time.time() - t0 > budget:
                    print(f"   budget reached (+{n_new}); rerun to continue",
                          flush=True)
                    report(rows)
                    return
                H, _ = streams[si]
                h = H[l, t]
                rec = dict(source=source, sent=si, layer=l, pos=t)
                try:
                    cc = curvature_at(lm, h, K_DIM, top_k=TOP_K, n_planes=16)
                    rec["K"] = cc["K_median"]
                    rec["R"] = cc["scalar_R"]
                    rec["pca"] = pca_curvature(H[l], t,
                                               n_neighbors=min(20, H.shape[1] - 1))
                    rec["king"] = contextual_angle(H, l, t)
                    # Frenet curvature is defined at INTERIOR trajectory points
                    # only, so entry j corresponds to layer j+1 -- the same
                    # indexing run_stage4.py uses when it builds man_map.
                    # Layers 0 and L have none.
                    kap = frenet_curvature(layer_trajectory(H, token=t), G_man)
                    rec["manson"] = (float(kap[l - 1]) if 1 <= l <= len(kap)
                                     else float("nan"))
                    lp = torch.log_softmax(lm.logits(h), -1)
                    rec["entropy"] = float(-(lp.exp() * lp).sum())
                except Exception as e:                            # noqa: BLE001
                    rec["error"] = repr(e)
                rows.append(rec)
                done[key] = rec
                n_new += 1
                with ckpt.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
            print(f"   layer {l:>2}  ({time.time()-t0:.0f}s, +{n_new})", flush=True)
    report(rows)


def report(rows):
    ok = [r for r in rows if "error" not in r and "K" in r]
    print(f"\n{'='*80}\nE2 + RQ3a ON TWO CORPORA   n={len(ok)}\n{'='*80}")

    INSTR = [("intrinsic K", "K"), ("intrinsic R", "R"),
             ("local-PCA (Mabrok)", "pca"), ("Frenet U^T U (Manson)", "manson"),
             ("Euclid angle (King)", "king")]

    print(f"\nRQ3a -- each instrument's Spearman against next-token entropy")
    print(f"   {'instrument':<24}" + "".join(f"{s:>16}" for s in SOURCES)
          + f"{'difference':>13}")
    out = {}
    for name, key in INSTR:
        vals = {}
        for s in SOURCES:
            sub = [r for r in ok if r["source"] == s]
            vals[s] = spearman([r[key] for r in sub], [r["entropy"] for r in sub])
        d = vals[SOURCES[1]] - vals[SOURCES[0]]
        out[name] = vals
        print(f"   {name:<24}" + "".join(f"{vals[s]:>16.3f}" for s in SOURCES)
              + f"{d:>+13.3f}")
    for s in SOURCES:
        print(f"   n({s}) = {len([r for r in ok if r['source']==s])}")

    print(f"\nE2 -- pairwise Spearman among the four instruments")
    pairs = [("K", "pca"), ("K", "manson"), ("K", "king"),
             ("pca", "manson"), ("pca", "king"), ("manson", "king"),
             ("K", "R")]
    print(f"   {'pair':<24}" + "".join(f"{s:>16}" for s in SOURCES))
    e2 = {}
    for a, b in pairs:
        vals = {}
        for s in SOURCES:
            sub = [r for r in ok if r["source"] == s]
            vals[s] = spearman([r[a] for r in sub], [r[b] for r in sub])
        e2[f"{a} vs {b}"] = vals
        tag = f"{a} vs {b}" + ("   [positive control]" if b == "R" else "")
        print(f"   {tag:<24}" + "".join(f"{vals[s]:>16.3f}" for s in SOURCES))

    print(f"\n{'='*80}\nDO THE CONCLUSIONS SURVIVE THE CORPUS CHANGE?\n{'='*80}")
    wt = SOURCES[1]
    intrinsic = max(abs(out["intrinsic K"][wt]), abs(out["intrinsic R"][wt]))
    proxy = max(abs(out[n][wt]) for n in
                ("local-PCA (Mabrok)", "Frenet U^T U (Manson)", "Euclid angle (King)"))
    print(f"   RQ3a on {wt}: best intrinsic |rho| = {intrinsic:.3f}, "
          f"best proxy |rho| = {proxy:.3f}   "
          f"-> intrinsic {'WINS' if intrinsic > proxy else 'DOES NOT WIN'} "
          f"({intrinsic/proxy if proxy else float('inf'):.1f}x)")
    cross = max(abs(v[wt]) for k, v in e2.items() if not k.endswith("vs R"))
    print(f"   E2 on {wt}: largest cross-instrument |rho| = {cross:.3f}   "
          f"positive control K vs R = {e2['K vs R'][wt]:+.3f}")
    print(f"   -> {'no instrument tracks any other' if cross < 0.4 else 'SOME PAIR NOW AGREES -- investigate'}")

    (OUT / "corpus_compare.json").write_text(json.dumps(
        dict(rq3a=out, e2=e2, n=len(ok), rows=rows), indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'corpus_compare.json'}")


if __name__ == "__main__":
    main()
