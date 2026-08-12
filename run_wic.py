"""
RQ3b on a SENSE-ANNOTATED corpus -- WiC, with the control the hand-written set
never had.

07-stage4-log.md S3b measures where the model resolves lexical ambiguity, and it
is the strongest result in the project: a ~22-layer disagreement between the
Fisher-Rao metric and the raw flat ones.  It rests on 64 minimal pairs written
by the author.  Two things are missing from that design and WiC supplies both:

  1. LABELS.  WiC (SuperGLUE) gives 319 DIFFERENT-sense pairs and 319 SAME-sense
     pairs on the same target words, in the validation split alone.  The
     hand-written set has only different-sense pairs, so it has no control:
     nothing there rules out the possibility that ANY two occurrences of a word
     in different sentences separate the same way.  Same-sense pairs are exactly
     that control, and they are the point of using WiC.

  2. Sentences nobody in this project wrote.

WHAT CANNOT BE CARRIED OVER, and this is the interesting complication
-------------------------------------------------------------------
The hand-written corpus places the target word LAST, because under causal
attention a hidden state sees only what precedes it (corpus.POLYSEMY documents
the first version of the experiment being vacuous for exactly this reason).

WiC makes no such guarantee -- the target sits wherever it naturally falls.  So
some pairs have little or no disambiguating context BEFORE the target, and those
must separate weakly no matter how good the instrument is.  That is not a defect
to hide: `preceding context length` is recorded per pair, and the analysis is
reported against it.  If separation grows with preceding context and same-sense
pairs stay flat, the instrument is doing what it claims.

    d_FR    2 arccos(sum sqrt(p q))  -- exact geodesic distance on the simplex
    d_UU    ||h_A - h_B|| under Manson's constant G = U^T U
    d_euc   plain Euclidean

Usage:  python run_wic.py [n_per_class] [model_id]
"""

from __future__ import annotations

import json
import pathlib
import sys

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fisherrao import LM, DEFAULT_MODEL, manson_metric

OUT = pathlib.Path("results/wic")


def load_wic() -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("super_glue", "wic/validation-00000-of-00001.parquet",
                           repo_type="dataset")
    return pq.read_table(path).to_pylist()


def token_at_char(lm, text: str, start: int, end: int):
    """(token index, n preceding tokens) for the token covering [start, end).

    Uses the fast tokenizer's offset mapping rather than string matching -- the
    target word can appear more than once in a sentence, and WiC's offsets say
    WHICH occurrence is meant.
    """
    enc = lm.tok(text, return_offsets_mapping=True, return_tensors="pt")
    offs = enc["offset_mapping"][0].tolist()
    hit = None
    for i, (a, b) in enumerate(offs):
        if a == b:                       # special / empty token
            continue
        if a < end and b > start:        # overlaps the target span
            hit = i                      # keep the LAST overlapping token:
    if hit is None:                      # a multi-token word is decided at its
        return None, 0                   # final piece
    return hit, hit                      # index doubles as "tokens before it"


def fisher_rao(p, q):
    bc = (p.sqrt() * q.sqrt()).sum().clamp(-1.0, 1.0)
    return float(2.0 * torch.arccos(bc))


def half_depth(vals: list[float]) -> int:
    """First layer index reaching 50% of the final value."""
    final = vals[-1]
    if final <= 0:
        return len(vals) - 1
    for i, v in enumerate(vals):
        if v >= 0.5 * final:
            return i
    return len(vals) - 1


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    model_id = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL

    lm = LM(model_id)
    G_man = manson_metric(lm.U)
    print(lm.summary())

    data = load_wic()
    gen = torch.Generator().manual_seed(0)
    rows, skipped = [], 0
    for label in (0, 1):
        pool = [r for r in data if r["label"] == label]
        pick = torch.randperm(len(pool), generator=gen)[:n_per].tolist()
        for j in pick:
            r = pool[j]
            ia, ca = token_at_char(lm, r["sentence1"], r["start1"], r["end1"])
            ib, cb = token_at_char(lm, r["sentence2"], r["start2"], r["end2"])
            if ia is None or ib is None:
                skipped += 1
                continue
            Ha, _ = lm.residual_stream(r["sentence1"])
            Hb, _ = lm.residual_stream(r["sentence2"])
            if ia >= Ha.shape[1] or ib >= Hb.shape[1]:
                skipped += 1
                continue
            # SURFACE-FORM CONFOUND.  WiC targets are lemmas, so the two
            # sentences may carry different inflections ("stripe" / "stripes")
            # and therefore DIFFERENT TOKENS.  Two different tokens differ at
            # layer 0 with no context at all, which would be read as
            # "separation" by every instrument here.  Record it and split on it
            # rather than averaging over both cases.
            ta = int(lm.tok(r["sentence1"], return_tensors="pt")["input_ids"][0][ia])
            tb = int(lm.tok(r["sentence2"], return_tensors="pt")["input_ids"][0][ib])

            per = []
            for l in range(lm.n_layers + 1):
                ha, hb = Ha[l, ia], Hb[l, ib]
                pa = lm.next_token_probs(ha)
                pb = lm.next_token_probs(hb)
                d = ha - hb
                per.append(dict(
                    d_fr=fisher_rao(pa, pb),
                    d_uu=float(torch.sqrt((d @ G_man @ d).clamp_min(0))),
                    d_euc=float(torch.linalg.vector_norm(d))))
            rows.append(dict(word=r["word"], label=label, same_token=(ta == tb),
                             ctx=min(ca, cb), layers=per))
        print(f"   label={label}: {len([x for x in rows if x['label']==label])} pairs",
              flush=True)
    print(f"   skipped {skipped}")

    # Restrict everything below to pairs whose target is the SAME TOKEN in both
    # sentences.  Otherwise the comparison is partly about tokenisation.
    clean = [r for r in rows if r["same_token"]]
    print(f"\n   same-token pairs: {len(clean)}/{len(rows)} "
          f"({len(rows)-len(clean)} dropped as different surface forms)")

    print(f"\n{'='*78}\nWiC -- HALF-SEPARATION DEPTH, {lm.n_layers} layers"
          f"   (same-token pairs only, n={len(clean)})\n{'='*78}")
    print(f"   {'instrument':<12} {'DIFFERENT sense':>18} {'SAME sense (control)':>22}"
          f" {'gap':>7}")
    summary = {}
    for key in ("d_fr", "d_uu", "d_euc"):
        cells = {}
        for label in (0, 1):
            sub = [r for r in clean if r["label"] == label]
            hd = [half_depth([pl[key] for pl in r["layers"]]) for r in sub]
            cells[label] = (float(torch.tensor(hd, dtype=torch.float64).mean())
                            if hd else float("nan"))
        summary[key] = cells
        print(f"   {key:<12} {cells[0]:>18.2f} {cells[1]:>22.2f} "
              f"{cells[1]-cells[0]:>+7.2f}")
    for label, name in ((0, "DIFFERENT"), (1, "SAME")):
        print(f"   n({name}) = {len([r for r in clean if r['label']==label])}")

    print(f"\n{'='*78}\nTHE CONTROL -- do SAME-sense pairs separate as much?\n{'='*78}")
    print(f"   {'instrument':<12} {'final sep, DIFFERENT':>22} {'final sep, SAME':>18}"
          f" {'ratio':>8}")
    for key in ("d_fr", "d_uu", "d_euc"):
        fin = {}
        for label in (0, 1):
            sub = [r for r in clean if r["label"] == label]
            v = torch.tensor([r["layers"][-1][key] for r in sub], dtype=torch.float64)
            fin[label] = float(v.median()) if len(v) else float("nan")
        summary.setdefault("final", {})[key] = fin
        print(f"   {key:<12} {fin[0]:>22.4f} {fin[1]:>18.4f} "
              f"{fin[0]/fin[1] if fin[1] else float('nan'):>8.2f}x")

    print(f"\n{'='*78}\nDOES SEPARATION TRACK PRECEDING CONTEXT?\n{'='*78}")
    print("   Under causal attention a target with nothing before it cannot be")
    print("   disambiguated.  WiC does not control target position, so this is")
    print("   the check that the measurement is reading context at all.")
    print(f"   {'ctx tokens':<14} {'n':>5} {'d_FR final (diff)':>20} "
          f"{'d_FR final (same)':>20}")
    bins = [(0, 1), (1, 3), (3, 6), (6, 12), (12, 10**6)]
    for lo, hi in bins:
        cell = []
        for label in (0, 1):
            sub = [r for r in clean if r["label"] == label and lo <= r["ctx"] < hi]
            v = torch.tensor([r["layers"][-1]["d_fr"] for r in sub],
                             dtype=torch.float64)
            cell.append((len(sub), float(v.median()) if len(v) else float("nan")))
        tag = f"[{lo},{hi})" if hi < 10**6 else f"[{lo},inf)"
        print(f"   {tag:<14} {cell[0][0]+cell[1][0]:>5} {cell[0][1]:>20.4f} "
              f"{cell[1][1]:>20.4f}")

    (OUT / f"wic_{model_id.split('/')[-1]}.json").write_text(json.dumps(
        dict(model=model_id, n=len(rows), skipped=skipped, summary=summary,
             rows=rows), indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / f'wic_{model_id.split(chr(47))[-1]}.json'}")


if __name__ == "__main__":
    main()
