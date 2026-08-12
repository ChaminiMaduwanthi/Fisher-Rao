"""
RQ3b -- does the geometry localise the layer at which lexical ambiguity resolves?

For each polysemy minimal pair the SAME surface word appears in two contexts that
fix opposite senses.  Early in the network the two hidden states should be nearly
identical (the model has only seen the same token); as context is integrated they
must separate.  The question is where, and how sharply -- and whether the
Fisher-Rao metric localises it differently from the flat alternatives.

The Fisher-Rao distance between two categorical distributions is CLOSED FORM,

    d_FR(p, q) = 2 arccos( sum_i sqrt(p_i q_i) )

so no geodesic integration is needed: the predictive distributions are points on
the simplex and this is the exact geodesic distance there (radius-2 sphere
convention, matching validate_curvature.py).  That makes this the cheapest
genuinely-Fisher measurement in the whole project -- no metric assembly, no
autodiff, one forward pass per state.

Three instruments, same states, for comparison:
    d_FR    exact Fisher-Rao geodesic distance between predictive distributions
    d_UU    ||h_A - h_B|| under Manson's constant metric U^T U
    d_euc   plain Euclidean ||h_A - h_B||

Usage:  python run_polysemy.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fisherrao import LM, manson_metric
from fisherrao import corpus
from fisherrao.corpus import POLYSEMY

OUT = pathlib.Path("results/polysemy")


def find_token(lm, toks: list[str], word: str) -> int | None:
    """Index of the token realising `word` (byte-level BPE marks a leading space)."""
    w = word.lower()
    for i, t in enumerate(toks):
        if lm.tok.convert_tokens_to_string([t]).strip().lower() == w:
            return i
    for i, t in enumerate(toks):                      # fallback: prefix match
        if t.replace("Ġ", "").replace("▁", "").lower().startswith(w[:4]):
            return i
    return None


def fisher_rao_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    """2 arccos( sum sqrt(p q) ) -- exact geodesic distance on the simplex."""
    bc = (p.sqrt() * q.sqrt()).sum().clamp(-1.0, 1.0)
    return float(2.0 * torch.arccos(bc))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lm = LM()
    G_man = manson_metric(lm.U)
    L = lm.n_layers
    layers = list(range(L + 1))

    print(f"{lm.model_id}: {L} layers\n")
    print(f"{len(POLYSEMY)} minimal pairs; separation measured at the ambiguous "
          f"token in each context.\n")

    rows, skipped = [], []
    for word, sent_a, sent_b in POLYSEMY:
        Ha, ta = lm.residual_stream(sent_a)
        Hb, tb = lm.residual_stream(sent_b)
        ia, ib = find_token(lm, ta, word), find_token(lm, tb, word)
        if ia is None or ib is None:
            skipped.append(word)
            continue
        per_layer = []
        for l in layers:
            ha, hb = Ha[l, ia], Hb[l, ib]
            pa = lm.next_token_probs(ha)
            pb = lm.next_token_probs(hb)
            diff = ha - hb
            per_layer.append(dict(
                layer=l,
                d_fr=fisher_rao_distance(pa, pb),
                d_uu=float(torch.sqrt((diff @ G_man @ diff).clamp_min(0))),
                d_euc=float(torch.linalg.vector_norm(diff)),
            ))
        final = per_layer[-1]["d_fr"]
        # GUARD.  If the two states are identical the pair is vacuous -- almost
        # always because the disambiguating context follows the target word, which
        # causal attention cannot see.  See the warning in corpus.POLYSEMY.
        if final < 1e-9:
            skipped.append(f"{word} (zero separation -- context after target?)")
            continue
        rows.append(dict(word=word, pos_a=ia, pos_b=ib, layers=per_layer))
        print(f"  {word:<8} at token {ia:>2}/{ib:<2}  "
              f"d_FR: L0={per_layer[0]['d_fr']:.4f} -> L{L}={final:.4f}")

    if skipped:
        print(f"\n  ⚠ skipped: {skipped}")
    if not rows:
        print("\nNO PAIRS ANALYSED.  If all pairs had zero separation, the corpus")
        print("places the disambiguating context AFTER the ambiguous word; causal")
        print("attention makes that unobservable.  Rewrite so context comes first.")
        return

    # ---- where does separation grow fastest? ------------------------------
    def profile(key):
        M = torch.tensor([[pl[key] for pl in r["layers"]] for r in rows],
                         dtype=torch.float64)
        M = M / M[:, -1:].clamp_min(1e-30)            # normalise per pair
        return M

    print(f"\n{'='*70}\nNORMALISED SEPARATION (each pair scaled by its final-layer value)\n{'='*70}")
    print(f"{'layer':>6} {'d_FR':>10} {'d_UU':>10} {'d_euc':>10}")
    P = {k: profile(k) for k in ("d_fr", "d_uu", "d_euc")}
    for j, l in enumerate(layers):
        print(f"{l:>6} " + "".join(f"{float(P[k][:, j].mean()):>10.4f}"
                                   for k in ("d_fr", "d_uu", "d_euc")))

    print(f"\n{'='*70}\nDISAMBIGUATION LAYER  (steepest rise in normalised separation)\n{'='*70}")
    print(f"{'instrument':<10} {'mean layer':>11} {'std':>7} {'median':>8}")
    summary = {}
    for k in ("d_fr", "d_uu", "d_euc"):
        M = P[k]
        step = M[:, 1:] - M[:, :-1]
        arg = step.argmax(dim=1) + 1                  # layer of steepest increase
        a = arg.to(torch.float64)
        print(f"{k:<10} {float(a.mean()):>11.2f} {float(a.std()):>7.2f} "
              f"{float(a.median()):>8.1f}")
        summary[k] = dict(mean=float(a.mean()), std=float(a.std()),
                          median=float(a.median()),
                          per_pair={rows[i]["word"]: int(arg[i]) for i in range(len(rows))})
    print("\nper-pair steepest-rise layer under d_FR:")
    for w, l in summary["d_fr"]["per_pair"].items():
        print(f"    {w:<8} layer {l}")

    # half-separation depth: the layer at which each pair reaches 50% of final
    print(f"\n{'='*70}\nHALF-SEPARATION DEPTH  (first layer reaching 50% of final)\n{'='*70}")
    print(f"{'instrument':<10} {'mean':>7} {'median':>8}")
    for k in ("d_fr", "d_uu", "d_euc"):
        M = P[k]
        half = [(int((M[i] >= 0.5).nonzero()[0]) if (M[i] >= 0.5).any() else len(layers) - 1)
                for i in range(M.shape[0])]
        h = torch.tensor(half, dtype=torch.float64)
        print(f"{k:<10} {float(h.mean()):>7.2f} {float(h.median()):>8.1f}")
        summary[k]["half_depth_mean"] = float(h.mean())

    # ---- is the gap real, and is it carried by a subgroup? ----------------
    # At n=10 this experiment could report a point estimate and nothing else.
    # At n=64 the gap gets an interval, and the two obvious subgroup worries --
    # heteronyms, and one sense domain doing all the work -- get tested rather
    # than acknowledged in a limitations paragraph.
    half = {}
    for k in ("d_fr", "d_uu", "d_euc"):
        M = P[k]
        half[k] = torch.tensor(
            [(int((M[i] >= 0.5).nonzero()[0]) if (M[i] >= 0.5).any() else len(layers) - 1)
             for i in range(M.shape[0])], dtype=torch.float64)

    gap = half["d_uu"] - half["d_fr"]                 # paired, same pairs
    gen = torch.Generator().manual_seed(0)
    boot = torch.stack([gap[torch.randint(len(gap), (len(gap),), generator=gen)].mean()
                        for _ in range(10000)])
    lo, hi = (float(torch.quantile(boot, torch.tensor(q, dtype=torch.float64)))
              for q in (0.025, 0.975))
    n_pos = int((gap > 0).sum())
    print(f"\n{'='*70}\nIS THE GAP REAL?  paired, per-pair, d_UU minus d_FR\n{'='*70}")
    print(f"  mean gap {float(gap.mean()):.2f} layers   "
          f"95% bootstrap CI [{lo:.2f}, {hi:.2f}]   n = {len(gap)}")
    print(f"  d_FR is earlier in {n_pos}/{len(gap)} pairs "
          f"({100*n_pos/len(gap):.0f}%)   [50% = no effect]")

    print(f"\n{'-'*70}\nSUBGROUPS -- is the gap carried by one kind of word?\n{'-'*70}")
    words = [r["word"] for r in rows]
    het = [i for i, w in enumerate(words) if w in corpus.HETERONYMS]
    rest = [i for i, w in enumerate(words) if w not in corpus.HETERONYMS]
    print(f"  {'heteronyms (bass/bow/lead)':<34} n={len(het):>2}  "
          f"gap {float(gap[het].mean()):>6.2f}")
    print(f"  {'everything else':<34} n={len(rest):>2}  "
          f"gap {float(gap[rest].mean()):>6.2f}")
    by_dom: dict[str, list[int]] = {}
    for i, w in enumerate(words):
        for s in corpus.POLYSEMY_SENSES.get(w, ()):
            by_dom.setdefault(s, []).append(i)
    for dom, ix in sorted(by_dom.items(), key=lambda kv: -len(kv[1])):
        print(f"  {dom:<34} n={len(ix):>2}  gap {float(gap[ix].mean()):>6.2f}"
              f"   d_FR half-depth {float(half['d_fr'][ix].mean()):>5.2f}")
    summary["gap"] = dict(mean=float(gap.mean()), ci=[lo, hi], n=len(gap),
                          n_fr_earlier=n_pos,
                          heteronym_gap=float(gap[het].mean()) if het else None,
                          rest_gap=float(gap[rest].mean()),
                          by_domain={d: float(gap[ix].mean()) for d, ix in by_dom.items()})

    (OUT / "polysemy.json").write_text(
        json.dumps(dict(pairs=rows, summary=summary, skipped=skipped), indent=2),
        encoding="utf-8")
    print(f"\nwrote {OUT / 'polysemy.json'}")
    print("\nNOTE: d_FR is the exact Fisher-Rao distance between PREDICTIVE")
    print("distributions -- a different object from the intrinsic curvature of the")
    print("pullback metric.  It is the right instrument for 'how far apart are the")
    print("model's two beliefs', and it needs no k, no frame, no cutoff.")


if __name__ == "__main__":
    main()
