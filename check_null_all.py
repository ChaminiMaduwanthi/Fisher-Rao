"""
The null-direction falsification test, measured on EVERY model the paper claims.

WHY THIS SCRIPT EXISTS
----------------------
`check_architectures.py` runs the full four-rung check but only on three models,
and 09-e5-log.md S1 tabulates four while asserting "PASS on all six".  Table I
of the paper lists six.  A table in a paper must contain numbers that were
actually measured on the rows it names, so this script measures all six on one
code path and prints the table.

THE TEST
--------
`G(h)` is singular by construction: the normalisation layer annihilates one
direction (RMSNorm) or two (LayerNorm).  If the metric is the model's own, then
stepping along a direction the metric calls null must not move the model's
prediction.  The step is `eps * ||h||` with `eps = 1`, which DOUBLES the hidden
state -- a perturbation far outside the linear regime, so a pass is a property
of the architecture rather than of a small displacement.

Reported per model:
    null dirs found (expected)   1 for RMSNorm, 2 for LayerNorm
    KL along null                KL(p(h) || p(h + ||h|| n))       must be ~0
    KL along random              same step size, random direction  for contrast
    ratio                        the falsification statistic

A sign bug in the null basis previously made eps=1 land on the ORIGIN, which a
LayerNorm model answers with its bias -- reading as a spectacular failure that
was nothing of the kind (09-e5-log.md S1.1).  The basis is canonically signed
now, and the check below asserts <n0, h_hat> > 0 so that failure mode cannot
return silently.

Usage:  python check_null_all.py [model_id ...]
"""

from __future__ import annotations

import json
import pathlib
import sys
import warnings

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

from fisherrao import LM

MODELS = [
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "JackFram/llama-160m",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "gpt2",
    "EleutherAI/gpt-neo-125m",
]
PROMPT = "The capital of France is"
SHORT = {
    "HuggingFaceTB/SmolLM2-135M-Instruct": "SmolLM2-135M",
    "JackFram/llama-160m": "LLaMA-160M",
    "EleutherAI/pythia-70m": "Pythia-70M",
    "EleutherAI/pythia-160m": "Pythia-160M",
    "gpt2": "GPT-2",
    "EleutherAI/gpt-neo-125m": "GPT-Neo-125M",
}


def kl(p: torch.Tensor, q: torch.Tensor) -> float:
    m = p > 0
    return float((p[m] * (p[m].log() - q[m].log())).sum())


def one(model_id: str, gen: torch.Generator) -> dict:
    lm = LM(model_id)
    expect = 1 if "RMS" in lm.norm_kind else 2
    H, _ = lm.residual_stream(PROMPT)
    h = H[-1, -1].double()

    N = lm.null_directions(h)
    found = N.shape[1]

    # the sign guard: n0 must point ALONG h, not against it
    hhat = h / h.norm()
    align = float(N[:, 0] @ hhat)

    p0 = lm.next_token_probs(h)
    step = float(h.norm())                       # eps = 1 doubles the state

    kl_null = kl(p0, lm.next_token_probs(h + step * N[:, 0]))
    r = torch.randn(lm.d, generator=gen, dtype=torch.float64)
    r = r / r.norm()
    kl_rand = kl(p0, lm.next_token_probs(h + step * r))

    return dict(model=model_id, short=SHORT.get(model_id, model_id),
                norm=lm.norm_kind, d=lm.d, layers=lm.n_layers, N=lm.N,
                expect=expect, found=found, align=align,
                kl_null=abs(kl_null), kl_rand=abs(kl_rand),
                ratio=abs(kl_null) / abs(kl_rand) if kl_rand else float("nan"))


def main() -> int:
    ids = sys.argv[1:] or MODELS
    gen = torch.Generator().manual_seed(0)
    rows, ok = [], True

    for mid in ids:
        try:
            r = one(mid, gen)
        except Exception as e:                                   # noqa: BLE001
            print(f"  {mid}: FAILED TO LOAD -- {type(e).__name__}: {e}")
            ok = False
            continue
        good = (r["found"] == r["expect"] and r["align"] > 0.99
                and r["kl_null"] < 1e-8)
        r["pass"] = good
        ok &= good
        rows.append(r)

    print(f"\n{'='*96}\nNULL-DIRECTION FALSIFICATION TEST -- every model the paper names\n{'='*96}")
    print(f"{'model':<15}{'norm':<12}{'d':>5}{'L':>4}{'null (exp)':>12}"
          f"{'KL null':>12}{'KL random':>11}{'ratio':>10}   verdict")
    for r in rows:
        print(f"{r['short']:<15}{r['norm']:<12}{r['d']:>5}{r['layers']:>4}"
              f"{str(r['found'])+' ('+str(r['expect'])+')':>12}"
              f"{r['kl_null']:>12.2e}{r['kl_rand']:>11.3f}{r['ratio']:>10.1e}"
              f"   {'PASS' if r['pass'] else '**FAIL**'}")

    print(f"\n  step size: eps = 1, i.e. ||dh|| = ||h|| -- the perturbation DOUBLES the state")
    print(f"  sign guard <n0, h_hat>: {min(r['align'] for r in rows):.6f} .. "
          f"{max(r['align'] for r in rows):.6f}   (must be > 0.99)")
    print(f"\n{len(rows)}/{len(ids)} models measured; verdict: {'PASS' if ok else 'FAIL'}")

    out = pathlib.Path("results/nulltest")
    out.mkdir(parents=True, exist_ok=True)
    (out / "nulltest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"saved -> {out/'nulltest.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
