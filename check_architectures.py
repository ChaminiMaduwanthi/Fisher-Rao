"""
Does the pipeline actually work on GPT-2 and Pythia, or only on SmolLM2?

Stage 4's exit criteria need layer-wise profiles for >= 3 models, and RQ3a as
literally posed needs GPT-2 / Pythia.  Both were blocked on TLS
(05-stage0-log.md S0, fixed in fisherrao/net.py).  With downloads working, the
next question is whether anything downstream survives the architecture change --
and the honest answer before this script was "unknown", because every line of
model.py had only ever run against one RMSNorm Llama.

Four checks, in dependency order.  A failure at any rung makes the rungs below
it meaningless, so they are reported together and none is skipped on failure.

    1  LOAD          base model, blocks and final norm located at all
    2  FORWARD       our float64 apply_norm reproduces the model's own norm
    3  JACOBIAN      norm_jacobian matches autograd, and annihilates exactly the
                     null directions the algebra predicts (1 for RMSNorm, 2 for
                     LayerNorm -- radial AND all-ones)
    4  LOGIT LENS    logits(H[-1]) reproduces the model's real output logits.
                     This is the end-to-end check: it fails if the residual
                     stream is captured at the wrong point, if the final norm is
                     applied twice, or if the unembedding is wrong.  It is the
                     check that caught the double-normalisation bug in Stage 0.

Usage:  python check_architectures.py [model_id ...]
"""

from __future__ import annotations

import sys
import warnings

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

from fisherrao import LM
from fisherrao.metrics import radial_null_check

DEFAULT = ["HuggingFaceTB/SmolLM2-135M-Instruct", "gpt2", "EleutherAI/pythia-160m"]
PROMPT = "The capital of France is"


def check(model_id: str) -> dict:
    out = {"model": model_id}
    lm = LM(model_id)
    out["norm_kind"] = lm.norm_kind
    out["d"], out["N"], out["layers"] = lm.d, lm.N, lm.n_layers
    out["bias"] = lm.bias is not None
    out["gain"] = lm.g is not None

    H, toks = lm.residual_stream(PROMPT)
    out["H_shape"] = tuple(H.shape)
    out["n_blocks_ok"] = H.shape[0] == lm.n_layers + 1

    h = H[-1, -1]
    nc = lm.norm_check(h)
    out.update(forward_err=nc["forward_rel_err"],
               jac_err=nc["jacobian_rel_err"],
               n_null=nc["n_null"],
               null_resid=nc["null_residual"])

    # ---- 4. logit lens against the model's own forward pass ----------------
    enc = lm.tok(PROMPT, return_tensors="pt")
    with torch.no_grad():
        ref = lm.model(**enc).logits[0, -1].to(torch.float64)
    mine = lm.logits(h)
    out["logit_lens_err"] = float(torch.linalg.vector_norm(mine - ref)
                                  / torch.linalg.vector_norm(ref))
    out["argmax_match"] = int(mine.argmax()) == int(ref.argmax())
    out["top_token"] = lm.tok.decode([int(ref.argmax())])

    # ---- the RQ4 null-space falsification, per architecture ---------------
    rn = radial_null_check(lm, H[lm.n_layers // 2, -1])
    out["kl_ratio"] = rn["kl_radial"] / rn["kl_random"] if rn["kl_random"] else float("nan")
    return out


def main():
    ids = sys.argv[1:] or DEFAULT
    rows = []
    for mid in ids:
        try:
            rows.append(check(mid))
        except Exception as exc:                                  # noqa: BLE001
            rows.append({"model": mid, "error": f"{type(exc).__name__}: {exc}"})

    print(f"\n{'='*78}\nARCHITECTURE COMPATIBILITY\n{'='*78}")
    for r in rows:
        print(f"\n{r['model']}")
        if "error" in r:
            print(f"   FAILED: {r['error']}")
            continue
        print(f"   1 LOAD       {r['norm_kind']:<16} d={r['d']} N={r['N']} "
              f"layers={r['layers']}  gain={r['gain']} bias={r['bias']}")
        print(f"                residual stream {r['H_shape']}  "
              f"block count {'OK' if r['n_blocks_ok'] else 'MISMATCH'}")
        print(f"   2 FORWARD    apply_norm vs the model's own norm : "
              f"{r['forward_err']:.2e}   {_v(r['forward_err'], 1e-6)}")
        print(f"   3 JACOBIAN   norm_jacobian vs autograd          : "
              f"{r['jac_err']:.2e}   {_v(r['jac_err'], 1e-10)}")
        print(f"                null directions = {r['n_null']}  "
              f"(expected {1 if 'RMS' in r['norm_kind'] else 2}), "
              f"||A N||/||A|| = {r['null_resid']:.2e}   {_v(r['null_resid'], 1e-6)}")
        print(f"   4 LOGIT LENS logits(H[-1]) vs real output logits: "
              f"{r['logit_lens_err']:.2e}   {_v(r['logit_lens_err'], 1e-5)}")
        print(f"                argmax {'matches' if r['argmax_match'] else 'DIFFERS'}"
              f"   top token = {r['top_token']!r}")
        print(f"   + RQ4        KL(radial)/KL(random) = {r['kl_ratio']:.2e}   "
              f"{_v(r['kl_ratio'], 1e-10)}")

    bad = [r for r in rows if "error" in r
           or r["jac_err"] > 1e-10 or r["logit_lens_err"] > 1e-5
           or r["forward_err"] > 1e-6]
    print(f"\n{'='*78}")
    print(f"{len(rows)-len(bad)}/{len(rows)} architectures fully pass"
          if not bad else
          f"FAILURES: {', '.join(r['model'] for r in bad)}")
    return 1 if bad else 0


def _v(x, tol):
    return "OK" if x == x and x < tol else "**FAIL**"


if __name__ == "__main__":
    sys.exit(main())
