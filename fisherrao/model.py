"""
Model loading, residual-stream extraction, and the logit lens.

Stage 0.3.  Everything here is deliberately explicit -- no hidden reshapes,
no silent dtype changes -- because every downstream geometric quantity
inherits whatever this module gets wrong.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


class LM:
    """A decoder-only LM exposing its residual stream and its unembedding map.

    Attributes
    ----------
    U : (N, d) float64      unembedding matrix
    g : (d,)  float64       final-norm gain, or None if the norm has no weight
    eps : float             final-norm epsilon
    """

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
        self.model.eval().to(device)

        cfg = self.model.config
        self.d = cfg.hidden_size
        self.N = cfg.vocab_size
        self.n_layers = cfg.num_hidden_layers
        self.tied = bool(getattr(cfg, "tie_word_embeddings", False))

        # ---- unembedding -------------------------------------------------
        head = self.model.get_output_embeddings()
        if head is None:
            raise ValueError(f"{model_id} has no output embedding / lm_head")
        self.U = head.weight.detach().to(torch.float64)          # (N, d)
        self.bias = None if head.bias is None else head.bias.detach().to(torch.float64)

        # ---- the final norm sitting between residual stream and lm_head ----
        # This is NOT optional.  The predictive map is
        #     h -> norm(h) -> U -> softmax
        # so the Jacobian of the norm belongs in every pullback metric.
        # Omitting it is a real error, not an approximation; see metrics.py.
        self.norm = self._find_final_norm()
        w = getattr(self.norm, "weight", None)
        self.g = None if w is None else w.detach().to(torch.float64)
        self.eps = float(
            getattr(self.norm, "variance_epsilon", None)
            or getattr(self.norm, "eps", 1e-5)
        )
        self.norm_kind = type(self.norm).__name__

    # ------------------------------------------------------------------
    # Architecture plumbing.
    #
    # The three architectures this project needs disagree on every name:
    #
    #     Llama / SmolLM2   .model        .layers   .norm               RMSNorm
    #     GPT-2             .transformer  .h        .ln_f               LayerNorm
    #     GPT-NeoX / Pythia .gpt_neox     .layers   .final_layer_norm   LayerNorm
    #
    # Hardcoding Llama's names silently blocked the cross-architecture
    # comparison Stage 4 needs (it read as "could not locate the final norm").
    # Go through HF's own base_model_prefix and then try each known spelling.
    # ------------------------------------------------------------------
    def _base(self):
        return getattr(self.model, self.model.base_model_prefix, self.model)

    def _find(self, kind: str, names: tuple[str, ...]):
        base = self._base()
        for attr in names:
            mod = getattr(base, attr, None)
            if mod is not None:
                return mod
        raise ValueError(
            f"could not locate the {kind} on {self.model_id} "
            f"(base={type(base).__name__}, tried {names}); "
            f"add its attribute name to LM._find")

    def _find_final_norm(self):
        return self._find("final norm",
                          ("norm", "final_layernorm", "final_layer_norm", "ln_f"))

    def _find_blocks(self):
        return self._find("transformer blocks", ("layers", "h", "blocks"))

    # ------------------------------------------------------------------
    def residual_stream(self, text: str) -> tuple[torch.Tensor, list[str]]:
        """Return (L+1, T, d) float64 RAW residual stream and the token strings.

        Index 0 is the embedding output; index l is the output of block l-1.
        Every entry is pre-final-norm, so `logits()` applies the norm to all of
        them uniformly.

        Why hooks rather than `output_hidden_states=True`
        -------------------------------------------------
        HF's LlamaModel appends `self.norm(hidden_states)` as the LAST entry of
        `hidden_states`.  So `out.hidden_states[-1]` is ALREADY normalised while
        every other entry is raw.  Applying the norm again double-normalises,
        which silently corrupts the last layer of every downstream quantity.
        The logit-lens sanity check in run_stage0.py caught this at rel.err
        2.07 -- an inconsistency of that size does not announce itself in a
        curvature plot.  Capturing block outputs directly avoids depending on
        that convention at all, and generalises across architectures.

        Entry 0 is captured by a forward PRE-hook on block 0, i.e. it is
        literally whatever enters the first block.  A hook on the token
        embedding would be wrong on GPT-2, where the residual stream at that
        point is wte + wpe (+ dropout) rather than wte alone -- the positional
        term would go missing and layer 0 would be quietly incorrect.
        """
        enc = self.tok(text, return_tensors="pt").to(self.device)
        blocks = self._find_blocks()

        captured: list[torch.Tensor] = []
        emb_out: list[torch.Tensor] = []

        def hook(_mod, _inp, out):
            captured.append((out[0] if isinstance(out, tuple) else out).detach())

        def pre_hook(_mod, inp):
            emb_out.append((inp[0] if isinstance(inp, tuple) else inp).detach())

        handles = [b.register_forward_hook(hook) for b in blocks]
        handles.append(blocks[0].register_forward_pre_hook(pre_hook))
        try:
            with torch.no_grad():
                self.model(**enc)
        finally:
            for hd in handles:
                hd.remove()

        H = torch.stack(emb_out + captured).squeeze(1).to(torch.float64)
        toks = self.tok.convert_ids_to_tokens(enc["input_ids"][0])
        return H, toks

    # ------------------------------------------------------------------
    def apply_norm(self, h: torch.Tensor) -> torch.Tensor:
        """Apply the final norm in float64, matching the model's own formula.

            RMSNorm    y = g * h / sqrt(mean(h^2) + eps)
            LayerNorm  y = g * (h - mu) / sqrt(var + eps) + b

        The bias is added AFTER the gain, not before.  An earlier version added
        it inside the normalised term and then multiplied the whole thing by g,
        computing g*z + g*b instead of g*z + b.  That is exactly zero error on
        SmolLM2 (RMSNorm, no bias) and a silent, unnoticeable-looking error on
        every LayerNorm model -- i.e. on GPT-2 and Pythia, the two models the
        cross-architecture comparison needs.  Verified against the model's own
        norm module by `norm_check()`.
        """
        if "RMS" in self.norm_kind:
            r = torch.sqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
            out = h / r
            return out if self.g is None else out * self.g

        mu = h.mean(-1, keepdim=True)
        var = h.var(-1, unbiased=False, keepdim=True)
        out = (h - mu) / torch.sqrt(var + self.eps)
        if self.g is not None:
            out = out * self.g
        b = getattr(self.norm, "bias", None)
        if b is not None:
            out = out + b.detach().to(torch.float64)
        return out

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """Logit lens: read any layer's hidden state through the output head."""
        z = self.apply_norm(h) @ self.U.T
        return z if self.bias is None else z + self.bias

    def next_token_probs(self, h: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(h), dim=-1)

    def norm_jacobian(self, h: torch.Tensor) -> torch.Tensor:
        """Jacobian A = d(norm(h))/dh, a (d, d) matrix.

        RMSNorm, gain g, r = sqrt(mean(h^2) + eps):

            norm(h)_m = g_m h_m / r
            A = diag(g) * (1/r) * ( I - h h^T / (r^2 d) )

        ONE exact null direction: h itself.  This is the whole basis of the
        quotient formulation -- see the metrics.py header.

        LayerNorm, gain g, bias b, s = sqrt(var + eps), z = (h - mu)/s:

            norm(h)_m = g_m z_m + b_m
            A = diag(g) * (1/s) * ( I - 11^T/d - z z^T/d )

        TWO null directions, and this is why the RMSNorm result must not be
        assumed to transfer:

            1_d   from the mean subtraction   ( (I - 11^T/d) 1 = 0, and z.1 = 0 )
            h     the radial one, as before   ( z^T z = d*var/(var+eps) ~ d, so
                                                the bracket kills z, hence h - mu )

        So on a LayerNorm model the metric is rank <= d-2, the quotient is by a
        2-plane rather than a line, and any code that removes only the radial
        direction leaves an exact null direction in the retained subspace --
        which silently poisons the Christoffel symbols.  Use null_directions()
        to get the right projector for whichever norm the model has.

        The bracket is symmetric; diag(g) is not, so A is symmetric only when g
        is constant.  Verified against autograd by norm_check().
        """
        d = h.shape[-1]
        I = torch.eye(d, dtype=h.dtype)
        if "RMS" in self.norm_kind:
            r = torch.sqrt(h.pow(2).mean() + self.eps)
            A = (I - torch.outer(h, h) / (r**2 * d)) / r
        else:
            mu = h.mean()
            s = torch.sqrt(h.var(unbiased=False) + self.eps)
            z = (h - mu) / s
            ones = torch.ones(d, dtype=h.dtype)
            A = (I - torch.outer(ones, ones) / d - torch.outer(z, z) / d) / s
        return A if self.g is None else self.g[:, None] * A

    def null_directions(self, h: torch.Tensor) -> torch.Tensor:
        """Orthonormal (d, m) basis of the EXACT null space of the norm Jacobian.

        m = 1 for RMSNorm (radial), m = 2 for LayerNorm (radial + all-ones).
        Project these out before selecting a curvature subspace; see
        norm_jacobian() for why.

            !!  THE SIGN OF EACH COLUMN IS FIXED DELIBERATELY  !!

        `torch.linalg.qr` is free to return -q for any column, and it does:
        measured, the raw factor gives -hhat at 12/12 layers of gpt2, 10/12 of
        pythia-160m and 19/30 of SmolLM2 -- data- and LAPACK-dependent, with no
        pattern.

        For a PROJECTOR that is harmless (N N^T is sign-blind), which is why
        curvature was never affected.  For anything that STEPS along the
        returned vector it is not:

            h + eps*||h||*(-hhat)  =  (1 - eps)*h

        so eps=1 lands on the ORIGIN rather than perturbing along a null
        direction, and eps>1 lands on a NEGATIVE multiple of h -- which is not
        null at all, since norm(-ch) = -norm(ch).  E5's falsification test read
        KL = 4.15 against a random direction's 3.86 because of exactly this, and
        the median hid it on the model where the signs happened to come out
        mixed.

        Columns are therefore signed so that column 0 points along +h and, on a
        LayerNorm model, column 1 along +1.
        """
        v = [h / torch.linalg.vector_norm(h)]
        if "RMS" not in self.norm_kind:
            v.append(torch.ones(h.shape[-1], dtype=h.dtype))
        ref = torch.stack(v, dim=1)
        Q, _ = torch.linalg.qr(ref)
        sign = torch.sign((Q * ref).sum(0))
        sign[sign == 0] = 1.0
        return (Q * sign).contiguous()

    def norm_check(self, h: torch.Tensor) -> dict:
        """apply_norm vs the model's own module, and norm_jacobian vs autograd.

        Both are silent-failure sites: a wrong bias placement or a missing term
        in A produces plausible numbers everywhere downstream.  Run this on any
        new architecture BEFORE trusting a single curvature value.
        """
        with torch.no_grad():
            ref = self.norm(h.to(torch.float32)).to(torch.float64)
        mine = self.apply_norm(h)
        A_auto = torch.autograd.functional.jacobian(
            self.apply_norm, h.clone().requires_grad_(True))
        A_mine = self.norm_jacobian(h)
        N = self.null_directions(h)
        return dict(
            norm_kind=self.norm_kind,
            n_null=N.shape[1],
            forward_rel_err=float(torch.linalg.vector_norm(mine - ref)
                                  / torch.linalg.vector_norm(ref)),
            jacobian_rel_err=float(torch.linalg.matrix_norm(A_mine - A_auto)
                                   / torch.linalg.matrix_norm(A_auto)),
            null_residual=float(torch.linalg.matrix_norm(A_mine @ N)
                                / torch.linalg.matrix_norm(A_mine)),
        )

    # ------------------------------------------------------------------
    def summary(self) -> str:
        return (
            f"{self.model_id}\n"
            f"  d = {self.d}, N = {self.N}, layers = {self.n_layers}\n"
            f"  tied embeddings : {self.tied}\n"
            f"  final norm      : {self.norm_kind} (eps={self.eps:g}, "
            f"gain={'yes' if self.g is not None else 'no'})\n"
            f"  lm_head bias    : {'yes' if self.bias is not None else 'no'}"
        )
