"""The blessed logit-lens primitive — one door for hidden-state -> logits.

The recurring trap: `transformers`' ``output_hidden_states`` returns the residual
stream PRE final-norm for intermediate layers, but the LAST entry is already POST
final-norm. The logit lens is ``lm_head(final_norm(residual))``; applying the norm to
the already-post-norm last entry double-norms it and *silently* produces garbage — a
valid-shaped rank curve with a wrong argmax. It has bitten the homing study and the
"between" experiment, in both cases while someone was focused on the new thing.

The fix is not to remember the convention harder. It is to route every readout
through this one function, which (a) handles the norm stage correctly and (b) checks
itself against the one place ground truth exists — the model's real final-layer
output — and RAISES on mismatch. Silent-wrong becomes loud-wrong. Do not hand-roll a
lens; call this.
"""
from __future__ import annotations


def lens_logits(out, model, pos: int = -1, check: bool = True):
    """Per-layer logit lens at token ``pos`` for a completed forward pass.

    ``out`` must be ``model(..., output_hidden_states=True)``. Returns a tensor of
    shape ``[n_layers, vocab]``. Intermediate layers use ``lm_head(final_norm(h))``;
    the final layer uses the model's TRUE logits, because its ``hidden_states`` entry
    is already post-norm and norming it again is the trap. With ``check=True``
    (default) it verifies the final-layer lens top-1 equals the model's real
    prediction and raises on mismatch.
    """
    import torch

    hs = out.hidden_states
    if hs is None:
        raise ValueError("lens_logits needs a forward with output_hidden_states=True")
    inner = getattr(model, "model", model)
    norm = getattr(inner, "norm", None)
    lm = getattr(model, "lm_head", None)
    if norm is None or lm is None:
        raise ValueError("lens_logits expects *.model.norm and *.lm_head (llama-family)")

    n = len(hs) - 1                       # n decoder layers; hs[0] = embeddings

    if check:
        # Verify the hidden states actually reproduce the model's own output under a
        # KNOWN norm stage — otherwise the readout is untrustworthy. This is a real
        # ground-truth check, not a tautology: it fires if hs[-1] is neither the
        # expected post-norm state nor a pre-norm one (wrong model, mismatched
        # norm/lm_head, corrupted or foreign hidden states, a library-version change).
        true_id = int(out.logits[0, pos].argmax())
        last = hs[-1][0, pos, :]
        post_ok = int(lm(last).argmax()) == true_id            # hs[-1] already normed
        pre_ok = int(lm(norm(last)).argmax()) == true_id       # hs[-1] still pre-norm
        if not (post_ok or pre_ok):
            raise RuntimeError(
                "lens_logits: hidden states do not reproduce the model's own output "
                "under either norm stage — the readout is untrustworthy (wrong model, "
                "mismatched norm/lm_head, or corrupted/foreign hidden states).")

    rows = []
    for l in range(n):
        if l < n - 1:                     # pre-norm residual -> apply norm exactly once
            rows.append(lm(norm(hs[l + 1][0, pos, :])))
        else:                             # hs[-1] is post-norm -> use the true logits
            rows.append(out.logits[0, pos, :])
    return torch.stack([r.detach() for r in rows])
