"""Necessity loss — answer-margin hinge between the with-latent and latent-ablated forward.

Keeps the model's L_answer + L_patch and ADDS a term that forces the answer to causally depend on the
latents Z: ablating Z (replacing the <lvr> embeddings with an in-distribution substitute) must make the
GOLD answer at least `margin` NATS harder to predict.

    necessity = mean_b max(0, margin - (NLL_ablated - NLL_with))

NLL is over the gold answer span. `NLL_with` is DETACHED (L_answer already optimizes the with-latent
answer; letting necessity also push it would double-count and let the model satisfy necessity by making
the *with-Z* answer worse). Gradient flows only through the ablated forward, pushing its gold prob DOWN.
The hinge makes `margin` a TARGET: once the gap >= margin the term is 0 and stops pushing (bounded, so it
nudges the answer to depend on Z rather than lobotomizing the image-only path).

No bottleneck: both forwards are unmasked; the ablated forward differs only in the <lvr> embeddings, via
the forward's `override_latent_embeds` hook. Ablation is in-distribution (shuffled/mean latents), NEVER
zeros — a zero vector is a trivially detectable gate trigger.
"""

import torch
import torch.nn.functional as F

from src.harness.spans import get_spans


def _answer_nll(logits_b, labels_b, s, e):
    """Mean CE (nats) of the gold answer tokens [s, e) at their predicting positions logits[s-1:e-1].

    Same causal-shift slice as distill_loss / harness answer_nll: the distribution predicting token t sits
    at logits[t-1], so answer span [s, e) is scored by logits[s-1:e-1] against labels[s:e].
    """
    ans_logits = logits_b[s - 1:e - 1].float()   # [n_ans, V]
    targets = labels_b[s:e]                       # [n_ans]; answer span is the non-ignore gold region
    return F.cross_entropy(ans_logits, targets)


def necessity_margin_over_answer(with_logits, ablated_logits, input_ids, labels, config, margin=1.0):
    """Answer-margin necessity hinge, averaged over the examples in the batch.

    Args:
        with_logits: [B, L, V] logits from the with-latent (clean) forward. Used only via `.detach()` — no
            gradient flows here (L_answer owns the with-latent answer).
        ablated_logits: [B, L, V] logits from the latent-ablated forward. Gradient flows through THIS branch
            (the hinge pushes the ablated gold prob DOWN).
        input_ids, labels: [B, L]; the gold answer span per example comes from get_spans.
        config: model.config (image_token_id, lvr_id, lvr_start_id, lvr_end_id).
        margin: target NLL gap in nats (see module docstring). Default 1.0 (~2.7x harder without Z).

    Returns:
        Scalar mean_b max(0, margin - (nll_ablated - nll_with.detach())). Gradient only through ablated.
    """
    B = input_ids.shape[0]
    total = ablated_logits.new_zeros(())
    n = 0
    for b in range(B):
        spans = get_spans(
            input_ids[b], labels[b],
            image_token_id=config.image_token_id, lvr_id=config.lvr_id,
            lvr_start_id=config.lvr_start_id, lvr_end_id=config.lvr_end_id,
        )
        s, e = spans["answer"].start, spans["answer"].end
        if e - s < 1 or s < 1:
            continue
        nll_with = _answer_nll(with_logits[b], labels[b], s, e).detach()   # reference; no grad
        nll_ablated = _answer_nll(ablated_logits[b], labels[b], s, e)      # grad flows here
        total = total + torch.clamp(margin - (nll_ablated - nll_with), min=0.0)
        n += 1

    if n == 0:
        return ablated_logits.new_zeros(())
    return total / n
