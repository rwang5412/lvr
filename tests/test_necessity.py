"""Mechanical done-test for the necessity (answer-margin) loss.

Pure part (needs torch, CPU — run anywhere):
  - necessity is finite and >= 0 (hinge, clamped at 0).
  - the ABLATED branch carries gradient; the with-latent branch is DETACHED (inverse of distill's teacher
    test — here it's the ablated pass we push, and NLL_with is detached).
  - necessity == 0 when ablating already makes the gold answer >= margin harder (hinge satisfied).

The `necessity_weight=0` path reproducing baseline is guaranteed by construction (the `if necessity_weight
> 0` guard skips the ablated pass entirely). Model-level integration (shuffle ablation via
override_latent_embeds, memory) is verified by the 1-GPU smoke run, not here.

    python tests/test_necessity.py        # pure part (CPU)
"""

import torch

from src.train.necessity_loss import necessity_margin_over_answer
from types import SimpleNamespace

IMG, LVR, LVR_START, LVR_END, IGN = 100, 200, 201, 202, -100
CONFIG = SimpleNamespace(image_token_id=IMG, lvr_id=LVR, lvr_start_id=LVR_START, lvr_end_id=LVR_END)


def _fixture():
    """[img*3][q*2][lvr_start][lvr*2][lvr_end][ans*3]; answer span = [9,12), predicted at logits[8:11]."""
    seq = [IMG, IMG, IMG, 300, 301, LVR_START, LVR, LVR, LVR_END, 400, 401, 402]
    labs = [IGN] * 5 + [LVR_START, LVR, LVR, LVR_END, 400, 401, 402]
    return torch.tensor([seq]), torch.tensor([labs])


def test_necessity_pure():
    input_ids, labels = _fixture()
    B, L, V = 1, input_ids.shape[1], 512   # V > max answer token id (402) — they are cross_entropy targets
    with_logits = torch.randn(B, L, V, requires_grad=True)
    ablated = torch.randn(B, L, V, requires_grad=True)

    # margin=5.0 with ~random logits (gap ~ 0) guarantees the hinge is active, so grad actually flows.
    nec = necessity_margin_over_answer(with_logits, ablated, input_ids, labels, CONFIG, margin=5.0)
    assert torch.isfinite(nec) and nec.item() >= 0, nec
    nec.backward()
    assert with_logits.grad is None, "with-latent branch must be DETACHED (NLL_with.detach())"
    assert ablated.grad is not None and ablated.grad.abs().sum() > 0, "ablated branch must receive real grad"
    print(f"PASS necessity_pure: nec={nec.item():.4f} finite>=0, with-branch detached, ablated has grad")


def test_necessity_zero_when_satisfied():
    """If ablating already makes the gold answer >> margin harder, the hinge clamps to 0."""
    input_ids, labels = _fixture()
    L, V = input_ids.shape[1], 512   # V > max answer token id (402); gold tokens indexed below
    with_logits = torch.zeros(1, L, V)                       # uniform -> nll_with = log(V)
    ablated = torch.zeros(1, L, V)
    for pred_pos, gold_tok in [(8, 400), (9, 401), (10, 402)]:   # gold token very unlikely at its predicting pos
        ablated[0, pred_pos, gold_tok] = -50.0
    nec = necessity_margin_over_answer(with_logits, ablated, input_ids, labels, CONFIG, margin=1.0)
    assert nec.item() == 0.0, f"hinge should be 0 when the gap >= margin, got {nec.item()}"
    print("PASS necessity_zero_when_satisfied")


if __name__ == "__main__":
    test_necessity_pure()
    test_necessity_zero_when_satisfied()
    print("\nAll necessity pure tests passed.")
