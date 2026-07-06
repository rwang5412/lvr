"""Branch 2 done-test: the answer->image attention bottleneck.

Two parts:

1. test_mask_structure()  — PURE (needs torch, no model/GPU). Verifies build_bottleneck_mask puts
   -inf exactly on (answer-row, image-col), keeps latent->image OPEN, and carries the standard causal
   + padding masking. This is the silent-leak guard: an off-by-one here would mask the wrong tokens.

2. test_mask_applied()    — MODEL (needs a checkpoint + GPU; run on Palmetto). Verifies the mask is
   actually consumed by HF/SDPA (toggling the bottleneck changes the logits) AND that it REDUCES the
   image->answer sensitivity (perturbing the image, with latents held fixed via override, changes the
   answer less with the bottleneck on than off).

   NOTE: it does NOT assert "answer unchanged". latent->image is open by design, so image info still
   reaches the answer THROUGH the latents (that is the intended routing). The bottleneck removes the
   DIRECT answer->image path, so sensitivity drops but is not zero.

Usage:
    python tests/test_bottleneck.py                              # structure test only (CPU)
    PYTHONPATH=. python tests/test_bottleneck.py --checkpoint weights/LVR-7B \
        --image-folder /scratch/haizhow/lvr_images --heldout /scratch/haizhow/heldout_harness.json
"""

import argparse

import torch

from src.train.monkey_patch_forward_lvr import build_bottleneck_mask

IMG, LVR, LVR_START, LVR_END = 100, 200, 201, 202


def test_mask_structure():
    """[image*3][q*2][lvr_start][lvr*2][lvr_end][ans*2], one row + one padded row."""
    dtype = torch.float32
    min_val = torch.finfo(dtype).min
    # row 0: full 11-token example; row 1: same but last 2 tokens are padding
    seq = [IMG, IMG, IMG, 300, 301, LVR_START, LVR, LVR, LVR_END, 400, 401]
    input_ids = torch.tensor([seq, seq], dtype=torch.long)
    attn2d = torch.ones(2, 11, dtype=torch.long)
    attn2d[1, 9:] = 0  # pad the last two columns of row 1

    m = build_bottleneck_mask(input_ids, attn2d, image_token_id=IMG, lvr_id=LVR, dtype=dtype)
    assert m.shape == (2, 1, 11, 11), m.shape

    # image cols = {0,1,2}; last latent index = 7; answer rows = {8,9,10}
    for r in (8, 9, 10):
        for c in (0, 1, 2):
            assert m[0, 0, r, c] == min_val, f"answer row {r} should be blocked from image col {c}"
    # latent->image is OPEN (latent row 6, image col 0; image is causally in the past)
    assert m[0, 0, 6, 0] == 0, "latent->image must stay open"
    # answer->latent and answer->question OPEN
    assert m[0, 0, 9, 6] == 0, "answer->latent must stay open"
    assert m[0, 0, 9, 3] == 0, "answer->question must stay open"
    # causal: a future key (k>q) is blocked
    assert m[0, 0, 0, 5] == min_val, "causal: pos 0 must not see future pos 5"
    assert m[0, 0, 3, 4] == min_val, "causal: pos 3 must not see future pos 4"
    # a normal past, non-image attention is open
    assert m[0, 0, 4, 3] == 0, "pos 4 -> past pos 3 (question) should be open"
    # padding: row 1's padded key columns 9,10 are blocked everywhere
    assert (m[1, 0, :, 9] == min_val).all(), "padded col 9 must be blocked"
    assert (m[1, 0, :, 10] == min_val).all(), "padded col 10 must be blocked"

    print("PASS test_mask_structure")


# --------------------------------------------------------------------- model-level (Palmetto) -----

def test_mask_applied(checkpoint, image_folder, heldout):
    from evaluation.run_harness import load_model_and_processor, _forward, _to_device
    from src.params import DataArguments
    from src.harness import data as hdata
    from src.harness.spans import get_spans

    model, processor, config = load_model_and_processor(checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_args = DataArguments(image_folder=image_folder)
    records = hdata.load_records(heldout)[:1]
    ds = hdata.build_dataset(records, processor, data_args, model_id=checkpoint)
    collator = hdata.build_collator(processor)
    batch = _to_device(hdata.collate_one(collator, ds[0]), device)

    spans = get_spans(batch["input_ids"][0], batch["labels"][0],
                      image_token_id=config.image_token_id, lvr_id=config.lvr_id,
                      lvr_start_id=config.lvr_start_id, lvr_end_id=config.lvr_end_id)
    a_s, a_e = spans["answer"].start, spans["answer"].end

    # hold latents fixed across all conditions: use the clean captured ROI targets as the override
    clean = _forward(model, batch, override_latent_embeds=None)
    T_fixed = clean.latent_target_embeds.to(device)

    def ans_logits(zero_image):
        return _forward(model, batch, override_latent_embeds=T_fixed, zero_image=zero_image).logits[0, a_s:a_e].float()

    # 1. mask is APPLIED: toggling the bottleneck changes the logits (HF consumed the 4D mask)
    model.config.use_bottleneck = False
    off = ans_logits(zero_image=False)
    model.config.use_bottleneck = True
    on = ans_logits(zero_image=False)
    d_toggle = (off - on).abs().max().item()
    assert d_toggle > 1e-3, f"toggling bottleneck did not change logits (max|Δ|={d_toggle}) — 4D mask not applied"
    print(f"PASS applied: toggling bottleneck changes answer logits (max|Δ|={d_toggle:.3e})")

    # 2. bottleneck REDUCES image->answer sensitivity (perturb image, latents fixed)
    model.config.use_bottleneck = False
    d_off = (ans_logits(zero_image=False) - ans_logits(zero_image=True)).abs().max().item()
    model.config.use_bottleneck = True
    d_on = (ans_logits(zero_image=False) - ans_logits(zero_image=True)).abs().max().item()
    print(f"image->answer sensitivity: bottleneck OFF={d_off:.3e}  ON={d_on:.3e}  (ratio {d_on/max(d_off,1e-9):.3f})")
    # Soft: the direct path is removed, so sensitivity should drop clearly. The residual is the
    # latent-mediated route (latent->image is open by design), so we don't require it to reach 0.
    assert d_on < 0.95 * d_off, f"bottleneck did not clearly reduce image->answer sensitivity (on={d_on}, off={d_off})"
    print("PASS reduced: bottleneck cuts the direct image->answer path (residual is the latent route)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--image-folder")
    ap.add_argument("--heldout", default="data/lvr_data/heldout_harness.json")
    args = ap.parse_args()

    test_mask_structure()
    if args.checkpoint:
        test_mask_applied(args.checkpoint, args.image_folder, args.heldout)
        print("\nAll bottleneck tests passed.")
    else:
        print("\nStructure test passed. Pass --checkpoint/--image-folder to run the model-level test.")
