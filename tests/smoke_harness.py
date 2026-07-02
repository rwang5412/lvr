"""Plumbing smoke test for the Branch-1 harness — run on Palmetto (needs checkpoint + GPU).

Validates the MODEL EDIT and harness wiring on 1-2 real examples, independent of any scientific
result. It answers "is the override/capture correct?", not "does the swap work?". Fast (a couple of
forwards), so run it BEFORE the full 300-example harness.

    PYTHONPATH=. python tests/smoke_harness.py \
        --checkpoint /path/to/lvr_checkpoint \
        --image-folder /path/to/images \
        --heldout data/lvr_data/heldout_harness.json

Checks:
  1. capture shapes  — latent_hidden_states / latent_target_embeds are [L, H] with L == #<lvr> tokens
                       == get_spans latent length (the shared contract lines up with the model).
  2. determinism     — two clean forwards give identical logits (eval mode, no sampling).
  3. override reaches the model — override=zeros changes the logits (the hook is live).
  4. override idempotence — override=clean-targets reproduces the clean logits exactly (the override
                       writes the RIGHT positions with the RIGHT semantics; this is the strongest
                       single correctness check for the edit).
  5. image corruption — zeroing pixels changes the logits (image path is live, for the NIE denominator).
  6. metrics finite  — answer_nll is finite for clean and corrupted.
"""

import argparse

import torch

from evaluation.run_harness import load_model_and_processor, _forward, _to_device
from src.params import DataArguments
from src.harness import data as hdata, metrics
from src.harness.spans import get_spans


def _max_abs_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--heldout", default="data/lvr_data/heldout_harness.json")
    args = ap.parse_args()

    model, processor, config = load_model_and_processor(args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_token_id, lvr_id = config.image_token_id, config.lvr_id

    data_args = DataArguments(image_folder=args.image_folder)
    records = hdata.load_records(args.heldout)[:2]
    ds = hdata.build_dataset(records, processor, data_args, model_id=args.checkpoint)
    collator = hdata.build_collator(processor)

    batch = _to_device(hdata.collate_one(collator, ds[0]), device)
    spans = get_spans(batch["input_ids"][0], batch["labels"][0],
                      image_token_id=image_token_id, lvr_id=lvr_id)
    n_lvr = len(spans["latent"])
    print(f"[smoke] example 0: seq_len={batch['input_ids'].shape[1]}  latent tokens={n_lvr}  "
          f"answer span={spans['answer'].as_tuple()}")

    # --- clean forward (twice) ---
    clean = _forward(model, batch, override_latent_embeds=None)
    clean2 = _forward(model, batch, override_latent_embeds=None)

    Z = clean.latent_hidden_states
    T = clean.latent_target_embeds
    assert Z is not None and T is not None, "captures are None — lvr_tokens not reaching the forward?"

    # 1. capture shapes line up with the span contract
    assert Z.shape[0] == n_lvr, f"latent_hidden_states rows {Z.shape[0]} != #<lvr> {n_lvr}"
    assert T.shape[0] == n_lvr, f"latent_target_embeds rows {T.shape[0]} != #<lvr> {n_lvr}"
    assert Z.shape[1] == T.shape[1], "Z and target hidden dims differ"
    print(f"[smoke] PASS 1 capture shapes: Z={tuple(Z.shape)}  T={tuple(T.shape)}")

    # 2. determinism
    d = _max_abs_diff(clean.logits, clean2.logits)
    assert d < 1e-4, f"clean forwards not deterministic (max|Δ|={d})"
    print(f"[smoke] PASS 2 determinism: max|Δ|={d:.2e}")

    # 3. override reaches the model (zeros must change logits)
    zeros = torch.zeros(n_lvr, Z.shape[1], dtype=clean.logits.dtype, device=device)
    zc = _forward(model, batch, override_latent_embeds=zeros)
    d_zero = _max_abs_diff(clean.logits, zc.logits)
    assert d_zero > 1e-4, f"override=zeros did NOT change logits (max|Δ|={d_zero}) — hook not live"
    print(f"[smoke] PASS 3 override reaches model: max|Δ|(zero vs clean)={d_zero:.3e}")

    # 4. override idempotence: overriding with the clean targets reproduces clean exactly
    idem = _forward(model, batch, override_latent_embeds=T.to(device))
    d_idem = _max_abs_diff(clean.logits, idem.logits)
    assert d_idem < 1e-3, f"override=clean-targets did not reproduce clean (max|Δ|={d_idem})"
    print(f"[smoke] PASS 4 override idempotence: max|Δ|(targets vs clean)={d_idem:.2e}")

    # 5. image corruption changes logits (NIE denominator path is live)
    ic = _forward(model, batch, override_latent_embeds=None, zero_image=True)
    d_img = _max_abs_diff(clean.logits, ic.logits)
    assert d_img > 1e-4, f"zeroing pixels did NOT change logits (max|Δ|={d_img})"
    print(f"[smoke] PASS 5 image corruption changes logits: max|Δ|={d_img:.3e}")

    # 6. metrics finite
    nll_clean = float(metrics.answer_nll(clean.logits, batch["labels"]))
    nll_zero = float(metrics.answer_nll(zc.logits, batch["labels"]))
    nll_img = float(metrics.answer_nll(ic.logits, batch["labels"]))
    assert all(map(lambda x: x == x and abs(x) < 1e6, [nll_clean, nll_zero, nll_img])), "non-finite NLL"
    print(f"[smoke] PASS 6 answer_nll finite: clean={nll_clean:.3f} "
          f"latent0={nll_zero:.3f} imgcorrupt={nll_img:.3f}")

    print("\n[smoke] ALL PLUMBING CHECKS PASSED — safe to run the full harness.")


if __name__ == "__main__":
    main()
