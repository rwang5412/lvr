"""Mechanical done-test for the self-distillation KL loss.

Pure part (needs torch, CPU — run anywhere):
  - KL is finite and >= 0.
  - teacher is DETACHED: after backward, gradient flows into the student logits only, not the teacher.
  - KL == 0 when student == teacher (sanity).

Model part (needs a checkpoint + GPU; run on Palmetto): one end-to-end student(bottleneck on) +
teacher(bottleneck off, no_grad) pass, KL finite on real logits.

Not verifying "proportion-mediated moved" — that's the harness's job on a real run. And
`distill_weight=0` reproducing Step 1 is guaranteed by construction (the `if distill_weight > 0` guard
skips the teacher pass entirely).

    python tests/test_distill.py                                  # pure part (CPU)
    PYTHONPATH=. python tests/test_distill.py --checkpoint weights/LVR-7B \
        --image-folder /scratch/haizhow/lvr_images --heldout /scratch/haizhow/heldout_harness.json
"""

import argparse
from types import SimpleNamespace

import torch

from src.train.distill_loss import distill_kl_over_answer

IMG, LVR, LVR_START, LVR_END, IGN = 100, 200, 201, 202, -100
CONFIG = SimpleNamespace(image_token_id=IMG, lvr_id=LVR, lvr_start_id=LVR_START, lvr_end_id=LVR_END)


def _fixture():
    """[img*3][q*2][lvr_start][lvr*2][lvr_end][ans*3]; response (lvr_start..ans) kept in labels."""
    seq = [IMG, IMG, IMG, 300, 301, LVR_START, LVR, LVR, LVR_END, 400, 401, 402]
    labs = [IGN] * 5 + [LVR_START, LVR, LVR, LVR_END, 400, 401, 402]
    return torch.tensor([seq]), torch.tensor([labs])


def test_kl_pure():
    input_ids, labels = _fixture()
    B, L, V = 1, input_ids.shape[1], 50
    student = torch.randn(B, L, V, requires_grad=True)
    teacher = torch.randn(B, L, V, requires_grad=True)

    kl = distill_kl_over_answer(student, teacher, input_ids, labels, CONFIG)
    assert torch.isfinite(kl) and kl.item() >= 0, kl
    kl.backward()
    assert teacher.grad is None, "teacher must be detached (no gradient)"
    assert student.grad is not None and torch.isfinite(student.grad).all(), "student must receive finite grad"
    print(f"PASS kl_pure: kl={kl.item():.4f} finite>=0, teacher detached, student has grad")


def test_kl_zero_when_equal():
    input_ids, labels = _fixture()
    logits = torch.randn(1, input_ids.shape[1], 50)
    kl = distill_kl_over_answer(logits, logits.clone(), input_ids, labels, CONFIG)
    assert abs(kl.item()) < 1e-5, f"KL should be ~0 when student==teacher, got {kl.item()}"
    print("PASS kl_zero_when_equal")


def test_model(checkpoint, image_folder, heldout):
    from evaluation.run_harness import load_model_and_processor, _forward, _to_device
    from src.params import DataArguments
    from src.harness import data as hdata

    model, processor, config = load_model_and_processor(checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_args = DataArguments(image_folder=image_folder)
    ds = hdata.build_dataset(hdata.load_records(heldout)[:2], processor, data_args, model_id=checkpoint)
    collator = hdata.build_collator(processor)
    batch = _to_device(collator([ds[0], ds[1]]), device)

    model.config.use_bottleneck = True
    student = _forward(model, batch)                    # student = bottleneck on
    model.config.use_bottleneck = False
    teacher = _forward(model, batch)                    # teacher = bottleneck off (already no_grad in _forward)
    model.config.use_bottleneck = True

    kl = distill_kl_over_answer(student.logits, teacher.logits, batch["input_ids"], batch["labels"], config)
    assert torch.isfinite(kl) and kl.item() >= 0, kl
    print(f"PASS model: end-to-end student/teacher passes, kl={kl.item():.4f} finite")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--image-folder")
    ap.add_argument("--heldout", default="data/lvr_data/heldout_harness.json")
    args = ap.parse_args()

    test_kl_pure()
    test_kl_zero_when_equal()
    if args.checkpoint:
        test_model(args.checkpoint, args.image_folder, args.heldout)
        print("\nAll distill tests passed.")
    else:
        print("\nPure tests passed. Pass --checkpoint/--image-folder for the model-level test.")
