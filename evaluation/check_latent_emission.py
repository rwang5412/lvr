"""Latent-emission smoke — does a checkpoint actually emit <|lvr_start|> + latent steps at inference?

PREREQUISITE for the CapImagine-style do(Z) accuracy harness: if the model does not emit latents on a
dataset, there is nothing to intervene on — do(Z) is a trivial no-op and you'd "reproduce" the paper's
null result for a boring reason. The paper itself only keeps "instances with a valid latent reasoning
process." So run this on ~5 examples of each target set BEFORE building the intervention harness.

Run it on BOTH an in-domain set (gqa held-out — the model was trained to reason here) and each OOD
benchmark (V*, HR-Bench, MME). If gqa emits latents but a benchmark doesn't, that benchmark is OOD for
latent reasoning and the do(Z) accuracy story has to live on gqa (or filter to valid-latent instances).

    PYTHONPATH=. python evaluation/check_latent_emission.py \
        --checkpoint /scratch/haizhow/ckpts/bottleneck_7b/checkpoint-833 \
        --image-folder /scratch/haizhow/vcot_dl \
        --records data/lvr_data/heldout_val_clean.json --limit 5 --lvr-steps 16

`--records` is a JSON list. Two accepted shapes:
  - gqa/viscot: {"image": ["path"], "conversations": [{"from":"human","value":"<image>\\nQ"}...]}
  - simple:     {"image": "path", "question": "Q"}
"""

import argparse
import json
import os

import torch  # noqa: F401  (ensures CUDA init before the eval import chain)

from evaluation.evaluation import load_model_and_processor, run_inference

LVR_START, LVR, LVR_END = "<|lvr_start|>", "<|lvr|>", "<|lvr_end|>"


def _question(rec):
    if "conversations" in rec:
        for c in rec["conversations"]:
            if c.get("from") == "human":
                v = c["value"]
                return v.split("\n", 1)[-1].strip() if v.startswith("<image>") else v.replace("<image>", "").strip()
    return rec.get("question", "")


def _image(rec):
    img = rec.get("image")
    return img[0] if isinstance(img, list) else img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--records", required=True, help="JSON list (gqa/viscot or {image,question})")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--lvr-steps", type=int, default=16, help="latent steps for 'steps' decoding")
    ap.add_argument("--decoding-strategy", default="steps")
    args = ap.parse_args()

    print(f"[emission] loading {args.checkpoint}")
    model, processor, _ = load_model_and_processor(args.checkpoint)
    recs = json.load(open(args.records))[: args.limit]

    first_img = os.path.join(args.image_folder, _image(recs[0]))
    if not os.path.exists(first_img):
        raise FileNotFoundError(f"first image not found: {first_img}\n--image-folder probably wrong.")

    n_latent = 0
    for i, rec in enumerate(recs):
        q = _question(rec)
        img = os.path.join(args.image_folder, _image(rec))
        out = run_inference(model, processor, img, q, args.lvr_steps, args.decoding_strategy)[0]
        has_start = LVR_START in out
        n_latent += int(has_start)
        print(f"\n[{i}] latent_start={has_start}  <lvr>x{out.count(LVR)}  lvr_end={LVR_END in out}  Q: {q[:70]}")
        print(f"    OUT: {out[:320]!r}")

    print(f"\n=== {n_latent}/{len(recs)} examples emitted <|lvr_start|> (entered latent reasoning) ===")
    if n_latent == 0:
        print("!! NO latents emitted -> the do(Z) accuracy harness is MOOT on this set.\n"
              "   Check: is --lvr-steps/--decoding-strategy right? is this set OOD for latent reasoning?")
    elif n_latent < len(recs):
        print("   (partial) -> the harness must FILTER to valid-latent instances and report N.")


if __name__ == "__main__":
    main()
