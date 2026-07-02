"""Carve a fixed, seeded held-out split for the harness (Branch 1).

Selects records that HAVE a bbox and a single image (the harness needs the ROI for latents/probe,
and get_spans assumes one contiguous image block). Writes:

    <out>          — the held-out records, in the training JSON format.
    <ids-out>      — the (dataset, split, question_id) keys, so Branch 4 can EXCLUDE them from the
                     swap-training subset (keeps the harness set genuinely held-out for arm compares).

Deterministic: same --seed → same split. Run once; commit the outputs.

    python evaluation/make_heldout_split.py \
        --source data/lvr_data/viscot_363k_lvr_formatted.json \
        --out data/lvr_data/heldout_harness.json \
        --ids-out data/lvr_data/heldout_harness_ids.json \
        --n 300 --seed 1234
"""

import argparse
import json
import random


def _has_bbox(rec):
    # Exactly ONE bbox: the training forward's enumerate(lvr_tokens) couples the group index to the
    # batch index, so multi-bbox examples misalign at batch size 1. Single-bbox is also the §4 default.
    b = rec.get("bboxes")
    return isinstance(b, list) and len(b) == 1


def _single_image(rec):
    img = rec.get("image")
    return isinstance(img, list) and len(img) == 1


def _has_lvr_and_answer(rec):
    convs = rec.get("conversations", [])
    if len(convs) < 2:
        return False
    return "<lvr>" in convs[0]["value"] + convs[1]["value"] and "<answer>" in convs[1]["value"]


def _key(rec):
    return [rec.get("dataset"), rec.get("split"), rec.get("question_id")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="data/lvr_data/viscot_363k_lvr_formatted.json")
    ap.add_argument("--out", default="data/lvr_data/heldout_harness.json")
    ap.add_argument("--ids-out", default="data/lvr_data/heldout_harness_ids.json")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    with open(args.source) as f:
        records = json.load(f)
    print(f"[split] loaded {len(records)} records from {args.source}")

    eligible = [r for r in records if _has_bbox(r) and _single_image(r) and _has_lvr_and_answer(r)]
    print(f"[split] {len(eligible)} eligible (bbox + single image + <lvr>/<answer>)")
    if len(eligible) < args.n:
        raise ValueError(f"only {len(eligible)} eligible records, need {args.n}")

    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    held = eligible[: args.n]

    with open(args.out, "w") as f:
        json.dump(held, f, indent=2)
    with open(args.ids_out, "w") as f:
        json.dump([_key(r) for r in held], f, indent=2)

    print(f"[split] wrote {len(held)} held-out records → {args.out}")
    print(f"[split] wrote {len(held)} exclusion keys     → {args.ids_out}")


if __name__ == "__main__":
    main()
