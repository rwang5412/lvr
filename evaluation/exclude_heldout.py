"""Freeze the harness held-out set OUT of a training JSON.

The harness measures the trained arms (Branch 2/4) on the fixed held-out 300. Those examples must
never appear in training, or the causal metric is "graded on its own homework". This filters any
training manifest by the (dataset, split, question_id) keys recorded in heldout_harness_ids.json.

    python evaluation/exclude_heldout.py \
        --source data/lvr_data/viscot_363k_lvr_formatted.json \
        --ids data/lvr_data/heldout_harness_ids.json \
        --out data/lvr_data/viscot_train_minus_heldout.json

Run this once to produce the training-safe manifest; point the swap-training scripts at --out.
"""

import argparse
import json


def _key(rec):
    return (rec.get("dataset"), rec.get("split"), rec.get("question_id"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="data/lvr_data/viscot_363k_lvr_formatted.json",
                    help="full training manifest to filter")
    ap.add_argument("--ids", default="data/lvr_data/heldout_harness_ids.json",
                    help="held-out keys emitted by make_heldout_split.py")
    ap.add_argument("--out", default="data/lvr_data/viscot_train_minus_heldout.json")
    args = ap.parse_args()

    with open(args.source) as f:
        records = json.load(f)
    with open(args.ids) as f:
        heldout = {tuple(k) for k in json.load(f)}

    kept = [r for r in records if _key(r) not in heldout]
    dropped = len(records) - len(kept)

    with open(args.out, "w") as f:
        json.dump(kept, f)

    print(f"[exclude] source records : {len(records)}")
    print(f"[exclude] held-out keys  : {len(heldout)}")
    print(f"[exclude] dropped        : {dropped}   (expected == held-out keys present in source)")
    print(f"[exclude] kept           : {len(kept)} -> {args.out}")
    if dropped != len(heldout):
        print(f"[exclude] WARNING: dropped ({dropped}) != held-out keys ({len(heldout)}). "
              f"Some held-out ids were not found in the source (different manifest?).")


if __name__ == "__main__":
    main()
