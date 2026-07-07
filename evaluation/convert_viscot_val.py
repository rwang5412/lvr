"""Convert a raw Visual-CoT val/test .jsonl into the LVR training-row format the harness ingests.

The raw val schema (deepcs233/Visual-CoT):
    {question, answer, image:"<file>.jpg", width, height, bboxs:[[x0,y0,x1,y1]](pixels), dataset, split, ...}

LVR harness rows need:
    {dataset, split, question_id, image:["viscot/<src>/<file>.jpg"],
     conversations:[{human:"<image>\\n<q>"},{gpt:"<lvr>\\n<answer> <a> </answer>"}],
     bboxes:[[x0/w,y0/h,x1/w,y1/h]](normalized)}

This is the CLEAN held-out (split=val, unseen by the Visual-CoT-train base). Point
make_slice.py heldout / run_harness.py at the output.

    python evaluation/convert_viscot_val.py \
        --src ~/Desktop/viscot_eval/gqa_cot_val.jsonl \
        --image-prefix viscot/gqa \
        --out data/lvr_data/viscot_gqa_val_lvr.json
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="raw Visual-CoT val jsonl")
    ap.add_argument("--image-prefix", default="viscot/gqa",
                    help="folder prefix prepended to the bare image filename")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out, skipped = [], 0
    for qid, line in enumerate(open(args.src)):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        w, h = r.get("width"), r.get("height")
        bboxs = r.get("bboxs") or []
        # harness contract: exactly one bbox, one image, valid dims
        if len(bboxs) != 1 or not w or not h:
            skipped += 1
            continue
        x0, y0, x1, y1 = bboxs[0]
        bbox_norm = [x0 / w, y0 / h, x1 / w, y1 / h]

        fname = r["image"].split("/")[-1]  # tolerate bare name or a path
        rel = f"{args.image_prefix}/{fname}"

        out.append({
            "dataset": r.get("dataset", "gqa"),
            "split": r.get("split", "val"),
            "question_id": qid,
            "image": [rel],
            "conversations": [
                {"from": "human", "value": "<image>\n" + r["question"].strip()},
                {"from": "gpt", "value": "<lvr>\n<answer> " + r["answer"].strip() + " </answer>"},
            ],
            "bboxes": [bbox_norm],
        })

    json.dump(out, open(args.out, "w"), ensure_ascii=False)
    print(f"[convert] {len(out)} rows -> {args.out}  (skipped {skipped} non-single-bbox)")
    if out:
        print("[convert] sample:", json.dumps(out[0], ensure_ascii=False)[:300])


if __name__ == "__main__":
    main()
