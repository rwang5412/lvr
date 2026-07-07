#!/usr/bin/env python3
"""Draw LVR training bboxes onto their images and write an annotated copy tree.

The Stage-1 LVR JSONs store, per row, one image path (relative to --image_folder)
and one normalized bbox (x_min, y_min, x_max, y_max) in [0, 1] -- the same
interpretation the dataset code uses (src/dataset/lvr_sft_dataset.py:make_bbox_masks_rgb).

Because the same image is reused across rows with different boxes, two output
layouts are supported:

  per-row   (default): one annotated copy per JSON row, named <stem>__row<idx>.<ext>,
            each showing that row's single box. 1:1 with training examples.
  per-image          : one annotated copy per unique image, mirroring the original
            relative path, with ALL of that image's boxes overlaid.

Example:
  python3 draw_bboxes.py \
      --json data/lvr_data/viscot_sroie_dude_lvr_formatted.json \
      --image_folder /Users/Richard/Desktop/SummerVLM/images \
      --out_folder   /Users/Richard/Desktop/SummerVLM/images_bbox \
      --mode per-row
"""
import argparse
import json
import os
from collections import defaultdict

from PIL import Image, ImageDraw


def draw_boxes(img, boxes, color=(255, 0, 0)):
    """Draw normalized xyxy boxes onto a PIL image (in place-ish); returns RGB image."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    width = max(2, round(0.004 * max(w, h)))
    for x0, y0, x1, y1 in boxes:
        px = [x0 * w, y0 * h, x1 * w, y1 * h]
        # clamp and normalize ordering
        x_min, x_max = sorted((max(0, min(w, px[0])), max(0, min(w, px[2]))))
        y_min, y_max = sorted((max(0, min(h, px[1])), max(0, min(h, px[3]))))
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=width)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--out_folder", required=True)
    ap.add_argument("--mode", choices=["per-row", "per-image"], default="per-row")
    ap.add_argument("--limit", type=int, default=0, help="process only first N rows (0 = all)")
    args = ap.parse_args()

    rows = json.load(open(args.json))
    if args.limit:
        rows = rows[: args.limit]

    written = skipped = 0

    if args.mode == "per-image":
        # collect all boxes per unique image first
        by_img = defaultdict(list)
        for r in rows:
            by_img[r["image"][0]].extend(r["bboxes"])
        items = by_img.items()
    else:
        items = None

    if args.mode == "per-image":
        for i, (rel, boxes) in enumerate(items):
            src = os.path.join(args.image_folder, rel)
            dst = os.path.join(args.out_folder, rel)
            try:
                img = Image.open(src)
                img = draw_boxes(img, boxes)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                img.save(dst)
                written += 1
            except Exception as e:
                skipped += 1
                if skipped <= 10:
                    print(f"  skip {rel}: {e}")
            if (i + 1) % 500 == 0:
                print(f"  {i + 1} images -> {written} written, {skipped} skipped")
    else:
        for idx, r in enumerate(rows):
            rel = r["image"][0]
            src = os.path.join(args.image_folder, rel)
            stem, ext = os.path.splitext(rel)
            dst = os.path.join(args.out_folder, f"{stem}__row{idx}{ext}")
            try:
                img = Image.open(src)
                img = draw_boxes(img, r["bboxes"])
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                img.save(dst)
                written += 1
            except Exception as e:
                skipped += 1
                if skipped <= 10:
                    print(f"  skip row {idx} ({rel}): {e}")
            if (idx + 1) % 1000 == 0:
                print(f"  {idx + 1} rows -> {written} written, {skipped} skipped")

    print(f"\nDone. {written} written, {skipped} skipped -> {args.out_folder}")


if __name__ == "__main__":
    main()
