#!/usr/bin/env python3
"""
Verify that the images referenced by the LVR training JSON(s) exist on disk.

Usage:
    python check_images.py --image-folder ~/Desktop/SummerVLM/images \
        data/lvr_data/viscot_363k_lvr_formatted.json \
        data/lvr_data/viscot_sroie_dude_lvr_formatted.json

    # exhaustive (check every row, not a sample):
    python check_images.py --image-folder ~/Desktop/SummerVLM/images --full <json>...

Per source-root (e.g. viscot/flickr30k) it reports how many referenced images
are present vs missing, and prints a few example missing paths. Exit code is
non-zero if anything is missing, so it can gate a training run.
"""
import argparse, json, os, random, collections, sys


def img_paths(row):
    img = row.get("image")
    if img is None:
        return []
    return img if isinstance(img, list) else [img]


def root_of(p):
    parts = p.split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else parts[0]


def main():
    DEFAULT_JSONS = [
        "data/lvr_data/viscot_363k_lvr_formatted.json",
        "data/lvr_data/viscot_sroie_dude_lvr_formatted.json",
    ]
    DEFAULT_IMAGE_FOLDER = "/Users/Richard/Desktop/SummerVLM/images"

    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="*", default=DEFAULT_JSONS,
                    help="training JSON file(s); defaults to the two Stage-1 files")
    ap.add_argument("--image-folder", default=DEFAULT_IMAGE_FOLDER,
                    help="root folder containing viscot/... (default: project images dir)")
    ap.add_argument("--sample", type=int, default=300,
                    help="images checked per source-root (default 300)")
    ap.add_argument("--full", action="store_true",
                    help="check every referenced image (overrides --sample)")
    args = ap.parse_args()

    image_folder = os.path.expanduser(args.image_folder)
    if not os.path.isdir(image_folder):
        print(f"ERROR: image folder not found: {image_folder}")
        sys.exit(2)

    # gather referenced paths grouped by source-root
    by_root = collections.defaultdict(list)
    for jf in args.jsons:
        data = json.load(open(jf))
        for row in data:
            for p in img_paths(row):
                by_root[root_of(p)].append(p)

    print(f"image_folder = {image_folder}\n")
    print(f"{'source root':26s}{'referenced':>12s}{'checked':>9s}{'missing':>9s}  status")
    print("-" * 70)

    total_missing = 0
    missing_examples = []
    for root in sorted(by_root):
        paths = by_root[root]
        uniq = list(dict.fromkeys(paths))  # de-dupe, keep order
        if args.full or len(uniq) <= args.sample:
            check = uniq
        else:
            check = random.sample(uniq, args.sample)
        miss = [p for p in check if not os.path.exists(os.path.join(image_folder, p))]
        total_missing += len(miss)
        if miss:
            missing_examples.extend(miss[:3])
        status = "OK" if not miss else f"** {len(miss)} MISSING **"
        print(f"{root:26s}{len(paths):>12,}{len(check):>9,}{len(miss):>9,}  {status}")

    print("-" * 70)
    if total_missing == 0:
        print("\nAll checked images present. Data is ready for the referenced sources.")
        sys.exit(0)
    else:
        print(f"\n{total_missing} missing (in the checked sample). Examples:")
        for p in missing_examples[:15]:
            print("   ", os.path.join(image_folder, p))
        print("\nFix the image folder / extraction before training, or filter these rows out.")
        sys.exit(1)


if __name__ == "__main__":
    main()
