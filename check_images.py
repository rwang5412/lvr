"""Verify that every training image referenced by the LVR JSONs exists under --image-folder.

Run this ON PALMETTO (where the images live) to answer "do we have all the training images?".
Dedupes to unique paths first (404k rows -> ~155k unique files), stats each once, and reports
present/missing broken down by source, with a sample of what's missing.

    python check_images.py \
        --data data/lvr_data/viscot_363k_lvr_formatted.json data/lvr_data/viscot_sroie_dude_lvr_formatted.json \
        --image-folder /scratch/haizhow/<parent_of_viscot>

Exit code 0 = all present, 1 = some missing (so it's usable in a gating script).
"""

import argparse
import json
import os
import sys
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="one or more LVR training JSON files")
    ap.add_argument("--image-folder", required=True, help="root that image paths are relative to (parent of viscot/)")
    ap.add_argument("--show-missing", type=int, default=20, help="how many missing paths to print")
    args = ap.parse_args()

    # collect unique relative image paths across all data files
    rel_paths = set()
    for f in args.data:
        recs = json.load(open(f))
        for r in recs:
            for img in r.get("image", []):
                rel_paths.add(img)
        print(f"[check] {f}: {len(recs)} rows")
    print(f"[check] {len(rel_paths)} unique image paths across all data\n")

    def source(rel):  # viscot/<source>/... -> <source>
        parts = rel.split("/")
        return parts[1] if len(parts) > 1 and parts[0] == "viscot" else parts[0]

    present, missing = Counter(), Counter()
    missing_samples = []
    for rel in rel_paths:
        full = os.path.join(args.image_folder, rel)
        s = source(rel)
        if os.path.exists(full):
            present[s] += 1
        else:
            missing[s] += 1
            if len(missing_samples) < args.show_missing:
                missing_samples.append(full)

    sources = sorted(set(present) | set(missing))
    print(f"  {'source':<18} {'present':>8} {'missing':>8} {'total':>8}")
    for s in sources:
        p, m = present[s], missing[s]
        flag = "  <-- MISSING" if m else ""
        print(f"  {s:<18} {p:>8} {m:>8} {p + m:>8}{flag}")
    tot_p, tot_m = sum(present.values()), sum(missing.values())
    print(f"  {'TOTAL':<18} {tot_p:>8} {tot_m:>8} {tot_p + tot_m:>8}")

    if tot_m:
        print(f"\n[check] MISSING {tot_m} images. Sample:")
        for p in missing_samples:
            print(f"    {p}")
        print(f"\n[check] RESULT: INCOMPLETE — {tot_m}/{tot_p + tot_m} images missing.")
        sys.exit(1)
    else:
        print(f"\n[check] RESULT: COMPLETE — all {tot_p} images present under {args.image_folder}.")
        sys.exit(0)


if __name__ == "__main__":
    main()
