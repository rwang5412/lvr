"""Stratified-proportional slice + fixed held-out for Step-1 (bottleneck + existing SFT).

Step 1 continue-finetunes from an existing checkpoint with the bottleneck ON and only the
existing L_answer + L_patch, on a REPRESENTATIVE subset, then runs the harness. This script
produces that subset. "Representative" here = source-stratified proportional sampling, seeded,
with a fixed held-out eval split carved out and asserted disjoint.

Two subcommands:

  heldout — draw a FIXED, seeded, source-stratified held-out eval set from the full data.
            Run ONCE; commit it; EVERY arm/draw evaluates on this same set (so the Step-1
            baseline number the ablation is measured against is global, not per-draw).
            --base-train-ids: assert the held-out is disjoint from the BASE checkpoint's
            training ids. A held-out row the base already trained on pushes both T and M
            toward 0 -> proportion-mediated degenerates into a memorization score. The
            plain train/eval disjoint-assert is BLIND to this; this guard is not.

  train   — draw a seeded, source-stratified TRAINING subset of size N, EXCLUDING the
            held-out. Run with two seeds for the (training-noise) stability check, or reuse
            one train draw + two held-out draws for the (cheaper, eval-noise) check.
            Prints a representativeness report: attribute-type histogram (full vs subset).
            Source-mix matches by construction (that's the sampling key), so the claim that
            actually needs validating is "source proxies attribute type" -> that histogram.

Deterministic: same --seed -> same draw. Key = (dataset, split, question_id) (verified unique).

    # once — CLEAN held-out from the VAL split (unseen by the base; no contamination guard needed):
    python evaluation/make_slice.py heldout --source data/lvr_data/viscot_gqa_val_lvr.json \
        --heldout-n 300 --seed 1234 --out data/lvr_data/heldout_val_clean.json
    # training slice — diverse, multi-source, TRAIN split. Held-out is the val split, so it's
    # disjoint by construction and --heldout-in is unnecessary. ~80k x1 epoch ~= 300-450 steps:
    python evaluation/make_slice.py train --source data/lvr_data/viscot_363k_lvr_formatted.json \
        --n 80000 --epochs 1 --seed 1234 --out data/lvr_data/slice_train.json
    # (the train command prints an ESTIMATED-STEPS block; keep total steps in the ~200-450 band)
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict


# ----------------------------------------------------------------- helpers -----

def _key(rec):
    return (rec.get("dataset"), rec.get("split"), rec.get("question_id"))


def _question(rec):
    for c in rec.get("conversations", []):
        if c["from"] == "human":
            v = c["value"]
            return (v.split("\n", 1)[-1] if v.startswith("<image>") else v).strip()
    return ""


def _eligible(rec):
    """Harness needs: exactly one bbox, one image, and <lvr>/<answer> present."""
    b, im, cv = rec.get("bboxes"), rec.get("image"), rec.get("conversations", [])
    if not (isinstance(b, list) and len(b) == 1):
        return False
    if not (isinstance(im, list) and len(im) == 1):
        return False
    if len(cv) < 2:
        return False
    return "<lvr>" in cv[0]["value"] + cv[1]["value"] and "<answer>" in cv[1]["value"]


def _attr_type(q):
    """Coarse keyword attribute-type — used ONLY for the representativeness check, never as the
    sampling key (24.7% land in 'other'; too crude to stratify on, fine to histogram-compare)."""
    q = q.lower()
    if re.search(r"what colou?r|which colou?r", q):
        return "color"
    if re.search(r"how many|number of|count", q):
        return "count"
    if re.search(r"\b(where|left|right|above|below|behind|front|next to|top|bottom|side)\b", q):
        return "spatial"
    if re.search(r"^is |^are |^does |^do |^can |^has |^have ", q):
        return "yes/no"
    if re.search(r"what (kind|type|is|are)|which", q):
        return "object/identity"
    if re.search(r"doing|action|activity|holding|wearing", q):
        return "action"
    if re.search(r"date|address|company|total|amount|invoice|document|title|percent|value", q):
        return "doc/OCR"
    return "other"


def _largest_remainder(strata_counts, target):
    """Proportional allocation of `target` across strata by their sizes, summing EXACTLY to target,
    capped by availability. Largest-remainder rounding."""
    total = sum(strata_counts.values())
    raw = {k: target * n / total for k, n in strata_counts.items()}
    floor = {k: min(int(v), strata_counts[k]) for k, v in raw.items()}
    used = sum(floor.values())
    # distribute the remainder to the largest fractional parts (that still have headroom)
    rem = sorted(strata_counts, key=lambda k: raw[k] - int(raw[k]), reverse=True)
    i = 0
    while used < target and i < len(rem) * 4:
        k = rem[i % len(rem)]
        if floor[k] < strata_counts[k]:
            floor[k] += 1
            used += 1
        i += 1
    return floor


def _stratified_draw(records, target, seed, strat_key="dataset"):
    """Return `target` records, allocated proportionally across strata, seeded & reproducible."""
    by = defaultdict(list)
    for r in records:
        by[r.get(strat_key)].append(r)
    counts = {k: len(v) for k, v in by.items()}
    alloc = _largest_remainder(counts, min(target, len(records)))
    rng = random.Random(seed)
    picked = []
    for k in sorted(by):
        pool = by[k][:]
        rng.shuffle(pool)
        picked.extend(pool[: alloc[k]])
    rng.shuffle(picked)
    return picked


def _dist(records, fn):
    c = Counter(fn(r) for r in records)
    tot = sum(c.values()) or 1
    return {k: v / tot for k, v in c.items()}, c


def _print_dist_compare(title, full_recs, sub_recs, fn):
    fp, _ = _dist(full_recs, fn)
    sp, sc = _dist(sub_recs, fn)
    keys = sorted(set(fp) | set(sp), key=lambda k: -fp.get(k, 0))
    print(f"\n  {title:<18} {'full%':>7} {'subset%':>8} {'Δpp':>6}  {'n_sub':>6}")
    worst = 0.0
    for k in keys:
        dpp = 100 * (sp.get(k, 0) - fp.get(k, 0))
        worst = max(worst, abs(dpp))
        print(f"    {str(k):<16} {100*fp.get(k,0):7.1f} {100*sp.get(k,0):8.1f} {dpp:+6.1f}  {sc.get(k,0):6d}")
    print(f"  max |Δpp| = {worst:.1f}")
    return worst


# ----------------------------------------------------------------- commands -----

def cmd_heldout(args):
    records = json.load(open(args.source))
    print(f"[heldout] loaded {len(records)} records from {args.source}")
    eligible = [r for r in records if _eligible(r)]
    print(f"[heldout] {len(eligible)} eligible")

    held = _stratified_draw(eligible, args.heldout_n, args.seed)
    print(f"[heldout] drew {len(held)} stratified held-out (seed={args.seed})")

    # contamination guard against the BASE checkpoint's training set
    if args.base_train_ids:
        base = {tuple(k) for k in json.load(open(args.base_train_ids))}
        overlap = [r for r in held if _key(r) in base]
        if overlap:
            raise SystemExit(
                f"[heldout] FAIL: {len(overlap)} held-out rows are in the BASE checkpoint's "
                f"training set — proportion-mediated would be a memorization score. "
                f"Exclude these from the base or draw the held-out from data the base never saw."
            )
        print(f"[heldout] OK: disjoint from base training ids ({len(base)} keys)")
    else:
        print("[heldout] WARNING: --base-train-ids NOT given. Held-out cleanliness against the "
              "base checkpoint is UNVERIFIED. If the base trained on any of these rows, the causal "
              "metric is contaminated. This is a CORRECTNESS gate, not a nicety.")

    # per-stratum counts — rare strata contribute ~0 effective N (T>eps); don't per-interpret them
    print("\n  held-out per source (rare strata are noise-only for per-attribute reads):")
    for k, v in sorted(Counter(r["dataset"] for r in held).items(), key=lambda x: -x[1]):
        flag = "  <- too few to interpret" if v < 5 else ""
        print(f"    {k:<16} {v:4d}{flag}")

    json.dump(held, open(args.out, "w"), indent=1, ensure_ascii=False)
    ids_out = args.out.replace(".json", "_ids.json")
    json.dump([list(_key(r)) for r in held], open(ids_out, "w"), indent=1)
    print(f"\n[heldout] wrote {len(held)} records -> {args.out}")
    print(f"[heldout] wrote ids            -> {ids_out}")


def _report_steps(n, epochs, global_batch, pack_factor):
    """Estimate optimizer steps for the continue-FT reroute and flag under/over-training.

    Packing decouples example-count from steps: ~global_batch * pack_factor instances land per
    optimizer step, so a 'few thousand, 1 epoch' run can be ~15 steps -> the reroute never develops
    and you get a false 'no effect'. Target ~200-450 steps for the bottleneck adaptation."""
    inst_per_step = global_batch * pack_factor
    steps_per_epoch = n / inst_per_step
    total = steps_per_epoch * epochs
    print(f"\n  ESTIMATED TRAINING STEPS  (the knob that governs whether the reroute develops)")
    print(f"    global_batch={global_batch} x pack_factor={pack_factor} -> ~{inst_per_step:.0f} instances/step")
    print(f"    {n} rows x {epochs} epoch(s) -> ~{steps_per_epoch:.0f} steps/epoch -> ~{total:.0f} TOTAL steps")
    if total < 200:
        print(f"    !! UNDERTRAINED: ~{total:.0f} < 200 steps. The reroute may not develop -> false 'no effect'.")
        print(f"       Fix: raise --n or --epochs to reach ~200-450 steps.")
    elif epochs > 2 and n < 20000:
        print(f"    !! OVERFIT risk: {epochs} epochs on a small slice ({n}) memorizes the slice -> corrupts the")
        print(f"       causal signal. Prefer a bigger slice at 1 epoch.")
    elif total > 800:
        print(f"    note: ~{total:.0f} steps exceeds the ~200-450 reroute band — fine, just longer than needed.")
    else:
        print(f"    OK: ~{total:.0f} steps is in the ~200-450 reroute target band.")
    print(f"    -> in the FT script set MAX_STEPS~={round(total)}  (or num_train_epochs={epochs}, MAX_STEPS=-1).")


def cmd_train(args):
    records = json.load(open(args.source))
    if args.heldout_in:
        held_ids = {tuple(k) for k in json.load(open(args.heldout_in.replace(".json", "_ids.json")))}
        print(f"[train] loaded {len(records)} records; held-out to exclude: {len(held_ids)}")
        pool = [r for r in records if _key(r) not in held_ids]
    else:
        held_ids = set()
        print(f"[train] loaded {len(records)} records; no --heldout-in given — train/val disjointness "
              f"relies on the split field (OK when the held-out is the val split).")
        pool = records

    subset = _stratified_draw(pool, args.n, args.seed)
    print(f"[train] drew {len(subset)} stratified training rows (seed={args.seed})")

    # fail-fast: eval must never appear in train (only checkable if an id-set was given)
    if held_ids:
        leak = {_key(r) for r in subset} & held_ids
        if leak:
            raise SystemExit(f"[train] FAIL: {len(leak)} training rows overlap the held-out set — leak.")
        print(f"[train] OK: training slice disjoint from held-out")

    # representativeness: source matches by construction; the claim to validate is source->attr-type
    _print_dist_compare("SOURCE (sanity)", records, subset, lambda r: r.get("dataset"))
    worst = _print_dist_compare("ATTR-TYPE (real check)", records, subset, lambda r: _attr_type(_question(r)))
    if worst > args.tol:
        print(f"  NOTE: attr-type max|Δpp|={worst:.1f} > tol={args.tol} — source is a weaker attr-type "
              f"proxy than hoped at this size; grow --n or eyeball which type skews.")

    _report_steps(len(subset), args.epochs, args.global_batch, args.pack_factor)

    json.dump(subset, open(args.out, "w"), ensure_ascii=False)
    print(f"\n[train] wrote {len(subset)} rows -> {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("heldout")
    h.add_argument("--source", default="data/lvr_data/viscot_363k_lvr_formatted.json")
    h.add_argument("--heldout-n", type=int, default=300)
    h.add_argument("--seed", type=int, default=1234)
    h.add_argument("--base-train-ids", default=None,
                   help="ids the BASE checkpoint trained on; asserts held-out disjoint (contamination guard)")
    h.add_argument("--out", default="data/lvr_data/slice_heldout.json")
    h.set_defaults(func=cmd_heldout)

    t = sub.add_parser("train")
    t.add_argument("--source", default="data/lvr_data/viscot_363k_lvr_formatted.json",
                   help="TRAIN-split data to slice (diverse, multi-source)")
    t.add_argument("--n", type=int, default=80000,
                   help="training subset size. ~80k (20%% of 363k) x1 epoch ~= 300-450 steps (reroute target)")
    t.add_argument("--seed", type=int, default=1234)
    t.add_argument("--heldout-in", default=None,
                   help="held-out ids to EXCLUDE. Optional: when the held-out is the VAL split, "
                        "train/val are disjoint by split and this isn't needed.")
    t.add_argument("--tol", type=float, default=3.0, help="max |Δpp| tolerance for attr-type match")
    t.add_argument("--epochs", type=int, default=1, help="epochs — feeds the step estimate + the FT script")
    t.add_argument("--global-batch", type=int, default=64,
                   help="batch_per_device * num_devices * grad_accum (Stage-1 7B default = 1*8*8 = 64)")
    t.add_argument("--pack-factor", type=float, default=3.0,
                   help="avg instances packed per batch element (1=no packing, ~3-4 for short VQA)")
    t.add_argument("--out", default="data/lvr_data/slice_train.json")
    t.set_defaults(func=cmd_train)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
