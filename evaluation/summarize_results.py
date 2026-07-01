"""Summarize saved evaluation results into permanent summary files.

`evaluation.py` writes per-sample predictions to JSON in each dataset's
`out_dir`, but the accuracy figures are only ever `print()`ed to the terminal.
This script recomputes those figures from the saved per-sample JSONs and writes
them to disk as `summary.json` (structured) and `summary.txt` (human-readable,
mirroring the terminal format), so every run self-documents its metrics.

It reuses the exact `accuracy_reward` logic from `evaluation.py`, so the numbers
match the terminal output. It needs no GPU and no re-inference — it only reads
the JSON files already on disk, so it can summarize past runs too.

Usage:
    python evaluation/summarize_results.py <path> [<path> ...] [--print]

    <path>   a specific out_dir, or any parent directory to walk recursively
             (defaults to the current directory). Every directory containing
             per-sample result JSONs gets its own summary.json + summary.txt.
    --print  also echo each summary to stdout.
"""

import argparse
import json
import os
import re


def accuracy_reward(response, ground_truth):
    """Identical to evaluation.py:accuracy_reward — kept in sync deliberately."""
    given_answer = response.split('<answer>')[-1]
    given_answer = given_answer.split('</answer')[0].strip()
    if " " in given_answer:
        given_answer = given_answer.split(" ")[0]
    if len(given_answer) > 1:
        given_answer = given_answer[0]
    return given_answer == ground_truth


def _step_sort_key(filename):
    """Sort step files (e.g. steps004.json) by their numeric step count."""
    nums = re.findall(r"\d+", os.path.basename(filename))
    return (int(nums[0]) if nums else -1, filename)


def _is_result_list(obj):
    """A per-sample result file is a non-empty list of dicts with prediction+label."""
    return (
        isinstance(obj, list)
        and len(obj) > 0
        and isinstance(obj[0], dict)
        and "prediction" in obj[0]
        and "label" in obj[0]
    )


def summarize_file(path):
    """Return (total, correct, by_category) for one per-sample result JSON."""
    with open(path) as f:
        result = json.load(f)

    total = 0
    correct = 0
    by_category = {}
    for res in result:
        pred = res["prediction"]
        pred = pred[0] if isinstance(pred, list) else pred
        ok = bool(accuracy_reward(pred, res["label"]))
        total += 1
        correct += int(ok)

        cat = res.get("category")
        if cat is not None:
            bucket = by_category.setdefault(cat, {"total": 0, "correct": 0})
            bucket["total"] += 1
            bucket["correct"] += int(ok)

    return total, correct, by_category


def summarize_dir(out_dir):
    """Summarize every result JSON in out_dir; return a summary dict or None."""
    result_files = []
    for name in sorted(os.listdir(out_dir)):
        if not name.endswith(".json") or name == "summary.json":
            continue
        fpath = os.path.join(out_dir, name)
        try:
            with open(fpath) as f:
                obj = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if _is_result_list(obj):
            result_files.append(fpath)

    if not result_files:
        return None

    result_files.sort(key=_step_sort_key)

    per_file = {}
    categories = []
    for fpath in result_files:
        key = os.path.splitext(os.path.basename(fpath))[0]
        total, correct, by_category = summarize_file(fpath)
        per_file[key] = {
            "total": total,
            "correct": correct,
            "accuracy": (correct / total * 100.0) if total else 0.0,
            "by_category": {
                cat: {
                    "total": c["total"],
                    "correct": c["correct"],
                    "accuracy": (c["correct"] / c["total"] * 100.0) if c["total"] else 0.0,
                }
                for cat, c in by_category.items()
            },
        }
        for cat in by_category:
            if cat not in categories:
                categories.append(cat)

    return {"out_dir": os.path.abspath(out_dir), "per_file": per_file, "categories": sorted(categories)}


def render_txt(summary):
    """Render a human-readable summary mirroring the terminal output format."""
    lines = [summary["out_dir"], ""]
    per_file = summary["per_file"]

    for key, m in per_file.items():
        lines.append(f"{key} - Accuracy: {m['correct']}/{m['total']} = {m['accuracy']:.2f}")

    lines.append("")
    lines.append("Overall accuracy by file:")
    lines.append(",".join(f"{m['accuracy']:.2f}" for m in per_file.values()))

    if summary["categories"]:
        lines.append("")
        for cat in summary["categories"]:
            per_file_acc = []
            for m in per_file.values():
                if cat in m["by_category"]:
                    per_file_acc.append(f"{m['by_category'][cat]['accuracy']:.2f}")
            lines.append(f"{cat}," + ",".join(per_file_acc))

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", default=["."], help="out_dir(s) or parent dir(s) to walk")
    parser.add_argument("--print", dest="echo", action="store_true", help="also print summaries to stdout")
    args = parser.parse_args()

    summarized = 0
    for root in args.paths:
        if not os.path.isdir(root):
            print(f"skip (not a directory): {root}")
            continue
        for dirpath, _dirnames, _filenames in os.walk(root):
            summary = summarize_dir(dirpath)
            if summary is None:
                continue

            with open(os.path.join(dirpath, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            txt = render_txt(summary)
            with open(os.path.join(dirpath, "summary.txt"), "w") as f:
                f.write(txt)

            summarized += 1
            print(f"wrote summary: {os.path.join(dirpath, 'summary.txt')}")
            if args.echo:
                print(txt)

    if summarized == 0:
        print("No result JSONs found. Point this at a dataset's out_dir or a parent of the results tree.")
    else:
        print(f"Done. Summarized {summarized} director{'y' if summarized == 1 else 'ies'}.")


if __name__ == "__main__":
    main()
