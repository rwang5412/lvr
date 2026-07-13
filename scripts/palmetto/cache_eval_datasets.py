"""One-shot: download ALL benchmark datasets into the scratch HF cache AND dump each schema.

Run on the LOGIN node (has internet; compute nodes are air-gapped). It populates
$HF_HOME so evaluation.py can then run fully offline, and it prints the feature schema +
one example row of each dataset so the loaders can be pinned to the real field names.

    conda activate train
    export HF_HOME=/scratch/haizhow/cache/huggingface
    python scripts/palmetto/cache_eval_datasets.py 2>&1 | tee /scratch/haizhow/eval_schemas.txt

Then paste the printed schema block back so the vstar / MMVP loaders can be finalized.
"""

import os

from datasets import load_dataset


def dump(name, ds):
    print(f"\n===== {name}  (n={len(ds)}) =====", flush=True)
    try:
        print("features:", ds.features)
        ex = ds[0]
        print("example field types:", {k: type(v).__name__ for k, v in ex.items()})
        for k, v in ex.items():
            if k == "image":
                continue
            print(f"  {k!r}: {str(v)[:140]!r}")
    except Exception as e:
        print(f"  (could not introspect: {type(e).__name__}: {e})")


def try_load(name, fn):
    try:
        fn()
    except Exception as e:
        import traceback
        print(f"\n===== {name}  !! FAILED: {type(e).__name__}: {e} =====")
        traceback.print_exc()


print("HF_HOME =", os.environ.get("HF_HOME", "(unset!)"))

# --- already-working (confirm they're cached) --------------------------------------------------
try_load("hrbench", lambda: dump("hrbench_4k", load_dataset("DreamMr/HR-Bench", "hrbench_version_split")["hrbench_4k"]))
try_load("mme", lambda: dump("mme_realworld", (lambda d: d[list(d.keys())[0]])(load_dataset("yifanzhang114/MME-RealWorld-Lite"))))
try_load("blink", lambda: dump("blink_Counting", load_dataset("BLINK-Benchmark/BLINK", "Counting")["val"]))

# --- MMVP: needs wiring to HF (loader currently reads local /dockerx CSV) -----------------------
try_load("MMVP", lambda: dump("MMVP", (lambda d: d[list(d.keys())[0]])(load_dataset("MMVP/MMVP"))))

# --- vstar: two candidates. lmms-lab is Parquet (likely EMBEDDED images -> no image staging). ---
try_load("vstar_lmms", lambda: dump("vstar_lmms-lab", (lambda d: d[list(d.keys())[0]])(load_dataset("lmms-lab/vstar-bench"))))
try_load("vstar_craigwu", lambda: dump("vstar_craigwu", load_dataset("craigwu/vstar_bench")["test"]))

# craigwu images are loose files; grab the full snapshot and show its layout in case we need it.
try:
    from huggingface_hub import snapshot_download
    p = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    print(f"\n===== vstar craigwu snapshot =====\npath: {p}\ntop-level: {sorted(os.listdir(p))[:30]}")
    for sub in ("direct_attributes", "relative_position"):
        d = os.path.join(p, sub)
        if os.path.isdir(d):
            print(f"  {sub}/ -> {len(os.listdir(d))} files, e.g. {sorted(os.listdir(d))[:3]}")
except Exception as e:
    print(f"\n(snapshot_download failed: {type(e).__name__}: {e})")

print("\n=== DONE. All datasets cached to $HF_HOME; paste the schema blocks above. ===")
