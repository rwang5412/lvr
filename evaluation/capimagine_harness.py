"""CapImagine-style do(Z) ACCURACY harness (Finding 2) — matches the paper's Z->Y intervention.

Corrupt the latent tokens during FREE generation, then measure the change in task ACCURACY (not NLL),
exactly as the paper does. Four interventions matched to the empirical latent mean/std (no OOD shift):

    identical      : force every latent to one shared tensor (the mean latent)     [collapse]
    gauss_add      : inject Gaussian noise into the latent                          [perturb]
    gauss_replace  : replace the latent entirely with N(mu, sigma)                  [destroy]
    near_zero      : set the latent to a small value ~0                             [erase]

Runs on the SAME benchmark datasets as the paper (V*, HR-Bench, MME-RealWorld) via evaluation.py's
loaders (multiple-choice, letter-scored), and on an in-domain gqa held-out (free-form) for an
apples-to-apples tie to the causal NLL harness. Only VALID-latent instances (emitted <|lvr_start|>)
are scored; N is reported. Per-checkpoint: run on base / bottleneck / distill and compare deltas.

  # benchmark (paper-style) — datasets download from HF, so PRE-CACHE on the login node first:
  PYTHONPATH=.:./src python evaluation/capimagine_harness.py \
      --checkpoint <ckpt> --dataset hrbench_4k --limit 300 --use-bottleneck 0 --out evaluation/cap_bn_hr4k
  # gqa (in-domain, free-form):
  PYTHONPATH=.:./src python evaluation/capimagine_harness.py \
      --checkpoint <ckpt> --dataset gqa --records data/lvr_data/heldout_val_clean.json \
      --image-folder /scratch/haizhow/vcot_dl --limit 300 --out evaluation/cap_bn_gqa
"""

import argparse
import importlib.util
import json
import os
import re

import torch
from transformers import AutoConfig, AutoProcessor
from qwen_vl_utils import process_vision_info

from src.model.qwen_lvr_model import QwenWithLVR
from src.train.monkey_patch_forward_lvr import replace_qwen2_5_with_mixed_modality_forward_lvr

LVR_START, LVR, LVR_END, LVR_LATENT_END = "<|lvr_start|>", "<|lvr|>", "<|lvr_end|>", "<|lvr_latent_end|>"
STRATEGIES = ["identical", "gauss_add", "gauss_replace", "near_zero"]
MC_TASK_INSTRUCTION = "\nAnswer with the option's letter from the given choices directly."
BENCH_LOADERS = {  # ds_name -> evaluation.py loader fn name; all return (data,image_dir,out_dir,ds_name)
    "vstar": "load_vstar_dataset", "mmvp": "load_mmvp_dataset", "blink": "load_blink_dataset",
    "hrbench_4k": "load_hrbench_4k_dataset", "hrbench_8k": "load_hrbench_8k_dataset",
    "mme_realworld": "load_mme_realworld_dataset",
}


def _eval_module():
    """Load evaluation.py by path (evaluation/ isn't a package, so a plain import clashes)."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation.py")
    spec = importlib.util.spec_from_file_location("lvr_evaluation_mod", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------------------ model / generation --
def load_model_and_processor(chkpt, use_bottleneck):
    config = AutoConfig.from_pretrained(chkpt)
    replace_qwen2_5_with_mixed_modality_forward_lvr(inference_mode=True, lvr_head=config.lvr_head)
    model = QwenWithLVR.from_pretrained(
        chkpt, config=config, trust_remote_code=True,
        torch_dtype="auto", attn_implementation="sdpa", device_map="auto",
    ).eval()
    model.config.use_bottleneck = bool(use_bottleneck)   # match the eval regime explicitly (like run_harness)
    print(f"[capimagine] use_bottleneck={model.config.use_bottleneck}")
    return model, AutoProcessor.from_pretrained(chkpt)


def generate(model, processor, image, q, lvr_steps, latent_intervention=None):
    # `image` may be a file path OR a PIL image (benchmarks embed PIL) — process_vision_info handles both.
    msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": q}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msgs)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=512, decoding_strategy="steps",
                             lvr_steps=[lvr_steps], latent_intervention=latent_intervention)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
    return processor.batch_decode(trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]


# ------------------------------------------------------------------------------ do(Z) callbacks -----
def make_capturer(store):
    def fn(h, mode):
        idx = mode.nonzero(as_tuple=True)[0]
        if idx.numel():
            store.append(h[idx].detach().float().cpu())
        return h
    return fn


def make_corruptor(strategy, mu, sigma, eps=1e-2):
    def fn(h, mode):
        m = mode.view(-1, 1).to(h.dtype)
        mu_d, sig_d = mu.to(h.device, h.dtype), sigma.to(h.device, h.dtype)
        if strategy == "identical":
            new = mu_d.expand_as(h)
        elif strategy == "near_zero":
            new = torch.full_like(h, eps)
        elif strategy == "gauss_replace":
            new = mu_d + torch.randn_like(h) * sig_d
        elif strategy == "gauss_add":
            new = h + torch.randn_like(h) * sig_d
        else:
            raise ValueError(strategy)
        return h * (1 - m) + new * m
    return fn


# ------------------------------------------------------------------------------ scoring -------------
def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def _pred_text(out):
    mm = re.search(r"<answer>(.*?)</answer>", out, re.S)
    if mm:
        return mm.group(1).strip()
    a = out
    for t in (LVR_START, LVR, LVR_LATENT_END, LVR_END, "<|im_end|>", "<|endoftext|>"):
        a = a.replace(t, "")
    return a.strip()


def score(out, gold, is_mc):
    pred = _pred_text(out)
    if is_mc:
        # take the first A-D letter the model produced; compare to the gold letter (evaluation.py logic)
        m = re.search(r"[A-Da-d]", pred)
        return (m.group(0).upper() == str(gold).strip().upper()[:1]) if m else False
    p, g = _norm(pred), _norm(gold)
    return bool(p) and bool(g) and (p == g or g in p or p in g)   # lenient free-form (gqa)


# ------------------------------------------------------------------------------ datasets ------------
def load_items(args):
    """Return a list of unified items: {image, question, gold, is_mc}. Two kinds:
    - gqa: free-form, from a records JSON (+ --image-folder).
    - benchmarks: multiple-choice, via evaluation.py loaders (download from HF; pre-cache on login)."""
    if args.dataset == "gqa":
        recs = json.load(open(args.records))
        items = []
        for r in recs:
            img = r["image"][0] if isinstance(r.get("image"), list) else r.get("image")
            q = ""
            for c in r.get("conversations", []):
                if c.get("from") == "human":
                    v = c["value"]
                    q = v.split("\n", 1)[-1].strip() if v.startswith("<image>") else v.replace("<image>", "").strip()
            gold = ""
            for c in r.get("conversations", []):
                if c.get("from") == "gpt":
                    mm = re.search(r"<answer>(.*?)</answer>", c["value"], re.S)
                    gold = (mm.group(1) if mm else c["value"]).strip()
            items.append({"image": os.path.join(args.image_folder, img), "question": q, "gold": gold, "is_mc": False})
        return items

    # benchmark: reuse evaluation.py's loader (returns data,image_dir,out_dir,ds_name)
    ev = _eval_module()
    data, image_dir, _out, ds_name = getattr(ev, BENCH_LOADERS[args.dataset])(False, "capimagine", "steps")
    items = []
    for d in data:
        img = d["image"]
        if isinstance(img, str) and image_dir:            # filename -> full path (e.g. MMVP)
            img = os.path.join(image_dir, img)
        elif isinstance(img, list):                       # some benches wrap the image in a list
            img = img[0]
        items.append({"image": img, "question": d["query"] + MC_TASK_INSTRUCTION, "gold": d["label"], "is_mc": True})
    return items


# ------------------------------------------------------------------------------ main ----------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True, choices=["gqa"] + list(BENCH_LOADERS))
    ap.add_argument("--records", help="gqa records JSON (only for --dataset gqa)")
    ap.add_argument("--image-folder", default="", help="image root for gqa")
    ap.add_argument("--use-bottleneck", type=int, default=0, help="1 = apply answer->image mask at eval")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--lvr-steps", type=int, default=16)
    ap.add_argument("--out", default="evaluation/capimagine_report")
    args = ap.parse_args()

    model, processor = load_model_and_processor(args.checkpoint, args.use_bottleneck)
    items = load_items(args)[: args.limit]
    print(f"[capimagine] dataset={args.dataset}  items={len(items)}  lvr_steps={args.lvr_steps}")

    # ---- Phase 0: clean generation — capture latents, clean accuracy, valid-latent filter ----
    store = []
    capt = make_capturer(store)
    clean = []
    for i, it in enumerate(items):
        out = generate(model, processor, it["image"], it["question"], args.lvr_steps, latent_intervention=capt)
        clean.append({"valid": LVR_START in out, "correct": score(out, it["gold"], it["is_mc"]), "it": it})
        if (i + 1) % 20 == 0:
            print(f"[capimagine] clean {i + 1}/{len(items)}")

    valid = [i for i, r in enumerate(clean) if r["valid"]]
    N = len(valid)
    if N == 0:
        raise SystemExit(f"No valid-latent instances on {args.dataset} (no <|lvr_start|>). do(Z) is moot here.")
    Z = torch.cat(store, dim=0)
    mu, sigma = Z.mean(0), Z.std(0)
    clean_acc = sum(clean[i]["correct"] for i in valid) / N
    print(f"[capimagine] valid-latent N={N}/{len(items)} | clean acc={clean_acc:.4f} | latents={tuple(Z.shape)}")

    # ---- Phase 1-4: do(Z) interventions on the valid instances ----
    results = {}
    for strat in STRATEGIES:
        corr = make_corruptor(strat, mu, sigma)
        n_correct = flip_to_wrong = flip_to_right = 0
        for i in valid:
            r = clean[i]
            out = generate(model, processor, r["it"]["image"], r["it"]["question"], args.lvr_steps, latent_intervention=corr)
            c = score(out, r["it"]["gold"], r["it"]["is_mc"])
            n_correct += int(c)
            flip_to_wrong += int(r["correct"] and not c)
            flip_to_right += int((not r["correct"]) and c)
        acc = n_correct / N
        results[strat] = {"acc": acc, "delta": acc - clean_acc,
                          "flip_to_wrong": flip_to_wrong, "flip_to_right": flip_to_right}
        print(f"[capimagine] {strat:14s} acc={acc:.4f}  Δ={acc-clean_acc:+.4f}  (→wrong {flip_to_wrong}, →right {flip_to_right})")

    report = {"checkpoint": args.checkpoint, "dataset": args.dataset, "use_bottleneck": bool(args.use_bottleneck),
              "n_total": len(items), "n_valid_latent": N, "lvr_steps": args.lvr_steps,
              "clean_accuracy": clean_acc, "interventions": results}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 66)
    print(f"CapImagine do(Z) ACCURACY — {args.dataset}  use_bottleneck={bool(args.use_bottleneck)}")
    print(f"{args.checkpoint}")
    print(f"valid-latent N = {N}/{len(items)}   clean acc = {clean_acc:.4f}")
    print("-" * 66)
    for s in STRATEGIES:
        r = results[s]
        print(f"  {s:14s}  acc {r['acc']:.4f}   Δ {r['delta']:+.4f}   flips→wrong {r['flip_to_wrong']}")
    print("=" * 66)
    print("Read: Δ≈0 -> latents don't matter for the answer (the paper's finding).")
    print("      Δ<0 (accuracy DROPS under do(Z)) -> latents are load-bearing (the fix worked).")
    print(f"[capimagine] wrote {args.out}.json")


if __name__ == "__main__":
    main()
