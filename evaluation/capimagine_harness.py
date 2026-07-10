"""CapImagine-style do(Z) ACCURACY harness (Finding 2).

Replicates the paper's Z->Y intervention in ITS units — corrupt the latent tokens during FREE
generation, then measure the change in task ACCURACY (not NLL). Four interventions, matched to the
empirical latent mean/std so there's no OOD shift, exactly as the paper specifies:

    identical      : force every latent to one shared tensor (the mean latent)     [collapse]
    gauss_add      : inject Gaussian noise into the latent                          [perturb]
    gauss_replace  : replace the latent entirely with N(mu, sigma)                  [destroy]
    near_zero      : set the latent to a small value ~0                             [erase]

Only instances with a VALID latent trace (emitted <|lvr_start|>) are scored — do(Z) on a zero-latent
example is a no-op that would fake the paper's null result. N kept is reported.

This is a per-checkpoint tool: run it on base / bottleneck / distill and compare the deltas. The claim
is "after bottleneck/distill, corrupting Z hurts accuracy MORE" — i.e. latents became load-bearing.
Sensitive detection lives in run_harness.py (NLL); this is the accuracy story in the paper's terms.

    PYTHONPATH=.:./src python evaluation/capimagine_harness.py \
        --checkpoint /scratch/haizhow/ckpts/bottleneck_7b/checkpoint-833 \
        --image-folder /scratch/haizhow/vcot_dl \
        --records data/lvr_data/heldout_val_clean.json \
        --limit 100 --lvr-steps 16 --out evaluation/capimagine_bottleneck
"""

import argparse
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


# ------------------------------------------------------------------------------ model / generation --
def load_model_and_processor(chkpt):
    config = AutoConfig.from_pretrained(chkpt)
    replace_qwen2_5_with_mixed_modality_forward_lvr(inference_mode=True, lvr_head=config.lvr_head)
    model = QwenWithLVR.from_pretrained(
        chkpt, config=config, trust_remote_code=True,
        torch_dtype="auto", attn_implementation="sdpa", device_map="auto",
    ).eval()
    return model, AutoProcessor.from_pretrained(chkpt)


def _messages(img, q):
    return [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q}]}]


def generate(model, processor, img, q, lvr_steps, latent_intervention=None):
    msgs = _messages(img, q)
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
    """Records the latent hidden states (rows in latent mode) so we can fit mu/sigma. No modification."""
    def fn(h, mode):
        idx = mode.nonzero(as_tuple=True)[0]
        if idx.numel():
            store.append(h[idx].detach().float().cpu())
        return h
    return fn


def make_corruptor(strategy, mu, sigma, eps=1e-2):
    """Returns fn(h[B,H], mode[B]bool) -> h with latent-mode rows corrupted per `strategy`."""
    def fn(h, mode):
        m = mode.view(-1, 1).to(h.dtype)                       # [B,1] gate
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
def _gold(rec):
    if "answer" in rec:
        return rec["answer"]
    for c in rec.get("conversations", []):
        if c.get("from") == "gpt":
            mm = re.search(r"<answer>(.*?)</answer>", c["value"], re.S)
            return (mm.group(1) if mm else c["value"]).strip()
    return ""


def _pred(out):
    mm = re.search(r"<answer>(.*?)</answer>", out, re.S)
    if mm:
        return mm.group(1).strip()
    a = out
    for t in (LVR_START, LVR, LVR_LATENT_END, LVR_END, "<|im_end|>", "<|endoftext|>"):
        a = a.replace(t, "")
    return a.strip()


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def _correct(pred, gold):
    p, g = _norm(pred), _norm(gold)
    if not p or not g:
        return False
    return p == g or g in p or p in g          # lenient free-form match (gqa word answers)


def _question(rec):
    if "conversations" in rec:
        for c in rec["conversations"]:
            if c.get("from") == "human":
                v = c["value"]
                return v.split("\n", 1)[-1].strip() if v.startswith("<image>") else v.replace("<image>", "").strip()
    return rec.get("question", "")


def _image(rec):
    img = rec.get("image")
    return img[0] if isinstance(img, list) else img


# ------------------------------------------------------------------------------ main ----------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--lvr-steps", type=int, default=16)
    ap.add_argument("--out", default="evaluation/capimagine_report")
    args = ap.parse_args()

    print(f"[capimagine] loading {args.checkpoint}")
    model, processor = load_model_and_processor(args.checkpoint)
    recs = json.load(open(args.records))[: args.limit]
    print(f"[capimagine] {len(recs)} records | lvr_steps={args.lvr_steps}")

    # ---- Phase 0: clean generation — capture latents, clean accuracy, valid-latent filter ----
    store = []
    capt = make_capturer(store)
    clean = []
    for i, rec in enumerate(recs):
        img = os.path.join(args.image_folder, _image(rec))
        q, gold = _question(rec), _gold(rec)
        out = generate(model, processor, img, q, args.lvr_steps, latent_intervention=capt)
        clean.append({"valid": LVR_START in out, "pred": _pred(out), "gold": gold,
                      "correct": _correct(_pred(out), gold), "q": q, "img": img})
        if (i + 1) % 20 == 0:
            print(f"[capimagine] clean {i + 1}/{len(recs)}")

    valid = [i for i, r in enumerate(clean) if r["valid"]]
    N = len(valid)
    if N == 0:
        raise SystemExit("No valid-latent instances (no <|lvr_start|> emitted). do(Z) is moot here.")
    Z = torch.cat(store, dim=0)                       # [n_latents, H]
    mu, sigma = Z.mean(0), Z.std(0)
    clean_acc = sum(clean[i]["correct"] for i in valid) / N
    print(f"[capimagine] valid-latent N={N}/{len(recs)} | clean acc={clean_acc:.4f} | latents={Z.shape}")

    # ---- Phase 1-4: do(Z) interventions on the valid instances ----
    results = {}
    for strat in STRATEGIES:
        corr = make_corruptor(strat, mu, sigma)
        n_correct = flip_to_wrong = flip_to_right = 0
        for i in valid:
            r = clean[i]
            out = generate(model, processor, r["img"], r["q"], args.lvr_steps, latent_intervention=corr)
            c = _correct(_pred(out), r["gold"])
            n_correct += int(c)
            flip_to_wrong += int(r["correct"] and not c)
            flip_to_right += int((not r["correct"]) and c)
        acc = n_correct / N
        results[strat] = {"acc": acc, "delta": acc - clean_acc,
                          "flip_to_wrong": flip_to_wrong, "flip_to_right": flip_to_right}
        print(f"[capimagine] {strat:14s} acc={acc:.4f}  Δ={acc-clean_acc:+.4f}  "
              f"(→wrong {flip_to_wrong}, →right {flip_to_right})")

    report = {
        "checkpoint": args.checkpoint, "records": args.records,
        "n_total": len(recs), "n_valid_latent": N, "lvr_steps": args.lvr_steps,
        "clean_accuracy": clean_acc, "interventions": results,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 64)
    print(f"CapImagine do(Z) ACCURACY — {args.checkpoint}")
    print(f"valid-latent N = {N}/{len(recs)}   clean acc = {clean_acc:.4f}")
    print("-" * 64)
    for s in STRATEGIES:
        r = results[s]
        print(f"  {s:14s}  acc {r['acc']:.4f}   Δ {r['delta']:+.4f}   flips→wrong {r['flip_to_wrong']}")
    print("=" * 64)
    print("Read: Δ≈0 -> latents don't matter for the answer (the paper's finding).")
    print("      Δ<0 (accuracy DROPS under do(Z)) -> latents are load-bearing (the fix worked).")
    print(f"[capimagine] wrote {args.out}.json")


if __name__ == "__main__":
    main()
