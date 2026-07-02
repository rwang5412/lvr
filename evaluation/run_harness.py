"""LVR causal-validation harness — entry point (Branch 1).

Runs the offline battery on a checkpoint over a held-out viscot split and writes a report:

    proportion mediated / NIE   — teacher-forced answer-NLL mediation (the disconnect metric)
    latent diversity            — effective rank + participation ratio + avg pairwise distance of Z
    target diversity (§7.1 ref) — same, over the ROI supervision targets
    linear probe R²             — how well Z linearly encodes the ROI target (richness)
    directed flip-to-target     — does splicing a partner's latents make the PARTNER answer likelier

Gating done-test: run on the current (un-fine-tuned) LVR checkpoint and confirm proportion-mediated
≈ 0 (dummies barely change the answer) — reproducing the CapImagine disconnect AND revealing any
pre-existing collapse.

Runs on Palmetto (needs the checkpoint + GPU). Clean letter-accuracy (the brittleness guard) is
produced separately by the existing evaluation/evaluation.py on vstar/blink — not re-implemented here.

Usage:
    PYTHONPATH=. python evaluation/run_harness.py \
        --checkpoint /path/to/lvr_checkpoint \
        --heldout data/lvr_data/heldout_harness.json \
        --image-folder /path/to/images \
        --out evaluation/harness_report \
        --corruption mean --limit 300
"""

import argparse
import json
import os

import torch

from src.params import DataArguments
from src.train.monkey_patch_forward_lvr import replace_qwen2_5_with_mixed_modality_forward_lvr
from src.model.qwen_lvr_model import QwenWithLVR
from transformers import AutoConfig, AutoProcessor

from src.harness.spans import get_spans
from src.harness import metrics
from src.harness import interventions as itv
from src.harness import data as hdata


# ----------------------------------------------------------------------------------- loading -----

def load_model_and_processor(checkpoint: str):
    """Load the checkpoint with the TRAINING forward installed (so latent positions are filled from
    lvr_tokens and our override/capture additions are live). Mirrors evaluation.py:169-187 but with
    inference_mode=False and the harness-edited forward.
    """
    config = AutoConfig.from_pretrained(checkpoint)

    # The override/capture edits live ONLY in qwen2_5_mixed_modality_forward_lvr (no head, no
    # latent-end). Fail fast if this checkpoint needs a different forward variant.
    if getattr(config, "lvr_head", False):
        raise NotImplementedError(
            "Harness override/capture is implemented for lvr_head=False checkpoints. "
            "Add the same 3 edits to qwen2_5_mixed_modality_forward_lvr_with_head for head checkpoints."
        )
    if getattr(config, "latent_end_token", False):
        raise NotImplementedError("Harness assumes latent_end_token=False (stage-1 default).")

    replace_qwen2_5_with_mixed_modality_forward_lvr(
        inference_mode=False, coconut=True, lvr_head=False,
        mode_switch_loss=False, latent_end_token=False,
    )
    model = QwenWithLVR.from_pretrained(
        checkpoint, config=config, torch_dtype="auto",
        attn_implementation="sdpa", device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(checkpoint)
    return model, processor, config


# --------------------------------------------------------------------------------- forwards ------

def _to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        elif isinstance(v, list):
            # lvr_tokens is a list of index tensors (collator output). They index image_embeds on the
            # model device inside the forward, so they must be moved too — else a device mismatch.
            out[k] = [x.to(device) if torch.is_tensor(x) else x for x in v]
        else:
            out[k] = v
    return out


@torch.no_grad()
def _forward(model, batch, override_latent_embeds=None, zero_image=False):
    """One teacher-forced forward. Returns the model output (logits + latent captures).
    labels are NOT passed to the model (we compute answer NLL ourselves), so it skips its own loss.
    """
    kwargs = dict(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        pixel_values=batch["pixel_values"],
        image_grid_thw=batch["image_grid_thw"],
        lvr_tokens=batch["lvr_tokens"],
        labels=None,
        override_latent_embeds=override_latent_embeds,
        return_dict=True,
    )
    if zero_image:
        kwargs["pixel_values"] = torch.zeros_like(batch["pixel_values"])
    return model(**kwargs)


# ------------------------------------------------------------------------------------- main -------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--heldout", default="data/lvr_data/heldout_harness.json")
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--out", default="evaluation/harness_report")
    ap.add_argument("--corruption", choices=["mean", "zero"], default="mean",
                    help="mediator (latent) corruption for NIE. mean = replace with dataset-mean latent.")
    ap.add_argument("--limit", type=int, default=None, help="cap number of held-out examples")
    ap.add_argument("--skip-flip", action="store_true", help="skip directed flip-to-target")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor, config = load_model_and_processor(args.checkpoint)
    image_token_id, lvr_id = config.image_token_id, config.lvr_id

    data_args = DataArguments(image_folder=args.image_folder)
    records = hdata.load_records(args.heldout)
    if args.limit:
        records = records[: args.limit]

    # Fail fast on a wrong --image-folder: the held-out records store relative paths, so a bad root
    # would fail every example identically. Check the first image resolves before the 300-run.
    first_img = os.path.join(args.image_folder, records[0]["image"][0])
    if not os.path.exists(first_img):
        raise FileNotFoundError(
            f"first held-out image not found: {first_img}\n"
            f"--image-folder ({args.image_folder}) probably points at the wrong root."
        )

    dataset = hdata.build_dataset(records, processor, data_args, model_id=args.checkpoint)
    collator = hdata.build_collator(processor)

    n = len(dataset)
    print(f"[harness] {n} held-out examples | corruption={args.corruption} | device={device}")

    nll_clean, nll_image_corrupt = [], []
    nll_latent_corrupt = [None] * n
    Z_all, T_all = [], []          # model latents Z, ROI targets T (both [L_i, H] per example)
    example_L = [0] * n            # latent count per example (for splice alignment)

    # -------- Phase 1: clean + image-corruption; capture Z and targets --------
    for i in range(n):
        batch = _to_device(hdata.collate_one(collator, dataset[i]), device)
        spans = get_spans(batch["input_ids"][0], batch["labels"][0],
                          image_token_id=image_token_id, lvr_id=lvr_id)

        clean = _forward(model, batch, override_latent_embeds=None)
        nll_clean.append(float(metrics.answer_nll(clean.logits, batch["labels"])))

        img = _forward(model, batch, override_latent_embeds=None, zero_image=True)
        nll_image_corrupt.append(float(metrics.answer_nll(img.logits, batch["labels"])))

        Z = clean.latent_hidden_states.float().cpu()
        T = clean.latent_target_embeds.float().cpu()
        Z_all.append(Z)
        T_all.append(T)
        example_L[i] = Z.shape[0]

        if args.corruption == "zero":
            over = itv.zero_latents(Z.shape[0], Z.shape[1], dtype=clean.logits.dtype, device=device)
            zc = _forward(model, batch, override_latent_embeds=over)
            nll_latent_corrupt[i] = float(metrics.answer_nll(zc.logits, batch["labels"]))

        if (i + 1) % 25 == 0:
            print(f"[harness] phase1 {i + 1}/{n}")

    Z_cat = torch.cat(Z_all, dim=0)
    T_cat = torch.cat(T_all, dim=0)

    # -------- Phase 2 (mean corruption): replace latents with the global mean target --------
    if args.corruption == "mean":
        mean_vec = T_cat.mean(dim=0).to(device)
        for i in range(n):
            batch = _to_device(hdata.collate_one(collator, dataset[i]), device)
            over = itv.mean_replace_latents(example_L[i], mean_vec)
            mc = _forward(model, batch, override_latent_embeds=over)
            nll_latent_corrupt[i] = float(metrics.answer_nll(mc.logits, batch["labels"]))
            if (i + 1) % 25 == 0:
                print(f"[harness] phase2 {i + 1}/{n}")

    # -------- Aggregate metrics --------
    mediation = metrics.proportion_mediated(nll_clean, nll_latent_corrupt, nll_image_corrupt)
    latent_div = {
        "effective_rank": metrics.effective_rank(Z_cat),
        "participation_ratio": metrics.participation_ratio(Z_cat),
        "avg_pairwise_cosine_distance": metrics.avg_pairwise_cosine_distance(Z_cat),
        "n_latents": int(Z_cat.shape[0]),
        "hidden_dim": int(Z_cat.shape[1]),
    }
    target_div = {
        "effective_rank": metrics.effective_rank(T_cat),
        "participation_ratio": metrics.participation_ratio(T_cat),
        "avg_pairwise_cosine_distance": metrics.avg_pairwise_cosine_distance(T_cat),
    }
    probe = {"r2_latent_to_roi": metrics.linear_probe_r2(Z_cat, T_cat)}

    report = {
        "checkpoint": args.checkpoint,
        "n_examples": n,
        "corruption": args.corruption,
        "clean_answer_nll_mean": float(sum(nll_clean) / n),
        "mediation": mediation,
        "latent_diversity": latent_div,
        "target_diversity_ref": target_div,
        "linear_probe": probe,
    }

    # -------- Directed flip-to-target (post-gating metric) --------
    if not args.skip_flip:
        report["directed_flip"] = _flip_to_target(
            model, collator, dataset, records, T_all, example_L,
            image_token_id, lvr_id, device,
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(report, f, indent=2)
    with open(args.out + ".txt", "w") as f:
        f.write(_render(report))
    print("\n" + _render(report))
    print(f"[harness] wrote {args.out}.json / .txt")


def _flip_to_target(model, collator, dataset, records, T_all, example_L,
                    image_token_id, lvr_id, device):
    """For each (i, j) with different answers: build example i's sequence but with j's answer
    teacher-forced, then score j's answer NLL under (a) i's clean latents and (b) j's latents spliced
    in. A causal latent → splicing j's latents lowers j's answer NLL.
    """
    pairs = hdata.build_partner_pairs(records)
    nll_clean, nll_splice = [], []
    skipped = 0
    for (i, j) in pairs:
        ex_i = dataset[i]
        ex_j = dataset[j]
        # j's answer token ids = ex_j.input_ids where ex_j.labels != IGNORE.
        j_ids = ex_j["input_ids"].tolist()
        j_labs = ex_j["labels"].tolist()
        j_answer = [t for t, l in zip(j_ids, j_labs) if l != -100]
        if not j_answer or example_L[j] == 0:
            skipped += 1
            continue

        # Rebuild example i with j's answer replacing i's answer span.
        spans_i = get_spans(ex_i["input_ids"], ex_i["labels"],
                            image_token_id=image_token_id, lvr_id=lvr_id)
        prompt_ids = ex_i["input_ids"][: spans_i["answer"].start].tolist()
        new_input_ids = torch.tensor(prompt_ids + j_answer, dtype=ex_i["input_ids"].dtype)
        new_labels = torch.tensor([-100] * len(prompt_ids) + j_answer, dtype=ex_i["labels"].dtype)
        flip_ex = dict(ex_i)
        flip_ex["input_ids"] = new_input_ids
        flip_ex["labels"] = new_labels

        batch = _to_device(hdata.collate_one(collator, flip_ex), device)
        clean = _forward(model, batch, override_latent_embeds=None)
        nll_clean.append(float(metrics.answer_nll(clean.logits, batch["labels"])))

        partner_targets = T_all[j].to(device)
        over = itv.align_partner_latents(partner_targets, example_L[i])
        spl = _forward(model, batch, override_latent_embeds=over)
        nll_splice.append(float(metrics.answer_nll(spl.logits, batch["labels"])))

    result = metrics.directed_flip_scores(nll_clean, nll_splice)
    result["skipped_pairs"] = skipped
    return result


def _render(report: dict) -> str:
    m = report["mediation"]
    ld = report["latent_diversity"]
    lines = [
        f"LVR HARNESS REPORT — {report['checkpoint']}",
        f"examples={report['n_examples']}  corruption={report['corruption']}",
        "",
        f"clean answer NLL (mean)      : {report['clean_answer_nll_mean']:.4f}",
        "-- causal (mediation) --",
        f"latent effect  (M, NLL rise) : {m['latent_effect']:.4f}",
        f"image  effect  (T, NLL rise) : {m['image_effect']:.4f}",
        f"proportion mediated (M/T)    : {m['proportion_mediated']:.4f}   "
        f"(defined on {m['n_defined']}/{m['n_total']})   [baseline expectation: ~0]",
        "-- latent representation --",
        f"effective rank               : {ld['effective_rank']:.2f}  of dim {ld['hidden_dim']}",
        f"participation ratio          : {ld['participation_ratio']:.2f}",
        f"avg pairwise cosine distance : {ld['avg_pairwise_cosine_distance']:.4f}",
        f"linear probe R² (Z→ROI)      : {report['linear_probe']['r2_latent_to_roi']:.4f}",
    ]
    if "directed_flip" in report:
        d = report["directed_flip"]
        lines += [
            "-- directed flip-to-target --",
            f"mean partner-NLL drop        : {d['mean_partner_nll_drop']:.4f}  (positive = causal)",
            f"flip rate                    : {d['flip_rate']:.3f}  over {d['n_pairs']} pairs "
            f"({d['skipped_pairs']} skipped)",
        ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
