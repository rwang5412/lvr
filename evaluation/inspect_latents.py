"""Latent inspection: where is each latent "looking"?

Two views, both run on any checkpoint (read-only diagnostics, no training):

  #2 latent_nearest_patches  — what each latent ENCODES. For each <lvr> position, cosine-match its
     produced hidden state Z_i against every image-patch embedding and return the nearest patch
     (+ the supervised-target patch from lvr_tokens, for comparison). Cheap (one matmul). This is the
     linear probe made visual and per-position.

  #3 latent_attention_flow   — what each latent READS FROM. Runs the forward with output_attentions
     and returns, per latent, its attention distribution over the image patches (mean over heads,
     selected layers). WARNING: output_attentions materializes [B, H, L, L] per layer — run on a
     SHORT example (few hundred tokens), not the full 2600-token ones, or it will OOM.

Both return normalized [0,1] xyxy boxes for patches, so they're directly comparable to the record's
`bboxes` (same frame). Patch geometry: Qwen2.5-VL merges patch_size(14) x spatial_merge_size(2) =>
each token is a ~28px cell; the merged grid is (Hgrid//2, Wgrid//2).

    PYTHONPATH=. python evaluation/inspect_latents.py --checkpoint weights/LVR-7B \
        --image-folder /scratch/haizhow/lvr_images --heldout /scratch/haizhow/heldout_harness.json \
        --index 0 [--attention]
"""

import argparse

import torch
import torch.nn.functional as F

from evaluation.run_harness import load_model_and_processor, _forward, _to_device
from src.params import DataArguments
from src.harness import data as hdata


def _merged_grid(config, image_grid_thw):
    """(Hm, Wm) merged-token grid for a single image (t=1)."""
    merge = getattr(getattr(config, "vision_config", config), "spatial_merge_size", 2)
    t, h, w = [int(x) for x in image_grid_thw[0].tolist()]
    return h // merge, w // merge


def _patch_to_normbox(p, Hm, Wm):
    """Merged-token index -> normalized [x1,y1,x2,y2] (same frame as bboxes)."""
    row, col = p // Wm, p % Wm
    return [col / Wm, row / Hm, (col + 1) / Wm, (row + 1) / Hm]


def _image_patch_embeds(model, batch):
    emb = model.model.get_image_features(batch["pixel_values"], batch["image_grid_thw"])
    if isinstance(emb, (list, tuple)):
        emb = torch.cat(emb, dim=0)
    return emb.float()                                    # [n_patches, H]


# ---------------------------------------------------------------------------------- #2 nearest -----

@torch.no_grad()
def latent_nearest_patches(model, batch, config):
    """For each latent, the image patch its produced Z most resembles (+ the supervised target)."""
    out = _forward(model, batch, override_latent_embeds=None)   # captures latent_hidden_states
    Z = out.latent_hidden_states.float()                        # [L, H]
    img = _image_patch_embeds(model, batch)                     # [n_patches, H]

    sim = F.normalize(Z, dim=1) @ F.normalize(img, dim=1).t()   # [L, n_patches] cosine
    nearest = sim.argmax(dim=1)                                 # [L]
    nearest_sim = sim.max(dim=1).values                         # [L]
    target = torch.cat([t.flatten() for t in batch["lvr_tokens"]]).to(nearest.device)  # [L] supervised

    Hm, Wm = _merged_grid(config, batch["image_grid_thw"])
    rows = []
    for i in range(Z.shape[0]):
        npatch, tpatch = int(nearest[i]), int(target[i])
        rows.append({
            "latent": i,
            "nearest_patch": npatch,
            "nearest_box": _patch_to_normbox(npatch, Hm, Wm),
            "nearest_cos": float(nearest_sim[i]),
            "target_patch": tpatch,
            "target_box": _patch_to_normbox(tpatch, Hm, Wm),
            "cos_to_target": float(sim[i, tpatch]),
            "hits_target": npatch == tpatch,
        })
    hit_rate = sum(r["hits_target"] for r in rows) / len(rows)
    return {"grid": [Hm, Wm], "n_patches": img.shape[0], "hit_rate": hit_rate, "latents": rows}


# ------------------------------------------------------------------------------ #3 attention flow --

@torch.no_grad()
def latent_attention_flow(model, batch, config, layers=None):
    """Per-latent attention distribution over image patches (mean over heads, selected layers).

    WARNING: output_attentions materializes [B, H, L, L] per layer. Use a SHORT example.
    """
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        pixel_values=batch["pixel_values"],
        image_grid_thw=batch["image_grid_thw"],
        lvr_tokens=batch["lvr_tokens"],
        labels=None,
        output_attentions=True,
        return_dict=True,
    )
    atts = out.attentions                                       # tuple[n_layers] of [B, H, L, L]
    if atts is None:
        raise RuntimeError("model returned no attentions (output_attentions not honored)")

    ids = batch["input_ids"][0]
    lat_pos = (ids == config.lvr_id).nonzero(as_tuple=True)[0]
    img_pos = (ids == config.image_token_id).nonzero(as_tuple=True)[0]
    layers = list(range(len(atts))) if layers is None else layers

    flow = torch.zeros(len(lat_pos), len(img_pos))
    for l in layers:
        a = atts[l][0].mean(0)                                  # [L, L] mean over heads
        flow += a[lat_pos][:, img_pos].float().cpu()
    flow /= len(layers)                                         # [n_lat, n_img] each row sums ~1

    Hm, Wm = _merged_grid(config, batch["image_grid_thw"])
    peak = flow.argmax(dim=1)                                   # patch each latent attends to most
    rows = [{
        "latent": i,
        "peak_patch": int(peak[i]),
        "peak_box": _patch_to_normbox(int(peak[i]), Hm, Wm),
        "peak_weight": float(flow[i, peak[i]]),
    } for i in range(flow.shape[0])]
    return {"grid": [Hm, Wm], "n_image_tokens": len(img_pos), "layers": layers,
            "latents": rows, "flow": flow}                      # flow: [n_lat, n_img] full heatmap


# ------------------------------------------------------------------------------------- main --------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--heldout", default="data/lvr_data/heldout_harness.json")
    ap.add_argument("--index", type=int, default=0, help="which held-out example to inspect")
    ap.add_argument("--attention", action="store_true", help="also run #3 (needs a short example)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor, config = load_model_and_processor(args.checkpoint)
    data_args = DataArguments(image_folder=args.image_folder)
    records = hdata.load_records(args.heldout)
    ds = hdata.build_dataset(records[args.index:args.index + 1], processor, data_args, model_id=args.checkpoint)
    collator = hdata.build_collator(processor)
    batch = _to_device(hdata.collate_one(collator, ds[0]), device)

    near = latent_nearest_patches(model, batch, config)
    print(f"[#2 nearest-patch] grid={near['grid']}  n_patches={near['n_patches']}  "
          f"hit_rate(nearest==target)={near['hit_rate']:.2f}")
    for r in near["latents"]:
        print(f"  latent {r['latent']:2d}: nearest={r['nearest_patch']:4d} cos={r['nearest_cos']:.3f} "
              f"target={r['target_patch']:4d} cos_to_target={r['cos_to_target']:.3f} "
              f"{'HIT' if r['hits_target'] else '   '}  nearest_box={[round(x,3) for x in r['nearest_box']]}")

    if args.attention:
        flow = latent_attention_flow(model, batch, config)
        print(f"\n[#3 attention-flow] grid={flow['grid']}  n_image_tokens={flow['n_image_tokens']}")
        for r in flow["latents"]:
            print(f"  latent {r['latent']:2d}: peak patch={r['peak_patch']:4d} weight={r['peak_weight']:.3f} "
                  f"box={[round(x,3) for x in r['peak_box']]}")


if __name__ == "__main__":
    main()
