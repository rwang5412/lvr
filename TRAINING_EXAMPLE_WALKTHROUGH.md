# One Training Example, Line by Line

This traces a **single Stage-1 SFT training step** end to end, on one concrete data row, citing every function in the call path by `file:line`. Companion to [CODEBASE_EXPLAINED.md](CODEBASE_EXPLAINED.md) (which explains the *ideas*); this explains the *mechanics* of one step.

Concrete numbers used throughout: 7B model (hidden = **3584**), the Stage-1 script settings (`MIN_TOKEN=128`, `MAX_TOKEN=5120`, `loss_lvr_fct=mse`, `λ=0.1`, `lvr_head=False`).

---

## The training data point

This is the row we follow — a real example from the README ([README.md:62-79](README.md#L62-L79)), one object out of the ~438k-row Visual-CoT JSON list that `--data_path` points at:

```json
{
  "dataset": "flickr30k",
  "split": "train",
  "question_id": 31593,
  "image": ["viscot/flickr30k/2618322793.jpg"],
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nCan you describe the lower apparel of the child on the swing?\nProvide a short and direct response."
    },
    {
      "from": "gpt",
      "value": "<lvr>\n<answer> The child on the swing is wearing dark blue denim shorts. </answer>"
    }
  ],
  "bboxes": [[0.382, 0.456, 0.718, 0.656]]
}
```

What each field does:

| Field | Role | Consumed by code? |
|---|---|---|
| `dataset`, `split`, `question_id` | provenance / metadata | ignored by the loader |
| `image` | path(s) relative to `--image_folder` (list, but a bare string is also accepted) | ✅ loaded into pixels |
| `conversations` | LLaVA turns: `human`→user (the prompt it reads), `gpt`→assistant (the target it learns to produce) | ✅ |
| `bboxes` | one `[x_min, y_min, x_max, y_max]` per `<lvr>`, **normalized to [0,1]** | ✅ drives slot count **and** the MSE targets |

The two placeholders are what the pipeline rewrites:
- **`<image>`** (in the *human* turn — it's input you give) → `<|vision_start|>` + N×`<|image_pad|>` + `<|vision_end|>`.
- **`<lvr>`** (in the *gpt* turn — it's behavior you want produced) → `<|lvr_start|>` + N×`<|lvr|>` + `<|lvr_end|>`.

**What this row resolves to** (derived below; the soft numbers depend on the real image dimensions, assumed ~500×333):

| Quantity | Value | Where it's computed |
|---|---|---|
| Resized image | 504 × 336 | Step 1a |
| Image-token grid | 18 × 12 = **216** tokens | Step 1b |
| Bbox → grid cells | cols 6–12 × rows 5–7 = 7×3 = **21** | Step 1b |
| `idxs` (lvr targets) | `[96–102, 114–120, 132–138]` | Step 1b |
| `<\|lvr\|>` slots emitted | **21** | Step 1c |

`idxs` (those 21 numbers) is the spine of the whole example — computed once, reused to size the `<|lvr|>` block, to pick the MSE targets, and to place them.

---

## The call path

```
Trainer loop
 └─ DataLoader → SupervisedDatasetLVR.__getitem__(i)      ← builds ONE example   [Step 1]
       ├─ get_image_info()            (load+resize image)
       ├─ bbox_to_token_idxs_manual() (box → 21 indices)
       └─ llava_to_openai_lvr()       (expand <image>/<lvr> placeholders)
 └─ DataCollatorForSupervisedDatasetLVR.__call__(batch)   ← pads + flattens      [Step 2]
 └─ QwenLVRSFTTrainer.compute_loss(model, inputs)         ← calls model + adds   [Step 3]
       └─ qwen2_5_mixed_modality_forward_lvr(**inputs)    ← the heart            [Step 4]
```

---

## Step 1 — `__getitem__` ([lvr_sft_dataset.py:207-350](src/dataset/lvr_sft_dataset.py#L207-L350))

### 1a. Load & resize the image

- **210** `sources = self.list_data_dict[i]` — grabs our one JSON dict.
- **215** `if "image" in sources:` — true; sets `grid_key="image_grid_thw"`, `pixel_key="pixel_values"` (**217-218**).
- **220** `image_files = sources["image"]` → `["viscot/flickr30k/2618322793.jpg"]`.
- **223-224** wraps a bare string in a list (here already a list, no-op).
- **228-232** loop over files; **229-231** prepends `--image_folder` if the path isn't absolute/URL; **232** `get_image_info(...)` does the actual load+resize → appends a PIL image to `images`.

**`get_image_info` ([data_utils.py:120-144](src/dataset/data_utils.py#L120-L144))**
- **125-130** builds a Qwen "content" dict with `min_pixels=100352`, `max_pixels=4014080` (the `128/5120 × 28²` from the script).
- **132-134** would pin an exact width/height if `--image_resized_*` were set; they aren't, so skipped.
- **142** `process_vision_info(messages)` — Qwen's util runs `smart_resize` (rounds to multiples of 28, clamps pixel count to the min/max) and returns the resized PIL image → our 504×336.
- **144** returns it.

### 1b. Box → token indices

- **240** `bboxes = sources['bboxes']` → `[[0.382, 0.456, 0.718, 0.656]]`.
- **246** `lvr_token_idxs_list_manual = self.bbox_to_token_idxs_manual(images, bboxes)`.

**`bbox_to_token_idxs_manual` ([lvr_sft_dataset.py:131-204](src/dataset/lvr_sft_dataset.py#L131-L204))**
- **149** `token_idx_list = []` — accumulates one index-array per (image, box) pair.
- **150** `for img, bbox in zip(images, bboxes):` — one iteration here.
- **159** `patch_size = 14`.
- **160-161** `image_width=504, image_height=336` (the *resized* dims — that's why normalized boxes are mandatory; see the warning at **151-157**).
- **163-164** `grid_height = 336//14 = 24`, `grid_width = 504//14 = 36` (raw 14px patches).
- **166-167** `token_grid_height = 24//2 = 12`, `token_grid_width = 36//2 = 18` (after the 2×2 merge → the 216-token grid). *Note: it divides by `temporal_patch_size` (=2), numerically equal to `merge_size` here.*
- **170** `x1,y1,x2,y2 = 0.382, 0.456, 0.718, 0.656`.
- **171-175** the "if any coord > 1, divide by W/H" fallback — **not** triggered (ours are already normalized).
- **178-179** clamp to `[0,1]` — no-op here.
- **183** `token_x1 = int(0.382*18) = 6`.
- **184** `token_y1 = int(0.456*12) = 5`.
- **185** `token_x2 = min(ceil(0.718*18), 18) = 13`.
- **186** `token_y2 = min(ceil(0.656*12), 12) = 8`.
- **189-192** "at least one token" guards — not needed (box is non-empty).
- **197-201** double loop builds 1-D indices `token_idx = y*18 + x` for `y∈{5,6,7}, x∈{6..12}` → `[96,97,98,99,100,101,102, 114,…,120, 132,…,138]` = **21 indices**.
- **202** `np.array(token_indices)` appended.
- **204** returns `[array(21 indices)]`.

### 1c. Expand the placeholders into real tokens

- **251** `sources = copy.deepcopy(llava_to_openai_lvr(sources['conversations'], is_video=False, lvr_token_idxs_list=lvr_token_idxs_list_manual))`.

**`llava_to_openai_lvr` ([data_utils.py:57-73](src/dataset/data_utils.py#L57-L73))**
- **64** loops the two turns.
- **65** `replace_image_tokens(value)` — on the **human** turn.
- **66** `replace_lvr_tokens(value, idxs_list, None, None)` — on the **gpt** turn.
- **67-70** remaps `human→user`, `gpt→assistant`.

**`replace_image_tokens` ([data_utils.py:23-31](src/dataset/data_utils.py#L23-L31))** — **28-29** regex-replaces `<image>` (and its surrounding `\n`) with `<|vision_start|><|image_pad|><|vision_end|>`. (Just **one** `<|image_pad|>` for now — the processor multiplies it to 216 in Step 1d.)

**`replace_lvr_tokens` ([data_utils.py:33-53](src/dataset/data_utils.py#L33-L53))**
- **35-36** detects the `<lvr>` placeholder.
- **37** `input_segments = split(LVR_PLACEHOLDER)[1:]` — text *after* each `<lvr>` (one segment: `"\n<answer> … </answer>"`).
- **39** `fixed_num_of_lvr_tokens is None` → take the **else** branch (count-the-box mode).
- **45** `for seg, idxs in zip(input_segments, lvr_token_idxs_list):` — pairs our one segment with the 21-index array.
- **46** `latent_end_token is None` → **49** `replacement = LVR_START + LVR_TOKEN*len(idxs) + LVR_END` = `<|lvr_start|>` + **21**×`<|lvr|>` + `<|lvr_end|>`.
- **50** prepends it to the segment; **51** joins.

After Step 1c the two turns are plain strings with all special tokens spelled out (image still a single `<|image_pad|>`).

### 1d. Tokenize + build labels (the masking loop)

- **253-257** init accumulator lists.
- **260-266** prepends the system turn `<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n`; **263** its labels are all `IGNORE_INDEX` (−100) — never trained on.
- **268** `for j in range(0, len(sources), 2):` — one (user, assistant) pair.
- **269-270** split into `user_input` / `gpt_response`.
- **272** wraps the user side: `"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"` — note it **includes the assistant header**, so everything up to "assistant\n" is prompt.
- **273** `gpt_response = f"{content}<|im_end|>\n"` — the part to be learned (our `<|lvr_start|>…<|lvr_end|>\n<answer>…</answer><|im_end|>\n`).
- **275** `if DEFAULT_IMAGE_TOKEN in user_input:` — `<|image_pad|>` is present → **276** `processor(text=[user_input], images=images, …)`. **This is where the single `<|image_pad|>` explodes into 216** (the processor reads `image_grid_thw=[1,24,36]`, computes `24*36/2² = 216`, and repeats the pad token). **277** `prompt_input_ids` now holds those 216; **278-279** stash `pixel_values` (shape `[864, 1176]`) and `image_grid_thw` (`[[1,24,36]]`).
- **294-295** (the `else`) would tokenize a text-only prompt — not taken here.
- **299** `response_input_ids = tokenizer(gpt_response)` — the 21 `<|lvr|>` ids + answer ids. (Special tokens already in vocab → one id each.)
- **301** `input_ids = cat([prompt_input_ids, response_input_ids])` → full sequence `[S≈260]`.
- **302-308** `labels = cat([IGNORE×len(prompt), response_input_ids])` — **prompt masked, response kept.** The 21 `<|lvr|>` ids sit in the kept region *for now* (CE-masked later in Step 4f).
- **310-311** append to the batch-of-turns lists.
- **315-316** `cat` everything → final `input_ids`, `labels`, both `[S]`, `long`.
- **321** `attention_mask = (input_ids > -1000000)` — all ones (no padding yet).

### 1e. Package the LVR targets + return

- **323-329** rebuilds `lvr_tokens` as a **list of lists of tensors**: `[[tensor([96,…,138])]]` (per-image, per-box grouping).
- **333-338** `data_dict = {input_ids, attention_mask, labels, lvr_tokens}`.
- **340-344** adds `pixel_values` (`[864,1176]`) and `image_grid_thw` (`[[1,24,36]]`).
- **350** returns the dict. **This is one fully-built example.**

---

## Step 2 — Collator `__call__` ([lvr_sft_dataset.py:358-417](src/dataset/lvr_sft_dataset.py#L358-L417))

(With packing on, batch-per-device=1, but the collation logic is the same — assume a small batch.)

- **368-378** loops examples; **373-375** collects `pixel_values` + `image_grid_thw`; **377-378** collects `input_ids`/`labels`.
- **383-385** `pad_sequence(..., padding_value=pad_token_id)` right-pads all `input_ids` to the longest → `[B, S_max]`.
- **387** `attention_mask = input_ids != pad_token_id` — now the pad positions are False.
- **388** pads `labels` with `IGNORE_INDEX` (so pads aren't trained on).
- **390** `lvr_tokens = [example['lvr_tokens'] …]` — list per example.
- **391** `lvr_tokens_all_local_indices = [tensor(idx) for group in lvr_tokens for idx in group]` — **flattens** to a flat list of index-tensors, one per box across the whole batch → `[tensor([96,…,138])]`.
- **393-406** assembles `data_dict` with batched `input_ids`, `labels`, `attention_mask`, `lvr_tokens`, `pixel_values` (`cat` along patches → `[ΣP, 1176]`), `image_grid_thw` (`cat` → `[B,3]`).
- **417** returns the batch dict — exactly the kwargs the forward expects.

---

## Step 3 — `compute_loss` ([lvr_trainer.py:225-254](src/trainer/lvr_trainer.py#L225-L254))

- **227-232** if packing, just logs batch/token counts.
- **234** `outputs = model(**inputs)` — invokes the monkey-patched forward (Step 4).
- **236-237** pulls `loss_ce` and `loss_lvr` off the custom output object (**238** `loss_mode_switch` is `None` in this config).
- **240** `if self.args.mode_switch_loss:` — false here.
- **243** `loss = loss_ce + loss_lvr_lambda * loss_lvr` with `loss_lvr_lambda=0.1` → **`loss = loss_ce + 0.1·loss_lvr`**.
- **246-251** logs the three components.
- **254** returns `loss` (the scalar autograd uses for `.backward()`).

---

## Step 4 — `qwen2_5_mixed_modality_forward_lvr` ([monkey_patch_forward_lvr.py:118-350](src/train/monkey_patch_forward_lvr.py#L118-L350))

*(This exact function is selected by [the dispatcher at lines 65-67](src/train/monkey_patch_forward_lvr.py#L65-L67) because the run is `coconut=True, lvr_head=False, not inference, not rl`.)*

### 4a. Signature & embed
- **137** `lvr_tokens` = our `[tensor([96,…,138])]` (training targets-by-index).
- **139-140** `lvr_mode_switch` / `last_position_hidden_state` are **inference-only** → `None` now.
- **143-147** default the `output_*`/`return_dict` flags.
- **149-150** `inputs_embeds = embed(input_ids)` → `[B, S, 3584]`. Every token (including the 216 image-pad and 21 lvr placeholders) gets its *text* embedding for now; the next steps overwrite the visual ones.

### 4b. Inference-only fills (skipped)
- **152-163** the two `inputs_embeds[lvr_mode_switch,-1,:] = …` lines only run at inference (`lvr_mode_switch` is None) → **skipped in training**. These are the "recycle my own thought" lines.

### 4c. Dummy-image guard (skipped)
- **167-177** only fires when a batch has *no* pixels (avoids a DeepSpeed hang); we have pixels → skipped.

### 4d. Real image features + scatter
- **179** `if pixel_values is not None:` — true.
- **181-182** `image_embeds = get_image_features(pixel_values, image_grid_thw)` → vision encoder + projector (both **frozen**) → `[216, 3584]`.
- **184-189** sanity check: #`<|image_pad|>` ids == #image features (216 == 216).
- **197** `image_mask = input_ids == image_token_id` → `[B,S]` with 216 Trues.
- **200-201** expands the mask to hidden width.
- **208** `inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)` — drops the 216 real image vectors into their slots.

### 4e. LVR training-wheel fill (the crux)
- **211** `if lvr_tokens is not None:` — true (training).
- **216** `total_tokens = image_mask.sum(dim=1)` → `tensor([216])` (image-token count per example).
- **220** `lvr_mask = input_ids == self.config.lvr_id` → `[B,S]` with 21 Trues.
- **223** `batch_indices, seq_positions = nonzero(lvr_mask, as_tuple=True)` — two length-21 tensors: which example, and the **absolute sequence positions** of the 21 slots.
- **225** `isinstance(lvr_tokens, list)` → true → the "extract from original image" branch.
- **228-230** `image_token_offsets = cumsum(pad(total_tokens,(1,0)))[:-1]` → `tensor([0])` (start offset of each example's block inside the concatenated `image_embeds`; one example → 0).
- **234-237** for each example, `global_lvr_token_indices = idxs + offset` → `[96,…,138] + 0`.
- **238** `cat` → `[L_total=21]`.
- **241** `selected_lvr_embeds = image_embeds[global_lvr_token_indices]` → **`[21, 3584]` — THE TARGETS** (the actual projected image-pieces over the child's legs).
- **244** `inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds` — the 21 `<|lvr|>` input slots now literally hold the **true** image-pieces (teacher forcing / "training wheels").

### 4f. Position ids / RoPE
- **255-286** computes `position_ids` via Qwen's `get_rope_index` (the 3-D M-RoPE for vision). Mechanical; unchanged from stock Qwen. **268-276** is the prefill path taken in training.

### 4g. Language model + heads
- **288-299** `outputs = self.model.language_model(inputs_embeds=inputs_embeds, …)` — the transformer stack runs on the assembled embeddings → `hidden_states` `[B,S,3584]`.
- **301** `hidden_states = outputs[0]`.
- **302** `last_position_hidden_state = outputs.last_hidden_state[:,-1,:]` — saved for inference recycling; unused in training loss.
- **303** `logits = lm_head(hidden_states)` → `[B,S,vocab]`.
- **305** `lvr_loss_fct = set_lvr_loss_fct("mse")` → returns `MSELoss()` ([lines 102-103](src/train/monkey_patch_forward_lvr.py#L102-L103)).

### 4h. Card 1 — cross-entropy (words)
- **310** `if labels is not None:` — true.
- **312** upcast logits to fp32.
- **314-315** standard causal shift: `shift_logits = logits[:, :-1]`, `shift_labels = labels[:, 1:]` (position *t* predicts token *t+1*).
- **317-319** flatten to `[(B·(S-1)), vocab]` and `[(B·(S-1))]`.
- **321** `shift_labels.masked_fill(shift_labels == lvr_id, IGNORE_INDEX)` — **the 21 `<|lvr|>` targets are removed from word-loss** (they aren't words). The system + human prompt were already −100 from Step 1d.
- **325** `loss_ce = CrossEntropyLoss()(shift_logits, shift_labels)` — graded only on `<answer> … dark blue denim shorts. </answer><|im_end|>`.

### 4i. Card 2 — LVR MSE (re-picturing), with the `−1` shift
- **329** `seq_positions_start = seq_positions - 1` — step **back one** from each lvr slot. Because position *p* predicts *p+1*, the thought that should *produce* slot *i* lives at *i−1*. For slot 0 that's `<|lvr_start|>`'s position; for later slots it's the previous lvr slot (which holds the true previous piece — teacher forcing).
- **331** `selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(float32)` → `[21, 3584]` — the model's **predicted** thoughts.
- **332** `selected_lvr_embeds` (the targets from line 241) → fp32.
- **334** `loss_lvr = MSELoss(predicted_thoughts, true_image_pieces)` → one scalar; gradients pull the model's pre-slot hidden states toward the real leg-region image-pieces.

### 4j. Return
- **340-349** packs `loss_ce`, `loss_lvr`, `logits`, `past_key_values`, `hidden_states`, `last_position_hidden_state` into the custom `Qwen2_5_VLCausalLMOutputWithPast`. (Note **341** `loss=` is commented out — the combination is deferred to the trainer.)

→ back to **Step 3, line 243**: `loss = loss_ce + 0.1·loss_lvr`, then `.backward()`. Only the **language model** updates; the vision tower + projector are frozen.

---

## The two facts that make it cohere

1. **`idxs` (21 numbers) is computed once in Step 1b and reused three times** — to size the `<|lvr|>` block (1c), to pick the MSE targets (4e:241), and implicitly to place them (4e:244).
2. **The `−1` shift (4i:329) is the hinge** between "the slot" and "the thought that should predict it" — identical alignment to how, at inference, the hidden state at *p−1* gets fed *into* slot *p* (the 4b lines that were skipped here).
