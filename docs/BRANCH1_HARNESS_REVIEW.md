# Branch 1 (`harness`) — Implementation Review

In-depth review of every change made to build the LVR causal-validation harness. Read alongside the
plan (`~/.claude/plans/here-is-a-design-validated-sunset.md`).

---

## 1. Context & scope

LVR emits latent "imagination" tokens that (per CapImagine) the answer causally **ignores** — the
model shortcuts to the visible image. The larger project adds a bottleneck + directed
counterfactual-swap loss to force `Z → Y` causality. **Branch 1 builds the measurement foundation
first**: the shared `get_spans` contract, an offline validation harness, and a baseline run that
must reproduce the disconnect. It is the acceptance test for Branches 2–4 and makes **no training
change** (the model additions are inert unless the harness passes its new kwargs).

Two decisions were locked with the user before implementation:
- **Causal metrics via held-out viscot + teacher-forced NLL** (not benchmark letter-accuracy).
- **Gating-slice-first build order**: `get_spans` → NIE + clean-ref → baseline → then probe /
  diversity / flip.

**Execution reality:** this dev machine has no `train` env, GPU, or checkpoint. Pure-Python pieces
(`get_spans`, its test, the split generator) were **run and verified here**; everything torch/model
was **syntax-checked (`py_compile`) only** and must be validated on Palmetto (§7).

---

## 2. Summary of all changes

| File | Type | What |
|---|---|---|
| `src/train/monkey_patch_forward_lvr.py` | **modified** | 4 additive changes to `qwen2_5_mixed_modality_forward_lvr` only |
| `src/harness/__init__.py` | new | package surface |
| `src/harness/spans.py` | new | `get_spans` — the shared segment-range contract |
| `src/harness/metrics.py` | new | NIE, diversity, probe (pure torch/numpy) |
| `src/harness/interventions.py` | new | latent-override builders (zero / mean / splice) |
| `src/harness/data.py` | new | held-out loading + partner pairing (reuses training dataset) |
| `evaluation/run_harness.py` | new | entry point: runs the battery, writes report |
| `evaluation/make_heldout_split.py` | new | carves the seeded held-out split (ran locally) |
| `data/lvr_data/heldout_harness.json` | new (generated) | 300 held-out records |
| `data/lvr_data/heldout_harness_ids.json` | new (generated) | exclusion keys for Branch 4 |
| `tests/test_spans.py` | new | Test #2 — span correctness (ran locally, green) |

No existing behavior changed: the model additions default to `None`/no-op, so training and the
existing `evaluation/evaluation.py` are unaffected.

---

## 3. The model edit (the one risky change) — in depth

All edits are confined to **`qwen2_5_mixed_modality_forward_lvr`** (the no-head training forward,
[monkey_patch_forward_lvr.py:123](../src/train/monkey_patch_forward_lvr.py#L123)). This is the
correct and only target because stage-1 checkpoints are `lvr_head=False`, `latent_end_token=False`
(verified in both `finetune_lvr_stage1_{3b,7b}.sh`). The other 7 forward variants are untouched.

**Change 1 — output dataclass fields** (`Qwen2_5_VLCausalLMOutputWithPast`, ~line 90):
```python
latent_hidden_states: Optional[torch.FloatTensor] = None   # model's produced Z, [L_total, H]
latent_target_embeds: Optional[torch.FloatTensor] = None   # ROI supervision targets, [L_total, H]
```
Both default `None`, so the other variants that construct this dataclass without them are unaffected.

**Change 2 — signature param** (~line 146):
```python
override_latent_embeds: Optional[torch.FloatTensor] = None,
```

**Change 3 — override the latent fill, right before the LM call** (~line 296):
```python
if override_latent_embeds is not None and lvr_tokens is not None:
    inputs_embeds[batch_indices, seq_positions] = override_latent_embeds.to(inputs_embeds.dtype)
```
Placed *after* the normal supervision fill and *before* `self.model.language_model(...)`, so it
cleanly replaces whatever is at the `<lvr>` positions. `batch_indices/seq_positions` come from
`torch.nonzero(input_ids == lvr_id)` computed earlier in the same function — the override's row order
must match that (documented in-code).

**Change 4 — capture Z and targets, after the LM call** (~line 316):
```python
latent_hidden_states = None
latent_target_embeds = None
if lvr_tokens is not None:
    latent_hidden_states = hidden_states[batch_indices, seq_positions - 1].detach()  # model's Z
    latent_target_embeds = selected_lvr_embeds.detach()                              # ROI target
```
`hidden_states[batch_indices, seq_positions - 1]` is **exactly the tensor `L_patch` supervises**
([the loss uses the same indices at line ~350](../src/train/monkey_patch_forward_lvr.py#L350)) — i.e.
the model's produced latent representation. Both captures are `.detach()`ed (diagnostic only).

**Why this is safe / disambiguation:** the fill site and signature are byte-identical across 6
variants, so edits were anchored on text unique to this function — the `language_model(...)` call
that closes **without** `**kwargs` (present only here; the inference variants add `**kwargs`), and a
capture→return span made unique by the new capture lines. Verified with `str.count(...) == 1` before
each edit. The file compiles.

---

## 4. The harness package

### `spans.py` — `get_spans(input_ids, labels, *, image_token_id, lvr_id)`
The shared contract. Returns `{image, question, latent, answer}` as half-open `Span(start, end)`.
- `image` = contiguous run of `image_token_id`; `latent` = contiguous run of `lvr_id` (the repeated
  `<|lvr|>` only — `<|lvr_start|>`/`<|lvr_end|>` are different ids and excluded); `answer` =
  contiguous `labels != -100` (the dataset masks everything else, verified at
  lvr_sft_dataset.py:302-308); `question` = the text between image and latent blocks.
- **Pure-Python core** (converts tensors→list of ints), so it has no torch/model dependency and its
  test runs anywhere. **Fail-fast**: raises `ValueError` on non-contiguous or missing segments —
  off-by-one must crash here, not leak downstream. `validate_spans(...)` asserts non-overlap,
  ordering, and optional count matches.

### `metrics.py` (pure torch/numpy — env has no sklearn/scipy)
- `answer_nll(logits, labels)` — mean teacher-forced CE over the answer span (standard causal shift,
  `ignore_index=-100`). Per-example.
- `proportion_mediated(clean, latent_corrupt, image_corrupt)` — NIE mediation: indirect effect
  `M = NLL(latent-corrupt) − NLL(clean)` over total effect `T = NLL(image-corrupt) − NLL(clean)`,
  averaged only where `T > eps`. Also returns raw `M`, `T`.
- `directed_flip_scores(...)` — partner-answer NLL drop when the partner's latents are spliced in
  (mean drop + flip rate).
- `effective_rank`, `participation_ratio`, `avg_pairwise_cosine_distance` — latent spread (§7.2).
- `linear_probe_r2(Z, targets)` — ridge least-squares `Z→ROI`, in-sample R² (richness).

### `interventions.py`
`zero_latents`, `mean_replace_latents(n, mean_vector)`, `align_partner_latents(partner, target_n)`
(truncate/tile for whole-set splice). These build the `[L, H]` tensors fed to `override_latent_embeds`.

### `data.py`
Reuses `SupervisedDatasetLVR` + `DataCollatorForSupervisedDatasetLVR` so assembled examples are
identical to training. `build_partner_pairs` groups by normalized gold answer and pairs each example
with a **different-answer** partner (same-answer pairs excluded — they'd teach nothing; the
Branch-4 mined map is the training-time version). `gold_answer_text` strips `<lvr>/<answer>` tags.

### `run_harness.py` (entry point)
Installs the training forward, asserts `lvr_head=False`/`latent_end_token=False` (fail-fast, §3),
then per example runs up to three forwards and aggregates. See §5 for the methodology.

---

## 5. Methodology — how each metric is computed (and its caveats)

**Everything reduces to "score a fixed answer under a chosen set of latents."** The forward is run
teacher-forced (`labels=None` to the model; NLL computed in-harness from `logits` + the batch labels
over `get_spans().answer`). The three mediation conditions:

| Condition | pixel_values | `override_latent_embeds` | Measures |
|---|---|---|---|
| **clean** | real | `None` (supervision ROI latents) | reference NLL; also captures `Z`, `targets` |
| **latent-corrupt** (M) | real | mean-vector or zeros | answer's dependence on latent content |
| **image-corrupt** (T) | zeroed | `None` | total reliance on the visible image |

`proportion_mediated = M/T`. **Baseline expectation ≈ 0**: if the answer shortcuts to the image,
corrupting latents (M) barely moves NLL while corrupting the image (T) moves it a lot.

- **Latent diversity** (§7.2): effective-rank / participation-ratio / pairwise-cosine over all
  captured `Z` (`latent_hidden_states`). **Target diversity** (§7.1 reference): same over the ROI
  `targets`. Low rank ⇒ collapse.
- **Linear probe**: `linear_probe_r2(Z, targets)` — does the model's latent encode the supervised ROI?
- **Directed flip-to-target**: rebuild example *i*'s prompt+latents with **partner *j*'s answer**
  teacher-forced, then score *j*'s answer NLL under (a) *i*'s clean latents vs (b) *j*'s latents
  spliced in. Causal latents ⇒ splicing lowers *j*'s answer NLL. Guards against a gate (we score the
  *specific* partner answer, not just "changed").

**Faithfulness caveats (documented honestly):**
1. **"Clean" latents = supervision ROI embeds**, not the model's autoregressively-generated `Z` used
   at deployment. Because `L_patch` trains the model's produced `Z` toward exactly these targets, the
   two are close for a trained checkpoint; and for the *baseline disconnect* the conclusion
   (proportion ≈ 0) is invariant to which one is "clean." A deployed-circuit upgrade (inject the
   autoregressive `Z` captured from a `generate` pass) is noted for later and needs no new model edit
   beyond capturing during the decoding loop.
2. **Image-corruption zeroes pixels**, which also zeroes the latent-fill in the *image-corrupt*
   condition — this is correct for the mediation *total effect* (image + its downstream), which is
   what `T` should be.
3. **`mean` corruption is two-pass** (pass 1 captures targets → global mean; pass 2 scores
   latent-corrupt). `zero` is single-pass.

---

## 6. Data — the held-out split

`make_heldout_split.py` carved a **seeded, deterministic** 300-record split from
`viscot_363k_lvr_formatted.json` (404,120 records; all single-image, single-bbox, with `<lvr>`/
`<answer>`). Filter requires **exactly one bbox** (the forward's `enumerate(lvr_tokens)` couples the
bbox-group index to the batch index, so multi-bbox misaligns at batch=1; single-bbox is also the §4
default). Outputs: `heldout_harness.json` (records) and `heldout_harness_ids.json` (the
`(dataset, split, question_id)` keys **Branch 4 must exclude** from the swap-training subset to keep
this set genuinely held-out for arm comparisons).

---

## 7. Verification status — verified vs must-validate-on-Palmetto

**Verified locally (ran, green):**
- `tests/test_spans.py` — 6 cases: exact spans, non-overlap/order, list/nested inputs, varied sizes,
  and fail-fast on non-contiguous / missing segments. **All pass.** This is Test #2, the shared
  contract.
- `make_heldout_split.py` — produced the 300-record split + exclusion keys.
- `py_compile` on **all** changed/added files — clean.

**Must be validated on Palmetto (needs checkpoint + GPU) — NOT yet run:**
- The model edit's runtime behavior (override actually replaces latent embeds; captures have the
  right shape/order).
- End-to-end `run_harness.py` (forward calls, collator keys, device movement, NLL numerics).
- **The done-test**: proportion-mediated ≈ 0 on the current un-fine-tuned checkpoint.
- The mediation sign/magnitude sanity: latent effect `M` small, image effect `T` clearly positive.

Nothing model-dependent has been executed; treat §5's numeric behavior as designed-and-compiled, not
empirically confirmed.

---

## 8. How to run (on Palmetto)

```bash
# 1) (already done locally, regenerate if desired) build the held-out split
python evaluation/make_heldout_split.py --n 300 --seed 1234

# 2) span contract test — should be green anywhere
python tests/test_spans.py

# 3) baseline harness run (the done-test) on the current LVR checkpoint
PYTHONPATH=. python evaluation/run_harness.py \
    --checkpoint /path/to/current_lvr_checkpoint \
    --heldout data/lvr_data/heldout_harness.json \
    --image-folder /path/to/images \
    --out evaluation/harness_report --corruption mean --limit 300

# 4) clean letter-accuracy guard (existing script, separate)
PYTHONPATH=. python evaluation/evaluation.py   # vstar/blink/mmvp
```
Expected read at baseline: `proportion mediated ≈ 0`, and report latent effective-rank so any
**pre-existing collapse** is visible before Branches 2–4.

---

## 9. The five silent-bug tests — status

1. **Bottleneck sanity** — Branch 2 (not this branch).
2. **Spans correct** — ✅ `tests/test_spans.py`, ran green locally.
3. **Partners differ** — enforced in `build_partner_pairs` (same-answer excluded); to re-confirm on
   real data during the flip run.
4. **Latents detached** — swap is Branch 4; here all captures are `.detach()`ed.
5. **Held-out corruption ≠ training corruption** — ✅ by construction: harness uses mean/zero;
   Branch-4 training uses cross-input swap.

---

## 10. Assumptions, fail-fast points, limitations

- **Checkpoint must be `lvr_head=False`, `latent_end_token=False`** — else `run_harness` raises
  `NotImplementedError` (the override/capture edits live only in the no-head forward). Extending to
  head checkpoints = the same 4 edits in `..._with_head`.
- **Single image, single bbox per example** — enforced by the split filter.
- **Batch size 1** in the harness (per-example NLL). Simpler and exact; slower but fine for ~300.
- **`proportion_mediated` is NaN** for examples where `T ≤ eps` (image doesn't matter) — excluded and
  counted, not silently dropped.
- **Probe R² is in-sample** (train-fit ridge). For a stricter richness estimate, add a train/val
  split of the latents later.
- **"Clean" = supervision latents**, not deployed autoregressive `Z` (§5 caveat 1).

---

## 11. Not in this branch (later)

Bottleneck attention mask (Branch 2 — needs SDPA, already defaulted on), partner mining +
target-diversity gating (Branch 3), collator + two-pass swap loss (Branch 4), `L_spread` (Branch 5).
The `get_spans` contract and the `override_latent_embeds` / capture hooks are the seams those
branches plug into.
