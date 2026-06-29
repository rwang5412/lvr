# LVR Codebase, Explained (Friendly Edition)

This guide explains what this repository does and how it works, written for someone who is **new to AI models and this kind of code**. We'll define jargon as we go, use everyday analogies, and walk through what each part is actually doing and *why*.

The big question this guide keeps answering: **which parts are the normal, off-the-shelf AI model (Qwen2.5-VL), and which parts are the new idea this project adds (Latent Visual Reasoning, or "LVR")?**

---

## 1. What is this project, in plain English?

Imagine an AI that can look at a picture and answer questions about it ("What color is the sign in the corner?"). That's a **Vision-Language Model** (VLM). A popular one is called **Qwen2.5-VL** — it's the "off-the-shelf" model this project starts from.

The problem: these models often *glance* at an image but miss small details. They tend to "talk themselves into" an answer using words, instead of really looking closely at the relevant spot in the picture.

This project's idea — **Latent Visual Reasoning (LVR)** — is to teach the model to **pause and mentally re-picture the important part of the image before answering**, kind of like how you'd squint and visualize a detail in your mind's eye. The clever part: it does this re-picturing using the model's internal "thought vectors," not by generating words.

So the whole repository is: **take Qwen2.5-VL, add the ability to mentally re-picture image regions, and train it to do that well.**

---

## 2. A few concepts you need first

Before we compare "normal model" vs "new idea," here are three ideas everything rests on. Don't skip these — the rest of the guide builds on them.

**Embedding (a "meaning vector").** A computer can't work with the word "dog" or a patch of pixels directly. It first turns each one into a long list of numbers — say 3,584 numbers — called an *embedding*. Think of it as a point on a giant map where similar meanings sit close together: "dog" lands near "puppy" and far from "math." Both words *and* image-pieces get turned into embeddings.

**Hidden state (the model's "current thought").** The model processes a sequence of embeddings through many layers and, at each position, produces another vector called a *hidden state* — its summary of "what I'm thinking right here." Crucially, **a hidden state is the same shape (3,584 numbers) as an embedding.** Normally the model turns its latest hidden state into the next word and moves on. Remember this matching-shape fact — it's the trick the whole project depends on.

**Token.** A "token" is just one unit in the sequence — usually a word-piece, but in a VLM it can also be one little patch of the image. The model reads tokens and produces tokens, one at a time.

---

## 3. How the normal Qwen2.5-VL model works (the part LVR does NOT change)

Stock Qwen2.5-VL has **three parts**, like an assembly line:

1. **Vision Encoder** — the "eyes." It cuts the image into a grid of small squares (each 28×28 pixels, called *patches*) and turns each square into an embedding vector. So a photo becomes a few hundred "image tokens." *(In the code: called via `get_image_features`, [monkey_patch_forward_lvr.py:181](src/train/monkey_patch_forward_lvr.py#L181).)*

2. **Projector** (also called the "merger") — the "translator." The eyes and the language part speak slightly different number-languages. The projector converts the image vectors so they live in the **same map** as the word vectors. After this, an image-piece and a word are directly comparable points. *(In the code: [:208](src/train/monkey_patch_forward_lvr.py#L208).)*

3. **Language Model (LLM)** — the "brain." It takes the combined stream of `[image tokens] + [word tokens]` and predicts the answer one token at a time. *(In the code: [:288](src/train/monkey_patch_forward_lvr.py#L288).)*

**The single most important consequence:** because the projector puts images and words on the *same map*, and a hidden state lives on that same map too, **a hidden state can stand in for an image-piece.** That's the door LVR walks through.

One more thing: in this project the **eyes and translator are "frozen"** — their settings are locked and never change during training. Only the **brain** is trained. ("Frozen" literally means "don't update these numbers." Set by `--freeze_vision_tower True --freeze_merger True`, enforced at [train_lvr.py:153-154](src/train/train_lvr.py#L153-L154).)

---

## 4. So what exactly does LVR add?

Here's the whole modification in a table. The left column is normal Qwen2.5-VL; the right is the new stuff.

| Topic | Normal Qwen2.5-VL | What LVR adds |
|---|---|---|
| Vocabulary | Normal words | 4 brand-new "marker" words (explained below) |
| Generating an answer | Always: think → pick a word → repeat | Can sometimes: think → **feed the thought back in** → repeat (no word!) |
| Training goal | "Predict the next word correctly" | That **plus** "make your thoughts match the important image region" |
| Extra brain parts | — | **None by default!** It reuses the brain's own thoughts directly |

That last row surprises people: the winning version of LVR adds *almost no new machinery*. It mostly **re-routes** what the model already produces. The repository looks big only because re-routing thoughts (instead of words) means rewriting the parts of the model that assumed "words in, words out."

---

## 5. The 4 new marker words

LVR invents four new "words" that don't mean anything in English — they're more like punctuation that tells the model what mode it's in. *(Defined in [constants.py:12-16](src/constants.py#L12-L16).)*

- `<|lvr_start|>` — "begin mentally re-picturing now."
- `<|lvr_end|>` — "stop re-picturing, go back to writing the answer."
- `<|lvr|>` — a single *blank slot* reserved for one piece of mental imagery. (A big region needs many of these in a row.)
- `<|lvr_latent_end|>` — a special "stop" signal used only by one experimental variant (more later).

A training example, written out, looks like this:

```
[image] What color is the sign?  <|lvr_start|> <|lvr|><|lvr|><|lvr|> <|lvr_end|>  <answer>red</answer>
```

Read it as: *here's the image and question; now re-picture the relevant region (using 3 slots); now give the answer.*

---

## 6. The three mechanisms that make LVR work

LVR is really three cooperating pieces. We'll call them (a), (b), and (c).

### (a) Create a "re-picturing zone" — the marker words

Two small jobs:
- **Teach the model the 4 new words** and remember their ID numbers, so the rest of the code can recognize them. *(In [train_lvr.py:172-190](src/train/train_lvr.py#L172-L190).)*
- **Decide how many blank `<|lvr|>` slots to lay down** for each example. (Covered in detail in Section 8 — it's based on how big the important region is.) *(In [data_utils.py:33-50](src/dataset/data_utils.py#L33-L50).)*

Think of (a) as drawing the empty boxes the model will later fill with mental imagery.

### (b) Re-picture by recycling thoughts — the heart of LVR

This is the actual "thinking in pictures." Recall from Section 2: a hidden state (a thought) is the same shape as an embedding (an input). So you can take the model's latest thought and **feed it straight back in as the next input**, skipping the usual "turn it into a word" step.

- **Normal mode:** thought → pick a word → feed that word back in.
- **Re-picturing mode:** thought → **feed the thought back in directly.**

When the model produces `<|lvr_start|>`, the code flips it into re-picturing mode for a few steps, recycling its own thoughts. Then it flips back and writes the answer. *(The recycle happens at [monkey_patch_forward_lvr.py:395](src/train/monkey_patch_forward_lvr.py#L395); the on/off switch logic is in the generation loop at [qwen_lvr_model.py:528-535](src/model/qwen_lvr_model.py#L528-L535).)*

**Important subtlety — training is different from real use:**
- During **real use (inference)**, the model recycles its *own* thoughts, as just described.
- During **training**, the model is given the *correct* image-pieces to fill those slots — like training wheels — so it can be graded on whether its thoughts were close to correct. *(Training-wheels filling at [monkey_patch_forward_lvr.py:220-244](src/train/monkey_patch_forward_lvr.py#L220-L244).)*

### (c) Teach it *what* to re-picture — the grading

Mechanism (b) is the engine, but how does the model learn to re-picture something *useful* instead of nonsense? That's (c), and it only happens during training.

Each training picture comes with a **bounding box** — a rectangle marking the exact region that answers the question (the sign in the corner). The training does this:

1. **Figure out which image-pieces are inside that box.** *(In [lvr_sft_dataset.py:131-204](src/dataset/lvr_sft_dataset.py#L131-L204).)*
2. **Compare the model's thoughts to those true image-pieces** using a "how far apart are these vectors?" measure (called MSE — Mean Squared Error). Closer = better. The model is nudged to make them closer over millions of examples. *(In [monkey_patch_forward_lvr.py:329-334](src/train/monkey_patch_forward_lvr.py#L329-L334).)*
3. **Don't penalize the re-picturing slots for not being words** — the model isn't supposed to say a word there, so those slots are excluded from the normal word-grading. *(In [:321-325](src/train/monkey_patch_forward_lvr.py#L321-L325).)*

So the model has **two report cards** that get added together: "did you say the right words?" (cross-entropy loss) plus "did you re-picture the right region?" (MSE loss), with a knob (λ = 0.1) controlling how much the second one counts. *(Added together at [lvr_trainer.py:243](src/trainer/lvr_trainer.py#L243).)*

**The key insight tying (b) and (c) together:** they act on the *same* `<|lvr|>` slots. In training those slots are filled with the correct image-pieces and graded (c); in real use they're filled with the model's own recycled thoughts (b). Same slots, two situations.

---

## 7. "Monkey patches" — why the code rewrites parts of Qwen at startup

A **monkey patch** is when code *replaces* a function inside a library at runtime, without editing the library's source files. This project uses several, because stock Qwen2.5-VL assumes "words in, words out," and LVR needs to bend that.

If you're ever debugging and the model behaves differently from normal Qwen, **look here first.**

| File | What it swaps in | Used when |
|---|---|---|
| [monkey_patch_forward_lvr.py](src/train/monkey_patch_forward_lvr.py) | The main brain-step: adds the re-picturing grading + thought-recycling | LVR training & testing |
| [monkey_patch_forward_lvr_rl.py](src/train/monkey_patch_forward_lvr_rl.py) | A version for the reinforcement-learning stage | Stage 2 (below) |
| [monkey_patch_forward.py](src/train/monkey_patch_forward.py) | A plain version with no LVR | The "baseline" comparison model |
| [monkey_patch_patch_emb.py](src/train/monkey_patch_patch_emb.py) | A numerical-stability fix for the eyes (prevents math errors that produce "NaN", i.e. broken numbers) | All training |
| [monkey_patch_dataloader.py](src/train/monkey_patch_dataloader.py) | A custom data-feeding routine (for "packing," explained below) | Most training |

---

## 8. How the model decides how many `<|lvr|>` slots to use

A natural question: how does it know to lay down 3 slots vs 30? **Answer: it counts how many image-pieces fall inside the bounding box.** Bigger important region → more slots.

Step by step *(all in [lvr_sft_dataset.py:131-204](src/dataset/lvr_sft_dataset.py#L131-L204))*:
1. Work out the grid of image-pieces the eyes will produce (image size ÷ patch size).
2. Map the bounding-box rectangle onto that grid → a smaller rectangle of grid cells.
3. Count the cells inside it. That count = the number of `<|lvr|>` slots. *(The slots get created at [data_utils.py:45-49](src/dataset/data_utils.py#L45-L49): `LVR_TOKEN * len(idxs)`.)*

This is why #slots always equals #targets: both come from the same list of grid cells. (There's also an optional "just use a fixed number every time" mode, but the default scripts use this count-the-box method.)

---

## 9. Training happens in two stages

### Stage 1: Supervised Fine-Tuning (SFT) — "learn the basic skill"
"Supervised" means we show the model the right answers and grade it. This stage uses both report cards from Section 6(c).

- Run by [train_lvr.py](src/train/train_lvr.py); script `scripts/finetune_lvr_stage1_7b.sh`.
- Data: **Visual-CoT**, ~438,000 picture-question pairs, each with a bounding box.
- Uses **data packing** — a speed trick that bundles several short examples into one batch so the GPU isn't wasted. *(In [lvr_sft_dataset_packed.py](src/dataset/lvr_sft_dataset_packed.py).)*

### Stage 2: Reinforcement Learning (RL) — "polish it by trial and error"
Here we stop giving the model bounding boxes. Instead it *tries* answers, and we **reward** good ones, letting it discover its own re-picturing style. The method is called **GRPO** (a popular RL recipe), adapted for LVR.

- Run by [train_grpo.py](src/train/train_grpo.py); script `scripts/finetune_lvr_stage2_3b.sh`. It starts from a Stage-1 model.
- **The tricky bit:** normal GRPO grades the *probability of each word*. But re-picturing steps don't produce words, so they have no probability. The fix (`score_with_lvr_replay`, [grpo_trainer.py:1324](src/trainer/grpo_trainer.py#L1324)) **re-plays the recorded thoughts** so the math still works, and the re-picturing steps are **left out of the grade** ([:757](src/trainer/grpo_trainer.py#L757)).
- **Rewards** ([reward_funcs.py](src/train/reward_funcs.py)): one point if the response used the `<|lvr_start|>…<|lvr_end|>…<answer>` format, one point if the answer is correct.

There's also a **vanilla SFT baseline** ([train_sft.py](src/train/train_sft.py)) — the *same* training but with LVR turned off — used to prove LVR actually helps.

---

## 10. When the model is actually used: 3 ways to decide when to stop re-picturing

A real difficulty: once the model starts re-picturing, *when should it stop?* The project tried three strategies. *(All in `generate()`, [qwen_lvr_model.py:100](src/model/qwen_lvr_model.py#L100).)*

- **Fixed number of steps** (`"steps"`) — "re-picture for exactly N steps (4, 8, or 16), then stop." Simple and **works best**. This is what testing uses.
- **Learned stop signal** (`"latent"`) — a trainable "I'm done" vector; stop when the thought gets close to it. **Unstable** — often fails to stop. *(Code: [qwen_lvr_model.py:583](src/model/qwen_lvr_model.py#L583).)*
- **Stop-prediction loss** — train the model to predict when to stop. **Didn't work.**

The lesson: the simplest approach (just count steps) won.

---

## 11. Testing the model (the evaluation script)

[evaluation/evaluation.py](evaluation/evaluation.py) measures how good the model is on standard quizzes (benchmarks named BLINK, V*, and MMVP).

**What it does:** loads a trained model and, for every quiz question, has it answer **three times** — once re-picturing for 4 steps, once for 8, once for 16. Then it checks each answer against the correct one and prints a score.

**What it produces:**
- **Files** of every answer it gave (saved as JSON, so you can inspect them later).
- **Printed accuracy tables** — the numbers you'd compare against the paper's results.

**Heads-up if you run it:** it has some hard-coded file paths (like `/dockerx/...`) and even some **leaked cloud passwords** ([lines 17-21](evaluation/evaluation.py#L17-L21)) baked into the file — these need to be removed/replaced before it'll run on your machine. It also does *not* save the model's internal thoughts, only its final answers.

---

## 12. The supporting cast (other folders)

- **Datasets** ([src/dataset/](src/dataset/)) — code that reads the picture-question files and turns them into model input. Includes the bounding-box-to-slots logic.
- **Bounding-box math** ([lvr_utils.py](src/lvr_utils.py)) — converts rectangles into grid-cell lists.
- **Saving models to the cloud** ([s3_checkpoints_lvr.py](src/s3_checkpoints_lvr.py)) — uploads checkpoints to online storage when enabled.
- **Settings** ([params.py](src/params.py)) — defines all the `--flags` you can pass to training.
- **Optional "head" parts** ([lvr_heads.py](src/model/lvr_heads.py)) — small extra layers tested in experiments but turned off in the best version (they made things slightly worse).

---

## 13. The whole thing in one picture

```
   Picture ─▶  Eyes (Vision Encoder) ─▶ Translator (Projector) ─┐      ← normal Qwen2.5-VL
                                                                ▼          (frozen, unchanged)
   Question ─▶ word vectors ─▶ [ image-pieces  +  words ]
                                                                │
                                                                ▼
                                                   Brain (Language Model)
                                                                │
                          normal mode:  ────────────────────────┴─▶ next word
                          re-picture mode:  thought ─▶ recycled back in   (b)   ┐
                                            graded against the real region (c)  │  ← the LVR additions
                                            marked off by <|lvr_start/end|> (a) ┘
```

Everything in the top two rows is normal, frozen Qwen2.5-VL. The only new things are: the marker words (a), recycling thoughts instead of words (b), and grading those thoughts against the right image region (c). **That's the entire invention — one idea, bolted cleanly onto a standard model.**

---

## 14. Where to find things (cheat sheet)

| If you want to understand… | Look at |
|---|---|
| The model and how it generates / stops | [src/model/qwen_lvr_model.py](src/model/qwen_lvr_model.py) |
| The brain-step: re-picturing + grading | [src/train/monkey_patch_forward_lvr.py](src/train/monkey_patch_forward_lvr.py) |
| Stage 1 training (learn the skill) | [src/train/train_lvr.py](src/train/train_lvr.py) |
| Stage 2 training (RL polish) | [src/train/train_grpo.py](src/train/train_grpo.py) + [src/trainer/grpo_trainer.py](src/trainer/grpo_trainer.py) |
| How the two report cards combine | [src/trainer/lvr_trainer.py](src/trainer/lvr_trainer.py) |
| How bounding boxes become slots/targets | [src/dataset/lvr_sft_dataset.py](src/dataset/lvr_sft_dataset.py) |
| The 4 marker words | [src/constants.py](src/constants.py) |
| Testing on benchmarks | [evaluation/evaluation.py](evaluation/evaluation.py) |
| Launch commands & GPU settings | [scripts/](scripts/) |

---

## Mini-glossary

- **VLM (Vision-Language Model):** an AI that takes images + text and produces text.
- **Embedding:** a list of numbers representing a word or image-piece's meaning.
- **Hidden state:** the model's internal "thought" at one position; same shape as an embedding.
- **Token:** one unit in the sequence (a word-piece or an image patch).
- **Latent:** "internal / hidden, not turned into words." LVR reasons in this internal space.
- **SFT (Supervised Fine-Tuning):** training by showing the model correct answers and grading it.
- **RL / GRPO:** training by rewarding good attempts rather than showing exact answers.
- **Loss:** a score of how wrong the model is; training tries to make it smaller.
- **MSE (Mean Squared Error):** a "how far apart are these two vectors" measure.
- **Frozen:** a part of the model whose numbers are locked and not trained.
- **Monkey patch:** replacing a library's function at runtime without editing its source.
- **Bounding box:** a rectangle marking the important region of an image.
- **Checkpoint:** a saved copy of the model's trained numbers.
```
