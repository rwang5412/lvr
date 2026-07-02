"""`get_spans` — the single source of truth for LVR segment token ranges.

Returns the (start, end) index ranges of the four segments in one assembled example:

    image     — the Qwen image-placeholder tokens (== config.image_token_id)
    question  — the instruction/prompt text between the image block and the latent block
    latent    — the repeated <|lvr|> tokens (== config.lvr_id); <|lvr_start|>/<|lvr_end|> excluded
    answer    — the response span (contiguous labels != IGNORE_INDEX)

This is a SHARED CONTRACT. Branch 2's bottleneck masks answer→image using `image` and `answer`;
Branch 4's swap splices partner latents into `latent`. Off-by-one here fails *silently* (a leaking
mask or a mis-spliced latent), so this ships with `tests/test_spans.py`.

Design note — verified against the codebase:
- Token IDs live on `model.config`: `image_token_id`, `lvr_id` (set in train_lvr.py:172-185).
- The dataset masks system/image/question/latent to IGNORE_INDEX and keeps only the response
  (lvr_sft_dataset.py:302-308), so the answer span is exactly the contiguous non-ignored region.
- Latent positions are `input_ids == lvr_id` (monkey_patch_forward_lvr.py:220-223); the count equals
  `sum(len(g) for g in lvr_tokens)`.

The core is deliberately pure-Python (operates on lists of ints), so `input_ids`/`labels` may be a
torch tensor, a numpy array, or a plain list — and the correctness test runs without torch/GPU.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Union

IGNORE_INDEX = -100  # mirrors src/constants.py; kept local so this module has no torch/model deps


@dataclass(frozen=True)
class Span:
    """A half-open token range [start, end). `end` is exclusive."""

    start: int
    end: int

    def __post_init__(self):
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid Span(start={self.start}, end={self.end})")

    def __len__(self) -> int:
        return self.end - self.start

    def indices(self) -> range:
        return range(self.start, self.end)

    def as_tuple(self):
        return (self.start, self.end)


def _to_int_list(seq: Union[Sequence[int], "TensorLike"]) -> List[int]:
    """Accept a torch tensor / numpy array / list and return a flat python list of ints."""
    if hasattr(seq, "tolist"):  # torch.Tensor or np.ndarray
        seq = seq.tolist()
    flat = list(seq)
    if flat and isinstance(flat[0], list):  # a [1, L] nested list → take the row
        if len(flat) != 1:
            raise ValueError("get_spans expects a single example (1-D sequence), got a batch.")
        flat = flat[0]
    return [int(x) for x in flat]


def _contiguous_run(indices: List[int], name: str) -> Span:
    """Assert `indices` form one contiguous block and return it as a Span. Fail fast otherwise."""
    if not indices:
        raise ValueError(f"get_spans: no {name} tokens found in the sequence.")
    start, end = indices[0], indices[-1] + 1
    if indices != list(range(start, end)):
        raise ValueError(
            f"get_spans: {name} tokens are not contiguous "
            f"(found {len(indices)} tokens spanning [{start}, {end})). "
            f"get_spans supports one contiguous {name} block per example."
        )
    return Span(start, end)


def get_spans(
    input_ids: Union[Sequence[int], "TensorLike"],
    labels: Union[Sequence[int], "TensorLike"],
    *,
    image_token_id: int,
    lvr_id: int,
    ignore_index: int = IGNORE_INDEX,
) -> Dict[str, Span]:
    """Compute image/question/latent/answer spans for one assembled example.

    Args:
        input_ids: 1-D token ids (tensor / array / list) for a single example.
        labels: 1-D labels aligned to input_ids; only the answer span is != ignore_index.
        image_token_id: `model.config.image_token_id`.
        lvr_id: `model.config.lvr_id` (the repeated <|lvr|> token).
        ignore_index: label value marking masked (non-answer) tokens; -100 by convention.

    Returns:
        {"image": Span, "question": Span, "latent": Span, "answer": Span}.

    Raises:
        ValueError if the sequence violates the expected layout (non-contiguous image/latent/answer,
        missing segment, or interleaving) — fail fast; downstream masks/splices depend on exactness.
    """
    ids = _to_int_list(input_ids)
    labs = _to_int_list(labels)
    if len(ids) != len(labs):
        raise ValueError(f"input_ids ({len(ids)}) and labels ({len(labs)}) length mismatch.")

    image = _contiguous_run([i for i, t in enumerate(ids) if t == image_token_id], "image")
    latent = _contiguous_run([i for i, t in enumerate(ids) if t == lvr_id], "latent")
    answer = _contiguous_run([i for i, l in enumerate(labs) if l != ignore_index], "answer")

    # Question = the instruction text between the image block and the latent block. Everything else
    # in the prompt (system/role tokens) is neither image nor latent, and the bottleneck keeps all
    # non-image prompt tokens open to the answer, so this range is the useful "answer→question" one.
    if not (image.end <= latent.start):
        raise ValueError(
            f"get_spans: expected image block before latent block, got image={image.as_tuple()} "
            f"latent={latent.as_tuple()}."
        )
    question = Span(image.end, latent.start)

    # Layout sanity: latent precedes the answer, and the answer is the tail of the sequence.
    if not (latent.end <= answer.start):
        raise ValueError(
            f"get_spans: expected latent block before answer, got latent={latent.as_tuple()} "
            f"answer={answer.as_tuple()}."
        )

    return {"image": image, "question": question, "latent": latent, "answer": answer}


def validate_spans(
    spans: Dict[str, Span],
    *,
    seq_len: int,
    n_lvr_tokens: int = None,
    n_image_tokens: int = None,
) -> None:
    """Assert the span contract holds. Used by the span-correctness test (Test #2) and callers.

    Checks: the four segments are non-overlapping, ordered image < question ≤ latent < answer, the
    answer ends the sequence, and (optionally) latent/image token counts match expectations.
    """
    image, question, latent, answer = spans["image"], spans["question"], spans["latent"], spans["answer"]

    ordered = [("image", image), ("question", question), ("latent", latent), ("answer", answer)]
    for (na, a), (nb, b) in zip(ordered, ordered[1:]):
        if a.end > b.start:
            raise AssertionError(f"spans overlap / misordered: {na}={a.as_tuple()} then {nb}={b.as_tuple()}")

    if answer.end > seq_len:
        raise AssertionError(f"answer end {answer.end} exceeds seq_len {seq_len}")

    if n_lvr_tokens is not None and len(latent) != n_lvr_tokens:
        raise AssertionError(f"latent span has {len(latent)} tokens, expected {n_lvr_tokens}")
    if n_image_tokens is not None and len(image) != n_image_tokens:
        raise AssertionError(f"image span has {len(image)} tokens, expected {n_image_tokens}")
