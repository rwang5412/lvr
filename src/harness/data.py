"""Held-out data + partner pairing for the harness.

Reuses the training dataset/collator so assembled examples are byte-identical to training:
`SupervisedDatasetLVR` (lvr_sft_dataset.py:25) turns a record into the data_dict
(input_ids/labels/lvr_tokens/pixel_values/image_grid_thw); `DataCollatorForSupervisedDatasetLVR`
pads a batch. The harness runs one example at a time (batch of 1).

The harness builds its OWN eval pairs for flip-to-target — no training collator needed. Partner =
another held-out example with a DIFFERENT gold answer (splicing same-answer latents would teach
nothing; see Branch-4 mining for the training-time version).
"""

import json
import re
from typing import Dict, List, Tuple

from src.dataset.lvr_sft_dataset import (
    SupervisedDatasetLVR,
    DataCollatorForSupervisedDatasetLVR,
)

_TAG_RE = re.compile(r"</?(lvr|answer|image)[^>]*>", re.IGNORECASE)


def load_records(path: str) -> List[dict]:
    with open(path, "r") as f:
        return json.load(f)


def gold_answer_text(record: dict) -> str:
    """The free-form gold answer, tags stripped — used only for the different-answer partner check."""
    gpt = record["conversations"][1]["value"]
    return _TAG_RE.sub("", gpt).strip()


def build_dataset(records, processor, data_args, model_id) -> SupervisedDatasetLVR:
    """Assemble held-out examples exactly as training does."""
    return SupervisedDatasetLVR(
        data_path=records, processor=processor, data_args=data_args, model_id=model_id
    )


def build_collator(processor) -> DataCollatorForSupervisedDatasetLVR:
    return DataCollatorForSupervisedDatasetLVR(pad_token_id=processor.tokenizer.pad_token_id)


def collate_one(collator, example: Dict) -> Dict:
    """Wrap a single dataset item into a padded batch-of-one ready for the forward."""
    return collator([example])


def build_partner_pairs(records: List[dict]) -> List[Tuple[int, int]]:
    """Pair each example index with a partner index whose gold answer DIFFERS.

    Simple deterministic pairing (harness-local, not the Branch-4 mined map): group indices by
    normalized answer text, then pair each index with the first index from any other group. Examples
    with no different-answer partner (degenerate single-answer set) are EXCLUDED and reported by the
    caller — never given a same-answer fallback.
    """
    by_answer: Dict[str, List[int]] = {}
    for i, rec in enumerate(records):
        by_answer.setdefault(gold_answer_text(rec).lower(), []).append(i)

    pairs: List[Tuple[int, int]] = []
    for i, rec in enumerate(records):
        ans = gold_answer_text(rec).lower()
        partner = None
        for other_ans, idxs in by_answer.items():
            if other_ans != ans and idxs:
                partner = idxs[0]
                break
        if partner is not None:
            pairs.append((i, partner))
    return pairs
