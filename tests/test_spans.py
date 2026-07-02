"""Test #2 (the shared-contract guard): get_spans points at the right tokens.

Pure-Python — no torch / model / GPU — so it runs anywhere:

    python tests/test_spans.py

Off-by-one in get_spans fails silently downstream (a leaking bottleneck mask in Branch 2, a
mis-spliced latent in Branch 4), so this is the one test that must stay green.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.harness.spans import Span, get_spans, validate_spans

# Synthetic vocab ids for the fixture (values are arbitrary, only equality matters).
IMG = 1000          # config.image_token_id
LVR = 2000          # config.lvr_id  (the repeated <|lvr|>)
LVR_START = 2001    # <|lvr_start|>  (NOT a latent — must be excluded)
LVR_END = 2002      # <|lvr_end|>    (NOT a latent — must be excluded)
IGN = -100


def _fixture(n_img=4, n_lvr=3, n_q=3, n_ans=3, n_sys=2):
    """Build one assembled example matching the real layout:
    [system][image][question][lvr_start][<lvr>*n_lvr][lvr_end][answer]
    with labels masked everywhere except the answer span.
    """
    input_ids, labels = [], []

    def add(tokens, is_answer=False):
        input_ids.extend(tokens)
        labels.extend(tokens if is_answer else [IGN] * len(tokens))

    add([10 + i for i in range(n_sys)])          # system/role
    add([IMG] * n_img)                            # image block
    add([20 + i for i in range(n_q)])            # question text
    add([LVR_START])                             # <|lvr_start|>  (excluded from latent)
    add([LVR] * n_lvr)                           # <|lvr|> * n_lvr  (the latents)
    add([LVR_END])                               # <|lvr_end|>    (excluded from latent)
    add([30 + i for i in range(n_ans)], is_answer=True)  # answer
    return input_ids, labels


def test_basic_layout():
    input_ids, labels = _fixture(n_sys=2, n_img=4, n_q=3, n_lvr=3, n_ans=3)
    spans = get_spans(input_ids, labels, image_token_id=IMG, lvr_id=LVR)

    # image = positions 2..6 (after 2 system tokens); latent = the 3 <|lvr|> only; answer = last 3.
    assert spans["image"] == Span(2, 6), spans["image"]
    assert spans["latent"] == Span(10, 13), spans["latent"]      # start<|lvr_start|>=9, lvr=10,11,12
    assert spans["answer"] == Span(14, 17), spans["answer"]      # lvr_end=13, answer=14,15,16
    assert len(spans["image"]) == 4
    assert len(spans["latent"]) == 3
    assert len(spans["answer"]) == 3

    # latent count == number of <|lvr|> tokens == what lvr_tokens would carry.
    assert len(spans["latent"]) == input_ids.count(LVR)
    assert len(spans["image"]) == input_ids.count(IMG)

    # <|lvr_start|>/<|lvr_end|> are NOT in the latent span.
    assert LVR_START not in [input_ids[i] for i in spans["latent"].indices()]
    assert LVR_END not in [input_ids[i] for i in spans["latent"].indices()]

    validate_spans(spans, seq_len=len(input_ids), n_lvr_tokens=3, n_image_tokens=4)
    print("PASS test_basic_layout")


def test_non_overlap_and_order():
    input_ids, labels = _fixture()
    spans = get_spans(input_ids, labels, image_token_id=IMG, lvr_id=LVR)
    order = [spans["image"], spans["question"], spans["latent"], spans["answer"]]
    for a, b in zip(order, order[1:]):
        assert a.end <= b.start, (a, b)
    # answer is the tail of the sequence.
    assert spans["answer"].end == len(input_ids)
    print("PASS test_non_overlap_and_order")


def test_accepts_nested_and_listlike():
    input_ids, labels = _fixture()
    # a [1, L] nested list (as a batch of one) must be accepted.
    s1 = get_spans([input_ids], [labels], image_token_id=IMG, lvr_id=LVR)
    s2 = get_spans(input_ids, labels, image_token_id=IMG, lvr_id=LVR)
    assert s1 == s2
    print("PASS test_accepts_nested_and_listlike")


def test_varied_sizes():
    for n_img, n_lvr, n_ans in [(1, 1, 1), (16, 8, 5), (100, 2, 20)]:
        input_ids, labels = _fixture(n_img=n_img, n_lvr=n_lvr, n_ans=n_ans)
        spans = get_spans(input_ids, labels, image_token_id=IMG, lvr_id=LVR)
        assert len(spans["image"]) == n_img
        assert len(spans["latent"]) == n_lvr
        assert len(spans["answer"]) == n_ans
        validate_spans(spans, seq_len=len(input_ids), n_lvr_tokens=n_lvr, n_image_tokens=n_img)
    print("PASS test_varied_sizes")


def test_non_contiguous_latent_raises():
    # Corrupt the fixture so a stray <|lvr|> sits inside the answer → must fail fast, not silently.
    input_ids, labels = _fixture()
    input_ids[-1] = LVR  # a latent token where the answer should be
    try:
        get_spans(input_ids, labels, image_token_id=IMG, lvr_id=LVR)
    except ValueError as e:
        assert "not contiguous" in str(e), str(e)
        print("PASS test_non_contiguous_latent_raises")
        return
    raise AssertionError("expected ValueError for non-contiguous latent tokens")


def test_missing_segment_raises():
    input_ids, labels = _fixture()
    no_img = [t for t in input_ids]
    for i, t in enumerate(no_img):
        if t == IMG:
            no_img[i] = 999  # remove all image tokens
    try:
        get_spans(no_img, labels, image_token_id=IMG, lvr_id=LVR)
    except ValueError as e:
        assert "no image tokens" in str(e), str(e)
        print("PASS test_missing_segment_raises")
        return
    raise AssertionError("expected ValueError when image tokens are absent")


if __name__ == "__main__":
    test_basic_layout()
    test_non_overlap_and_order()
    test_accepts_nested_and_listlike()
    test_varied_sizes()
    test_non_contiguous_latent_raises()
    test_missing_segment_raises()
    print("\nAll span tests passed.")
