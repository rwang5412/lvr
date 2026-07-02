"""LVR causal-validation harness (Branch 1).

Offline measurement of whether the answer causally depends on the latent tokens. See
`docs/BRANCH1_HARNESS_REVIEW.md` for the full design. Public surface:

- `spans.get_spans` — the shared segment-range contract (image/question/latent/answer). Every later
  branch imports this; nobody recomputes spans locally.
"""

from .spans import Span, get_spans, validate_spans  # noqa: F401
