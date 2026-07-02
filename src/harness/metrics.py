"""Harness metrics — pure torch/numpy (the env has neither sklearn nor scipy).

Two families:

Causal (answer-level, teacher-forced NLL):
- `answer_nll`             — mean cross-entropy over the answer span for one example.
- `proportion_mediated`   — NIE via mediation: (latent-corruption effect) / (image-corruption effect).
                            Baseline (inert latents) ⇒ corrupting latents barely moves the answer
                            while corrupting the image moves it a lot ⇒ ≈ 0 (the disconnect).

Representation (latent-level):
- `effective_rank`        — entropy-based effective rank of a latent matrix (collapse detector, §7.2).
- `participation_ratio`   — a second, moment-based spread measure.
- `avg_pairwise_cosine_distance`
- `linear_probe_r2`       — ridge-regress the ROI target from the latent; R² = how much the latent
                            encodes the supervised ROI (richness / faithfulness).
"""

from typing import Dict, List

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------------------- causal ---

def answer_nll(logits: torch.Tensor, labels: torch.Tensor, answer_span, ignore_index: int = -100) -> torch.Tensor:
    """Mean teacher-forced NLL over the ANSWER-TEXT tokens of ONE example.

    The dataset's `labels` keep the WHOLE response (including <lvr> tokens), so we cannot just use
    `labels != ignore_index` — that would also score the latent tokens. We restrict scoring to
    `answer_span` (from get_spans), which is the answer text after the latent block.

    Args:
        logits: [1, L, V] or [L, V] float logits from the forward.
        labels: [1, L] or [L] — the batch labels (whole response non-ignored).
        answer_span: a Span(start, end) marking the answer-text token range.
    Returns:
        scalar tensor — mean cross-entropy over the answer span only. Lower = more confident answer
        given the latents currently in the sequence.
    """
    if logits.dim() == 3:
        logits = logits[0]
    if labels.dim() == 2:
        labels = labels[0]
    # Keep only the answer-text tokens as targets; ignore everything else (prompt, latents, tags).
    targets = torch.full_like(labels, ignore_index)
    targets[answer_span.start:answer_span.end] = labels[answer_span.start:answer_span.end]
    # standard causal shift: token t's logits predict token t+1
    shift_logits = logits[:-1, :].float()
    shift_labels = targets[1:].to(shift_logits.device)
    n = (shift_labels != ignore_index).sum()
    if n == 0:
        raise ValueError("answer_nll: empty answer span.")
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=ignore_index, reduction="mean")


def proportion_mediated(
    nll_clean: List[float],
    nll_latent_corrupt: List[float],
    nll_image_corrupt: List[float],
    eps: float = 1e-4,
) -> Dict[str, float]:
    """Aggregate NIE / proportion-mediated across the eval set.

    Mediation view: the answer's total reliance on the image is the "total effect" T (measured by
    corrupting the image); the part routed *through the latents* is the "indirect effect" M
    (measured by corrupting the latents). proportion_mediated = M / T.

    Returns a dict with:
        latent_effect  (mean M) — raw increase in answer NLL under latent corruption.
        image_effect   (mean T) — increase in answer NLL under image corruption.
        proportion_mediated (M / T, only over examples where T > eps).
    """
    import statistics

    clean = torch.tensor(nll_clean, dtype=torch.float64)
    latent = torch.tensor(nll_latent_corrupt, dtype=torch.float64)
    image = torch.tensor(nll_image_corrupt, dtype=torch.float64)

    M = (latent - clean)          # indirect effect (via latents), per example
    T = (image - clean)           # total effect (via image), per example

    mask = T > eps                # only where the image genuinely matters is the ratio defined
    per_example = (M[mask] / T[mask]).tolist() if mask.any() else []

    return {
        "latent_effect": float(M.mean()),
        "image_effect": float(T.mean()),
        "proportion_mediated": float(statistics.mean(per_example)) if per_example else float("nan"),
        "n_defined": int(mask.sum()),
        "n_total": len(nll_clean),
    }


def directed_flip_scores(
    nll_partner_under_clean: List[float],
    nll_partner_under_splice: List[float],
) -> Dict[str, float]:
    """Directed flip-to-target: splicing partner x2's latents into x should make x2's answer MORE
    likely (its NLL should DROP). Reports the mean drop and the fraction of pairs where it dropped.
    Guards against a gate: we score the *specific* partner answer, not just "the answer changed".
    """
    clean = torch.tensor(nll_partner_under_clean, dtype=torch.float64)
    splice = torch.tensor(nll_partner_under_splice, dtype=torch.float64)
    drop = clean - splice  # positive = partner answer became more likely after splice
    return {
        "mean_partner_nll_drop": float(drop.mean()),
        "flip_rate": float((drop > 0).float().mean()),
        "n_pairs": len(nll_partner_under_clean),
    }


# ------------------------------------------------------------------------------- representation ---

def _center(X: torch.Tensor) -> torch.Tensor:
    return X - X.mean(dim=0, keepdim=True)


def effective_rank(X: torch.Tensor, eps: float = 1e-12) -> float:
    """Entropy-based effective rank (Roy & Vetterli) of an [N, D] matrix.

    erank = exp(-Σ p_i log p_i), p_i = σ_i / Σσ. Ranges [1, min(N, D)]. Low ⇒ collapse.
    """
    X = _center(X.to(torch.float32))
    s = torch.linalg.svdvals(X)
    s = s[s > eps]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    entropy = -(p * (p + eps).log()).sum()
    return float(torch.exp(entropy))


def participation_ratio(X: torch.Tensor, eps: float = 1e-12) -> float:
    """Moment-based spread: (Σσ²)² / Σσ⁴ over centered singular values. Low ⇒ few dominant directions."""
    X = _center(X.to(torch.float32))
    s2 = torch.linalg.svdvals(X) ** 2
    denom = (s2 ** 2).sum()
    if denom < eps:
        return 0.0
    return float((s2.sum() ** 2) / denom)


def avg_pairwise_cosine_distance(X: torch.Tensor, max_rows: int = 4096) -> float:
    """Mean (1 - cosine) over all distinct row pairs of [N, D]. Subsamples if N > max_rows."""
    X = X.to(torch.float32)
    if X.shape[0] > max_rows:
        idx = torch.randperm(X.shape[0])[:max_rows]
        X = X[idx]
    Xn = F.normalize(X, dim=1)
    sim = Xn @ Xn.t()
    n = sim.shape[0]
    if n < 2:
        return 0.0
    off = (sim.sum() - sim.diagonal().sum()) / (n * (n - 1))
    return float(1.0 - off)


def linear_probe_r2(Z: torch.Tensor, targets: torch.Tensor, ridge: float = 1e-2) -> float:
    """Ridge-regress `targets` (ROI embeddings) from latents `Z`; return R² (in-sample, train-fit).

    Z: [N, Dz] latent hidden states. targets: [N, Dt] ROI target embeddings. High R² ⇒ the latent
    linearly encodes the supervised ROI content (rich/faithful); low ⇒ thin latents.
    """
    Z = Z.to(torch.float32)
    Y = targets.to(torch.float32)
    Zc = _center(Z)
    Yc = _center(Y)
    # W = (ZᵀZ + λI)⁻¹ ZᵀY
    A = Zc.t() @ Zc + ridge * torch.eye(Zc.shape[1], dtype=Zc.dtype, device=Zc.device)
    W = torch.linalg.solve(A, Zc.t() @ Yc)
    pred = Zc @ W
    ss_res = ((Yc - pred) ** 2).sum()
    ss_tot = (Yc ** 2).sum().clamp_min(1e-12)
    return float(1.0 - ss_res / ss_tot)
