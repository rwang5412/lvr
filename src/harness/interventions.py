"""Latent-set builders for the harness `override_latent_embeds` path.

Each returns an [L, H] tensor to splice into one example's <lvr> positions (row order matching
`torch.nonzero(lvr_mask)`), which the patched `qwen2_5_mixed_modality_forward_lvr` consumes.

- `zero_latents`         — dummy latents (hard corruption): does the answer notice they're gone?
- `mean_replace_latents` — replace every latent with a fixed mean vector (soft corruption / NIE).
- `align_partner_latents`— fit a partner's latents to this example's latent count (whole-set splice).

Corruption is deliberately DIFFERENT from Branch-4 training (which uses cross-input swap), so the
harness never grades its own training objective (Test #5).
"""

import torch


def zero_latents(n: int, hidden: int, *, dtype=None, device=None) -> torch.Tensor:
    """`n` dummy (zero) latent rows of width `hidden`."""
    return torch.zeros(n, hidden, dtype=dtype, device=device)


def mean_replace_latents(n: int, mean_vector: torch.Tensor) -> torch.Tensor:
    """`n` copies of a fixed `mean_vector` [H] — the mean-latent corruption used for NIE.

    `mean_vector` is computed ONCE over the held-out set's latents (see run_harness) so every example
    is corrupted toward the same constant, removing per-example latent information.
    """
    if mean_vector.dim() != 1:
        raise ValueError(f"mean_vector must be 1-D [H], got shape {tuple(mean_vector.shape)}")
    return mean_vector.unsqueeze(0).expand(n, mean_vector.shape[0]).contiguous()


def align_partner_latents(partner_latents: torch.Tensor, target_n: int) -> torch.Tensor:
    """Fit partner latents [Lp, H] to `target_n` rows for a whole-set splice into the target example.

    Truncates if the partner has more latents, tiles (repeats) if it has fewer. Whole-set (not
    per-position) alignment is the default — per-position splice would require position-consistent
    ROIs (§4), which we do not assume.
    """
    if partner_latents.dim() != 2:
        raise ValueError(f"partner_latents must be [Lp, H], got {tuple(partner_latents.shape)}")
    lp = partner_latents.shape[0]
    if lp == target_n:
        return partner_latents
    if lp > target_n:
        return partner_latents[:target_n]
    # lp < target_n: tile then trim
    reps = (target_n + lp - 1) // lp
    return partner_latents.repeat(reps, 1)[:target_n]
