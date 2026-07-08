"""Self-distillation KL loss — forward KL(teacher ‖ student) over the answer span.

Teacher = full-context answer distribution (bottleneck OFF, detached).
Student = latent-only answer distribution (bottleneck ON, with grad).

Forward KL pulls the student to COVER the teacher's answer distribution, so the latent-only path must
reproduce the full-context answer — pressuring the latents to encode the image. One term; the trainer
adds `distill_weight * this` to `L_answer + L_patch`. No paired data needed.
"""

import torch
import torch.nn.functional as F

from src.harness.spans import get_spans


def distill_kl_over_answer(student_logits, teacher_logits, input_ids, labels, config):
    """Forward KL(teacher ‖ student), averaged over the answer-text tokens in the batch.

    Args:
        student_logits, teacher_logits: [B, L, V]. The teacher is detached here — no gradient flows
            into it (the trainer also runs the teacher pass under no_grad; this is belt-and-braces).
        input_ids, labels: [B, L]; the answer span per example comes from get_spans.
        config: model.config (needs image_token_id, lvr_id, lvr_start_id, lvr_end_id).

    The distribution predicting answer token t sits at logits[t-1] (causal shift), so for answer span
    [s, e) we score logits[s-1:e-1] — the same slice answer_nll uses. Per answer token:
        KL = Σ_x p_teacher(x) · (log p_teacher(x) − log p_student(x))
    Gradient flows only through the student (log p_student).
    """
    B = input_ids.shape[0]
    total_kl = student_logits.new_zeros(())
    n_tokens = 0
    for b in range(B):
        spans = get_spans(
            input_ids[b], labels[b],
            image_token_id=config.image_token_id, lvr_id=config.lvr_id,
            lvr_start_id=config.lvr_start_id, lvr_end_id=config.lvr_end_id,
        )
        s, e = spans["answer"].start, spans["answer"].end
        if e - s < 1 or s < 1:
            continue
        s_logp = F.log_softmax(student_logits[b, s - 1:e - 1].float(), dim=-1)          # [n_ans, V]
        t_logp = F.log_softmax(teacher_logits[b, s - 1:e - 1].float().detach(), dim=-1)  # detached
        t_p = t_logp.exp()
        kl = (t_p * (t_logp - s_logp)).sum(dim=-1)                                       # [n_ans]
        total_kl = total_kl + kl.sum()
        n_tokens += kl.shape[0]

    if n_tokens == 0:
        return student_logits.new_zeros(())
    return total_kl / n_tokens
