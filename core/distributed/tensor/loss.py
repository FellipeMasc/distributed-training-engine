"""Vocab-parallel cross-entropy (Megatron-LM style).

When ``lm_head`` is a ``ColumnParallelLinear`` with ``gather_output=False``,
each tensor-parallel rank holds logits for only its slice of the vocabulary:
``[..., vocab_size // tp_size]``. Computing the loss directly on those shards
avoids materializing the full ``[B, S, vocab_size]`` logits on every rank.

The softmax denominator and the target logit are combined across ranks with
three small all-reduces on ``[B*S]``-sized tensors (max, target logit, sum of
exps) instead of gathering ``[B*S, vocab_size]``.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        # Subtract the global max for numerical stability.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=group)
        logits = vocab_parallel_logits - logits_max.unsqueeze(-1)

        # Targets outside this rank's vocab range (including ignore_index, which
        # is negative) contribute 0 to the local predicted logit.
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = (target - vocab_start_index).masked_fill(target_mask, 0)

        logits_2d = logits.view(-1, logits.size(-1))
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(logits_2d.size(0), device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits = predicted_logits_1d.view_as(target).clone()
        predicted_logits.masked_fill_(target_mask, 0.0)
        dist.all_reduce(predicted_logits, op=dist.ReduceOp.SUM, group=group)

        exp_logits = torch.exp(logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=group)

        loss = torch.log(sum_exp_logits) - predicted_logits

        # Turn exp_logits into the local softmax slice for the backward pass.
        exp_logits.div_(sum_exp_logits.unsqueeze(-1))
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        grad_input = softmax
        grad_2d = grad_input.view(-1, grad_input.size(-1))
        arange_1d = torch.arange(grad_2d.size(0), device=grad_2d.device)
        softmax_update = 1.0 - target_mask.view(-1).to(grad_2d.dtype)
        grad_2d[arange_1d, masked_target_1d] -= softmax_update
        grad_input.mul_(grad_output.unsqueeze(-1))
        return grad_input, None, None, None, None


def vocab_parallel_cross_entropy(
    vocab_parallel_logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
    group: dist.ProcessGroup,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Mean token cross-entropy over sharded logits.

    Matches ``F.cross_entropy(logits.float(), labels, ignore_index=...)`` on the
    full (gathered) logits, and therefore ``transformers``' ``ForCausalLMLoss``
    when it is called with pre-shifted labels.

    Args:
        vocab_parallel_logits: ``[..., vocab_size // tp_size]`` local logits.
        labels: ``[...]`` already-shifted target ids; ``ignore_index`` entries
            are excluded from the mean.
        vocab_start_index / vocab_end_index: global vocab range owned by this
            rank (``lm_head.partition_start`` / ``partition_end``).
        group: tensor-parallel process group.
    """
    logits = vocab_parallel_logits.float()
    labels = labels.to(logits.device)
    per_token_loss = _VocabParallelCrossEntropy.apply(
        logits, labels, vocab_start_index, vocab_end_index, group
    )
    valid = labels != ignore_index
    per_token_loss = per_token_loss.masked_fill(~valid, 0.0)
    return per_token_loss.sum() / valid.sum().clamp(min=1).to(per_token_loss.dtype)
