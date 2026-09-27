import torch
import torch.distributed as dist
from torch.autograd.variable import Variable

from core.distributed.data.comm import sync_grads
from core.distributed.pipeline.p2p import P2PCommunication
from core.distributed.pipeline.layers import PipelineParallelModule


def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    '''Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.

    Never call this on the last stage: there the "output" is the loss scalar,
    whose value is still needed for logging (and it is never sent anywhere).

    Supports multiple formats:
    - torch.Tensor: Deallocates the tensor directly
    - List[Tensor]: Recursively deallocates each element
    - Dict[str, Tensor]: Recursively deallocates each value (for multi-module pipelines)
    '''
    if (out is None) or (not deallocate_pipeline_outputs):
        return

    # Handle dict format (multi-module pipelines)
    if isinstance(out, dict):
        for value in out.values():
            deallocate_output_tensor(value, deallocate_pipeline_outputs)
        return

    # Handle list format
    if isinstance(out, list):
        for item in out:
            deallocate_output_tensor(item, deallocate_pipeline_outputs)
        return

    # Base case: deallocate tensor
    assert isinstance(out, torch.Tensor), "expected Tensor, found %s." % type(out).__name__
    assert out._base is None, "counter-productive to free a view of another tensor."
    out.data = torch.empty((1,), device=out.device, dtype=out.dtype)


def custom_backward(output: torch.Tensor, grad_output: torch.Tensor | None) -> None:
    '''Directly call C++ autograd engine.

    `torch.autograd.backward` checks that `output.shape == grad_output.shape`,
    which fails once `deallocate_output_tensor` has replaced the output's data
    with a 1-element placeholder. Calling the engine directly skips that check
    (this is what Megatron-LM does).
    '''
    assert output.numel() == 1 or grad_output is not None, (
        "implicit grad requires scalar output."
    )
    if grad_output is None:
        grad_output = torch.ones_like(output, memory_format=torch.preserve_format)

    Variable._execution_engine.run_backward(
        tensors=(output,),
        grad_tensors=(grad_output,),
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )


def backward_step(
    input_tensor: torch.Tensor | None,
    output_tensor: torch.Tensor,
    grad_output: torch.Tensor | None,
) -> torch.Tensor | None:
    """Backward through one microbatch.

    Returns gradient w.r.t. input_tensor (to send to previous stage),
    or None on the first stage.
    """
    if input_tensor is not None:
        input_tensor.retain_grad()

    custom_backward(output_tensor, grad_output)

    if input_tensor is None or input_tensor.grad is None:
        return None
    return input_tensor.grad.to(input_tensor.dtype)


def _forward_microbatch(
    model: PipelineParallelModule,
    input_tensor: torch.Tensor,
    labels: torch.Tensor | None,
    accumulated_loss: torch.Tensor,
) -> torch.Tensor:
    """Run this stage's forward for one microbatch.

    On the last stage the output is the microbatch's mean loss. We record it
    for logging and scale it by 1/num_microbatches before it enters backward,
    so the accumulated parameter gradients equal the gradient of the *mean*
    loss over the whole batch (matching the non-pipeline path) instead of a sum.
    """
    output_tensor = model(input_tensor, labels=labels)
    if model.is_last_stage:
        accumulated_loss += output_tensor.detach()
        output_tensor = output_tensor / model.num_microbatches
    return output_tensor


def training_step_1f1b(
    model: PipelineParallelModule,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    dtype: torch.dtype = torch.float32,
    dp_group: dist.ProcessGroup | None = None,
) -> float | None:
    """One optimizer step using the 1F1B pipeline schedule.

    Splits the batch into microbatches and orchestrates forward/backward
    across pipeline stages with warm-up, steady (1F1B), and cooldown phases.
    Gradients are averaged across the data-parallel group before the
    optimizer step.

    Returns:
        Average loss on the last stage, None on other stages.
    """
    if dp_group is None:
        dp_group = model.comm_groups_config.dp_group

    num_microbatches = model.num_microbatches
    if tokens.shape[0] % num_microbatches != 0:
        raise ValueError(
            f"Batch size {tokens.shape[0]} is not divisible by "
            f"num_microbatches={num_microbatches}."
        )
    num_warmup = min(model.num_stages - model.stage_id - 1, num_microbatches)
    num_steady = num_microbatches - num_warmup

    if model.is_first_stage:
        microbatches_tokens = model._split_batch(tokens)
        micro_batch_size = microbatches_tokens[0].shape[0]
        microbatches_labels = model._split_batch(labels)
    else:
        micro_batch_size = tokens.shape[0] // num_microbatches
        microbatches_tokens = [None] * num_microbatches
        microbatches_labels = (
            model._split_batch(labels)
            if model.is_last_stage
            else [None] * num_microbatches
        )

    p2p = P2PCommunication(model.model_config, micro_batch_size, model.seq_len, dtype)

    input_tensors: list[torch.Tensor | None] = []
    output_tensors: list[torch.Tensor] = []
    accumulated_loss = torch.tensor(0.0, device=model.comm_groups_config.device)

    optimizer.zero_grad(set_to_none=True)

    # ── Warm-up phase ──────────────────────────────────────────
    for i in range(num_warmup):
        if model.is_first_stage:
            input_tensor = microbatches_tokens[i]
        else:
            input_tensor = p2p.recv_forward()

        output_tensor = _forward_microbatch(
            model, input_tensor, microbatches_labels[i], accumulated_loss
        )
        p2p.send_forward(output_tensor)

        input_tensors.append(input_tensor if not model.is_first_stage else None)
        output_tensors.append(output_tensor)
        if not model.is_last_stage:
            deallocate_output_tensor(output_tensor, deallocate_pipeline_outputs=True)

    if num_steady > 0:
        input_tensor = (
            microbatches_tokens[num_warmup] if model.is_first_stage else p2p.recv_forward()
        )

    # ── Steady phase (1F1B) ────────────────────────────────────
    for i in range(num_steady):
        last_iteration = i == (num_steady - 1)
        fwd_idx = num_warmup + i
        bwd_idx = i

        output_tensor = _forward_microbatch(
            model, input_tensor, microbatches_labels[fwd_idx], accumulated_loss
        )

        # Send this microbatch's activations forward and receive the gradient
        # for an earlier microbatch in one batched p2p call.
        grad_output = p2p.send_forward_recv_backward(output_tensor)

        input_tensors.append(input_tensor if not model.is_first_stage else None)
        output_tensors.append(output_tensor)
        if not model.is_last_stage:
            deallocate_output_tensor(output_tensor, deallocate_pipeline_outputs=True)

        grad_input = backward_step(
            input_tensors[bwd_idx], output_tensors[bwd_idx], grad_output
        )

        if last_iteration:
            input_tensor = None
            p2p.send_backward(grad_input)
        elif model.is_first_stage:
            input_tensor = microbatches_tokens[fwd_idx + 1]
        else:
            input_tensor = p2p.send_backward_recv_forward(grad_input)

    # ── Cooldown phase ─────────────────────────────────────────
    for i in range(num_warmup):
        bwd_idx = num_steady + i

        grad_output = p2p.recv_backward() if not model.is_last_stage else None
        grad_input = backward_step(
            input_tensors[bwd_idx], output_tensors[bwd_idx], grad_output
        )
        p2p.send_backward(grad_input)

    sync_grads(model, dp_group)
    optimizer.step()
    print("Oi")
    if model.is_last_stage:
        print((accumulated_loss / num_microbatches).item())
        return (accumulated_loss / num_microbatches).item()
    return None
