import torch
import torch.distributed as dist
from typing import Optional
from core.process_groups_config import CommGroupsConfig

# NVIDIA MEGATRON P2P OPERATIONS
def _batched_p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    group: torch.distributed.ProcessGroup,
    prev_pipeline_rank: int,
    next_pipeline_rank: int,
):
    ops = []
    if tensor_send_prev is not None and prev_pipeline_rank is not None:
        send_prev_op = torch.distributed.P2POp(
            torch.distributed.isend, tensor_send_prev, prev_pipeline_rank, group
        )
        ops.append(send_prev_op)
    if tensor_recv_prev is not None and prev_pipeline_rank is not None:
        recv_prev_op = torch.distributed.P2POp(
            torch.distributed.irecv, tensor_recv_prev, prev_pipeline_rank, group
        )
        ops.append(recv_prev_op)
    if tensor_send_next is not None and next_pipeline_rank is not None:
        send_next_op = torch.distributed.P2POp(
            torch.distributed.isend, tensor_send_next, next_pipeline_rank, group
        )
        ops.append(send_next_op)
    if tensor_recv_next is not None and next_pipeline_rank is not None:
        recv_next_op = torch.distributed.P2POp(
            torch.distributed.irecv, tensor_recv_next, next_pipeline_rank, group
        )
        ops.append(recv_next_op)
    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
    else:
        reqs = []
    return reqs


class P2PCommunication:
    def __init__(self, model_config, micro_batch_size: int, seq_len: int, dtype: torch.dtype = torch.float32):
        self.model_config = model_config
        self.micro_batch_size = micro_batch_size
        self.seq_len = seq_len
        self.dtype = dtype
        self.comm_groups_config = CommGroupsConfig()
        self.rank = self.comm_groups_config.local_rank
        self.pp_group = self.comm_groups_config.pp_group
        self.prev_rank = self.comm_groups_config.pipeline_prev_rank
        self.next_rank = self.comm_groups_config.pipeline_next_rank
        self.is_first_stage = self.comm_groups_config.is_first_stage
        self.is_last_stage = self.comm_groups_config.is_last_stage
        self._pending_sends: list[dist.Work] = []

    def _communicate(self,
        tensor_send_next: Optional[torch.Tensor],
        tensor_send_prev: Optional[torch.Tensor],
        recv_prev: bool,
        recv_next: bool,
        wait_on_reqs: bool = True):
        tensor_recv_prev_func = None
        tensor_recv_next_func = None
        recv_prev_shape = self._get_activation_shape()
        recv_next_shape = self._get_activation_shape()

        def create_tensor_recv_prev():
            return torch.empty(
                recv_prev_shape,
                requires_grad=True,
                device=self.comm_groups_config.device,
                dtype=self.dtype,
            )

        def create_tensor_recv_next():
            return torch.empty(
                recv_next_shape,
                requires_grad=True,
                device=self.comm_groups_config.device,
                dtype=self.dtype,
            )

        if recv_prev:
            tensor_recv_prev_func = create_tensor_recv_prev

        if recv_next:
            tensor_recv_next_func = create_tensor_recv_next


        pp_group = self.pp_group
        next_rank = self.next_rank
        prev_rank = self.prev_rank

        reqs = []

        tensor_recv_prev = None
        tensor_recv_next = None
        if tensor_recv_prev_func is not None:
            tensor_recv_prev = tensor_recv_prev_func()

        if tensor_recv_next_func is not None:
            tensor_recv_next = tensor_recv_next_func()

        p2p_reqs = _batched_p2p_ops(
            tensor_send_prev=tensor_send_prev,
            tensor_recv_prev=tensor_recv_prev,
            tensor_send_next=tensor_send_next,
            tensor_recv_next=tensor_recv_next,
            group=pp_group,
            prev_pipeline_rank=prev_rank,
            next_pipeline_rank=next_rank,
        )
        if isinstance(p2p_reqs, list):
            reqs.extend(p2p_reqs)
        else:
            reqs.update(p2p_reqs)

        if wait_on_reqs and len(reqs) > 0:
            for req in reqs if isinstance(reqs, list) else reqs.values():
                req.wait()
            reqs = None

        return tensor_recv_prev, tensor_recv_next, reqs


    def send_forward(self, tensor: torch.Tensor) -> None:
        if self.is_last_stage:
            return
        self._communicate(
            tensor_send_next=tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=False,
            wait_on_reqs=True,
        )

    def recv_forward(self) -> torch.Tensor:
        if self.is_first_stage:
            raise RuntimeError("First stage should not recv forward activations")
        
        input_tensor, _, _ = self._communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=True,
            recv_next=False,
            wait_on_reqs=True,
        )
        return input_tensor


    def send_backward(self, grad_tensor: torch.Tensor) -> None:
        if self.is_first_stage:
            return
        self._communicate(
            tensor_send_next=None,
            tensor_send_prev=grad_tensor,
            recv_prev=False,
            recv_next=False,
            wait_on_reqs=True,
        )

    def recv_backward(self) -> torch.Tensor:
        if self.is_last_stage:
            raise RuntimeError("Last stage should not recv backward gradients")
        
        _, grad_tensor, _ = self._communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            wait_on_reqs=True,
        )
        return grad_tensor
    
    def send_forward_recv_backward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.is_last_stage:
            return None
        _, grad_tensor, _ = self._communicate(
            tensor_send_next=tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            wait_on_reqs=True,
        )
        return grad_tensor

    def send_backward_recv_forward(self, grad_tensor: torch.Tensor) -> torch.Tensor:
        if self.is_first_stage:
            return None
        input_tensor, _, _ = self._communicate(
            tensor_send_next=None,
            tensor_send_prev=grad_tensor,
            recv_prev=True, 
            recv_next=False,
            wait_on_reqs=True,
        )   
        return input_tensor

    def _get_activation_shape(self) -> tuple:
        return (self.micro_batch_size, self.seq_len, self.model_config.hidden_size)