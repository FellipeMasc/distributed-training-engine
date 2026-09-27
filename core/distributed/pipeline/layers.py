import torch
import torch.nn as nn

from core.parallel_state import (
    get_stage_id,
)
from core.process_groups_config import CommGroupsConfig
from core.utils import causal_lm_loss, distribute_layers_pp_stages
from transformers.masking_utils import create_causal_mask
from core.models import LlamaRotaryEmbedding



class PipelineParallelModule(nn.Module):
    """Pipeline parallel wrapper that partitions a model across stages.

    Each stage holds its slice of transformer layers, plus embedding
    on the first stage and norm/lm_head on the last stage.
    The forward pass runs only through this stage's layers.
    """

    def __init__(
        self,
        model: nn.Module,
        model_config,
        num_microbatches: int,
        seq_len: int,
    ):
        super().__init__()
        self.comm_groups_config = CommGroupsConfig()
        self.model_config = model_config
        self.num_microbatches = num_microbatches
        self.seq_len = seq_len

        self.pp_group = self.comm_groups_config.pp_group
        self.stage_id = get_stage_id()
        self.num_stages = self.comm_groups_config.num_stages
        self.prev_rank = self.comm_groups_config.pipeline_prev_rank
        self.next_rank = self.comm_groups_config.pipeline_next_rank

        self.is_first_stage = self.comm_groups_config.is_first_stage
        self.is_last_stage = self.comm_groups_config.is_last_stage

        self._partition_model(model)

    def _partition_model(self, model: nn.Module) -> None:
        pp_rank = self.stage_id
        pp_world_size = self.num_stages
        num_layers = self.model_config.num_hidden_layers
        layers = distribute_layers_pp_stages(num_layers, pp_rank, pp_world_size)

        self.embed_tokens = model.embed_tokens if self.is_first_stage else None
        self.rotary_emb = LlamaRotaryEmbedding(model.config)
        self.layers = nn.ModuleList([model.layers[i] for i in layers])
        self.norm = model.norm if self.is_last_stage else None
        # Reuse the model's head so a tensor-parallel swap applied to the
        # underlying model (ColumnParallelLinear) carries over to this stage.
        self.lm_head = model.lm_head if self.is_last_stage else None
        self.attention_mask = None
        self.past_key_values = None

    # ── Model forward (pure computation, no schedule) ──────────────

    def forward(
        self, input_tensor: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Run forward through this stage's layers.

        Args:
            input_tensor: token ids (first stage) or hidden states (middle/last stage)
            labels: target labels, only used on last stage for loss computation

        Returns:
            hidden states (first/middle stages) or loss scalar (last stage)
        """
        if self.is_first_stage:
            hidden_states = self.embed_tokens(input_tensor)
        else:
            hidden_states = input_tensor

        position_ids = torch.arange(
            hidden_states.shape[1], device=hidden_states.device
        ).unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.model_config,
            inputs_embeds=hidden_states,
            attention_mask=self.attention_mask,
            past_key_values=self.past_key_values,
            position_ids=position_ids,
        )
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                past_key_values=self.past_key_values,
            )

        if self.is_last_stage:
            hidden_states = self.norm(hidden_states)
            return causal_lm_loss(
                self.lm_head, hidden_states, labels, self.model_config.vocab_size
            )

        return hidden_states

    def _split_batch(self, batch: torch.Tensor) -> list[torch.Tensor]:
        return batch.chunk(self.num_microbatches, dim=0)