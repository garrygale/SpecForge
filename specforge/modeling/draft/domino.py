# coding=utf-8
"""Domino draft model entry point."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.func import functional_call

try:
    from .dflash import (
        DFlashDraftModel,
        _target_embed_tokens,
        _target_lm_head,
        sample,
    )
    from .registry import register_draft
except ImportError:  # exported top-level remote-code layout
    from dflash import (
        DFlashDraftModel,
        _target_embed_tokens,
        _target_lm_head,
        sample,
    )
    from registry import register_draft

try:
    from specforge.utils import get_device_type
except ImportError:  # exported remote-code fallback
    def get_device_type() -> str:
        return "cpu"


@register_draft
class DominoDraftModel(DFlashDraftModel):
    """DFlash backbone with Domino's GRU logits correction."""

    expected_projector_type = "domino"

    def __init__(self, config) -> None:
        dflash_config = getattr(config, "dflash_config", None) or {}
        projector_type = dflash_config.get("projector_type")
        if projector_type is None:
            dflash_config["projector_type"] = self.expected_projector_type
            config.dflash_config = dflash_config
        elif projector_type != self.expected_projector_type:
            raise ValueError(
                "DominoDraftModel requires " "dflash_config.projector_type='domino'."
            )
        super().__init__(config)

    def _init_draft_head(self, config, dflash_config: dict) -> None:
        self.emb_dim = int(dflash_config["emb_dim"])
        self.gru_hidden_dim = int(dflash_config["gru_hidden_dim"])
        self.pure_draft_prefix_len = int(dflash_config.get("pure_draft_prefix_len", 0))
        self.shift_label = bool(dflash_config.get("shift_label", False))

        self.prefix_gru = nn.GRU(
            input_size=self.target_hidden_size,
            hidden_size=self.gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bias=False,
        )
        in_dim = self.target_hidden_size + self.gru_hidden_dim
        use_embed = dflash_config.get("use_embed_proj", True)
        use_hidden = bool(dflash_config.get("use_hidden_proj", False))
        if use_embed:
            self.embed_proj = nn.Sequential(
                nn.Linear(in_dim, self.emb_dim, bias=False),
                nn.SiLU(),
                nn.Linear(self.emb_dim, config.vocab_size, bias=False),
            )
            self.hidden_proj = None
            self.emb_dim = int(self.emb_dim)
        else:
            self.embed_proj = None
            self.emb_dim = None

        if use_hidden:
            hidden_proj_dim = int(
                dflash_config.get("hidden_proj_dim", self.emb_dim or 256)
            )
            self.hidden_proj = nn.Sequential(
                nn.Linear(in_dim, hidden_proj_dim, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_proj_dim, self.target_hidden_size, bias=False),
            )
        else:
            self.hidden_proj = None
        if self.embed_proj is None and self.hidden_proj is None:
            raise ValueError(
                "domino requires use_embed_proj or use_hidden_proj"
            )

    @property
    def suffix_start(self) -> int:
        return (
            self.pure_draft_prefix_len
            if self.shift_label
            else (1 + self.pure_draft_prefix_len)
        )

    def _run_gru(self, gru_inputs: torch.Tensor) -> torch.Tensor:
        if get_device_type() == "npu" and gru_inputs.dtype == torch.bfloat16:
            # Ascend DynamicGRU does not accept bf16. Functionally substitute
            # fp16 views of the real parameters so autograd still reaches the
            # registered bf16 weights and FSDP never sees a mixed-dtype module.
            fp16_parameters = {
                name: parameter.to(dtype=torch.float16)
                for name, parameter in self.prefix_gru.named_parameters()
            }
            output, _ = functional_call(
                self.prefix_gru,
                fp16_parameters,
                (gru_inputs.to(dtype=torch.float16),),
                strict=True,
            )
            return output.to(dtype=gru_inputs.dtype)
        return self.prefix_gru(gru_inputs)[0]

    def _sample_draft_tokens(
        self,
        target: nn.Module,
        draft_hidden: torch.Tensor,
        block_output_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        """Sample a block while causally applying Domino's GRU correction."""
        target_embed = _target_embed_tokens(target)
        target_lm_head = _target_lm_head(target)
        base_logits = target_lm_head(draft_hidden)
        completed_ids = block_output_ids.clone()
        base_logits4d = base_logits.unsqueeze(1)
        hidden_states4d = draft_hidden.unsqueeze(1)

        for token_position in range(1, completed_ids.shape[1]):
            previous_embeddings = target_embed(completed_ids).unsqueeze(1)
            final_logits = self.apply_logits_head(
                base_logits4d,
                prev_token_embeddings=previous_embeddings,
                hidden_states=hidden_states4d,
            )
            head_position = token_position - 1 if self.shift_label else token_position
            next_token_logits = final_logits[:, 0, head_position, :].unsqueeze(1)
            completed_ids[:, token_position] = sample(next_token_logits).squeeze(1)

        return completed_ids[:, 1:]

    def apply_logits_head(
        self,
        base_logits: torch.Tensor,
        *,
        prev_token_ids: Optional[torch.Tensor] = None,
        prev_token_embeddings: Optional[torch.Tensor] = None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        del prev_token_ids
        correction_logits = self.compute_correction_logits(
            prev_token_embeddings=prev_token_embeddings,
            hidden_states=hidden_states,
        )
        prefix_logits = base_logits[:, :, : self.suffix_start, :]
        suffix_logits = base_logits[:, :, self.suffix_start :, :] + correction_logits
        return torch.cat([prefix_logits, suffix_logits], dim=2)

    def compute_correction_logits(
        self,
        *,
        prev_token_embeddings: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Return suffix-only Domino logits without materializing final logits."""
        if prev_token_embeddings is None:
            raise ValueError("DominoDraftModel requires prev_token_embeddings")

        bsz, n_blocks, block_size = hidden_states.shape[:3]
        if self.shift_label:
            gru_inputs = prev_token_embeddings.reshape(bsz * n_blocks, block_size, -1)
            gru_out = self._run_gru(gru_inputs)
            gru_out = gru_out.reshape(bsz, n_blocks, block_size, -1)
            prefix_states = gru_out[:, :, self.suffix_start :, :]
        else:
            gru_inputs = prev_token_embeddings[:, :, : block_size - 1, :].reshape(
                bsz * n_blocks, block_size - 1, -1
            )
            gru_out = self._run_gru(gru_inputs)
            gru_out = gru_out.reshape(bsz, n_blocks, block_size - 1, -1)
            prefix_states = gru_out[:, :, self.suffix_start - 1 :, :]

        z_n = hidden_states[:, :, self.suffix_start :, :]
        concat_features = torch.cat([z_n, prefix_states], dim=-1)
        return self.embed_proj(concat_features)


__all__ = ["DominoDraftModel"]
