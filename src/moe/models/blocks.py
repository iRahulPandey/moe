"""TransformerBlock — accepts any FFN module via injection.

This is the key seam: dense, naive_experts, moe, and balanced_moe all use the same block.
The only thing that changes is what gets passed as `ffn`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from moe.models.attention import CausalSelfAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_heads: int,
        block_size: int,
        dropout: float,
        ffn: nn.Module,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = (
            ffn  # FeedForward (dense), ExpertLayer (naive_experts), MoELayer (moe/balanced_moe)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
