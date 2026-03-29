"""Dense transformer — every token routes through the same FFN. The baseline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe.config import BaseConfig
from moe.models.blocks import TransformerBlock
from moe.types import ModelOutput


class FeedForward(nn.Module):
    """Standard dense FFN. Replaced by expert layers in naive_experts and beyond."""

    def __init__(self, n_embd: int, ffn_mult: int, dropout: float) -> None:
        super().__init__()
        hidden = n_embd * ffn_mult
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseTransformer(nn.Module):
    def __init__(self, cfg: BaseConfig, vocab_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    n_embd=cfg.n_embd,
                    n_heads=cfg.n_heads,
                    block_size=cfg.block_size,
                    dropout=cfg.dropout,
                    ffn=FeedForward(cfg.n_embd, cfg.ffn_mult, cfg.dropout),
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, vocab_size, bias=False)

        self._init_weights()
        total = sum(p.numel() for p in self.parameters())
        print(f"parameters: {total:,}")

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> ModelOutput:
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return ModelOutput(logits=logits, loss=loss)
