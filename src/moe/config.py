"""Hyperparameter configs for all training stages."""

from __future__ import annotations

from dataclasses import dataclass, field

from moe.utils import BATCH_SIZE_BY_DEVICE, detect_device


@dataclass
class BaseConfig:
    # --- Model architecture ---
    # Faster config — trains in ~5 min on MPS, val loss ~1.9.
    # Scale up for full quality:
    #   Full: block_size=256, n_embd=256, max_iters=3000  (~20 min, val loss ~1.8)
    block_size: int = 128  # context window (tokens per sample)
    n_embd: int = 128  # embedding dimension
    n_heads: int = 4  # attention heads (head_dim = n_embd // n_heads = 32)
    n_layers: int = 4  # number of transformer blocks
    ffn_mult: int = 4  # FFN hidden dim = n_embd * ffn_mult
    dropout: float = 0.1

    # --- Training ---
    # batch_size=0 means "auto" — __post_init__ picks the right value per device:
    #   cuda -> 256  (typical mid-range GPU)
    #   mps  -> 128  (Apple Silicon, MacBook Air 32GB)
    #   cpu  ->  32  (no GPU)
    # Override by passing batch_size=N explicitly: DenseConfig(batch_size=64)
    batch_size: int = 0
    lr: float = 3e-4
    grad_clip: float = 1.0  # gradient norm clipping — stabilises training
    max_iters: int = 2000

    # --- Evaluation ---
    eval_interval: int = 200  # 10 checkpoints over 2000 steps
    eval_iters: int = 50  # 50 batches to estimate loss

    # --- Device (auto-detected, override with device="cpu" etc.) ---
    device: str = field(default="", init=True)

    def __post_init__(self) -> None:
        if not self.device:
            self.device = detect_device()
        if self.batch_size == 0:
            self.batch_size = BATCH_SIZE_BY_DEVICE.get(self.device, 32)


@dataclass
class SharedMoEConfig(BaseConfig):
    """Extends BaseConfig with MoE-specific fields (naive_experts and beyond)."""

    n_experts: int = 4
    top_k: int = 2  # used by routed_moe and balanced_moe; ignored in naive_experts
    lb_coeff: float = 0.0  # load-balancing coefficient; >0 only in balanced_moe


@dataclass
class DenseConfig(BaseConfig):
    """Dense baseline — one shared FFN, every token takes the same path."""


@dataclass
class NaiveExpertsConfig(SharedMoEConfig):
    """N independent FFNs, no routing — equal weight to all experts."""


@dataclass
class RoutedMoEConfig(SharedMoEConfig):
    """Sparse MoE — learned router picks top-k experts per token."""

    top_k: int = 2


@dataclass
class BalancedMoEConfig(SharedMoEConfig):
    """Sparse MoE + load-balancing loss to prevent all tokens flooding one expert."""

    top_k: int = 2
    lb_coeff: float = 0.01
