# Mixture of Experts — From Scratch

A ground-up implementation of Mixture of Experts (MoE) built as a 4-stage educational journey.

## The Story

### Stage 1 — Dense Baseline (`train_dense.py`)
Build a tiny transformer where every token passes through the same Feed-Forward Network (FFN). This is your control — the simplest thing that works.

### Stage 2 — Naive Experts (`train_naive_experts.py`)
Replace the single FFN with `N_EXPERTS` identical FFNs. No routing yet — every token is averaged across all experts. The "dumb version" that proves the architecture works before adding intelligence.

### Stage 3 — Routed MoE (`train_moe.py`)
One linear layer + softmax + `topk()`. The router learns on its own which expert handles which "kind" of token. No human labeling required.

### Stage 4 — Balanced MoE (`train_balanced_moe.py`)
Without load balancing, all tokens flood expert 0 (rich-get-richer collapse). The fix from the Switch Transformer paper: penalize `f_i × P_i` — fraction of tokens routed to expert `i` times the mean routing probability. This is what Mixtral and DeepSeek use in production.

## Structure

```
src/moe/
  config.py              # DenseConfig, NaiveExpertsConfig, RoutedMoEConfig, BalancedMoEConfig
  data.py                # load_shakespeare(), get_batch()
  trainer.py             # train(), compute_loss(), sample() — shared across all stages
  types.py               # ModelOutput, ShakespeareData
  models/
    attention.py         # CausalSelfAttention
    blocks.py            # TransformerBlock (ffn injected — the key seam)
    dense.py             # FeedForward + DenseTransformer

train_dense.py           # Stage 1 entrypoint
train_naive_experts.py   # Stage 2 entrypoint
train_moe.py             # Stage 3 entrypoint
train_balanced_moe.py    # Stage 4 entrypoint
```

## Setup

```bash
uv sync
uv run python train_dense.py
```

## References

- [Switch Transformers (Fedus et al., 2021)](https://arxiv.org/abs/2101.03961)
- [Mixtral of Experts (Mistral AI, 2024)](https://arxiv.org/abs/2401.04088)
- [DeepSeekMoE (DeepSeek, 2024)](https://arxiv.org/abs/2401.06066)
