# Mixture of Experts — From Scratch

A ground-up implementation of Mixture of Experts (MoE) built as a 4-act educational journey.

## The Story

### Act 1 — Dense Baseline
Build a tiny transformer where every token passes through the same Feed-Forward Network (FFN). This is your control — the simplest thing that works.

### Act 2 — Experts Without Routing
Replace the single FFN with `N_EXPERTS` identical FFNs. No routing yet — every token still goes to every expert (or one at random). The "dumb version" that proves the architecture works before adding intelligence.

### Act 3 — Add the Router
One linear layer + softmax + `topk()`. The router learns on its own which expert handles which "kind" of token. No human labeling required.

### Act 4 — Load Balancing Loss
Without it, all tokens flood expert 0 (rich-get-richer collapse). The fix from the Switch Transformer paper: penalize `f_i × P_i` — fraction of tokens routed to expert `i` times the mean routing probability. This is what Mixtral and DeepSeek use in production.

## Structure

```
moe/
  act1_dense.py        # Tiny transformer with standard FFN
  act2_experts.py      # FFN replaced by N identical experts (no routing)
  act3_router.py       # Add linear router + topk selection
  act4_load_balance.py # Add Switch Transformer load balancing loss
```

## Setup

```bash
uv sync
uv run python act1_dense.py
```

## References

- [Switch Transformers (Fedus et al., 2021)](https://arxiv.org/abs/2101.03961)
- [Mixtral of Experts (Mistral AI, 2024)](https://arxiv.org/abs/2401.04088)
- [DeepSeekMoE (DeepSeek, 2024)](https://arxiv.org/abs/2401.06066)
