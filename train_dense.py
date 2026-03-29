"""
Stage 1: Dense Baseline Transformer
=====================================
This is your starting point — a plain transformer where EVERY token is processed
by the SAME feed-forward network (FFN). Think of it as a single "generalist" brain
that handles all kinds of tokens: vowels, consonants, punctuation, spaces.

This is the control experiment. We train it, record the loss, and use it as the
benchmark to compare against in the next stages when we introduce experts.

WHY START HERE?
---------------
Before we split knowledge across multiple "experts", we need to know:
  - What loss does a good baseline achieve?
  - How does the architecture behave?
  - What does reasonable generated text look like?

Without this baseline, we can't tell if adding experts actually helps.


ARCHITECTURE (data flows top to bottom)
-----------------------------------------

  Input: "To be or"   (a string of characters)
       |
       v
  [Character Tokenizer]
    Maps each char to an integer index.
    "T" -> 39, "o" -> 47, " " -> 1, ...
    Vocab size ~65 (all unique chars in Shakespeare)
       |
       v
  [Token Embedding]  +  [Positional Embedding]
    Each token index becomes a 128-dim vector.
    Each position (0, 1, 2, ...) also gets a 128-dim vector.
    These two are ADDED together so the model knows both
    WHAT the token is and WHERE it sits in the sequence.
       |
       v
  [TransformerBlock] x 4   <-- repeated 4 times
  |                        |
  |  [LayerNorm]           |   Normalize before attention (pre-norm style,
  |       |                |   more stable than post-norm).
  |  [CausalSelfAttention] |   Each token looks at all PREVIOUS tokens
  |       |                |   (not future ones — that would be cheating).
  |  [residual add]        |   Output is added back to the input.
  |                        |
  |  [LayerNorm]           |
  |       |                |
  |  [FeedForward (FFN)]   |   <-- THE PART WE REPLACE IN STAGE 2+
  |       |                |   A simple: Linear -> GELU -> Linear
  |  [residual add]        |   Every token goes through the SAME weights here.
  |________________________|
       |
       v
  [LayerNorm]  (final normalization)
       |
       v
  [Linear Head]  (128-dim -> vocab_size)
    Projects back to vocabulary size.
    Gives a score for every possible next character.
       |
       v
  [Softmax + Sample]
    Convert scores to probabilities, sample the next character.


WHAT IS THE FFN DOING?
-----------------------
The attention layer figures out WHICH tokens to look at.
The FFN decides WHAT to do with that information.

Concretely, it's just two linear layers with a GELU activation in between:

  FFN(x) = Linear_2( GELU( Linear_1(x) ) )

  Input dim:  128
  Hidden dim: 512  (ffn_mult=4, so 4x the embedding size)
  Output dim: 128

This is the "memory" of the model — where facts get stored.
It's ~66% of total parameters. In Stage 2, we'll replace this
single shared FFN with N independent FFNs ("experts").


WHAT TO EXPECT WHEN YOU RUN THIS
----------------------------------
Tuned for MacBook Air (Apple Silicon, 32GB unified memory).
  - Device:     MPS (auto-detected)
  - Batch size: 128  — MPS runs efficiently at this size
  - Block size: 256  — 2x more context than the naive default
  - Model size: ~5M parameters
  - Runtime:    ~15-20 minutes before thermal throttling kicks in

Training runs for 3000 steps, printing loss every 500 steps (6 checkpoints).

  Step    0 | train loss ~4.17  <- random (log(65) ≈ 4.17, model knows nothing)
  Step  500 | train loss ~2.40  <- learning character frequencies
  Step 1000 | train loss ~2.10  <- learning common sequences
  Step 1500 | train loss ~1.98  <- learning word shapes
  Step 2000 | train loss ~1.90  <- converging
  Step 2500 | train loss ~1.85
  Step 3000 | train loss ~1.80  <- baseline locked in

A final text sample is generated at the end. At this loss level,
the output looks "Shakespeare-shaped" but not coherent — that's expected.
The goal is not great text, it's a reproducible baseline number.

Example output you might see:
  "Whe hath the sonce the hath sorl that the king
   And the sore that the sore the sore the hath..."

NOTE: If the MacBook Air starts throttling (fans spin up, progress slows),
reduce batch_size to 64 in DenseConfig. It will train slower but cooler.

Save this final val loss — you'll compare it to Stage 2 onwards.


MODULES IMPORTED
-----------------
  DenseConfig      <- src/moe/config.py
    All hyperparameters in one place. Change block_size, n_embd, etc. here.

  load_shakespeare <- src/moe/data.py
    Downloads tiny_shakespeare from HuggingFace (~1MB).
    Builds char->int and int->char mappings.
    Returns train/val tensors (90/10 split).

  DenseTransformer <- src/moe/models/dense.py
    The full model: embeddings + 4x TransformerBlock + head.
    Each block uses FeedForward as the FFN slot.

  train            <- src/moe/trainer.py
    The training loop. Works for ALL stages — not just this one.
    If the model returns an aux_loss, it adds it to the main loss.
    (For Stage 1 there is no aux_loss — that comes in Stage 4.)
"""

from moe.config import DenseConfig
from moe.data import load_shakespeare
from moe.models.dense import DenseTransformer
from moe.trainer import train


def main() -> None:
    cfg = DenseConfig()
    print(f"device: {cfg.device}")

    # Load and tokenize the dataset. ~1MB download on first run, cached after.
    dataset = load_shakespeare()

    # Build the model and move it to the right device (cpu / mps / cuda).
    model = DenseTransformer(cfg, dataset.vocab_size).to(cfg.device)

    # Run the training loop. Prints loss every eval_interval steps.
    # Generates a sample when done.
    train(cfg, model, dataset)


if __name__ == "__main__":
    main()
