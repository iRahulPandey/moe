# Stage 1 — Dense Transformer: A Concrete Walkthrough

We trace the string **"To be or not"** step by step through every module in
`train_dense.py`. No hand-waving — actual shapes, actual numbers.

---

## The Task

The model learns one thing: **predict the next character given all previous characters**.

```
Input  X:  "To be or not to b"   (256 chars per sample)
Target Y:  "o be or not to be"   (same string, shifted right by 1)
```

At every position the model makes one prediction:

| Position | Context seen so far | True next char |
|----------|---------------------|----------------|
| 0        | `T`                 | `o`            |
| 1        | `To`                | ` `            |
| 2        | `To `               | `b`            |
| 3        | `To b`              | `e`            |
| 4        | `To be`             | ` `            |

One forward pass produces 256 predictions simultaneously (one per position).
Loss is averaged across all 256. This is why training is efficient.

---

## Step 1 — Tokenize (`data.py → load_shakespeare`)

Build a vocabulary of every unique character in the dataset (~65 chars).
Map each character to an integer index.

```
char → index
─────────────
'\n' →  0
' '  →  1
'!'  →  2
...
'T'  → 39
...
'b'  → 14
'e'  → 21
'o'  → 47
...
```

Our example string becomes:

```
"To be"  →  [39, 47, 1, 14, 21]
```

The full dataset (~1M chars) becomes one long integer tensor.
`get_batch()` slices random windows of length 256 from it.

```
X shape:  [128, 256]   # B=128 samples, T=256 chars each
Y shape:  [128, 256]   # same, shifted by 1
```

---

## Step 2 — Embed (`DenseTransformer.__init__`)

An integer by itself carries no meaning — `39` doesn't tell the model anything
useful about `'T'`. We need to turn it into a rich vector.

**Two lookup tables:**

```
tok_emb:  Embedding(65, 256)   # one 256-dim row per character
pos_emb:  Embedding(256, 256)  # one 256-dim row per position
```

For our token `'T'` at position `0`:

```
tok_emb[39] = [ 0.12, -0.34,  0.67,  0.05, -0.88, ... ]   # "what kind of char is T?"
pos_emb[ 0] = [ 0.05,  0.88, -0.23,  0.41,  0.62, ... ]   # "where in the sequence am I?"

combined    = [ 0.17,  0.54,  0.44,  0.46, -0.26, ... ]   # "T at position 0"
             ─────────────────────────────────────────
                        element-wise addition
```

After embedding the full batch:

```
x shape:  [128, 256, 256]   # B=128, T=256 positions, C=256 features
```

Both tables start random and are updated by backprop — by the end of training,
similar characters (vowels, punctuation, capitals) end up with similar vectors.

---

## Step 3 — Attention (`CausalSelfAttention`)

**The question attention answers:** for each token, which past tokens should
I mix information from?

### The causal mask

The model cannot peek at future tokens (that would be cheating at prediction).
A lower-triangular mask enforces this:

```
         T    o    _    b    e
    T  [ 1    0    0    0    0 ]   ← T can only see itself
    o  [ 1    1    0    0    0 ]   ← o can see T and itself
    _  [ 1    1    1    0    0 ]
    b  [ 1    1    1    1    0 ]
    e  [ 1    1    1    1    1 ]   ← e can see all 5 previous tokens
```

`0` positions are filled with `-inf` before softmax, so they get weight `~0`.

### What Q, K, V mean

The model learns three projections from each token's vector:

```
Q (query):  "What information am I looking for?"
K (key):    "What information do I contain?"
V (value):  "What do I actually pass along if selected?"
```

Attention weight between position `i` and position `j`:

```
score(i, j) = Q[i] · K[j] / sqrt(64)   # dot product, scaled by head_dim
weight(i,j) = softmax(score(i, :))      # normalise across all visible positions
```

Output at position `i` = weighted sum of all V vectors.

### Four heads in parallel

With `n_heads=4` and `n_embd=256`, each head works on a 64-dim slice:

```
head_dim = 256 / 4 = 64
```

Each head learns to pay attention to different things — one might track
syntactic structure, another might track rhyme, etc. Their outputs are
concatenated back to 256-dim.

```
x shape in:   [128, 256, 256]
x shape out:  [128, 256, 256]   # same — attention re-mixes, doesn't resize
```

The output at each position is a new 256-dim vector that has "seen" and mixed
in information from every allowed past position.

---

## Step 4 — FeedForward (`FeedForward` in `models/dense.py`)

After attention figures out *which* tokens to look at, the FFN decides
*what to do* with that information.

```python
FFN(x) = Linear(1024 → 256)(  GELU(  Linear(256 → 1024)(x)  )  )
```

Visualised per token:

```
 [256]  ──Linear──►  [1024]  ──GELU──►  [1024]  ──Linear──►  [256]
                      expand              nonlinearity          compress
```

The expansion to 1024 (`ffn_mult=4`) lets the model consider many combinations
of the 256 input features before compressing back. GELU introduces nonlinearity
so the model can learn non-linear patterns.

### The key fact about Stage 1

**Every token — `'T'`, `'o'`, `' '`, `','`, `'!'` — passes through the exact
same FFN weights.**

One generalist brain handles everything.

| Token type  | Example chars       | FFN treatment   |
|-------------|---------------------|-----------------|
| Vowel       | a, e, i, o, u       | same weights    |
| Consonant   | b, c, d, f, ...     | same weights    |
| Punctuation | . , ; ! ?           | same weights    |
| Space       | ` `                 | same weights    |
| Newline     | `\n`                | same weights    |

This works — the model reaches val loss ~1.80. But it's inefficient.
A single FFN must simultaneously be good at all of them.

> **This is the problem Stage 2 solves.** We give the model N separate FFNs
> and let the data decide which tokens go to which expert.

---

## Step 5 — Output Head + Loss (`DenseTransformer.forward`, `trainer.py`)

After 4 transformer blocks, we have a 256-dim vector for each of the 256
positions. We project to vocabulary size to get one score per character:

```
head: Linear(256 → 65)   # no bias

logits shape: [128, 256, 65]   # a score for each of 65 chars, at every position
```

**Example — predicting the next char after `'T'`:**

```
logits at position 0:
  'a' → -1.2
  'b' →  0.3
  ...
  'o' →  3.8   ← highest
  'p' →  0.1
  ...

softmax → probabilities:
  'o' → 0.42   (model is fairly confident)

true next char: 'o'  ✓
```

**Cross entropy loss** measures how wrong the model was:

```
loss = -log( probability assigned to the correct character )
     = -log(0.42)
     = 0.87
```

Lower is better. The theoretical minimum for a random model is:

```
-log(1/65) = log(65) ≈ 4.17
```

That's exactly what you see at step 0. If your step-0 loss is not ~4.17,
something is wrong with the initialisation.

---

## Step 6 — The Training Loop (`trainer.py → train`)

```
repeat 3000 times:
  1. get_batch()                      → X [128,256], Y [128,256]
  2. model(X, Y)                      → logits, loss (CE averaged over all 128×256 predictions)
  3. optimizer.zero_grad()
  4. loss.backward()                  → compute gradient for every weight
  5. clip_grad_norm_(params, 1.0)     → cap gradient magnitude (prevents explosions)
  6. optimizer.step()                 → nudge every weight in the right direction
```

Every 500 steps we pause to measure loss on held-out validation data:

```
step    0  |  train 4.17  val 4.17   ← pure random, model knows nothing
step  500  |  train 2.40  val 2.43   ← learned character frequencies
step 1000  |  train 2.10  val 2.14   ← learning common bigrams (th, he, in...)
step 1500  |  train 1.98  val 2.02   ← learning word shapes
step 2000  |  train 1.90  val 1.95
step 2500  |  train 1.84  val 1.90
step 3000  |  train 1.80  val 1.86   ← baseline locked in
```

The small gap between train and val loss (0.06) tells us the model is
generalising — it hasn't just memorised the training set.

**Save this val loss: 1.86. Every future stage must beat it.**

---

## What a sample looks like at the end

After 3000 steps the model generates character-by-character from a blank context:

```
KING RICHARD:
Whe hath the sonce the hath sorl that the king
And the sore that the sore the sore the hath
I will the will the will the sore the sore
```

It has learned:
- Capital letters start lines
- Words are separated by spaces
- `KING`, `RICHARD` are common patterns
- Common bigrams and trigrams

But it can't hold long-range coherent structure. That's fine —
our goal here was a working baseline, not great Shakespeare.

---

## Module Map (which file does what)

```
train_dense.py                  ← entry point, 15 lines
    │
    ├── DenseConfig             src/moe/config.py
    │     block_size=256, n_embd=256, n_heads=4, n_layers=4
    │     batch_size auto-set: cuda=256, mps=128, cpu=32
    │
    ├── load_shakespeare()      src/moe/data.py
    │     downloads tiny_shakespeare (~1MB, cached after first run)
    │     returns ShakespeareData(train, val, stoi, itos, vocab_size)
    │
    ├── DenseTransformer        src/moe/models/dense.py
    │     uses CausalSelfAttention   ← src/moe/models/attention.py
    │     uses TransformerBlock      ← src/moe/models/blocks.py
    │     uses FeedForward           ← same file (dense.py)
    │
    └── train()                 src/moe/trainer.py
          shared across all 4 stages — only the model changes
```

---

## Parameter count

With `n_embd=256, n_layers=4, ffn_mult=4`:

| Component          | Parameters            |
|--------------------|-----------------------|
| tok_emb            | 65 × 256 = 16,640     |
| pos_emb            | 256 × 256 = 65,536    |
| Attention (×4)     | 4 × (4 × 256²) ≈ 1M  |
| FFN (×4)           | 4 × (2 × 256 × 1024) ≈ 2M |
| Head               | 256 × 65 = 16,640     |
| **Total**          | **~5M parameters**    |

Small enough to train in ~15 minutes on a MacBook Air (MPS).
Large enough to produce recognisable Shakespeare.
