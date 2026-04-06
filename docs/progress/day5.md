# Day 5: Feed-Forward Network (MLP) and Prediction

## Overview
Implemented the **Feed-Forward Network (MLP)** and connected it to the output head for **generating predictions**. This is where the model actually makes decisions—after attention understands relationships, the MLP processes them to generate the next word probability distribution.

---

## Problem Statement: From Understanding to Action

After self-attention (Day 4), the model understands **relationships between tokens**. But understanding isn't enough. The model needs to:

1. **Process the information** deeper (MLP layer)
2. **Convert to predictions** (output head)
3. **Pick the next token** (argmax)

Think of it like reading:
- **Attention**: Understanding what each word relates to
- **MLP**: Thinking deeply about what comes next
- **Output Head**: Deciding which word should come next

---

## The MLP Block: SwiGLU Activation

TinyLlama uses **SwiGLU**, a gating mechanism that's more sophisticated than simple feedforward networks.

### Traditional MLP (Outdated)
```
Output = ReLU(Input @ W1) @ W2
```

Simple, but less expressive.

### SwiGLU (Modern)
```
Output = (SiLU(Input @ W1) * (Input @ W3)) @ W2
```

Three projection matrices instead of two! Let's understand each:

---

## The Components: W1, W3 (Gate and Up), and W2 (Down)

### Gate Projection (W1)
```python
gate = SiLU(embedding @ W1.T)
```

- **Function**: Applies a smooth activation (SiLU) and learns **what to let through**
- **W1 shape**: [2048, 5632] (expands dimension)
- **SiLU function**: Smooth version of ReLU
  ```
  SiLU(x) = x * sigmoid(x)
  
  Behavior:
  - If x < 0: Small positive value (not killed like ReLU)
  - If x = 0: 0 (like ReLU)
  - If x > 0: Larger than x (amplified)
  ```

**Why this matters**: Unlike ReLU which fully kills negative values, SiLU keeps some information, making training smoother.

### Up Projection (W3)
```python
up = embedding @ W3.T
```

- **Function**: Another expansion that learns **what information to carry**
- **W3 shape**: [2048, 5632] (same as W1, expands dimension)
- **No activation**: Raw linear projection

**Why two expansions?**
```
Dual pathway:
- W1: "Which neurons should be active?" (gating)
- W3: "What information should flow?" (information pathway)

The multiplication: gate * up
Combines: "Use this gate level" × "This information"
```

### Down Projection (W2)
```python
output = (gate * up) @ W2.T
```

- **Function**: Contracts back to original dimension
- **W2 shape**: [5632, 2048] (contracts back down)

**The full flow**:
```
2048 (embedding dimension)
  ↓
[Expand via W1 and W3 to 5632]
  ↓
[Apply gating and element-wise multiply]
  ↓
[Contract back via W2 to 2048]
```

This expansion-contraction pattern allows the MLP to **learn non-linear transformations** in a higher-dimensional space.

---

## RMSNorm: Keeping the Signal Clean

Before and after MLP blocks, the model uses **RMSNorm** (Root Mean Square Normalization):

```python
def rms_norm(x, weight):
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)) * weight
```

### Why Normalize?
```
Without normalization:
- After embedding: vectors might be [0.1, 1000, 0.05, ...]  (huge variance)
- After attention: vectors might be [-100, 0.001, 50, ...]  (different scale)
- Model becomes unstable

With RMSNorm:
- Keeps all vectors centered around "normal" scale
- Prevents some dimensions from dominating
- Makes training stable
```

### What RMSNorm Does
```
1. Calculate: sqrt(mean(x^2))  — RMS (root mean square)
2. Divide: x / RMS   — Normalize to unit scale
3. Multiply: * weight  — Learned per-dimension scaling

Special case:
- Input: [0.1, 1000, 0.05, ...]
- RMS: ~500
- After norm: [0.0002, 2, 0.0001, ...]  (now similar scale)
- After weight scaling: [w1*0.0002, w2*2, w3*0.0001, ...]  (learnable per dim)
```

The weight vector is learned during training, allowing the model to re-scale each dimension as needed.

---

## Layer Structure: Attention + MLP Block

In TinyLlama, each layer consists of:

```
Input
  ↓
[Self-Attention Layer]
  ↓
[Add layer input (residual connection)]
  ↓
[RMSNorm]
  ↓
[MLP Block (Gate-Up-Down)]
  ↓
[Add layer input (residual connection)]
  ↓
Output
```

**Residual connections ("skip connections")**: 
- `output = input + sublayer_output`
- Allows gradients to flow directly through layers
- Prevents vanishing gradient problem

---

## The Output Head: From Hidden States to Predictions

### What is the Output Head?
The output head is a **simple linear layer** that projects from model dimension to vocabulary size:

```python
logits = last_token_vector @ W_output.T
```

- **Input**: [2048] (the last token's hidden state)
- **Weight matrix**: [2048, 32000] (32,000 = vocabulary size)
- **Output**: [32000] (score for each possible next token)

### Why Only the Last Token?
```
Input sequence: "Building an inference engine is cool!"
Tokens:         [1, 2, 3, 4, 5, 6]

For prediction, we only care about:
"Given tokens 1-6, what comes next (token 7)?"

So we extract the hidden state of token 6 (position -1 / last)
and project it to vocabulary space.

This is called "last-token-pooling" or "decoder" prediction.
```

---

## Converting Logits to Predictions

### Step 1: Get Logits
```python
logits = embedding @ W_output.T
# Shape: [1, 32000]
# Values: some positive, some negative (unbounded)
```

**Logits are not probabilities yet.**

### Step 2: Convert to Probabilities (Optional)
```python
probabilities = softmax(logits)
# Shape: [1, 32000]
# Values: between 0 and 1, sum to 1
```

**Probabilities show confidence in each token.**

### Step 3: Make a Decision
```python
# Option A: Greedy (always pick highest)
next_token_id = argmax(logits)

# Option B: Temperature sampling (random, but biased toward high logits)
next_token_id = sample(softmax(logits / temperature))

# Option C: Top-k sampling (only sample from top k tokens)
next_token_id = sample(top_k(softmax(logits)))
```

In our implementation, we use **greedy** (Option A): pick the token with highest logit.

---

## Full Prediction Pipeline: One Token at a Time

### Processing the Prompt
```
Input: "Building an inference engine is"
Tokens: [1, 2, 3, 4, 5, 6]

Process through:
  Tokenizer: string → token IDs
  Embedding: token IDs → dense vectors
  22 Transformer Layers: vectors → context-aware vectors
  Output Head: vectors → logits
  Argmax: logits → next token ID

Output: Token ID 234 (let's say this is "cool")
```

### Generating More Tokens (Autoregressive)
```
Step 1: Predict after "Building": → gets "an"
Step 2: Predict after "Building an": → gets "inference"
Step 3: Predict after "Building an inference": → gets "engine"
...

Each step:
1. Add previous prediction to sequence
2. Process entire sequence through model
3. Look at logits for last token
4. Predict next token
5. Repeat
```

---

## The Complete Information Flow

```
[User Input]
    ↓
[Tokenizer (SentencePiece)]
    ↓
[Token Embedding]  ← Day 2
    ↓
[Layer 1: Attention + MLP]
    ↓    ← Self-Attention (Day 4)
    ↓    ← RoPE Rotation (Day 3)
    ↓    ← MLP Processing (Day 5)
[Layer 2: Attention + MLP]
    ↓
[Layer 3: Attention + MLP]
    ↓
... (22 layers total)
    ↓
[Layer 22: Attention + MLP]
    ↓
[Final RMSNorm]  ← Prepare for output head
    ↓
[Output Head @ W_output]  ← Day 5
    ↓
[Logits: 32,000 scores]
    ↓
[Argmax: Pick highest score]
    ↓
[Token ID]
    ↓
[Decode with SentencePiece]
    ↓
[Human-Readable Token]
    ↓
[Output]
```

---

## The Code: Step by Step

### Step 1: RMSNorm After Attention
```python
norm_output = rms_norm(final_attention_output, weights["model.layers.0.post_attention_layernorm.weight"])
```

Prepares attention output for MLP processing.

### Step 2: Gate Projection
```python
ffn_gate = F.silu(norm_output @ w1.T)
```

Squeezing through SiLU activation to create a gating mechanism.

### Step 3: Up Projection
```python
ffn_up = norm_output @ w3.T
```

Parallel information pathway.

### Step 4: Multiply Gate and Up
```python
combined = ffn_gate * ffn_up
```

Element-wise multiplication: letting gate control information flow.

### Step 5: Down Projection
```python
ffn_output = combined @ w2.T
```

Compress back to 2048 dimensions.

### Step 6: Input Normalization
```python
final_norm_output = rms_norm(last_token_vector, weights["model.norm.weight"])
```

Final normalization before output head (typically done at model's final layer).

### Step 7: Output Head
```python
logits = final_norm_output @ weights["lm_head.weight"].T
```

Project to vocabulary size: [2048] → [32000]

### Step 8: Prediction
```python
next_token_id = torch.argmax(logits).item()
predicted_word = sp_model.id_to_piece(next_token_id)
```

Convert logits to token ID, then decode to human-readable form.

---

## Practical Example from Code

```python
# Input: "Building an inference engine is cool!"
# After tokenization and embedding
# After 22 layers of attention and MLP

# Extract last token's hidden state (position -1)
last_hidden = layer_output[:, -1, :]  # [1, 2048]

# Apply final norm
normed = rms_norm(last_hidden, model.norm.weight)  # [1, 2048]

# Project to vocab
logits = normed @ lm_head.weight.T  # [1, 32000]

# Find best token
scores = logits[0]  # [32000]
best_idx = argmax(scores)  # scalar

# Decode
next_word = tokenizer.id_to_piece(best_idx)  # string

print(f"Next token: {next_word}")  # e.g., " " or "!" or "This"
```

---

## Key Insights

1. **MLP is the "thinking" layer**: After attention understands relationships, MLP performs complex computations
2. **SwiGLU is powerful**: Gating mechanism learns both what to process and how to process it
3. **Residual connections enable deep learning**: Skip connections let gradients flow and prevent instability
4. **RMSNorm keeps scales consistent**: Prevents exploding/vanishing values
5. **Output head is just a projection**: Simple linear layer after 22 sophisticated layers
6. **Greedy decoding is simplest**: Picks highest probability token sequentially

---

## Why This Matters

Together, Days 1-5 show the complete inference pipeline:
- **Day 1**: Load the model's learned knowledge
- **Day 2**: Convert text to vectors
- **Day 3**: Encode position information
- **Day 4**: Understand relationships between tokens
- **Day 5**: Think deeply and make predictions

This is how GPTs work! The architecture you implemented is the same fundamental structure used by ChatGPT, Claude, and other modern large language models—just with more layers and more parameters.
