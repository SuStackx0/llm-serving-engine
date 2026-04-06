# Day 4: Multi-Head Self-Attention with RoPE

## Overview
Implemented **Multi-Head Self-Attention**, the core mechanism that allows the model to understand relationships between tokens in a sequence. This is the "brain" that actually *thinks* about what each token should pay attention to.

## Problem Statement
After embedding tokens as vectors, the model needs to understand **which tokens should focus on which other tokens**. A token like "bank" could mean:
- A financial institution (pay attention to "money", "account")
- A river bank (pay attention to "river", "water")

The context determines meaning. Self-attention is how the model learns this.

---

## The Concept: Query, Key, and Value

Think of self-attention like a **search engine for token relationships**:

### Query (Q)
- **What am I asking about?** 
- Each token asks: "What other tokens are relevant to me?"
- Derived by multiplying embeddings by Q weight matrix: `Q = Embedding × W_q`

### Key (K)
- **What am I offering?**
- Each token broadcasts: "Here's what I am, in case it's relevant to someone"
- Derived by multiplying embeddings by K weight matrix: `K = Embedding × W_k`

### Value (V)
- **What information do I carry?**
- Each token says: "If you're interested in me, here's the actual information you get"
- Derived by multiplying embeddings by V weight matrix: `V = Embedding × W_v`

---

## The Mechanism: Scaled Dot-Product Attention

### Step 1: Compute Attention Scores
```
Scores = (Q × K^T) / √(head_dim)
```

- **Q × K^T**: How similar is each query to each key?
- **√(head_dim)**: Scale down so values don't get too large (prevents extreme softmax values)

**Example meaning:**
```
If Q[position 3] = [0.5, 0.1, 0.9, ...]  (token 3's query)
If K[position 0] = [0.4, 0.2, 0.8, ...]  (token 0's key)

Similarity Score = 0.5*0.4 + 0.1*0.2 + 0.9*0.8 + ...
                 = 0.2 + 0.02 + 0.72 + ...
                 = High score → Token 3 should attend to Token 0
```

### Step 2: Convert Scores to Weights (Softmax)
```
Weights = Softmax(Scores)
```

- Softmax converts scores to probabilities (sum to 1)
- High scores → higher probability
- Prevents any single token from overwhelming the others

**Example:**
```
Raw Scores:   [1.5, 0.2, 0.8]
Softmax:      [0.7, 0.1, 0.2]
Interpretation: Token should pay 70% attention to token 0,
                10% to token 1, 20% to token 2
```

### Step 3: Apply Weights to Values
```
Output = Weights × V
```

- Multiply the attention weights by the value vectors
- High-weight values contribute more to the output

**Example:**
```
Weights:     [0.7, 0.1, 0.2]
V_vectors:   [[1, 0], [0, 1], [1, 1]]

Output = 0.7*[1, 0] + 0.1*[0, 1] + 0.2*[1, 1]
       = [0.7, 0] + [0, 0.1] + [0.2, 0.2]
       = [0.9, 0.3]  ← Weighted combination
```

---

## Multi-Head Attention: The Power

Why use multiple attention heads instead of just one?

### Single Head = One Type of Relationship
```
Single Head might learn:
"Pay attention based on semantic meaning"
(All heads focus on the same relationship type)
```

### Multiple Heads = Multiple Relationship Types
```
Head 1: "Pay attention based on semantic similarity"
Head 2: "Pay attention based on grammatical role"
Head 3: "Pay attention based on distance (nearby tokens)"
Head 4: ... 32 heads total

Result: Model understands many types of relationships simultaneously
```

### The Numbers in TinyLlama
- **32 Query Heads**: 32 different ways to ask "what's relevant?"
- **4 Key/Value Heads**: More efficient (Grouped-Query Attention)
- **Each head operates on 64-dimensional space** (2048 / 32 = 64)

---

## Grouped-Query Attention (GQA): Efficiency

The model uses an optimization: instead of 32 heads for K and V (which is expensive), it uses only 4 heads and repeats them 8 times.

```
Q heads:  32 (independent queries)
K heads:   4 (shared keys)
V heads:   4 (shared values)

Why work?
- Each group of 8 Q heads shares the same K and V
- Reduces computation from 32×32 to 32×4
- Still allows diverse queries to interact with shared knowledge
```

---

## RoPE Integration: Position Information

Recall from Day 3: **RoPE encodes position by rotating vectors**.

### Why Apply RoPE to Q and K Only?
```
Q <- Apply RoPE  (Each query knows its position)
K <- Apply RoPE  (Each key knows its position)
V <- No RoPE     (Values just carry information)
```

Position-aware similarities are what matter for attention!

### The Effect
```
Without RoPE:
  Token at position 0 and position 100 look identical
  → Model can't distinguish between "1st word" and "100th word"

With RoPE:
  Token at position 0: rotated by 0°
  Token at position 100: rotated by 100° (on different frequency bands)
  → Attention automatically prefers nearby tokens (relative positions)
```

---

## The Implementation in TinyLlama

### Setup Dimensions
```python
n_heads_q = 32   # Query heads
n_heads_kv = 4   # Key/Value heads
head_dim = 64    # Each head's dimension (2048 / 32)
```

### Step 1: Project Embeddings to Q, K, V
```python
q = input_embeddings @ W_q.T  # [batch, seq, 2048] @ [2048, 2048] → [batch, seq, 2048]
k = input_embeddings @ W_k.T  # [batch, seq, 2048] @ [2048, 512] → [batch, seq, 512]
v = input_embeddings @ W_v.T  # [batch, seq, 2048] @ [2048, 512] → [batch, seq, 512]
```

Note: K and V have fewer dimensions (512 vs 2048) because they're shared across head groups.

### Step 2: Reshape for Multi-Head Processing
```python
# [batch, seq, feature] → [batch, heads, seq, head_dim]
q = q.reshape(batch, seq, 32, 64).transpose(1, 2)   # [batch, 32, seq, 64]
k = k.reshape(batch, seq, 4, 64).transpose(1, 2)    # [batch, 4, seq, 64]
v = v.reshape(batch, seq, 4, 64).transpose(1, 2)    # [batch, 4, seq, 64]
```

### Step 3: Apply RoPE Rotations
```python
q = apply_rotary_emb(q, freqs_cis)  # Rotate each position's query
k = apply_rotary_emb(k, freqs_cis)  # Rotate each position's key
```

### Step 4: Expand K and V for GQA
```python
# Repeat K and V 8 times so all 32 Q heads have compatible K and V
k = k.repeat_interleave(8, dim=1)  # [batch, 4, seq, 64] → [batch, 32, seq, 64]
v = v.repeat_interleave(8, dim=1)  # [batch, 4, seq, 64] → [batch, 32, seq, 64]
```

### Step 5: Compute Attention
```python
# Scaled dot-product: (Q @ K^T) / sqrt(head_dim)
scores = (q @ k.transpose(-2, -1)) / sqrt(64)

# Convert to probabilities
weights = softmax(scores)

# Apply to values
attention_output = weights @ v  # [batch, 32, seq, 64]
```

### Step 6: Merge Heads and Project Output
```python
# Concatenate all 32 head outputs
attention_output = attention_output.transpose(1, 2)               # [batch, seq, 32, 64]
attention_output = attention_output.reshape(batch, seq, 2048)    # [batch, seq, 2048]

# Project back to embedding space
output = attention_output @ W_o.T  # [batch, seq, 2048] @ [2048, 2048] → [batch, seq, 2048]
```

---

## Key Insights

1. **Attention is learned similarity**: The model learns what Q, K, V weight matrices should be
2. **Position matters**: RoPE makes position an intrinsic part of the similarity calculation
3. **Multiple perspectives**: 32 heads learn different ways to relate tokens
4. **Efficiency**: GQA reduces computation while maintaining expressiveness
5. **Residual flow**: The attention output is added back to the input (happens in the larger architecture)

---

## Practical Example: "The cat sat on the mat"

### What Each Head Might Learn
- **Head 1**: "Pronouns should attend to nouns they refer to"
  - "the" → high attention to "cat", "mat"
  
- **Head 2**: "Verbs should attend to their subjects"
  - "sat" → high attention to "cat"
  
- **Head 3**: "Articles should attend to following nouns"
  - "the" → high attention to "cat"
  
- **Head 4-32**: Other linguistic and semantic patterns

### Combined Effect
The model learns that different aspects of language are captured by different attention patterns, and all happen simultaneously in parallel!

---

## Verification from Code

After implementation:
```python
print(f"Query shape after rotation: {q.shape}")      # [batch, 32, seq_len, 64]
print(f"Key shape after repeat: {k.shape}")          # [batch, 32, seq_len, 64]
print(f"Attention weights shape: {weights.shape}")   # [batch, 32, seq_len, seq_len]
print(f"Final output shape: {output.shape}")         # [batch, seq_len, 2048]
```

This completes the self-attention layer—the foundation of transformer power!
