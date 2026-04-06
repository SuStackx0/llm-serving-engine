# Week 1: Complete LLM Inference Theory & Architecture

## Table of Contents
1. [The Overall Vision](#the-overall-vision)
2. [The Complete Data Flow](#the-complete-data-flow)
3. [Model Architecture](#model-architecture)
4. [Deep Dive: Rotary Position Embeddings (RoPE)](#deep-dive-rotary-position-embeddings-rope)
5. [Deep Dive: Self-Attention Mechanism](#deep-dive-self-attention-mechanism)
6. [Deep Dive: Feed-Forward Networks](#deep-dive-feed-forward-networks)
7. [Deep Dive: Prediction & Generation](#deep-dive-prediction--generation)
8. [The Layer Architecture](#the-layer-architecture)
9. [Complete Information Flow](#complete-information-flow)

---

## The Overall Vision

### What is an LLM?

A **Large Language Model (LLM)** is fundamentally a **sophisticated pattern predictor**. It learned from billions of words to understand:
- How words relate to each other
- What typically comes after a given sequence
- The structure and meaning of language

But it's **not** conscious or truly understanding. It's a mathematical machine that's extremely good at predicting probabilities.

### The Core Insight

```
LLM = Neural Network Trained on Text Prediction

Training: "Given the first N words, predict word N+1"
Inference: Same task, but we autoregress—use our prediction as next input

Example:
Training data points:
  "The cat sat on the" → [prediction: "mat"]
  "The quick brown" → [prediction: "fox"]
  "Hello how" → [prediction: "are"]
  
The model learns patterns and can apply them to new sequences:
  "The dog jumped over the" → [prediction: "fence"]
```

### From Theory to Implementation

You're building an **inference engine**—the machinery that takes a trained model and generates text with it. This week, you implemented the complete pipeline from text input to text output.

---

## The Complete Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ START: User Input                                           │
│ Example: "Building an inference engine is cool"             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Tokenization (Day 2)                                │
│ Converts text → Token IDs                                   │
│                                                             │
│ "Building an inference engine is cool"                      │
│         ↓                                                   │
│ [2534, 381, 4823, 8234, 338, 12890]                        │
│                                                             │
│ Tool: SentencePiece Tokenizer                              │
│ Output: List of integers (0-32000 range)                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Embedding Lookup (Day 2)                            │
│ Converts Token IDs → Dense Vectors                          │
│                                                             │
│ Token ID: 2534                                              │
│      ↓ (lookup in embedding table)                          │
│ Vector: [0.234, -0.891, 0.045, ..., 0.123]                │
│         (2048 numbers per token)                            │
│                                                             │
│ With 6 tokens:                                              │
│ Input shape: [batch=1, seq_len=6, hidden_dim=2048]         │
│                                                             │
│ Tool: model.embed_tokens.weight (32000 × 2048 lookup table)│
│ Output: Dense vectors                                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3-24: Transformer Layers (Days 3, 4, 5)              │
│ Refines representations 22 times                            │
│                                                             │
│ Each layer contains:                                        │
│ • Self-Attention (understands relationships)               │
│ • Feed-Forward MLP (thinks deeply)                          │
│ • Residual connections (preserves information)             │
│ • RMSNorm (keeps signals stable)                           │
│                                                             │
│ Input:  [1, 6, 2048] (batch, seq_len, dim)                │
│ Output: [1, 6, 2048] (same shape, refined values)         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 25: Final Normalization (Day 5)                        │
│ Prepares for output head                                    │
│                                                             │
│ rms_norm(layer_output)                                      │
│ Same shape: [1, 6, 2048]                                   │
│                                                             │
│ Tool: model.norm.weight                                     │
│ Output: Normalized hidden states                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 26: Extract Last Token (Day 5)                         │
│ Only the last token matters for next-token prediction      │
│                                                             │
│ Shape [1, 6, 2048] → [1, 2048]                            │
│ (extracting position 6)                                    │
│                                                             │
│ Output: 2048 numbers representing the final token context  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 27: Output Head (Day 5)                                │
│ Projects hidden state → Vocabulary predictions              │
│                                                             │
│ [1, 2048] @ [2048, 32000] = [1, 32000]                    │
│                                                             │
│ Each of 32000 positions is a logit:                        │
│ logit[token_id] = score for how likely that token is       │
│                                                             │
│ Tool: lm_head.weight [32000, 2048]                         │
│ Output: Logits (unbounded scores)                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 28: Argmax / Decoding (Day 5)                          │
│ Converts logits → Token ID → Human-readable text           │
│                                                             │
│ Logits: [0.2, 15.3, 0.1, ..., 2.1]  (32000 values)        │
│    ↓ (find index of max)                                   │
│ Token ID: 1234  (the one with logit 15.3)                 │
│    ↓ (reverse lookup in tokenizer)                         │
│ Token: " cool"  (or whatever token 1234 represents)        │
│    ↓ (append to sequence)                                  │
│ New prediction: "Building an inference engine is cool "    │
│                                                             │
│ Tool: argmax() and tokenizer.id_to_piece()                │
│ Output: Human-readable text                                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ Optional: Repeat for Autoregressive Generation              │
│                                                             │
│ To generate multiple tokens:                                │
│ 1. Take the new full sequence                              │
│ 2. Process through entire pipeline again                   │
│ 3. Get next token prediction                               │
│ 4. Append and repeat (loop back to STEP 1)                │
│                                                             │
│ This is how text generation works!                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ OUTPUT: Generated Text                                      │
│ "Building an inference engine is cool. It's a challenging..." │
└─────────────────────────────────────────────────────────────┘
```

---

## Model Architecture

### TinyLlama-1.1B Specifications

```
Model: TinyLlama-1.1B-Chat-v1.0
├─ Parameters: ~1.1 Billion
│
├─ Embedding Layer
│  └─ Input Dimension: 32,000 (vocabulary)
│  └─ Output Dimension: 2,048 (hidden size)
│
├─ 22 Transformer Layers
│  └─ Each layer contains:
│     ├─ Self-Attention Block
│     │  ├─ 32 Query Heads
│     │  ├─ 4 Key/Value Heads (Grouped-Query Attention)
│     │  └─ Head Dimension: 64
│     └─ MLP Block (Feed-Forward)
│        ├─ Gate Projection to 5,632 dimensions
│        ├─ Up Projection to 5,632 dimensions
│        └─ Down Projection back to 2,048
│
├─ Final Layer Normalization
│  └─ RMSNorm with learned scaling
│
└─ Output Head
   └─ Projects [2,048] → [32,000]
      (32,000 = vocabulary size for next-token prediction)
```

---

## Deep Dive: Rotary Position Embeddings (RoPE)

### The Problem: Why Do We Need Position Information?

Original Transformers used **absolute position embeddings**: add a vector based on position.

```
Token "cat" at position 0:     [0.1, 0.2, 0.3, ...] + pos_embed_0
Token "cat" at position 100:   [0.1, 0.2, 0.3, ...] + pos_embed_100

Problem: Position embeddings lose effectiveness for long sequences
         Model struggles to understand relative positions
```

### RoPE Solution: Geometric Rotation

**Key Insight**: Use complex number multiplication to **rotate** vectors based on position.

```
Complex number multiplication = 2D rotation

If we treat embedding pairs as complex numbers:
  (a + bi) × e^(iθ) rotates the vector by angle θ

RoPE applies position-dependent rotations:
  Position 0:   Rotate by 0°
  Position 1:   Rotate by θ
  Position 7:   Rotate by 7θ
  Position 100: Rotate by 100θ
```

### How RoPE Works: Step by Step

#### Step 1: Precompute Rotation Matrices

For each dimension pair, calculate rotation frequency. Different dimensions get different frequencies.

**Mathematical Formula**:
$$\theta_j = 10000^{-2j/d}$$

Where:
- $j$ = dimension pair index (0 to d/2)
- $d$ = embedding dimension (64 per head)
- $\theta$ = base (10,000)

This creates a **frequency spectrum**: low frequencies for early dimensions, high frequencies for later dimensions.

**Why different frequencies?**
- Lower dimensions capture local patterns (nearby tokens)
- Higher dimensions capture global patterns (distant tokens)

#### Step 2: Apply Rotations to Q and K

```python
def apply_rotary_emb(x, freqs_cis, seq_len):
    # x shape: [batch, seq_len, heads, head_dim]
    # freqs_cis shape: [seq_len, head_dim//2] (precomputed rotations)
    
    # Step 1: Reshape x as complex numbers
    # Pair up dimensions: [a, b] → a + bi
    x_complex = reshape_as_complex(x)
    
    # Step 2: Multiply by rotation (this is the magic!)
    # a + bi × e^(iθ) = rotated vector
    x_rotated = x_complex * freqs_cis
    
    # Step 3: Convert back to real numbers
    return as_real(x_rotated)
```

### Why This is Brilliant

#### 1. Relative Position Awareness

```
Without RoPE:
  Token at pos 0 and pos 100: same vector
  
With RoPE:
  Token at pos 0:   rotated by 0°   → unchanged
  Token at pos 100: rotated by 100° → rotated version
  
In attention: Dot product of rotated vectors encodes DISTANCE!
  pos_0 • pos_0:   Dot product with self ≈ 1 (aligned)
  pos_0 • pos_1:   Dot product slightly lower (7° apart)
  pos_0 • pos_100: Dot product much lower (huge angle apart)
```

This means attention naturally prefers nearby tokens!

#### 2. Extrapolation to Longer Sequences

```
RoPE was trained on seq_len=2048

Can it handle seq_len=4096?
  Position 3000: Rotate by 3000°
  
Because RoPE is relative, it still works!
  The angle difference between positions still encodes distance
  Model doesn't care if we're at positions 0-7 or 3000-3007
  (the relative angle is the same)
```

#### 3. Frequency Spectrum Interpretation

```
Lower freq dims:    Rotate slowly
  "Who's near me?" (capture local structure)
  
Mid freq dims:      Rotate at medium speed
  "What's in my context window?" (medium-range dependencies)
  
Higher freq dims:   Rotate quickly
  "What's the overall document structure?" (long-range patterns)
```

### Practical Example

```
Embedding dimension: 64 (8 pairs of complex numbers)

Position 0:
  Input:  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
  Pair 0: 0.1 + 0.2i  × e^(i·0°)        = 0.1 + 0.2i  (unchanged)
  Pair 1: 0.3 + 0.4i  × e^(i·0.1°)      ≈ 0.3 + 0.4i  (tiny rotation)
  Result: Almost unchanged from input

Position 7:
  Input:  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
  Pair 0: 0.1 + 0.2i  × e^(i·7°)        ≈ [rotated]
  Pair 1: 0.3 + 0.4i  × e^(i·0.7°)      ≈ [slightly rotated]
  Result: Clearly different from input (position 0)
```

### Verification Code

```python
# Position 0: rotation angle ≈ 0
# Change should be tiny
diff_pos_0 = norm(original - rotated)  # ≈ 0.0002

# Position 7: rotation angle ≈ 7° to 0.07°
# Change should be noticeable
diff_pos_7 = norm(original - rotated)  # ≈ 0.85

# Position 100: even more rotation
# Change should be large
diff_pos_100 = norm(original - rotated)  # ≈ 1.2
```

The magnitude of change proves RoPE is encoding position!

---

## Deep Dive: Self-Attention Mechanism

### The Core Question Self-Attention Answers

**"Given my current token, which other tokens in the sequence are important to understand me?"**

### The Analogy: Search Engine for Token Relationships

Imagine a search engine where:
1. **You ask a question** (Query)
2. **Documents broadcast what they contain** (Keys)
3. **Relevant documents provide information** (Values)

```
Query: "What about animals?"
  ↓
Search through all documents
  ↓
Document 1 relevance: 0.8 (has "dogs", "cats")
Document 2 relevance: 0.1 (has "numbers")
Document 3 relevance: 0.7 (has "birds", "fish")
  ↓
Extract info from documents weighted by relevance
  ↓
Result: Mostly about animals
```

This is self-attention!

### Mathematical Formulation

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Breaking it down:

1. **Compute Similarity**: $QK^T$
   - Multiply each query by all keys
   - Result: How similar is each query to each key?
   - Shape: [seq_len, seq_len]

2. **Scale by Dimension**: $\div \sqrt{d_k}$
   - Prevents values from becoming too extreme
   - Keeps softmax stable
   - $d_k = 64$ (head_dim)

3. **Convert to Probabilities**: $\text{softmax}(..., \dim=-1)$
   - Normalizes across keys
   - Each query now has a probability distribution over keys
   - Shape: [seq_len, seq_len] with each row summing to 1

4. **Aggregate Values**: $(\ldots) V$
   - Weight-average all values by attention probabilities
   - Shape: [seq_len, d_k]

### Why Multiple Heads?

Different heads learn different types of relationships:

```
Head 1: Semantic similarity
  "cat" ↔ "animals", "pet"

Head 2: Syntactic relationships
  "the" ↔ nouns
  "quickly" ↔ verbs

Head 3: Named entity relationships
  "John" ↔ proper nouns

Head 32: Some learned pattern
  ...

Combined: Richer understanding!
```

### Grouped-Query Attention (GQA)

Standard multi-head attention has 32 Q, K, V heads each.

GQA optimization:
```
32 Query heads    ← Learn 32 different question types
4 Key heads       ← Learn 4 ways to summarize content
4 Value heads     ← Learn 4 ways to provide info

Each Q head shares a KV pair:
  Q_heads 0-7   → K_head0, V_head0
  Q_heads 8-15  → K_head1, V_head1
  Q_heads 16-23 → K_head2, V_head2
  Q_heads 24-31 → K_head3, V_head3

Benefit: 8x fewer KV computation
         Still have diverse queries
```

### Full Attention Block Flow

```
Input: Embeddings [1, 6, 2048]

├─ Project to Q: @ W_q [2048, 2048] → [1, 6, 2048]
├─ Project to K: @ W_k [2048, 512]  → [1, 6, 512]
├─ Project to V: @ W_v [2048, 512]  → [1, 6, 512]

├─ Reshape for heads:
│  ├─ Q: [1, 6, 2048] → [1, 32, 6, 64]
│  ├─ K: [1, 6, 512]  → [1, 4, 6, 64]
│  └─ V: [1, 6, 512]  → [1, 4, 6, 64]

├─ Apply RoPE rotations
│  ├─ Q ← apply_rotary_emb(Q)
│  └─ K ← apply_rotary_emb(K)

├─ Expand K and V for GQA
│  ├─ K: [1, 4, 6, 64] → [1, 32, 6, 64]  (repeat 8x)
│  └─ V: [1, 4, 6, 64] → [1, 32, 6, 64]  (repeat 8x)

├─ Compute Attention Scores:
│  scores = (Q @ K^T) / sqrt(64)          → [1, 32, 6, 6]
│  weights = softmax(scores)               → [1, 32, 6, 6]  (per-head)

├─ Apply to Values
│  attention_out = weights @ V             → [1, 32, 6, 64]

├─ Merge Heads
│  merged = [1, 32, 6, 64] → [1, 6, 2048]

└─ Project Output: @ W_o [2048, 2048] → [1, 6, 2048]
   Final output: [1, 6, 2048]  ← Same shape as input!
```

### Why Attention is Powerful

1. **Parallelizable**: All tokens compute attention simultaneously
2. **Position-aware**: RoPE adds position information
3. **Learnable**: Weight matrices learned during training
4. **Interpretable**: Attention weights show which tokens matter
5. **Flexible**: Can attend to any position in sequence

---

## Deep Dive: Feed-Forward Networks

### The Role of MLP in Transformers

After attention (which handles **relationships**), MLP (which handles **computation**).

**Analogy**: 
- Attention = "Understanding what matters"
- MLP = "Thinking deeply about the implications"

### Architecture: SwiGLU

```
Input [batch, seq, 2048]
  ↓
├─ Gate:  SiLU(Input @ W_gate)      → [batch, seq, 5632]
├─ Up:    (Input @ W_up)            → [batch, seq, 5632]
│
└─ Multiply: Gate * Up              → [batch, seq, 5632]
             (element-wise)
  ↓
  Down: (Gate * Up) @ W_down        → [batch, seq, 2048]
  ↓
Output [batch, seq, 2048]
```

### Why Expand to 5632?

```
Input space:   2048 dimensions
Expanded:      5632 dimensions  (2.75x expansion)

Why expand?
- Allows learning non-linear transformations
- More "computational power" in hidden space
- Like letting neural network think in higher dimensions
- Then contract back down to output

Analogy: 
  Working in 2048D: can only do simple transformations
  Working in 5632D: can do complex transformations
  Then project back to 2048D with learned answer
```

### SiLU vs ReLU Activation

```
ReLU(x) = max(0, x)
  Problem: Kills all negative values (information loss)
  
SiLU(x) = x * sigmoid(x)
  Behavior:
    x < 0:  Small positive value (keeps some info)
    x = 0:  0
    x > 0:  x multiplied by value < 1
  
  Advantage: Smoother, keeps more information
             Better gradients during training
```

### RMSNorm: The Stabilizer

```
def rms_norm(x, weight, eps=1e-6):
  rms = sqrt(mean(x^2) + eps)
  normalized = x / rms
  return normalized * weight
```

**Step 1: Calculate scale**
```
x = [1000, 0.1, 500]
x^2 = [1000000, 0.01, 250000]
mean(x^2) = 416666.67
rms = sqrt(416666.67) ≈ 645.5
```

**Step 2: Normalize**
```
normalized = [1000/645.5, 0.1/645.5, 500/645.5]
           = [1.55, 0.00015, 0.77]
```

**Step 3: Learned scaling**
```
weight = [0.5, 2.0, 0.8]  (learned during training)
output = [0.775, 0.0003, 0.616]
```

**Why this matters**: 
- Keeps all dimensions at similar magnitude
- Prevents some dimensions from dominating
- Allows stable training with large models

### The Complete Layer

Each transformer layer:
```
Input: [batch, seq, 2048]
  ↓
┌─── Attention Block ───┐
│ Q, K, V projections   │─ Apply RoPE
│ Multi-head attention  │
│ Output projection     │
└──────────────────────┘
  ↓
  Add residual (input + attention_out)
  ↓
  RMSNorm
  ↓
┌──── MLP Block ────┐
│ Gate projection   │
│ * Up projection   │
│ Down projection   │
└──────────────────┘
  ↓
  Add residual (normed_input + mlp_out)
  ↓
Output: [batch, seq, 2048]
```

**Residual connections**: Skip connections that add input to output.
- Solves gradient flow problem
- Allows very deep networks (22 layers)
- Preserves information from earlier layers

---

## Deep Dive: Prediction & Generation

### From Hidden State to Token: The Output Head

```
Hidden state of last token: [2048 numbers]
  ↓
Multiply by weight matrix: [2048, 32000]
  ↓
Logits: [32000 numbers]
  
each logit[i] = score for token i
```

### Understanding Logits

Logits are not probabilities. They're unbounded scores.

```
Example logits: [0.2, 15.3, 0.1, ..., 2.1, -5.0]

Interpretation:
  Token 1 (logit=15.3): "Very likely" (highest score)
  Token 4 (logit=2.1):  "Somewhat likely"
  Token 0 (logit=0.2):  "Unlikely"
  Token n (logit=-5.0): "Very unlikely"
```

### Decoding Strategies

#### 1. Greedy Decoding
```python
next_token = argmax(logits)  # Always pick highest
```
✓ Fast, deterministic
✗ Can get stuck in repetitive loops

#### 2. Softmax + Sampling
```python
probs = softmax(logits)
next_token = sample(probs)
```
✓ Diverse outputs, still biased toward high-probability tokens
✗ Can generate nonsense

#### 3. Top-K Sampling
```python
probs = softmax(logits)
top_k_indices = argsort(logits)[:k]
probs[non_top_k] = 0
probs = renormalize(probs)
next_token = sample(probs)
```
✓ Balanced: diverse but not random
✗ Slightly slower

#### 4. Temperature Scaling
```python
logits = logits / temperature
probs = softmax(logits)
next_token = sample(probs)

Low temperature (0.1):  Sharper distribution → more confident
High temperature (2.0): Flatter distribution → more diverse
```

**We use greedy decoding** in our implementation.

### Autoregressive Generation

```
Iteration 1:
  Input: "Building an inference engine is cool"
  Output hidden state for last token (position 6)
  Predict next token: [sample] "!"
  Complete sequence: "Building an inference engine is cool!"

Iteration 2:
  Input: "Building an inference engine is cool!"
  Output hidden state for last token (position 7)
  Predict next token: [sample] " "
  Complete sequence: "Building an inference engine is cool! "

Iteration 3:
  Input: "Building an inference engine is cool! "
  Output hidden state for last token (position 8)
  Predict next token: [sample] "You"
  Complete sequence: "Building an inference engine is cool! You"

Continue until:
- Model predicts End-of-Sequence (EOS) token, or
- Reach max_tokens limit, or
- User stops
```

### Why We Use Last Token Only

```
Full hidden states: [1, 6, 2048]  (6 tokens, 2048 dimensions each)

For next token prediction:
- Token 1's context: can look back 0 positions
- Token 2's context: can look back 1 position
- ...
- Token 6's context: can look back 5 positions (full history!)

So using token 6's hidden state gives maximum historical context.
It's learned to integrate all previous information!
```

### One-Pass vs Iterative Generation

```
One-pass (used in our code):
  Input: "How are you?"
  Output: "I'm doing well"
  
  Efficiency: Just ONE forward pass through model
  Limitation: Can only predict ONE token ahead

Iterative (used in actual chat models):
  Step 1: Predict "I" after "How are you?"
  Step 2: Predict "'m" after "How are you? I"
  Step 3: Predict " doing" after "How are you? I 'm"
  ...
  
  Efficiency: N forward passes for N tokens
  Advantage: Can generate long sequences
```

---

## The Layer Architecture

### Complete Transformer Layer Structure

```
class TransformerLayer:
  input: [batch, seq_len, 2048]
  
  # Attention Block
  q = input @ W_q           # [batch, seq, 2048]
  k = input @ W_k           # [batch, seq, 512]
  v = input @ W_v           # [batch, seq, 512]
  
  q = reshape_heads(q)      # [batch, 32, seq, 64]
  k = reshape_heads(k)      # [batch, 4, seq, 64]
  v = reshape_heads(v)      # [batch, 4, seq, 64]
  
  q = apply_rope(q)         # Apply RoPE
  k = apply_rope(k)         # Apply RoPE
  
  k = repeat_interleave(k)  # [batch, 32, seq, 64]
  v = repeat_interleave(v)  # [batch, 32, seq, 64]
  
  scores = q @ k^T / sqrt(64)
  weights = softmax(scores)
  
  attention_out = weights @ v  # [batch, 32, seq, 64]
  attention_out = reshape_back(attention_out)  # [batch, seq, 2048]
  attention_out = attention_out @ W_o  # Project
  
  # Skip connection
  after_attention = input + attention_out
  
  # MLP Block  
  norm = rms_norm(after_attention)
  
  gate = silu(norm @ W_gate)  # [batch, seq, 5632]
  up = norm @ W_up            # [batch, seq, 5632]
  combined = gate * up        # [batch, seq, 5632]
  
  mlp_out = combined @ W_down  # [batch, seq, 2048]
  
  # Skip connection
  output = after_attention + mlp_out
  
  return output  # [batch, seq_len, 2048]
```

### 22 Layers in Sequence

```
Embedding: [batch, seq, 2048]
  ↓
Layer 1: Attention + MLP
  ↓
Layer 2: Attention + MLP
  ↓
... (layers 3-21)
  ↓
Layer 22: Attention + MLP
  ↓
Final RMSNorm
  ↓
Output Head (projection to 32k vocab)
  ↓
Logits: [batch, seq, 32000]
```

### Parameter Count Per Layer

```
One transformer layer contains:

Attention:
  Q projection: 2048 × 2048 = 4M
  K projection: 2048 × 512  = 1M
  V projection: 2048 × 512  = 1M
  O projection: 2048 × 2048 = 4M
  ─────────────────────────────
  Subtotal: ~10M

MLP:
  Gate: 2048 × 5632 = 11.5M
  Up:   2048 × 5632 = 11.5M
  Down: 5632 × 2048 = 11.5M
  ─────────────────────────────
  Subtotal: ~34.5M

Norms and biases: ~1M

Total per layer: ~45M

× 22 layers ≈ 990M (out of 1.1B total)
```

---

## Complete Information Flow

At the end of Day 5, you have a **complete but unoptimized LLM inference system**:

```
User Input: "Once upon a time"
           ↓
       Tokenizer
           ↓ [4 tokens]
       Embedding Layer
       + RoPE encoding
           ↓ [4, 2048]
       22× Transformer Blocks
           (each with multi-head attention)
           ↓ [4, 2048]
       Linear projection to logits
           ↓ [4, 50K]
       Softmax + Sample
           ↓ [1 token]
       First output: "there" (example)
           ↓
       Repeat: generate next token
           ↓
       Keep generating until EOS or max_length
           ↓
Output: "Once upon a time there was a kingdom..."
```

### The Information Transformation

```
Input: String of text  (~100 bytes)
   ↓
Tokens: Integer IDs  (~5-20 numbers)
   ↓
Embeddings: Dense vectors  ([5, 2048] = 10k numbers)
   ↓
After Layer 1: Refined vectors  ([5, 2048] = 10k numbers)
   ↓
After Layer 2: More refined  ([5, 2048] = 10k numbers)
   ↓
... (Layers 3-22)
   ↓
After Layer 22: Highly processed  ([5, 2048] = 10k numbers)
   ↓
Last token only: Final state  ([2048] = 2k numbers)
   ↓
Output logits: Probability scores  ([32000] = 32k numbers)
   ↓
Argmax: Single token ID  (1 number)
   ↓
Decoded: String of text  (~5 bytes)

What happened?
- Rich semantic information flowed through 22 layers
- Each layer refined understanding
- Attention learned relationships
- MLP processed deep patterns
- Output head made a decision
- Logits converted to prediction
```

---

## Summary: The Complete Model

You've implemented a **complete language model inference pipeline**:

### What You Built
1. **Day 1**: Model loading and weight management
2. **Day 2**: Text-to-vector conversion pipeline
3. **Day 3**: Position-aware representation via RoPE
4. **Day 4**: Relationship understanding via multi-head attention
5. **Day 5**: Deep processing via MLPs and token prediction

### What This Enables
- Taking any text prompt
- Processing it through 22 transformer layers
- Understanding context and relationships
- Making intelligent next-token predictions

### The Architecture You Understand
- **22 transformer layers**, each with attention + MLP
- **32 attention heads** (with 4 KV heads via GQA optimization)
- **RoPE encoding** of position information
- **SwiGLU gating** for non-linear processing
- **Residual connections** for stable deep learning
- **RMSNorm** for consistent signal magnitude

This is the same foundational architecture used by:
- GPT-2, GPT-3, GPT-4
- Claude, Llama
- Mistral, Phi, and thousands of other LLMs

The only differences in larger models are:
- More layers (80+ instead of 22)
- Larger dimensions (12,288 instead of  2,048)
- Bigger vocabulary
- More training data

**Congratulations—you understand how modern LLMs work!**

### Why Load into GPU?

**CPU**: Can do math, but SLOW (~100 tokens/sec)  
**GPU**: Designed for matrix multiplies, FAST (~1000+ tokens/sec)

We're loading into GPU because we want speed.

### How (Theory, Not Code)

1. **Download from HuggingFace Hub**
   - HuggingFace is like "GitHub for ML models"
   - You fetch: `meta-llama/Llama-2-7b-hf` or `TinyLlama/TinyLlama-1.1B`
   - Models are stored in `.safetensors` format (safe, fast binary format)

2. **Parse the config**
   - Extract metadata: "This model has 22 layers, 8 attention heads, hidden_dim=2048"
   - You need this to build the neural network structure

3. **Load weights into GPU**
   - Read binary files from disk
   - Convert to PyTorch tensors (PyTorch = deep learning library)
   - Move to GPU memory (`device='cuda'`)
   - Use FP16 (16-bit floats) to save memory (2x compression vs FP32)

### Prerequisites You Need to Understand

- **What is HuggingFace?** (Just: a model hub, like GitHub)
- **What's PyTorch?** (A library for tensor math)
- **What's GPU memory?** (Separate from CPU; needed for speed)
- **FP16 vs FP32?** (Floating point precision; FP16 is half the memory)

### How You'll Know It Works (Day 1 Success Criterion)

```
You run your code and see:
  ✓ Model downloaded (2.2 GB)
  ✓ Config says: "Layers: 22, Hidden: 2048, Heads: 8"
  ✓ Weights loaded to GPU
  ✓ Memory usage: ~4.5 GB (reasonable for 1.1B model)
  ✓ No errors
```

### Why This Matters

Without loaded weights, nothing else works. This is the **prerequisite prerequisite**. You're building the foundation.

---

## Day 2: Tokenizer & Input Preprocessing

### What is Tokenization?

Humans read words. Neural networks read **numbers**.

A tokenizer is a translator:
```
"Hello world" → [2882, 3186]
                (numbers representing those words)
```

### Why Do We Need This?

The LLM model **only understands numbers**. You can't feed it words directly.

Also, words are variable-length:
- "Hello" = 5 characters
- "world" = 5 characters
- But as tokens, they might be 1-2 tokens each

A tokenizer handles this ambiguity.

### How Tokenization Works (Theory)

There are different tokenization strategies:

**1. Character-level** (primitive)
```
"Hi" → [72, 105]  (ASCII codes for 'H', 'i')
Problem: Loss of word meaning
```

**2. Word-level** (used in old NLP)
```
"Hello world" → [vocabulary_index_hello, vocabulary_index_world]
Problem: Huge vocabulary (50K+ words), rare words unknown
```

**3. Byte-Pair Encoding (BPE)** (modern, what LLMs use)
```
Start with characters: H, e, l, l, o, space, w, o, r, l, d
Iteratively merge: "he" → token_42, "ll" → token_53, etc.
Result: "Hello world" → [token_for_He, token_for_llo, token_for_world]
Advantage: Handles unknown words, reasonable vocabulary (~50K tokens)
```

Modern LLMs like TinyLlama use **BPE tokenizers** (often via `tiktoken` or `sentencepiece`).

### What You're Building on Day 2

A **wrapper** around existing tokenizer that:

1. **Converts text → token IDs**
   - Input: "What is AI?"
   - Output: `[1, 1486, 338, 319, 29973]` (example token IDs)

2. **Handles padding**
   - Requests have different prompt lengths
   - Sometimes we need to "pad" shorter sequences with dummy tokens
   - Example:
     ```
     Request 1: "Hi" → 2 tokens → pad to 10: [2882, 3186, 0, 0, 0, 0, 0, 0, 0, 0]
     Request 2: "Hello world" → 3 tokens → pad to 10: [2882, 3186, 2154, 0, 0, 0, 0, 0, 0, 0]
     ```

3. **Special token handling**
   - `BOS` (Beginning of Sequence): Marks start
   - `EOS` (End of Sequence): Marks stop (when to stop generating)
   - `PAD` (Padding): Dummy token for sequences shorter than batch size

4. **Reverse operation (detokenization)**
   - Token IDs → text
   - The model outputs token 2882 → we convert back to "Hello"

### Prerequisites You Need

- **What is BPE?** (Byte-Pair Encoding; a text compression approach)
- **What are special tokens?** (BOS, EOS, PAD markers)
- **Why padding?** (ML prefers fixed-size batches)

### How You'll Know It Works (Day 2 Success Criterion)

```
You run your code with prompt "What is AI?" and see:
  ✓ Tokenized → [1, 1486, 338, 319, 29973]
  ✓ Token count: 5 (reasonable for short phrase)
  ✓ Detokenized → "What is AI?" (exact match back)
  ✓ Padding works: ["What is", "this is great"]
    both become [batch_size=2, sequence_length=5]
```

### Why This Matters

Tokenizers are the **input/output interface** of the model. Bad tokenization = garbage in, garbage out.

---

## Day 3: Basic Transformer Forward Pass (No KV Cache)

### What is a Transformer?

A transformer is a neural network architecture (invented in 2017, stands for **Attention Is All You Need**).

High-level structure:
```
Tokenized input [5 tokens from "What is AI?"]
           ↓
    Embedding Layer (token IDs → vectors of numbers)
           ↓
    Transformer Block (repeated 22 times for TinyLlama)
    - Self-Attention (tokens attend to each other)
    - Feed-Forward Network (dense matrix multiplies)
           ↓
    Output: one vector per input token
           ↓
    Linear layer (vector → logits over all possible next tokens)
           ↓
    Softmax (convert to probabilities)
           ↓
    Sample next token
```

### Why Build This?

Because it's the **core of any LLM**. Without understanding how inference works, you can't optimize it.

### How It Works (Theory)

**Step 1: Embedding**
```
Token IDs: [1, 1486, 338, 319, 29973]  (5 tokens)
After embedding (hidden_size=2048):
Result: [5, 2048] matrix (5 tokens, each is a 2048-dim vector)
```

A vector is just a list of numbers. It represents "meaning."

**Step 2: Transformer Block** (repeated 22 times)
```
Input: [5, 2048]
    ↓ Self-Attention
    Each token attends to all other tokens
    (token 0 asks: "what should I know from tokens 1,2,3,4?")
    ↓
    [5, 2048] (same shape, but now with context)
    ↓ Feed-Forward Network
    Two dense layers: hidden_size → 4*hidden_size → hidden_size
    ↓
    [5, 2048] (same shape, refined)
```

This repeats 22 times, progressively refining the vectors.

**Step 3: Output Logits**
```
After 22 blocks: [5, 2048]
Linear layer: hidden_size (2048) → vocab_size (~50K)
Result: [5, 50K] (5 tokens, each has a score for every possible next word)
```

**Step 4: Sampling**
```
Take the LAST token's logits: [50K]
Apply softmax: convert to probabilities (sum to 1)
Sample: pick next token based on probabilities
Result: token ID (e.g., 2882 = "This")
```

### Prerequisites You Need

- **What is self-attention?** (Each token pays attention to relevant context)
- **What's a linear layer?** (Matrix multiply: input × weight_matrix + bias)
- **What's softmax?** (Converts scores to probabilities)
- **What's a feed-forward network?** (Two dense layers with activation function)

### How You'll Know It Works (Day 3 Success Criterion)

```
You run your code with prompt "What is AI?" and see:
  ✓ Tokenize: 5 tokens
  ✓ Embed: [5, 2048]
  ✓ Pass through 22 transformer blocks
  ✓ Get output logits: [5, 50K]
  ✓ Last token logits used for next token prediction
  ✓ Output shape is correct (no errors)
  ✓ Numbers are in reasonable range (no NaN or Inf)
```

### Why This Matters

This is the **basic LLM loop**. Everything else this week builds on top of this foundation.

---

## Day 4: Rotary Positional Embeddings (RoPE)

### The Problem: How Does the Model Know Position?

Consider these two sequences:
```
Sequence 1: "The cat sat on the mat"
Sequence 2: "The mat sat on the cat"
```

The tokens are the same! But semantically they're different because **position matters**.

In sequence 1, "cat" is at position 1 and "mat" is at position 5.  
In sequence 2, they're reversed.

The model needs to know **which token is at which position**.

### The Old Way (Learned Embeddings)

Old transformers would add "position embeddings" directly:
```
Token embedding: [0.5, 0.2, 0.1, -0.3, ...]
Position embedding for pos=1: [0.1, 0.05, -0.2, 0.15, ...]
Add them: [0.6, 0.25, -0.1, -0.15, ...]  ← token+position

Problem: Fixed at training time (e.g., trained on seq_len=2048)
Can't handle longer sequences at inference!
```

### The New Way: Rotary Positional Embeddings (RoPE)

(This is why you'll implement RoPE manually on Day 4)

**Key Insight**: Instead of adding position info, **rotate** the token vectors based on position.

Think of it like a clock:
```
Hour hand = token vector
Rotation amount = position number

At position 0: no rotation (0°)
At position 1: rotate 10° (example)
At position 2: rotate 20°
...
```

Mathematical detail (you'll see this in ARCHITECTURAL_SPEC.md):
```
For dimension pair (2i, 2i+1) at position m:
[cos(m⋅θᵢ)  -sin(m⋅θᵢ)  ] [x_{2i}]
[sin(m⋅θᵢ)   cos(m⋅θᵢ)  ] [x_{2i+1}]
```

Translation: Apply a 2D rotation matrix to consecutive dimension pairs.

### Why RoPE is Better

1. **No learned parameters**: Just math, pure rotation
2. **Relative position natural**: If you rotate token_i and token_j by the same amount, their relative distance is preserved
3. **Extrapolates**: Can handle longer sequences at inference than training!
   - Trained on 4K tokens?
   - Can use 32K tokens at inference.
   - Why? Rotation math works for any position number.

### How You'll Implement It (Theory)

1. **Precompute rotation angles**
   - θᵢ = 10000^(-2i/d) for i = 0, 1, ..., d/2
   - Do this ONCE at initialization
   - Store sin/cos tables (fast lookup)

2. **At inference time**
   - For each position m in the sequence
   - Retrieve precomputed sin(m⋅θᵢ) and cos(m⋅θᵢ)
   - Apply rotation to query and key vectors
   - (You'll do this in the attention mechanism)

### Prerequisites You Need

- **What's a rotation matrix?** (A 2D transformation that rotates points)
- **Why apply to Q and K but not V?** (Positional info matters for attention, not for values)

### How You'll Know It Works (Day 4 Success Criterion)

```
You implement RoPE and run:
  ✓ Precompute sin/cos tables (no errors)
  ✓ Apply RoPE to Q and K vectors
  ✓ Test: relative distances are preserved
    - RoPE(vec, pos=0) · RoPE(vec, pos=5) 
    - RoPE(vec, pos=10) · RoPE(vec, pos=15)
    - These dot products should be similar (10 position gap preserved)
  ✓ Extrapolation works: can handle positions beyond training length
```

### Why This Matters

This is the **first optimization** on top of basic transformer. Modern LLMs (Llama2, Mistral) use RoPE. Understanding it deeply shows you're not just importing—you're **implementing research**.

---

## Day 5: Multi-Head Attention with RoPE Integration

### What is Attention?

Attention is the **core mechanism** of transformers.

Conceptually:
```
Query (Q): "What am I looking for?"
Key (K): "Here are labels for each token"
Value (V): "Here's the actual content of each token"

Attention = Look at all Keys, find best matches for my Query, 
            return Values of those matches
```

Mathematical:
```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
```

Translation:
```
Q @ K^T → Similarity matrix (how much does each query match each key?)
softmax → Convert to probabilities (which tokens should I attend to?)
... @ V → Weighted pool of values (I get the values of attended tokens)
```

### Why Multi-Head?

Instead of one attention head, use 8 or 16 heads in parallel:
```
Q: [seq_len, hidden_size]  (5 tokens, 2048 dimensions)
Split into 8 heads:
Q: [seq_len, 8 heads, 256 dimensions each]

Each head attends independently, then concatenate results
Benefit: Different heads learn different attention patterns
  - Head 1: "Look at the noun"
  - Head 2: "Look at the verb"
  - etc.
```

### How You'll Build It (Theory)

1. **Take token embeddings** (output from transformer blocks)
2. **Project to Q, K, V**
   - Three linear layers, each producing Q, K, V
   - Each is [seq_len, hidden_size]
3. **Reshape for multi-head**
   - [seq_len, hidden_size] → [seq_len, num_heads, head_dim]
   - head_dim = hidden_size / num_heads
4. **Apply RoPE to Q and K** (this is the integration from Day 4)
5. **Compute attention**
   - Score = Q @ K^T / sqrt(head_dim)
   - Probs = softmax(Score)
   - Output = Probs @ V
6. **Concatenate heads**
   - [seq_len, num_heads, head_dim] → [seq_len, hidden_size]
7. **Project back**
   - Linear layer to combine heads
   - Output: [seq_len, hidden_size] (same shape as input)

### Causal Masking (Important for LLMs)

LLMs generate left-to-right:
```
Generate token 1: look at token 0 only
Generate token 2: look at tokens 0,1 only
Generate token 3: look at tokens 0,1,2 only
... (cannot attend to future tokens!)
```

You implement this with a **causal mask**:
```
Attention scores before softmax:
Position:  0    1    2    3
Token 0:  [ 0.5  X    X    X  ]  (X = -∞, forces prob to 0)
Token 1:  [ 0.3  0.6  X    X  ]
Token 2:  [ 0.1  0.4  0.7  X  ]
Token 3:  [ 0.2  0.5  0.6  0.8]

After softmax with mask: future tokens have 0 probability ✓
```

### Prerequisites You Need

- **What's matrix multiplication?** (Basic linear algebra)
- **What does @ K^T mean?** (@ is multiply, T is transpose)
- **What's softmax?** (Converts scores to probabilities summing to 1)
- **Why causal mask?** (Can't attend to future in generative models)

### How You'll Know It Works (Day 5 Success Criterion)

```
You implement multi-head attention with RoPE:
  ✓ Forward pass runs without errors
  ✓ Output shape: [seq_len, hidden_size] (same as input)
  ✓ Causal mask works: future tokens have 0 attention
  ✓ Test prediction: "What is" should predict next token reasonably
    - Should NOT predict garbage
    - Should be a real word (use a reference model to check)
  ✓ RoPE integration: positions are properly encoded
```

### Why This Matters

Attention is the **differentiator** of transformers. Understanding it is essential before moving to caching optimizations (Week 2).

---

## End of Week 1: What Your System Looks Like

At the end of Day 5, you have a **complete but unoptimized LLM inference system**:

```
User Input: "Once upon a time"
           ↓
       Tokenizer
           ↓ [4 tokens]
       Embedding Layer
       + RoPE encoding
           ↓ [4, 2048]
       22× Transformer Blocks
           (each with multi-head attention)
           ↓ [4, 2048]
       Linear projection to logits
           ↓ [4, 50K]
       Softmax + Sample
           ↓ [1 token]
       First output: "there" (example)
           ↓
       Repeat: generate next token
           ↓
       Keep generating until EOS or max_length
           ↓
Output: "Once upon a time there was a kingdom..."
```

### Performance at End of Week 1

This system **works but is slow**:
- **TTFT**: ~3-5 seconds (terrible!)
  - Why? Recomputes full sequence for each token
  - Token 1: Process prompts tokens 0-3
  - Token 2: Process prompt tokens 0-3 AGAIN + new token 4
  - Wasteful!
- **TPOT**: ~1-2 seconds per output token (very slow)
- **Throughput**: ~1 token/second (crawling)

### What's Missing (You'll Add in Week 2-4)

- **KV Cache**: Cache attention values from prefill, reuse in decode (3x speedup)
- **Continuous Batching**: Process multiple requests at once (10x throughput)
- **Paged Attention**: Efficient memory management for long sequences

---

## Architectural Questions?

If you get stuck on:
- **How exactly does RoPE math work?** → See ARCHITECTURAL_SPEC.md > RoPE section
- **Multi-head attention details?** → See ARCHITECTURAL_SPEC.md > PagedAttention section (applies here too)
- **Where does tokenization fit?** → See ARCHITECTURAL_SPEC.md > (general flow)

## Prerequisites Learning Path

If you don't know these, spend 30 mins learning:

1. **PyTorch basics** (15 mins)
   - What's a Tensor?
   - How to create/reshape tensors
   - Tensor operations (reshape, transpose, matmul)

2. **Linear Algebra refresher** (15 mins)
   - Matrix multiplication
   - Transpose
   - What's a 2D rotation matrix?

3. **Transformer basics** (30 mins)
   - Read: "Attention Is All You Need" abstract + introduction
   - Watch: 3Blue1Brown transformer video (10 mins)

## Success Criteria for Week 1

By the end of Friday:
- ✅ Model loads and runs
- ✅ Can generate coherent text (5-10 tokens)
- ✅ Output is better than random gibberish
- ✅ All 5 days of components integrated
- ✅ You understand the data flow top-to-bottom

---

## Week 1 → Ready for Week 2

Week 2 builds on this foundation:

**Problem**: Current system recomputes K,V for every token generated.  
**Solution**: **KV Cache** - store and reuse them (next week)

But you need Week 1 working first. Don't skip it!

---

## Tips for Week 1

1. **Code incrementally**
   - Day 1: Load model
   - Day 2: Tokenize (call tokenizer, no forward pass yet)
   - Day 3: Run forward pass (ignore attention for now)
   - Day 4: Add RoPE
   - Day 5: Integrate into attention

2. **Test at each step**
   - Day 1: Print model shape
   - Day 2: Print tokenized input
   - Days 3-5: Print output shape

3. **Debug with small numbers**
   - Use small batch sizes (batch_size=1)
   - Use short prompts ("Hi" not "Explain quantum mechanics")
   - Print intermediate tensor shapes

4. **Read HuggingFace docs**
   - How to download models
   - How to use tokenizers
   - Standard patterns for transformers

5. **Refer to ARCHITECTURAL_SPEC.md when stuck**
   - It has pseudo-code and validation tests
   - Use those tests to verify your implementation

Good luck with Week 1! 🚀
