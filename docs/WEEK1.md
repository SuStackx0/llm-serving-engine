# Week 1: Foundation & Model Loading (Days 1-5)

## Overview: What You're Building This Week

By the end of Week 1, you'll have a **working "dumb" LLM inference engine** that can:
- Load a real LLM model from HuggingFace
- Take text input from a user
- Generate text one token at a time
- Output the generated text

**No optimization yet.** No fancy batching. Just the basic machinery that makes an LLM work.

Think of it like building the **chassis of a car** (engine, wheels, steering) before adding the turbo and aerodynamics.

---

## The Big Picture: What is an LLM Anyway?

Before diving in, let's demystify what's happening when you type "What is AI?" into ChatGPT:

```
User Input: "What is AI?"
           ↓
   Tokenizer (converts words to numbers)
           ↓
   LLM Model (neural network with billions of parameters)
           ↓
   First number output: 2048 (represents a token/word)
           ↓
   Detokenizer (converts number back to word: "AI")
           ↓
   User sees: "AI"
           ↓
   Repeat until user hits stop or we've generated enough tokens
```

That's it. LLMs are **fancy token prediction machines**. They predict: "Given these tokens, what comes next?"

Your job Week 1: Build the plumbing so this loop works.

---

## Day 1: Load Model Weights

### What Are "Weights"?

An LLM like TinyLlama-1.1B has **1.1 billion parameters**. These are just numbers (stored as floating-point values like 0.52893, -1.2847, etc.).

When you download TinyLlama from HuggingFace, you're downloading:
- **Model weights** (the actual numbers, ~2.2 GB as FP16)
- **Model config** (metadata: how many layers? heads? what's the hidden size?)

Think of it like this:
```
Model = Instructions (code)
      + Weights (learned knowledge from training on 4 trillion tokens)

Your Job: Load the weights into GPU memory so we can use them
```

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
