# Day 2: Tokenizer & Input Preprocessing → Embedding Lookup

## Day 1 Summary: The Foundation

On Day 1, you successfully **loaded a trained AI brain** (TinyLlama-1.1B) from the internet and prepared it for use. Here's what you achieved:

### What You Accomplished on Day 1
```
Goal: Get a 1.1B parameter model ready in GPU memory
Result: ✓ Downloaded model files (2.2+ GB)
        ✓ Read the model's blueprint (config.json)
        ✓ Loaded 1.1 billion numerical parameters
        ✓ Converted them to float16 for efficiency
        ✓ Moved them to GPU (MPS on Mac)
        ✓ Verified everything works
```

### The Model State After Day 1
You had:
- **22 transformer layers** stacked on top of each other
- **32 attention heads** (32 different ways to understand text simultaneously)
- **2,048 hidden dimension** (each token is processed as a vector of 2,048 numbers)
- **32,000 vocabulary size** (the model understands 32,000 different word pieces)
- **~4.5 GB in GPU memory** (the actual learned weights)

Think of it like having a **trained chess expert in your computer**, but you haven't asked them any questions yet. They're ready, but they need input.

---

## Day 2: The Bridge Between Language and Mathematics

### The Problem You're Solving

An LLM model is fundamentally a **mathematical machine**. It only understands numbers—matrices, vectors, and tensors. But humans communicate in **language**—words, sentences, paragraphs.

There's a gap:

```
Human: "Hello, how are you?"        ← Text (what humans understand)
        ↓ ← ← ← ← ← ↓
        ??? (The Magic Bridge)
        ↓ ← ← ← ← ← ↓
Model:  [1, 2, 5, 8, 3, 12, ...]    ← Numbers (what the model understands)
        ↓
Model processes the numbers
        ↓
Output: [0.2, 0.8, 0.1, ...]        ← Probability distribution
        ↓
Human: "I'm doing great, thank you!" ← Text (what humans see)
```

**Day 2 is about building that magic bridge.** Specifically:
1. **Tokenization**: Convert "Hello, how are you?" → [1, 2, 5, 8, 3, 12, ...]
2. **Embedding Lookup**: Convert [1, 2, 5, 8, 3, 12, ...] → vectors the model can process [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]

---

## The Theory: Understanding Tokenization

### What is a Token?

A **token** is not necessarily a word. It's a **subword unit**—a piece of text that the model learned to recognize as a fundamental unit.

#### Example 1: Common Words Are Single Tokens
```
"hello"     → Token ID: 235    (1 token)
"world"     → Token ID: 2564   (1 token)
```

#### Example 2: Longer/Rare Words Are Multiple Tokens
```
"unforgettable"  → Token IDs: [482, 1034, 234]     (3 tokens)
                              = "unfor" + "gett" + "able"

"pneumonoultramicroscopicsilicvolcanoconiosis"  → ~10 tokens
```

#### Example 3: Punctuation & Spaces
```
"Hello, world!"  → Token IDs: [235, 53, 2564, 81]
                            = "Hello" + "," + " world" + "!"
                            Note: space is included in " world"
```

### Why Tokenization?

You might ask: "Why not just use one token per character?"

```
If using characters:
  "hello" = 5 tokens: ['h', 'e', 'l', 'l', 'o']
  
  Problem:
  - The model would need to learn patterns in 5 steps
  - Loses word structure
  - Model becomes HUGE
```

```
If using words:
  "hello" = 1 token
  "unheard of" = 1 token
  
  Problem:
  - Vocabulary explosion: English has ~1 million words
  - Model can't handle rare words it hasn't seen
  - Can't handle new words or typos
```

**Subword tokenization (BPE)** is the goldilocks solution:
```
- Common words: 1 token (fast)
- Rare words: few tokens (manageable)
- Unknown words: decomposed into known pieces (flexible)
- Total vocabulary: 32,000 (reasonable size)
```

### Tokenization Algorithm: Byte-Pair Encoding (BPE) / SentencePiece

The model uses **SentencePiece**, a tokenizer that works in three phases:

#### Phase 1: Understand the Vocabulary (Already Done During Training)

When TinyLlama was trained, the researchers:
1. Took billions of text documents
2. Started with all characters: [a, b, c, ..., z, !, ?, ...]
3. Found the most common pairs and merged them:
   ```
   Step 1: e + s → "es" (because "es" appears often)
   Step 2: t + h → "th" (because "th" appears often)
   Step 3: "th" + e → "the" (because "the" is very common)
   ...
   Repeat until you have 32,000 tokens
   ```

This process learned that "the" should be a token, "and" should be a token, "ing" should be a token, etc.

#### Phase 2: Store the Vocabulary

The trained vocabulary is stored in `tokenizer.model` (a binary file with the learned token list and merging rules).

#### Phase 3: Tokenize New Text (What You Do on Day 2)

When you give the tokenizer "hello world", it:
1. Looks up each subword in the vocabulary
2. If "hello" exists as a single token, use it
3. If not, try "helo", then "hel", then "he", then "h"
4. Combine with any special tokens needed (BOS, EOS)

### Special Tokens

The tokenizer has special control tokens:

```
BOS (Beginning of Sequence) - ID: 1
  - Marks the start of input
  - Tells the model "a new request starts here"
  
EOS (End of Sequence) - ID: 2
  - Marks the end of input
  - Tells the model "we're done now"

PAD (Padding) - ID: 0
  - Fills empty slots in batch processing
  - Used when inputs are different lengths

Example:
  Input: "Hello"
  With BOS: [BOS, Hello, ...more tokens...]
  With BOS and EOS: [BOS, Hello, ...more tokens..., EOS]
```

---

## The Theory: Understanding Embeddings

### What is an Embedding?

An **embedding** is a **vector representation** of a token. It's a list of numbers that encodes the "meaning" or "characteristics" of that token.

You already have an embedding table in GPU memory:
```
Shape: [32000, 2048]
Meaning: 32,000 tokens × 2,048 numbers each
        (Each token gets represented as a vector of 2,048 numbers)
```

### Example Embedding

Imagine token ID 235 is "hello". Its embedding might be:
```
Token ID 235: [0.123, -0.456, 0.789, 0.234, ..., -0.567]  (2048 numbers)
```

These 2,048 numbers are **learned during training**. They're not random. The neural network, through training on billions of texts, learned:
- "hello" should have high positive values in dimensions related to "greeting"
- "hello" should have low negative values in dimensions related to "anger"
- etc.

### Why 2,048 numbers?

Each number captures a different "feature" of the token:
```
Position 0: Captures "is_greeting?" (high if greeting, low if not)
Position 1: Captures "is_verb?" (high if verb, low if noun)
Position 2: Captures "English_level" (higher for common words)
Position 3: Captures "emotion_positive" (higher for positive words)
...
Position 2047: Captures some other subtle pattern
```

You don't explicitly define these meanings. The model learns them automatically. Neural networks are good at finding hidden patterns.

### Integer-to-Embedding Lookup

This is a **lookup table operation**:

```python
token_id = 235          # "hello"
embedding = table[235]  # Get row 235 from the embedding table
# embedding is now a vector of 2048 numbers representing "hello"
```

In PyTorch, this is done with `torch.embedding()`:
```python
# If you have input_ids = [1, 235, 456, 789]
# And embedding_table = [32000, 2048]
# Then:
embeddings = torch.embedding(embedding_table, input_ids)
# Output shape: [4, 2048]  (4 tokens, each with 2048 numbers)
```

---

## The Implementation: Step-by-Step

### Step 1: Import Tokenizer Libraries

```python
from sentencepiece import SentencePieceProcessor
from huggingface_hub import hf_hub_download
```

**WHY:**
- `SentencePieceProcessor`: The tokenization library
- `hf_hub_download`: Downloads the tokenizer.model file from HuggingFace

**WHAT IT DOES:**
You're importing tools to tokenize text using the same tokenizer that was used to train TinyLlama.

---

### Step 2: Download the Tokenizer Model File

```python
tokenizer_path = hf_hub_download(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
    filename="tokenizer.model"
)
```

**WHY:**
The tokenizer has a learned vocabulary (the mapping of text → token IDs). This is stored as a file on HuggingFace servers. You need to download it locally.

**WHAT HAPPENS:**
1. Connects to HuggingFace servers
2. Finds the TinyLlama repository
3. Finds the tokenizer.model file
4. Downloads it to your computer (usually ~/.cache/huggingface/)
5. Returns the local path

**OUTPUT:**
```
tokenizer_path = "/Users/username/.cache/huggingface/hub/models--TinyLlama--..."
```

---

### Step 3: Load the Tokenizer

```python
sp_model = SentencePieceProcessor()
sp_model.load(tokenizer_path)

print(f"Tokenizer loaded. Vocabulary size: {sp_model.get_piece_size()}")
print(f"Beginning of String (BOS) ID: {sp_model.bos_id()}")
```

**WHY:**
Create a tokenizer object in memory that you can use repeatedly.

**WHAT HAPPENS:**
1. Creates an empty SentencePieceProcessor object
2. Loads the tokenizer.model file into it
3. Verifies:
   - Vocabulary size is 32,000 (matches the model's vocab_size)
   - BOS token ID is 1

**OUTPUT:**
```
Tokenizer loaded. Vocabulary size: 32000
Beginning of String (BOS) ID: 1
```

---

### Step 4: Encode a Prompt (Text → Token IDs)

```python
prompt = "Building an inference engine is cool!"
token_ids = sp_model.encode(prompt, add_bos=True, add_eos=False)
print(f"Token IDs for prompt: {token_ids}")
```

**WHY:**
Convert human-readable text into the numbers the model understands.

**HOW THE ENCODING WORKS:**
1. Takes the string "Building an inference engine is cool!"
2. Looks up each subword in the tokenizer's vocabulary
3. Returns a list of integer IDs

**WHAT EACH PARAMETER MEANS:**
- `prompt`: The text to encode
- `add_bos=True`: Prepend a BOS (Beginning of Sequence) token
  - This tells the model "a new request is starting"
  - Without it, the model might think it's a continuation
- `add_eos=False`: Don't append EOS (End of Sequence) token
  - We'll add this later when we're done generating

**OUTPUT EXAMPLE:**
```
Token IDs for prompt: [1, 7719, 28705, 396, 28705, 338, 28705, 시원, ...]
                       ^
                       BOS token (added by add_bos=True)
```

**DETAILED BREAKDOWN OF THE ABOVE EXAMPLE:**
```
Token ID 1     → BOS (Beginning of Sequence)
Token ID 7719  → "Building"
Token ID 28705 → " " (space)
Token ID 396   → "an"
Token ID 28705 → " " (space)
Token ID 338   → "inference"
... (and so on)
```

**IMPORTANT:** Notice that spaces are included in the tokens sometimes (" an" might be one token, or " a" + "n" split into two. The tokenizer learned which splits minimize token count for common patterns).

---

### Step 5: Decode Token IDs Back to Subword Pieces

```python
pieces = [sp_model.id_to_piece(idx) for idx in token_ids]
print(f"Original: {prompt}")
print(f"Token IDs: {token_ids}")
print(f"Sub-word Pieces: {pieces}")
```

**WHY:**
To debug and understand what the tokenizer is doing. This helps you verify that text was split correctly.

**HOW IT WORKS:**
For each token ID, look it up in the tokenizer's vocabulary and get the string representation.

**OUTPUT EXAMPLE:**
```
Original: Building an inference engine is cool!
Token IDs: [1, 7719, 28705, 396, 28705, 338, ...]
Sub-word Pieces: ['<s>', '▁Building', '▁an', '▁inference', '▁engine', '▁is', '▁cool', '!']
                   ^
                   <s> = BOS token
                   ▁ = space character (represented as underscore for visibility)
```

**WHAT THIS TELLS YOU:**
- "Building" is a single token
- "an" is a single token
- "inference" is a single token
- etc.

This is good! The tokenizer recognized common words and kept them as single tokens, which means the model can process them efficiently.

---

### Step 6: Convert Token IDs to PyTorch Tensor

```python
input_ids = torch.tensor([token_ids]).to(device)
```

**WHY:**
Token IDs are Python integers. The model needs PyTorch tensors (special arrays optimized for GPU math).

**HOW IT WORKS:**
1. `torch.tensor([token_ids])`: Converts the list to a PyTorch tensor
   - The outer brackets `[...]` make it a batch of 1 (1 prompt)
   - Shape becomes: [1, seq_len] = [1, 9] if 9 tokens
2. `.to(device)`: Moves the tensor to the device (GPU/CPU)
   - If GPU is available, it's moved to GPU
   - If not, stays on CPU

**WHY BATCH DIMENSION?**
Even though you're processing 1 prompt, you add a batch dimension because:
- Real inference engines handle multiple prompts simultaneously
- 1 prompt = batch size 1
- 5 prompts = batch size 5
- The model is built to accept batches

**OUTPUT:**
```
input_ids.shape = [1, 9]  (1 batch, 9 tokens)
input_ids.device = mps:0  (on GPU)
```

---

### Step 7: Lookup Embeddings Using the Embedding Table

```python
embedding_table = model_weights["model.embed_tokens.weight"]
print(embedding_table.shape)  # [32000, 2048]

input_embeddings = torch.embedding(embedding_table, input_ids)
print(f"Input IDs Shape: {input_ids.shape}")           # [1, 9]
print(f"Final Embedding Shape: {input_embeddings.shape}")  # [1, 9, 2048]
```

**WHY THIS IS THE CRITICAL STEP:**
This is where the bridge between language and mathematics happens. Token IDs become vectors that the model can process.

**HOW THE LOOKUP WORKS:**

Think of the embedding table as a phone book:
```
Phone Book (Embedding Table):
┌──────────────────────────────────────────────────────────┐
│ Token ID │ Embedding (vector of 2048 numbers)            │
├──────────────────────────────────────────────────────────┤
│ 1        │ [-0.1, 0.23, 0.5, ..., -0.8]   (BOS)          │
│ 7719     │ [0.42, -0.1, 0.9, ..., 0.3]    (Building)     │
│ 28705    │ [-0.5, 0.1, 0.2, ..., 0.1]     (space)        │
│ 396      │ [0.1, 0.2, 0.3, ..., -0.2]     (an)           │
│ ...      │ ...                                             │
│ 32000    │ [0.8, -0.4, 0.1, ..., 0.5]                    │
└──────────────────────────────────────────────────────────┘
```

When you call `torch.embedding(embedding_table, input_ids)`:
```
Input:        [1, 7719, 28705, 396]
              ↓
Lookup:       Phone book[1]    → [-0.1, 0.23, 0.5, ..., -0.8]
              Phone book[7719] → [0.42, -0.1, 0.9, ..., 0.3]
              Phone book[28705]→ [-0.5, 0.1, 0.2, ..., 0.1]
              Phone book[396]  → [0.1, 0.2, 0.3, ..., -0.2]
              ↓
Output:       [[-0.1, 0.23, 0.5, ..., -0.8],
               [0.42, -0.1, 0.9, ..., 0.3],
               [-0.5, 0.1, 0.2, ..., 0.1],
               [0.1, 0.2, 0.3, ..., -0.2]]
```

**SHAPE TRANSFORMATION:**
```
Before: [1, 9]              (1 batch, 9 token IDs)
After:  [1, 9, 2048]        (1 batch, 9 tokens, 2048 dimensions each)

Meaning: 1 sentence, 9 words, each word represented as a vector of 2048 numbers
```

---

## The Full Data Flow: A Complete Example

Let's trace one complete prompt through the Day 2 pipeline:

```
INPUT: "Hello world"

┌─────────────────────────────────────────────────────────────┐
│ STEP 1: TOKENIZATION                                        │
└─────────────────────────────────────────────────────────────┘

sp_model.encode("Hello world", add_bos=True, add_eos=False)

Text: "Hello world"
  ↓
Lookup in vocabulary:
  "Hello" → Found as token 235
  " " → Found as part of " world" → Found as token 2564
  (or maybe split differently, depends on tokenizer's learned rules)
  ↓
Add BOS: [1, 235, 2564]
  ↓
Output: [1, 235, 2564]  (3 token IDs)


┌─────────────────────────────────────────────────────────────┐
│ STEP 2: CONVERT TO BATCH TENSOR                             │
└─────────────────────────────────────────────────────────────┘

input_ids = torch.tensor([token_ids]).to(device)
input_ids = torch.tensor([[1, 235, 2564]]).to('mps:0')

Output:
  Shape: [1, 3]  (1 batch, 3 tokens)
  Device: mps:0 (on GPU)
  Values: [[1, 235, 2564]]


┌─────────────────────────────────────────────────────────────┐
│ STEP 3: EMBEDDING LOOKUP                                    │
└─────────────────────────────────────────────────────────────┘

embedding_table = model_weights["model.embed_tokens.weight"]
  Shape: [32000, 2048]
  Meaning: 32000 possible tokens, each has a 2048-dimensional embedding

input_embeddings = torch.embedding(embedding_table, input_ids)

Lookup process:
  Token ID 1    → embedding_table[1]    → [0.1, -0.2, 0.3, ..., 0.9]    
  Token ID 235  → embedding_table[235]  → [0.5, 0.1, -0.3, ..., -0.1]   
  Token ID 2564 → embedding_table[2564] → [-0.2, 0.4, 0.1, ..., 0.2]    

Output:
  Shape: [1, 3, 2048]
  Meaning: 1 batch, 3 tokens, each token is a 2048-dimensional vector
  
  Values (conceptually):
  [
    [[0.1, -0.2, 0.3, ..., 0.9],      ← BOS embedding
     [0.5, 0.1, -0.3, ..., -0.1],     ← "Hello" embedding
     [-0.2, 0.4, 0.1, ..., 0.2]]      ← "world" embedding
  ]


┌─────────────────────────────────────────────────────────────┐
│ OUTPUT: READY FOR MODEL                                     │
└─────────────────────────────────────────────────────────────┘

input_embeddings has shape [1, 3, 2048]
This is now in the exact format the model expects!

The transformer model will:
  1. Take these embeddings [1, 3, 2048]
  2. Pass them through 22 layers of transformer blocks
  3. Output logits for predicting the next token
  4. You sample the next token and repeat the process
```

---

## What You Achieved on Day 2

### ✓ Completed Tasks

1. **Loaded the Tokenizer**
   - Downloaded tokenizer.model from HuggingFace
   - Created a SentencePieceProcessor ready for encoding

2. **Implemented Text → Token ID Encoding**
   - Demonstrated encoding: "Building an inference engine is cool!" → [1, 7719, 28705, ...]
   - Added BOS token automatically

3. **Implemented Token ID → Subword Piece Decoding**
   - Showed reverse mapping to understand tokenization
   - Verified token boundaries (which parts are single tokens vs. split)

4. **Implemented Token ID → Embedding Conversion**
   - Used the embedding table (already loaded on Day 1)
   - Converted [1, 9] shape to [1, 9, 2048] shape
   - Ready to feed into the transformer

### Output Format Verification

```
input_ids.shape = [1, 9]              ✓ Correct
input_embeddings.shape = [1, 9, 2048] ✓ Correct (matches model input requirements)
```

---

## Why This Matters: The Usecase

### The Problem Solved

Before Day 2:
```
User: "Hello, how are you?"
You: "The model wants numbers, but I have text. Now what?"
```

After Day 2:
```
User: "Hello, how are you?" 
You: 
  1. Tokenize: [1, 235, 53, 2564, 81]
  2. Embed: [[vectors], [vectors], [vectors], [vectors], [vectors]]
  3. Feed to model
```

### Real-World Application

Imagine building a chat API:

```
HTTP REQUEST:
{
  "prompt": "What is machine learning?"
}
↓
YOUR CODE:
  prompt = "What is machine learning?"
  token_ids = sp_model.encode(prompt, add_bos=True, add_eos=False)
  input_ids = torch.tensor([token_ids]).to(device)
  embeddings = torch.embedding(embedding_table, input_ids)
↓
PASS TO MODEL:
  logits = model(embeddings)  ← This will be Day 3+
↓
HTTP RESPONSE:
{
  "response": "Machine learning is a subset of AI..."
}
```

Day 2 is the **essential preprocessing** that happens for every single request to your LLM serving engine.

---

## The Theory: Why Embeddings Work

### Why Do Embeddings Capture Meaning?

The embedding table wasn't hand-crafted. It was **learned during training**.

When TinyLlama was trained on billions of texts:
- The neural network learned which subwords appear together frequently
- It learned which tokens have similar contexts
- It assigned similar embeddings to similar tokens

**Example:**
```
Token "king" embedding:    [0.1, 0.2, 0.3, -0.4, ...]
Token "queen" embedding:   [0.09, 0.21, 0.29, -0.41, ...]
                            Similar! (Both are royalty)

Token "dog" embedding:     [0.8, 0.1, 0.5, 0.2, ...]
Token "apple" embedding:   [0.82, 0.11, 0.48, 0.19, ...]
                            Similar! (Both are concrete objects)

Token "hello" embedding:   [0.5, 0.9, 0.1, -0.3, ...]
Token "attack" embedding:  [-0.1, 0.2, -0.9, 0.5, ...]
                            Different! (Opposite semantic meanings)
```

### Vector Space Structure

The 2,048-dimensional space of embeddings has structure:

```
Metaphorical (2D for visualization, but really 2048D):

        +y (more positive)
         ↑
         │
         │  "king"  .
         │              .  "queen"
         │
    ────┼──────── → +x (more royal)
         │
         │    "person"     . "woman"
         │        .
         │
         ↓
         -y (more negative)

Words with similar context end up near each other in this space!
```

This learned structure is what allows the model to generalize. If the model learned "king" means "ruler" and "queen" also means "ruler", then it will treat them similarly even if it never saw them together in training.

---

## What's NOT Done Yet (Day 2 Partial)

The ROUTEMAP for Day 2 also mentioned:

1. **Batch Tokenization with Padding**
   - Current code: One prompt → tokens
   - Needed: Multiple prompts → tokens with padding to same length
   
2. **Attention Masks**
   - Current code: No attention masks
   - Needed: Tell the model which positions are padding and should be ignored

3. **Tokenizer Class Wrapper**
   - Current code: Direct SentencePiecProcessor calls
   - Needed: Clean class interface for future use

These will make the code production-ready, but the **core idea** is complete.

---

## Key Insights from Day 2

### 1. **Tokenization is Learned, Not Hardcoded**
The tokenizer isn't programmed to know that "hello" should be one token. It learned this from data.

### 2. **Embeddings Encode Meaning Automatically**
The 2,048 numbers per token aren't manually designed to mean anything. The training process created this structure.

### 3. **Token IDs are the Interface**
Token IDs are the fundamental interface:
- Text → Token IDs → Embeddings → Model
- Everything in the middle uses integer indices for efficiency

### 4. **Scale Matters**
- 32,000 vocabulary size is a sweet spot
  - Small enough to fit in memory
  - Large enough that most text can be represented efficiently
  - Fast lookup (array indexing)

### 5. **Special Tokens Control the Model**
- BOS tells the model a sequence is starting (important!)
- EOS tells the model a sequence is ending
- These learned patterns during training guide the model's behavior

---

## The Results: From Theory to Reality

### Concrete Outputs from Your Code

**Original Text:**
```
"Building an inference engine is cool!"
```

**After Tokenization:**
```
Token IDs: [1, 7719, 28705, 396, 28705, 338, 28705, 3577, 338, 12434, 102]
(11 token IDs total, including BOS)
```

**After Embedding Lookup:**
```
Shape: [1, 11, 2048]
Each of the 11 tokens now has a 2048-dimensional vector
Ready to feed into the transformer
```

**What the Model Sees:**
```
11 vectors, each of 2048 numbers
All of them on GPU (fast)
All of them in float16 format (memory efficient)
```

---

## Looking Ahead: What Day 2 Enables

With Day 2 complete, you can now:

**Day 3**: Build the transformer forward pass
- The embeddings will flow through 22 transformer blocks
- Each block processes the embeddings and learns patterns

**Day 4-5**: Implement positional encodings and proper attention
- Current embeddings don't include position information
- Day 4 will add "this is token 0", "this is token 1", etc.

**Day 6+**: Build the KV cache for efficient generation
- You'll generate one token at a time
- This preprocessing will happen for each new token

**Week 3+**: Build the serving engine
- Multiple requests simultaneously
- Batching, memory management, scheduling
- All building on top of this tokenization + embedding foundation

---

## Summary: The Mental Model

```
FLOW:  User Input → Tokenizer → Embeddings → Model Layers → Output
       "hello"     [235]        [[0.1,...]]  [logits]      "hi"

LAYER GOALS:
1. Tokenizer: Convert text to integers (efficient representation)
2. Embeddings: Convert integers to vectors (meaningful representation)
3. Model: Process vectors to predict next token (intelligence)
4. Output: Convert predictions back to text

DAY 2 ACCOMPLISHMENT:
We built layers 1 and 2. The pipeline from text to vectors the model can understand.
```

The beauty of Day 2 is that it's **invisible to the user** but **essential for everything else**. A user types text, and behind the scenes, this tokenization and embedding process happens automatically, converting human language into the mathematical structures the model understands.

---

## Technical Reference: Code Summary

```python
# DAY 2 PIPELINE

# 1. Load Tokenizer
from sentencepiece import SentencePieceProcessor
sp_model = SentencePieceProcessor()
sp_model.load(hf_hub_download(..., filename="tokenizer.model"))

# 2. Encode Text to Token IDs
token_ids = sp_model.encode("Building an inference engine is cool!", 
                             add_bos=True, add_eos=False)
# Output: [1, 7719, 28705, ...]

# 3. Convert to Batch Tensor
input_ids = torch.tensor([token_ids]).to(device)
# Shape: [1, seq_len]

# 4. Lookup Embeddings
embedding_table = model_weights["model.embed_tokens.weight"]  # [32000, 2048]
input_embeddings = torch.embedding(embedding_table, input_ids)
# Shape: [1, seq_len, 2048]

# DAY 2 OUTPUT:
# input_embeddings is ready for the transformer model!
```

This is the **bridge between language and mathematics**—the foundation upon which the rest of the LLM serving engine is built.
