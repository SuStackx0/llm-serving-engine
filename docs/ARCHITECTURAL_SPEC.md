# ARCHITECTURAL_SPEC.md: Deep Technical Specifications

For each core component, this document provides:
1. **Mathematical foundations** 
2. **Memory layout details**
3. **Pseudo-code logic**
4. **Performance characteristics**

---

## 1. Rotary Positional Embeddings (RoPE)

### Mathematical Foundation

**Goal**: Encode absolute position into Q,K tensors without learned embeddings.

**Key Insight**: Apply 2D rotation to consecutive dimension pairs.

```
For dimension pair (2i, 2i+1) at sequence position m:

┌─────────────┐     ┌──────────────────────────────────┐
│ x_2i        │  *  │ cos(m⋅θᵢ)  -sin(m⋅θᵢ) │
│ x_{2i+1}    │     │ sin(m⋅θᵢ)   cos(m⋅θᵢ) │
└─────────────┘     └──────────────────────────────────┘

where:
θᵢ = base^(-2i/d)
base = 10000 (standard)
d = hidden_dim / num_heads (per-head dimension)
```

**Why This Works**:
- Rotations preserve vector magnitude
- Relative position (i - j) is invariant under simultaneous rotation
- Enables extrapolation: model can attend beyond training sequence length

### Memory Layout

```python
# Forward computation: Q, K already shaped [batch, seq, heads, head_dim]
# head_dim is always even (e.g., 64)

# Precomputed once at initialization:
freqs = [θ_0, θ_1, ..., θ_{d/4}]  # d/4 because pairs

# At each forward pass:
m = [0, 1, 2, ..., seq_len]  # position indices
m_freqs = m[:, None] * freqs[None, :]  # [seq_len, d/4]
cos = np.cos(m_freqs)  # [seq_len, d/4]
sin = np.sin(m_freqs)  # [seq_len, d/4]

# Apply to Q, K:
def apply_rope(x, cos, sin):  # x: [batch, seq, heads, head_dim]
    # 2D rotation on consecutive pairs
    x_rot = torch.zeros_like(x)
    for i in range(0, head_dim, 2):
        x_rot[..., i] = (x[..., i] * cos[:, i//2] - 
                         x[..., i+1] * sin[:, i//2])
        x_rot[..., i+1] = (x[..., i] * sin[:, i//2] + 
                           x[..., i+1] * cos[:, i//2])
    return x_rot
```

### Pseudo-Code

```python
class RotaryEmbedding:
    def __init__(self, hidden_dim, base=10000, max_seq_length=4096):
        """
        Precompute rotation angles
        
        Args:
            hidden_dim: Per-head dimension (e.g., 64 for 8 heads of hidden=512)
            base: Theta base (standard 10000)
            max_seq_length: Cache up to this length
        """
        # θᵢ = base^(-2i/d) for i = 0, 1, ..., d/2
        inv_freq = 1.0 / (base ** (2.0 * torch.arange(hidden_dim // 2) / hidden_dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos/sin for max_seq_length
        t = torch.arange(max_seq_length, device=device)  # [0, 1, ..., max_seq-1]
        freqs = torch.outer(t, inv_freq)  # [max_seq, d/2]
        self.register_buffer("cos_cache", torch.cos(freqs))
        self.register_buffer("sin_cache", torch.sin(freqs))
    
    def forward(self, x, seq_idx):
        """
        Apply RoPE to query or key
        
        Args:
            x: [batch, seq_len, heads, head_dim]
            seq_idx: Indices into cached cos/sin (for KV cache)
        
        Returns:
            x_rotated: Same shape as x
        """
        # Retrieve precomputed cos/sin
        cos = self.cos_cache[seq_idx]  # [seq_len, d/2]
        sin = self.sin_cache[seq_idx]  # [seq_len, d/2]
        
        # Reshape for broadcasting: [seq_len, 1, 1, d/2]
        cos = cos[:, None, None, :]
        sin = sin[:, None, None, :]
        
        # Apply 2D rotation to pairs: (x_0, x_1), (x_2, x_3), ...
        x_rotated = (
            x[..., :x.shape[-1]//2] * cos - 
            x[..., x.shape[-1]//2:] * sin
        ) + (
            x[..., :x.shape[-1]//2] * sin + 
            x[..., x.shape[-1]//2:] * cos
        )
        
        return x_rotated
```

### Validation Test

```python
def test_rope_relative_invariance():
    """
    Test: RoPE preserves relative distances
    
    For two positions i, j:
    RoPE(x_i, i) · RoPE(x_j, j) should depend on |i - j|, not absolute positions
    """
    rope = RotaryEmbedding(hidden_dim=64)
    
    x = torch.randn(1, 1, 8, 64)  # [batch=1, seq=1, heads=8, dim=64]
    
    # Apply RoPE at position 0 and position 10
    x_rope_0 = rope.forward(x, seq_idx=torch.tensor([0]))
    x_rope_10 = rope.forward(x, seq_idx=torch.tensor([10]))
    
    # If we shift both by 5:
    x_rope_5 = rope.forward(x, seq_idx=torch.tensor([5]))
    x_rope_15 = rope.forward(x, seq_idx=torch.tensor([15]))
    
    # Similarity should be preserved:
    sim_1 = F.cosine_similarity(x_rope_0[0, 0, 0], x_rope_10[0, 0, 0])
    sim_2 = F.cosine_similarity(x_rope_5[0, 0, 0], x_rope_15[0, 0, 0])
    
    assert abs(sim_1 - sim_2) < 1e-5, "RoPE relative invariance failed"
    print(f"✓ RoPE test passed: sim_1={sim_1:.4f}, sim_2={sim_2:.4f}")
```

---

## 2. Physical Block Manager for KV Cache

### Conceptual Model

**Problem**: KV tensors grow with sequence length. Need efficient allocation.

**Solution**: Divide GPU VRAM into fixed-size blocks.

```
GPU VRAM (16GB for TinyLlama on consumer GPU)
├─ Model Weights: 4.5 GB (fixed)
├─ Compute Buffers: 1 GB (temporary)
└─ KV Cache Blocks: 10.5 GB  ← This part we manage

Each block holds:
  - K and V for ONE block of tokens (e.g., 4 tokens)
  - Across ALL transformer layers
  - Size per block = num_layers * heads * head_dim * block_size * 2 (K+V) * bytes

For TinyLlama (22 layers, 8 heads, 64 dim each):
  block_size = 4 tokens
  bytes_per_block = 22 * 8 * 64 * 4 * 2 * 2 (FP16)
                  = 22 * 8 * 64 * 4 * 2 * 2 = 45 KB per block
  
Num blocks = 10.5 GB / 45 KB = ~233,000 blocks (insane!)

Actually, in practice:
  num_blocks = 256-512 blocks (feasible allocation granularity)
  block_size = 16 tokens (larger blocks for efficiency)
```

### Memory Layout

```python
# Block Table Structure:
block_table[request_id] = {
    0: block_110,    # Tokens [0-15] stored in physical block 110
    1: block_205,    # Tokens [16-31] stored in physical block 205
    2: block_089,    # Tokens [32-47] stored in physical block 089
    ...
}

# Physical Block Layout:
physical_memory[block_110] = {
    "K": [num_layers, heads, 16, head_dim],  # K cache for 16 tokens, all layers
    "V": [num_layers, heads, 16, head_dim],  # V cache for 16 tokens, all layers
    "owner": request_id_5,
    "num_filled": 16  # How many of the 16 tokens are actually filled
}

# When request generates new token:
if block_table[request_id][-1]["num_filled"] < BLOCK_SIZE:
    # Append to last block
    block_table[request_id][-1]["num_filled"] += 1
else:
    # Need new block
    new_block_id = allocate_free_block()
    block_table[request_id].append(new_block_id)
```

### Pseudo-Code

```python
class PhysicalBlockManager:
    def __init__(self, total_gpu_memory_gb=16, block_size_tokens=16, 
                 model_config=None):
        """
        Initialize block manager
        
        Args:
            total_gpu_memory_gb: Total available GPU memory
            block_size_tokens: How many tokens per block
            model_config: Model architecture (layers, heads, etc.)
        """
        self.block_size = block_size_tokens
        
        # Calculate bytes per block
        bytes_per_block = self._calculate_block_size(model_config)
        
        # Allocate memory for all blocks
        reserved_for_weights = 4.5  # GB (approximate)
        reserved_for_compute = 1.0  # GB (temporary buffers)
        available_for_kv = total_gpu_memory_gb - reserved_for_weights - reserved_for_compute
        
        self.num_blocks = int((available_for_kv * 1e9) / bytes_per_block)
        
        # Track allocation state
        self.free_blocks = set(range(self.num_blocks))  # All initially free
        self.allocated_to = {}  # block_id -> request_id
    
    def allocate(self, request_id, num_blocks_needed):
        """
        Allocate blocks to a request
        
        Returns: List of block IDs allocated, or raises if insufficient
        """
        if len(self.free_blocks) < num_blocks_needed:
            raise RuntimeError("Insufficient free blocks for allocation")
        
        allocated = []
        for _ in range(num_blocks_needed):
            block_id = self.free_blocks.pop()
            self.allocated_to[block_id] = request_id
            allocated.append(block_id)
        
        return allocated
    
    def expand(self, request_id):
        """
        Allocate one more block to existing request
        """
        if not self.free_blocks:
            raise RuntimeError("No free blocks available")
        
        block_id = self.free_blocks.pop()
        self.allocated_to[block_id] = request_id
        return block_id
    
    def free(self, request_id):
        """
        Deallocate all blocks belonging to a request
        """
        blocks_to_free = [bid for bid, rid in self.allocated_to.items() if rid == request_id]
        for block_id in blocks_to_free:
            self.free_blocks.add(block_id)
            del self.allocated_to[block_id]
        
        return len(blocks_to_free)
    
    def get_free_blocks(self):
        return len(self.free_blocks)
```

### Validation Test

```python
def test_block_manager():
    """Verify block allocation without corruption or double-allocation"""
    manager = PhysicalBlockManager(total_gpu_memory_gb=8, block_size_tokens=16)
    initial_blocks = manager.get_free_blocks()
    
    # Allocate for request 0
    req0_blocks = manager.allocate(request_id=0, num_blocks_needed=10)
    assert len(req0_blocks) == 10
    assert manager.get_free_blocks() == initial_blocks - 10
    
    # Allocate for request 1
    req1_blocks = manager.allocate(request_id=1, num_blocks_needed=5)
    assert len(req1_blocks) == 5
    assert manager.get_free_blocks() == initial_blocks - 15
    
    # Blocks should be distinct
    assert len(set(req0_blocks) & set(req1_blocks)) == 0
    
    # Expand request 0
    new_block = manager.expand(request_id=0)
    assert new_block not in req0_blocks
    assert manager.get_free_blocks() == initial_blocks - 16
    
    # Free request 0
    freed = manager.free(request_id=0)
    assert freed == 11  # 10 original + 1 expanded
    assert manager.get_free_blocks() == initial_blocks - 5
    
    print("✓ Block manager test passed")
```

---

## 3. Paged Attention Implementation

### Algorithm Overview

**Challenge**: Query attention over K,V split across physical blocks.

**Solution**: Compute attention block-by-block, accumulate results.

```
Standard Attention:
  scores = Q @ K^T / sqrt(d)  # Dense [seq, seq] matrix
  probs = softmax(scores)
  output = probs @ V
  
Paged Attention:
  output = 0
  for each block in block_table[request_id]:
      # Gather physical K, V for this block
      K_block = physical_memory[block].K
      V_block = physical_memory[block].V
      
      # Attention over this block
      scores_block = Q @ K_block^T / sqrt(d)  # [seq, block_size]
      probs_block = softmax(scores_block)
      output += probs_block @ V_block
      
  # Normalize (approximately correct for causal attention)
  output = output / num_blocks
```

**Issue**: Softmax normalization is tricky. Need log-sum-exp trick.

### Pseudo-Code

```python
class PagedAttention(nn.Module):
    def __init__(self, num_heads, head_dim, block_size=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.scale = 1.0 / math.sqrt(head_dim)
    
    def forward(self, query, block_table, physical_kv, 
                kv_lengths, attn_mask=None):
        """
        Compute attention using paged KV cache
        
        Args:
            query: [batch, seq_len, heads, head_dim]
            block_table: {request_id: [block_ids]}
            physical_kv: Physical memory holding KV blocks
            kv_lengths: How many tokens filled in last block per request
            attn_mask: Causal mask, [seq_len, kv_seq_len]
        
        Returns:
            output: [batch, seq_len, heads, head_dim]
        """
        batch_size, seq_len, num_heads, head_dim = query.shape
        
        # Placeholder: assume single request for simplicity
        output = torch.zeros_like(query)
        
        blocks = block_table[0]  # Assuming single request
        
        for block_idx, block_id in enumerate(blocks):
            # Get KV for this block
            K_block = physical_kv[block_id]["K"]  # [heads, block_size, head_dim]
            V_block = physical_kv[block_id]["V"]  # [heads, block_size, head_dim]
            
            # How many tokens actually filled in this block?
            if block_idx == len(blocks) - 1:
                num_filled = kv_lengths[0]  # Last block might be partial
            else:
                num_filled = self.block_size
            
            K_block = K_block[:, :num_filled, :]  # Trim to filled
            V_block = V_block[:, :num_filled, :]
            
            # Q @ K^T: [batch, heads, seq_len, head_dim] @ 
            #          [batch, heads, head_dim, num_filled]
            #       -> [batch, heads, seq_len, num_filled]
            scores = torch.matmul(query, K_block.transpose(-1, -2))
            scores = scores * self.scale
            
            # Apply causal mask (if needed)
            if attn_mask is not None:
                # attn_mask: [seq_len, num_filled]
                scores = scores + attn_mask[:seq_len, :num_filled].unsqueeze(0).unsqueeze(0)
            
            # Softmax
            attn_probs = torch.softmax(scores, dim=-1)
            
            # Attention output for this block
            block_output = torch.matmul(attn_probs, V_block)  # [batch, heads, seq_len, head_dim]
            
            # Accumulate (need to be more careful for proper normalization in practice)
            output += block_output / len(blocks)
        
        return output
```

### Validation Test

```python
def test_paged_attention():
    """Verify paged attention matches dense attention"""
    
    batch_size, seq_len, num_heads, head_dim = 2, 128, 8, 64
    kv_seq_len = 100
    block_size = 16
    
    # Generate random Q, K, V
    Q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    K_dense = torch.randn(batch_size, kv_seq_len, num_heads, head_dim)
    V_dense = torch.randn(batch_size, kv_seq_len, num_heads, head_dim)
    
    # Dense attention (reference)
    scale = 1.0 / math.sqrt(head_dim)
    scores_dense = torch.matmul(Q, K_dense.transpose(-2, -1)) * scale
    attn_probs_dense = torch.softmax(scores_dense, dim=-1)
    output_dense = torch.matmul(attn_probs_dense, V_dense)
    
    # Paged attention (split K,V into blocks)
    physical_kv = {}
    num_blocks = (kv_seq_len + block_size - 1) // block_size
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min((block_idx + 1) * block_size, kv_seq_len)
        physical_kv[block_idx] = {
            "K": K_dense[:, start:end, :, :],
            "V": V_dense[:, start:end, :, :]
        }
    
    block_table = {0: list(range(num_blocks))}
    kv_lengths = {0: kv_seq_len % block_size or block_size}
    
    # Compute paged attention
    paged_attn = PagedAttention(num_heads, head_dim, block_size)
    output_paged = paged_attn(Q, block_table, physical_kv, kv_lengths)
    
    # Compare (should be very close)
    diff = torch.abs(output_dense - output_paged).mean()
    print(f"Dense vs Paged Attention difference: {diff:.6f}")
    assert diff < 0.01, f"Difference too large: {diff}"
    
    print("✓ Paged attention matches dense attention")
```

---

## 4. Continuous Batching Scheduler

### Key Concept: Iteration-Level Scheduling

```
Request lifecycle:

Request submitted ──> Enters prefill queue
                          ↓
                     Prefill iteration 0: Process all prompt tokens
                          ↓
                     Move to decode queue
                          ↓
                     Decode iteration 1: Generate 1 token
                     Decode iteration 2: Generate 1 token
                     ... (repeat until max_length)
                          ↓
                     Request finishes, blocks freed

Continuous Batching:
  Instead of processing all requests together,
  build dynamic batches that change every iteration!

Iteration 0:
  Batch = {req_0 prefill, req_1 prefill, req_2 prefill}
  
Iteration 1:
  req_0, req_1, req_2 all transition to decode
  req_3, req_4 arrive and go prefill
  Batch = {req_0 decode, req_1 decode, req_2 decode, req_3 prefill, req_4 prefill}
  BUT if GPU full, drop one request via preemption
  
Iteration 2:
  req_0 finishes (max_length reached)
  Batch = {req_1 decode, req_2 decode, req_3 decode, req_4 decode, req_5 prefill}
  
Key insight: Each iteration, batch composition changes.
This maximizes GPU utilization!
```

### Pseudo-Code

```python
class ContinuousBatchingScheduler:
    def __init__(self, max_prefill_batch=32, max_decode_batch=8):
        self.prefill_queue = deque()  # Requests waiting for prefill
        self.decode_batches = defaultdict(list)  # Active decode requests
        self.max_prefill_batch = max_prefill_batch
        self.max_decode_batch = max_decode_batch
        
        self.block_manager = PhysicalBlockManager()
    
    def submit_request(self, request):
        """Add new request to queue"""
        self.prefill_queue.append(request)
    
    def get_next_batch(self, current_iteration):
        """
        Build next batch for this iteration
        
        Returns: {
            "prefill_requests": [...],
            "decode_requests": [...]
        }
        """
        batch = {"prefill_requests": [], "decode_requests": []}
        
        # 1. Add decode requests that are mid-generation
        decode_candidates = self.get_active_decode_requests()
        for req in decode_candidates[:self.max_decode_batch]:
            if self.block_manager.get_free_blocks() > 0:
                batch["decode_requests"].append(req)
        
        # 2. Add prefill requests (have space after decode?)
        remaining_capacity = self.max_prefill_batch - len(batch["prefill_requests"])
        while self.prefill_queue and remaining_capacity > 0:
            req = self.prefill_queue.popleft()
            
            # Check if we have enough blocks for this request
            blocks_needed = (req.max_length + 15) // 16  # Estimate
            if self.block_manager.get_free_blocks() >= blocks_needed:
                batch["prefill_requests"].append(req)
                remaining_capacity -= 1
            else:
                # Preempt a low-priority decode request to make space
                victim = self.find_victim_for_preemption()
                if victim:
                    self.block_manager.free(victim.id)
                    self.end_request(victim)
                    batch["prefill_requests"].append(req)
                    remaining_capacity -= 1
                else:
                    # Queue full, can't take this request
                    self.prefill_queue.appendleft(req)
                    break
        
        return batch
    
    def process_batch(self, batch):
        """
        Execute a batch iteration
        """
        # Prefill: process entire prompt
        if batch["prefill_requests"]:
            prefill_batch = stack_requests(batch["prefill_requests"])
            logits = self.model.forward(prefill_batch)
            
            # Initialize KV cache for these requests
            for req in batch["prefill_requests"]:
                num_blocks_needed = (req.max_length + 15) // 16
                blocks = self.block_manager.allocate(req.id, num_blocks_needed)
                req.block_table = blocks
                req.phase = "decode"  # Move to decode phase
                req.num_tokens_generated = len(req.prompt_tokens)
        
        # Decode: generate next token
        if batch["decode_requests"]:
            decode_batch = stack_requests(batch["decode_requests"])
            
            # Only input is last token (KV cached)
            last_tokens = decode_batch["last_token"]
            logits = self.model.forward_decode(
                last_tokens,
                kv_blocks=decode_batch["block_tables"]
            )
            
            # Sample next token
            next_tokens = sample(logits, temperature=0.7)
            
            # Update requests
            for req, token in zip(batch["decode_requests"], next_tokens):
                req.tokens.append(token)
                req.num_tokens_generated += 1
                
                # Check if done
                if token == EOS_TOKEN or req.num_tokens_generated >= req.max_length:
                    self.block_manager.free(req.id)
                    self.end_request(req)
    
    def find_victim_for_preemption(self):
        """
        Choose which request to stop when GPU full
        
        Heuristics:
        - Lowest priority
        - Fewest tokens generated (minimize waste)
        - Longest waiting time (fairness)
        """
        candidates = [r for r in self.decode_batches if r.priority == MIN]
        if not candidates:
            # No minimum priority, preempt longest-running instead
            candidates = self.decode_batches
        
        victim = min(candidates, key=lambda r: r.num_tokens_generated)
        return victim
```

### Validation Test

```python
def test_continuous_batching():
    """Verify requests dynamically enter/exit batch"""
    scheduler = ContinuousBatchingScheduler(max_decode_batch=8)
    
    # Submit 3 requests
    for i in range(3):
        req = Request(id=i, prompt="Hello", max_length=100)
        scheduler.submit_request(req)
    
    # Iteration 0: All in prefill
    batch = scheduler.get_next_batch(0)
    assert len(batch["prefill_requests"]) == 3
    assert len(batch["decode_requests"]) == 0
    
    scheduler.process_batch(batch)
    
    # Iteration 1: All moved to decode, new request arrives
    req_new = Request(id=3, prompt="Hi", max_length=50)
    scheduler.submit_request(req_new)
    
    batch = scheduler.get_next_batch(1)
    assert len(batch["decode_requests"]) == 3
    assert len(batch["prefill_requests"]) == 1  # New request prefilled
    
    print("✓ Continuous batching scheduler test passed")
```

---

## 5. Request Preemption Strategies

### Preemption Scenarios

```
Scenario 1: GPU Memory Full
  - New high-priority request arrives
  - No blocks available
  - Preempt lowest-priority ongoing request
  - Free its blocks
  - Start new request

Scenario 2: Request Timeout
  - User cancels request
  - Free all blocks immediately
  - Remove from active batch

Scenario 3: Fairness-based Preemption  
  - Long-running decode request hogging GPU
  - Preempt to let other requests progress
  - Could implement round-robin
```

### Preemption Policy Options

```python
class PreemptionPolicy(Enum):
    NONE = "no_preemption"         # Reject new requests if full
    FIFO = "oldest_first"          # Preempt oldest
    PRIORITY = "priority_aware"    # Preempt lowest priority
    TOKEN_COUNT = "token_count"    # Preempt request with fewest output tokens
    MIXED = "mixed"                # Priority first, then token count

def select_victim(active_requests, policy):
    """Choose which request to preempt"""
    
    if policy == PreemptionPolicy.PRIORITY:
        return min(active_requests, key=lambda r: r.priority)
    
    elif policy == PreemptionPolicy.TOKEN_COUNT:
        return min(active_requests, key=lambda r: r.num_tokens_generated)
    
    elif policy == PreemptionPolicy.MIXED:
        # Prioritize by priority level, then by token count
        min_priority_reqs = [r for r in active_requests 
                            if r.priority == min(req.priority for req in active_requests)]
        return min(min_priority_reqs, key=lambda r: r.num_tokens_generated)
    
    else:  # Default: FIFO
        return min(active_requests, key=lambda r: r.start_time)
```

---

## 6. Memory Efficiency Optimizations

### Flash Attention Implementation (Tiled)

```python
def flash_attention_forward(Q, K, V, block_size=128):
    """
    FlashAttention: reduce memory I/O with tiling
    
    Standard: O(N^2) memory for scores matrix
    Flash: O(N) memory, 2-4x faster
    """
    batch_size, seq_len, heads, head_dim = Q.shape
    
    # Initialize output
    O = torch.zeros_like(Q)
    l = torch.zeros(batch_size, seq_len, heads)  # Log-sum-exp scaling
    
    # Iterate over blocks of K, V
    for j in range(0, seq_len, block_size):
        K_block = K[:, j:j+block_size]
        V_block = V[:, j:j+block_size]
        
        # Compute Q @ K_block^T
        S = torch.matmul(Q, K_block.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # Softmax on this block (log-sum-exp)
        S_max = S.max(dim=-1, keepdim=True)[0]
        P = torch.exp(S - S_max)
        
        # Accumulate into output  
        O_block = torch.matmul(P, V_block)
        
        # Update with scaling (tricky!)
        # l_new = log(e^(l_old) + sum(P))
        # O_new = (e^(l_old) * O_old + O_block) / e^(l_new)
        
        l_new = torch.logsumexp(
            torch.cat([l[:, :, None], S_max.squeeze(-1)], dim=-1),
            dim=-1
        )
        O = O + O_block
    
    return O
```

### Quantized KV Cache (INT8)

```python
class QuantizedKVCache:
    """Store K,V in INT8 to save memory"""
    
    def __init__(self, cache_shape, dtype=torch.int8):
        self.cache = torch.zeros(cache_shape, dtype=dtype, device='cuda')
        self.scale_factors = []  # Per-head scaling
    
    def store_kv(self, K, V):
        """Quantize and store K,V"""
        # Per-channel quantization
        K_scale = K.abs().max() / 127
        V_scale = V.abs().max() / 127
        
        K_quant = (K / K_scale).round().to(torch.int8)
        V_quant = (V / V_scale).round().to(torch.int8)
        
        self.cache[...] = torch.cat([K_quant, V_quant], dim=-1)
        self.scale_factors = [K_scale, V_scale]
    
    def dequant_for_attention(self):
        """Dequantize on-the-fly for attention"""
        cache_fp16 = self.cache.float()
        K_fp16 = cache_fp16[..., :cache_fp16.shape[-1]//2] * self.scale_factors[0]
        V_fp16 = cache_fp16[..., cache_fp16.shape[-1]//2:] * self.scale_factors[1]
        return K_fp16, V_fp16
```

---

## Summary: Component Dependencies

```
RoPE
  ↓ applied to Q,K in
Attention (Multi-Head)
  ↓ reads from
PagedAttention (Block-based)
  ↓ uses maps from
KV Cache Block Manager
  ↓ lives in
Physical Block Manager
  ↓ tracks via
Continuous Batch Scheduler
  ↓ executes
Inference Engine
  ↓ optimized with
Flash Attention / Quantization
```

Each component is independent, enabling modular testing and optimization.
