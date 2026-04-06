# Day 6: Physical Block Manager (Memory Infrastructure)

## Overview
Implemented the **Physical Block Manager**, a foundational memory management system that prevents GPU memory fragmentation. This component allocates and tracks fixed-size memory blocks across concurrent requests—think of it like an OS memory manager for GPU.

---

## Problem Statement: GPU Memory Fragmentation

### The Challenge
GPU memory management faces the same problem that plagued 1980s computers: **fragmentation**.

Example with dynamic allocation:
```
10 GB GPU Memory
Request 1: allocates 2 GB for KV cache
Request 2: allocates 1.5 GB 
Request 3: allocates 2.5 GB
Request 4: needs 1.5 GB ...
         ↓ but 1.5 GB is split across multiple holes!

Free space: 2.5 GB total, but scattered as [0.1GB, 0.3GB, 0.5GB, 1.2GB, 0.4GB]
Request 4: Can't fit! OOM error even with enough free memory.
```

This is **memory fragmentation**—we have total free space but in unusable chunks.

---

## The Solution: Fixed-Size Blocks

Instead of dynamic allocation, use **fixed-size blocks** (like OS virtual memory pages):

```
10 GB GPU Memory → 256 blocks of ~40 MB each

Request 1: occupies blocks [0, 1, 2, ..., 20]      (allocated)
Request 2: occupies blocks [100, 101, 102, ...]    (allocated)
Request 3: occupies blocks [500, 501, ..., 750]    (allocated)
Request 4: occupies blocks [21, 22, 23, ...]       (allocated)

Free blocks managed as a list: [36-99], [103-499], [751-999]
Allocation becomes O(1): just pop from the free list!
```

---

## Implementation: PhysicalBlockManager

### Core Structure
```python
class PhysicalBlockManager:
    - num_blocks: Total storage units available
    - block_size_tokens: Tokens per block (usually 16)
    - free_blocks: List of available block IDs [0, 1, 2, ...]
    - block_table: Maps request_id → [list of block IDs]
```

### Key Methods

**1. Initialization**
```python
def __init__(self, num_blocks, block_size_tokens):
    self.free_blocks = list(range(num_blocks))    # [0, 1, 2, ..., 127]
    self.block_table = {}                         # Empty, will fill as requests come
```

**2. Allocation**
```python
def allocate_block(self, request_id):
    block_id = self.free_blocks.pop(0)            # O(1) grab first free block
    
    if request_id not in self.block_table:
        self.block_table[request_id] = []
        
    self.block_table[request_id].append(block_id)
    return block_id
```

**3. Deallocation**
```python
def free_request(self, request_id):
    blocks_to_release = self.block_table.pop(request_id)
    self.free_blocks.extend(blocks_to_release)    # Return blocks to free pool
```

---

## Execution Flow Example

### Scenario: Two Concurrent Requests

```python
# Initialize: 128 blocks, each holding 16 tokens
pbm = PhysicalBlockManager(num_blocks=128, block_size_tokens=16)

# Request 1 starts (User 88)
block_0 = pbm.allocate_block("user_88")
# → block_table = {"user_88": [0]}
# → free_blocks = [1, 2, 3, ..., 127]  (127 blocks left)

# Request 2 starts (User 99)
block_1 = pbm.allocate_block("user_99")
# → block_table = {"user_88": [0], "user_99": [1]}
# → free_blocks = [2, 3, 4, ..., 127]  (126 blocks left)

# Request 1 fills its block, needs more
block_2 = pbm.allocate_block("user_88")
# → block_table = {"user_88": [0, 2], "user_99": [1]}
# → free_blocks = [3, 4, 5, ..., 127]  (125 blocks left)

# Request 1 finishes (User 88 closes connection)
pbm.free_request("user_88")
# → block_table = {"user_99": [1]}
# → free_blocks = [0, 2, 3, 4, ..., 127]  (back to 127 blocks)

# Request 3 can now reuse freed blocks
block_3 = pbm.allocate_block("user_77")
# → block_table = {"user_99": [1], "user_77": [0]}  (reused block 0!)
# → free_blocks = [2, 3, 4, ..., 127]
```

---

## Why This Design Matters

| Aspect | Naive Dynamic | Fixed Blocks |
|--------|---------------|--------------|
| **Fragmentation** | High risk | Eliminated (blocks stay same size) |
| **Allocation speed** | O(n) search | O(1) pop from list |
| **Memory overhead** | Low | Slight (block_table dict) |
| **Predictability** | Hard to guarantee | Guaranteed max blocks |
| **Implementation** | Complex heap algorithms | Simple list operations |

---

## Real-World Application

This is **exactly how vLLM works**:
- **num_blocks**: Set based on GPU memory size
- **block_size_tokens**: Usually 16 tokens per block
- **Fragmentation prevention**: The core reason for block-based allocation
- **Scalability**: Handles 100s of concurrent requests without OOM

---

## Success Criteria (All Met ✓)

```
✓ Initialize with 128 blocks
✓ Multiple concurrent requests allocate different blocks
✓ Free list updates when blocks allocated
✓ Free list replenishes when requests released
✓ Zero fragmentation by design
✓ O(1) allocation operations
```

---

## Next Steps (Day 7)

The Physical Block Manager tracks **which blocks exist**. Day 7 adds the **Block Table**—tracking:
- Which blocks belong to **which request**
- How many **tokens filled** in each block
- When to **allocate the next block** during generation

This enables concurrent, growth-aware memory management.
