"""
Token sampling strategies: greedy, temperature, top-k, top-p (nucleus).
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List

from src.core.types import SamplingParams


def sample_token(logits: Tensor, params: SamplingParams) -> int:
    """Sample next token id from logits [vocab_size] given SamplingParams."""
    if params.temperature == 0.0:
        return int(logits.argmax().item())

    # Temperature scaling
    logits = logits / params.temperature

    # Top-k filtering
    if params.top_k > 0:
        top_k = min(params.top_k, logits.size(-1))
        kth_val = logits.topk(top_k).values[..., -1]
        logits = logits.masked_fill(logits < kth_val, float("-inf"))

    # Top-p (nucleus) filtering
    if params.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        # Remove tokens once cumulative prob exceeds top_p
        sorted_remove = cumprobs - sorted_logits.softmax(dim=-1) >= params.top_p
        sorted_logits[sorted_remove] = float("-inf")
        logits = sorted_logits.scatter(0, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def batch_sample(logits: Tensor, params_list: List[SamplingParams]) -> List[int]:
    """Sample one token per row of logits [batch, vocab_size]."""
    return [sample_token(logits[i], params_list[i]) for i in range(logits.shape[0])]
