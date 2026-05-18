"""
OpenAI-compatible Pydantic schemas for the REST API.
"""

import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ── Request schemas ────────────────────────────────────────────────────

class CompletionRequest(BaseModel):
    model: str = "tinyllama"
    prompt: Union[str, List[str]]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=-1)
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    n: int = Field(default=1, ge=1, le=1)    # only n=1 supported


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "tinyllama"
    messages: List[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=-1)
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    n: int = Field(default=1, ge=1, le=1)


# ── Response schemas ───────────────────────────────────────────────────

class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    finish_reason: Optional[str] = None
    logprobs: None = None


class UsageStats(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:12]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "tinyllama"
    choices: List[CompletionChoice]
    usage: UsageStats


# ── Chat response ──────────────────────────────────────────────────────

class ChatCompletionMessage(BaseModel):
    role: str = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "tinyllama"
    choices: List[ChatCompletionChoice]
    usage: UsageStats


# ── Streaming chunk schemas ────────────────────────────────────────────

class DeltaContent(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class StreamingChoice(BaseModel):
    index: int = 0
    delta: DeltaContent
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "tinyllama"
    choices: List[StreamingChoice]


# ── Models list ────────────────────────────────────────────────────────

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "llm-serving-engine"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]


# ── Stats ──────────────────────────────────────────────────────────────

class EngineStats(BaseModel):
    model_id: str
    device: str
    num_running_requests: int
    num_waiting_requests: int
    kv_cache_blocks_used: int
    kv_cache_blocks_free: int
    kv_cache_utilization_pct: float
    throughput_tok_s: float
    avg_ttft_ms: Optional[float]
    avg_tpot_ms: Optional[float]
    total_requests_served: int
    total_tokens_in: int
    total_tokens_out: int
    uptime_s: float
    # Prefix cache (present only when prefix caching is enabled)
    prefix_cache_hits: Optional[int] = None
    prefix_cache_misses: Optional[int] = None
    prefix_cache_hit_rate_pct: Optional[float] = None
    prefix_cached_blocks: Optional[int] = None
