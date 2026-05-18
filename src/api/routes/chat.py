"""
POST /v1/chat/completions — with optional SSE streaming.

We apply a simple chat template to format messages into a single prompt.
If the tokenizer has a built-in chat template we use it; otherwise we
fall back to a human/assistant format that works for TinyLlama.
"""

import asyncio
import json
import time
import uuid
from typing import List

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    DeltaContent,
    StreamingChoice,
    UsageStats,
)
from src.core.types import Request as EngineRequest, SamplingParams, SequenceStatus

router = APIRouter()


def _format_prompt(messages, tokenizer) -> str:
    """Apply chat template or fall back to a simple format."""
    try:
        return tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback template
        parts = []
        for m in messages:
            if m.role == "system":
                parts.append(f"<<SYS>>{m.content}<</SYS>>")
            elif m.role == "user":
                parts.append(f"[INST] {m.content} [/INST]")
            elif m.role == "assistant":
                parts.append(m.content)
        return "\n".join(parts) + "\n"


def _build_sampling(body: ChatCompletionRequest) -> SamplingParams:
    stop = body.stop if isinstance(body.stop, list) else ([body.stop] if body.stop else [])
    return SamplingParams(
        temperature=body.temperature,
        top_p=body.top_p,
        top_k=body.top_k,
        max_tokens=body.max_tokens,
        stop=stop,
        stream=body.stream,
    )


@router.post("/v1/chat/completions")
async def create_chat_completion(body: ChatCompletionRequest, request: Request):
    engine = request.app.state.engine
    tokenizer = engine.tokenizer

    prompt = _format_prompt(body.messages, tokenizer)
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    sampling = _build_sampling(body)

    eng_req = EngineRequest(
        prompt=prompt,
        prompt_token_ids=token_ids,
        sampling_params=sampling,
    )

    loop = asyncio.get_event_loop()
    out_q = await loop.run_in_executor(None, engine.submit, eng_req)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # ── Streaming ─────────────────────────────────────────────────────
    if body.stream:
        async def sse_stream():
            # First chunk with role
            first = ChatCompletionChunk(
                id=completion_id,
                model=body.model,
                choices=[StreamingChoice(delta=DeltaContent(role="assistant"))],
            )
            yield f"data: {first.model_dump_json()}\n\n"

            output_ids = []
            while True:
                tok = await loop.run_in_executor(None, out_q.get)
                if tok is None:
                    break
                output_ids.append(tok)
                text = tokenizer.decode([tok], skip_special_tokens=True)
                chunk = ChatCompletionChunk(
                    id=completion_id,
                    model=body.model,
                    choices=[StreamingChoice(delta=DeltaContent(content=text))],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

            # Final chunk with finish_reason
            finish = "stop"
            last = ChatCompletionChunk(
                id=completion_id,
                model=body.model,
                choices=[StreamingChoice(
                    delta=DeltaContent(),
                    finish_reason=finish,
                )],
            )
            yield f"data: {last.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_stream(), media_type="text/event-stream")

    # ── Non-streaming ──────────────────────────────────────────────────
    output_ids: List[int] = []
    while True:
        tok = await loop.run_in_executor(None, out_q.get)
        if tok is None:
            break
        output_ids.append(tok)

    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    finish_reason = "stop"
    if eng_req.status == SequenceStatus.FINISHED_LENGTH:
        finish_reason = "length"

    return ChatCompletionResponse(
        id=completion_id,
        model=body.model,
        choices=[ChatCompletionChoice(
            message=ChatCompletionMessage(role="assistant", content=output_text),
            finish_reason=finish_reason,
        )],
        usage=UsageStats(
            prompt_tokens=len(token_ids),
            completion_tokens=len(output_ids),
            total_tokens=len(token_ids) + len(output_ids),
        ),
    )
