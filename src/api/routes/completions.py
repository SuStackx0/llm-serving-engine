import asyncio
import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    UsageStats,
)
from src.core.types import Request as EngineRequest, SamplingParams, SequenceStatus

router = APIRouter()


def _build_sampling_params(req: CompletionRequest) -> SamplingParams:
    stop = req.stop if isinstance(req.stop, list) else ([req.stop] if req.stop else [])
    return SamplingParams(
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        max_tokens=req.max_tokens,
        stop=stop,
        stream=req.stream,
    )


@router.post("/v1/completions")
async def create_completion(body: CompletionRequest, request: Request):
    engine = request.app.state.engine
    tokenizer = engine.tokenizer

    prompt = body.prompt if isinstance(body.prompt, str) else body.prompt[0]
    token_ids = tokenizer.encode(prompt, add_special_tokens=True)
    sampling = _build_sampling_params(body)

    eng_req = EngineRequest(
        prompt=prompt,
        prompt_token_ids=token_ids,
        sampling_params=sampling,
    )

    out_q = await asyncio.get_event_loop().run_in_executor(None, engine.submit, eng_req)

    if body.stream:
        async def event_stream():
            loop = asyncio.get_event_loop()
            output_ids = []
            while True:
                tok = await loop.run_in_executor(None, out_q.get)
                if tok is None:
                    break
                output_ids.append(tok)
                text = tokenizer.decode([tok], skip_special_tokens=True)
                chunk = f"data: {text}\n\n"
                yield chunk.encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming: wait for all tokens
    loop = asyncio.get_event_loop()
    output_ids: List[int] = []
    while True:
        tok = await loop.run_in_executor(None, out_q.get)
        if tok is None:
            break
        output_ids.append(tok)

    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    finish_reason = eng_req.status.value.replace("finished_", "") if eng_req.is_finished() else "stop"

    return CompletionResponse(
        model=body.model,
        choices=[CompletionChoice(text=output_text, finish_reason=finish_reason)],
        usage=UsageStats(
            prompt_tokens=len(token_ids),
            completion_tokens=len(output_ids),
            total_tokens=len(token_ids) + len(output_ids),
        ),
    )
