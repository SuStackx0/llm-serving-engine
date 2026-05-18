from fastapi import APIRouter, Request
from src.api.schemas import EngineStats

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/v1/stats", response_model=EngineStats)
async def engine_stats(request: Request):
    engine = request.app.state.engine
    raw = engine.stats()
    return EngineStats(**raw)
