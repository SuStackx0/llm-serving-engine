from fastapi import APIRouter, Request
from src.api.schemas import ModelCard, ModelList

router = APIRouter()


@router.get("/v1/models")
async def list_models(request: Request):
    engine = request.app.state.engine
    model_id = engine.model_config.model_id
    short_id = model_id.split("/")[-1].lower()
    return ModelList(data=[
        ModelCard(id=short_id),
        ModelCard(id=model_id),
    ])
