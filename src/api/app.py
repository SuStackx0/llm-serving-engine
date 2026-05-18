"""
FastAPI application factory.

Usage:
    app = create_app(engine)
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import completions, chat, models, health
from src.engine.inference_engine import LLMEngine


def create_app(engine: LLMEngine) -> FastAPI:
    app = FastAPI(
        title="LLM Serving Engine",
        description="vLLM-compatible LLM inference server with PagedAttention & Continuous Batching",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Attach engine to app state so routes can access it
    app.state.engine = engine

    app.include_router(completions.router)
    app.include_router(chat.router)
    app.include_router(models.router)
    app.include_router(health.router)

    @app.on_event("startup")
    async def _startup():
        if not engine._thread or not engine._thread.is_alive():
            engine.start()

    @app.on_event("shutdown")
    async def _shutdown():
        engine.stop()

    return app
