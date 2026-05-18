"""
FastAPI application factory.

Usage:
    app = create_app(engine)
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import completions, chat, models, health
from src.api.routes import debug
from src.engine.inference_engine import LLMEngine


def _configure_logging(level: str = "INFO") -> None:
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=fmt)
    # Engine / scheduler logs at requested level; uvicorn access logs at WARNING
    logging.getLogger("llm.engine").setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def create_app(engine: LLMEngine, log_level: str = "INFO") -> FastAPI:
    _configure_logging(log_level)

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

    app.state.engine = engine

    app.include_router(completions.router)
    app.include_router(chat.router)
    app.include_router(models.router)
    app.include_router(health.router)
    app.include_router(debug.router)

    @app.on_event("startup")
    async def _startup():
        if not engine._thread or not engine._thread.is_alive():
            engine.start()

    @app.on_event("shutdown")
    async def _shutdown():
        engine.stop()

    return app
