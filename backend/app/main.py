"""Gildongmu 테스트 앱 진입점.

실행: python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api import health, models, sessions
from .config import APP_VERSION, Settings
from .inference.registry import PipelineRegistry
from .services.session_service import SessionError, SessionService
from .services.storage_service import SessionStorage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        storage = SessionStorage(settings.sessions_dir, settings.save_frames)
        registry = PipelineRegistry(settings.model_dir, settings)
        service = SessionService(settings, storage, registry)
        service.recover_on_startup()
        app.state.service = service
        try:
            yield
        finally:
            if service.active_session_id:
                try:
                    service.stop(service.active_session_id)
                except Exception:  # noqa: BLE001
                    logging.getLogger(__name__).exception("stop on shutdown failed")

    app = FastAPI(title="Gildongmu Test App", version=APP_VERSION, lifespan=lifespan)
    app.state.settings = settings

    @app.exception_handler(SessionError)
    async def _session_error(_: Request, exc: SessionError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message, "detail": exc.detail}},
        )

    # API route를 정적 파일보다 먼저 등록한다
    app.include_router(health.router, prefix="/api")
    app.include_router(models.router, prefix="/api")
    app.include_router(sessions.router, prefix="/api")

    static_dir = settings.static_dir
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html", headers={"Cache-Control": "no-store"})

    return app


app = create_app()
