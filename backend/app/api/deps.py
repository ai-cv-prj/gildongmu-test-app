from __future__ import annotations

from fastapi import Request

from ..services.session_service import SessionService


def get_service(request: Request) -> SessionService:
    return request.app.state.service
