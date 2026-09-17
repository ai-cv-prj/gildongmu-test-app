from __future__ import annotations

from fastapi import APIRouter, Depends

from ..config import APP_VERSION
from ..schemas import HealthResponse
from ..services.session_service import SessionService
from .deps import get_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(svc: SessionService = Depends(get_service)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=APP_VERSION,
        active_session_id=svc.active_session_id,
        storage_writable=svc.storage.writable(),
        free_disk_gb=round(svc.storage.free_gb(), 1),
    )
