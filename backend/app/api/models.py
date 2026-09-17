from __future__ import annotations

from fastapi import APIRouter, Depends

from ..schemas import ModelInfo, ModelsResponse
from ..services.session_service import SessionService
from .deps import get_service

router = APIRouter()


@router.get("/models", response_model=ModelsResponse)
def list_models(svc: SessionService = Depends(get_service)) -> ModelsResponse:
    return ModelsResponse(
        models=[
            ModelInfo(id=s.id, mode=s.mode, name=s.name, version=s.version, available=s.available, is_mock=s.is_mock, note=s.note)
            for s in svc.registry.list_models()
        ]
    )
