from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from ..schemas import FrameResponse, SessionCreate, SessionCreated, SessionDetail, SessionList, StopResponse
from ..services.session_service import SessionError, SessionService
from .deps import get_service

router = APIRouter()


@router.post("/sessions", response_model=SessionCreated, status_code=201)
async def create_session(req: SessionCreate, svc: SessionService = Depends(get_service)) -> SessionCreated:
    result = await run_in_threadpool(svc.start, req)
    return SessionCreated(**result)


@router.post("/sessions/{session_id}/frames", response_model=FrameResponse)
async def upload_frame(
    session_id: str,
    image: UploadFile = File(...),
    frame_id: int = Form(...),
    captured_at_ms: int = Form(...),
    client_sent_at_ms: int = Form(0),
    svc: SessionService = Depends(get_service),
) -> FrameResponse:
    # 크기 검사는 스트림을 읽으면서 한도+1 까지만 읽는다
    limit = svc.settings.max_upload_bytes
    data = await image.read(limit + 1)
    if len(data) > limit:
        raise SessionError(413, "image_too_large", f"업로드 한도 {limit} bytes 초과")
    result = await run_in_threadpool(
        svc.process_frame, session_id, data, image.content_type or "", frame_id, captured_at_ms, client_sent_at_ms
    )
    return FrameResponse(**result)


@router.post("/sessions/{session_id}/stop", response_model=StopResponse)
async def stop_session(session_id: str, svc: SessionService = Depends(get_service)) -> StopResponse:
    row, already = await run_in_threadpool(svc.stop, session_id)
    return StopResponse(session=row, already_stopped=already)


@router.get("/sessions", response_model=SessionList)
def list_sessions(
    mode: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    svc: SessionService = Depends(get_service),
) -> SessionList:
    rows, total = svc.list(mode, status, limit, offset)
    return SessionList(sessions=rows, total=total)


@router.get("/sessions/{session_id}", response_model=SessionDetail)
def get_session(session_id: str, svc: SessionService = Depends(get_service)) -> SessionDetail:
    return SessionDetail(**svc.get(session_id))


@router.get("/sessions/{session_id}/results")
def get_results(
    session_id: str,
    after_frame_id: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=2000),
    svc: SessionService = Depends(get_service),
) -> dict:
    svc.get(session_id)
    results = svc.storage.read_results(session_id, after_frame_id, limit)
    return {"session_id": session_id, "results": results, "count": len(results)}


@router.get("/sessions/{session_id}/frames/{frame_id}.jpg")
def get_frame_image(session_id: str, frame_id: int, svc: SessionService = Depends(get_service)) -> FileResponse:
    svc.get(session_id)
    path = svc.storage.frame_path(session_id, frame_id)
    if not path.exists():
        raise SessionError(404, "frame_not_found", "프레임 이미지가 없습니다")
    return FileResponse(path, media_type="image/jpeg")
