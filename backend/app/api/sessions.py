"""
file_path: backend/app/api/sessions.py

세션의 프레임 처리, 실시간 녹화 업로드와 종료 API를 제공한다.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from ..schemas import FrameResponse, SessionCreate, SessionCreated, SessionDetail, SessionList, StopResponse
from ..services.session_service import SessionError, SessionService
from ..services.storage_service import StorageError
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


# 실행 중인 세션의 실시간 탐지 영상 저장
@router.post("/sessions/{session_id}/recording")
async def upload_recording(
    session_id: str,
    video: UploadFile = File(...),
    svc: SessionService = Depends(get_service),
) -> dict:
    """
    휴대폰에서 녹화된 WebM 영상을 현재 세션 폴더에 저장한다.
    """
    session = svc.get(session_id)
    if session["status"] != "running":
        raise SessionError(409, "session_not_running", "실행 중인 세션만 녹화를 저장할 수 있습니다")

    content_type = (video.content_type or "").lower()
    if not (content_type.startswith("video/webm") or content_type == "application/octet-stream"):
        raise SessionError(415, "invalid_video_type", "WebM 영상만 저장할 수 있습니다")

    limit = 250 * 1024 * 1024
    data = await video.read(limit + 1)
    if len(data) > limit:
        raise SessionError(413, "video_too_large", "녹화 영상은 250MB 이하여야 합니다")
    if not data:
        raise SessionError(400, "empty_video", "녹화 영상이 비어 있습니다")

    try:
        path = await run_in_threadpool(svc.storage.save_recording, session_id, data)
    except StorageError as exc:
        raise SessionError(507, "storage_failed", str(exc)) from exc
    return {"session_id": session_id, "recording_path": path, "size_bytes": len(data)}


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
