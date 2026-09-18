"""
file_path: tests/test_client_timings.py

지연 로그 API의 검증, 종료 후 저장과 JSONL 쓰기를 검사한다.
저장소를 모킹하여 실제 세션이나 폴더를 생성하지 않는다.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, mock_open, patch

import httpx
import pytest

from backend.app.api.deps import get_service
from backend.app.main import create_app
from backend.app.schemas import ClientInfo
from backend.app.services.session_service import SessionError
from backend.app.services.storage_service import SessionStorage, StorageError


# 정상 로그 묶음 생성
def sample_batch() -> dict:
    """프레임 한 건의 브라우저 측정값을 반환한다."""
    return {"batch_id": "batch-1", "records": [{
        "frame_id": 1, "captured_at_ms": 1000, "capture_started_ms": 10,
        "capture_ms": 5, "capture_backend": "worker", "request_started_ms": 15, "response_received_ms": 115,
        "request_ms": 100, "jpeg_bytes": 42000, "recording_active": True,
        "visibility": "visible", "status": "ok", "overlay_status": "drawn",
        "overlay_drawn_ms": 120, "response_to_overlay_ms": 5, "capture_to_overlay_ms": 110,
    }]}


# 파일을 만들지 않는 테스트 서비스
@pytest.fixture
def service():
    """종료된 세션과 메모리 기반 저장 대역을 제공한다."""
    return SimpleNamespace(get=Mock(return_value={"status": "completed"}), storage=Mock())


# 실제 ASGI 라우트를 통한 요청
def post(service, **kwargs):
    """서버 수명주기와 실제 네트워크 없이 로그 업로드 API를 호출한다."""
    app = create_app()
    app.dependency_overrides[get_service] = lambda: service

    # 비동기 HTTP 호출
    async def send():
        """인메모리 ASGI 전송으로 요청과 응답을 검증한다."""
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/api/sessions/test-session/client-timings", **kwargs)

    return asyncio.run(send())


# 종료 직후 마지막 로그 저장
def test_completed_session_accepts_final_batch(service):
    """종료된 세션에도 마지막 측정값을 연결해서 저장한다."""
    response = post(service, json=sample_batch())
    assert response.status_code == 200, response.text
    assert response.json()["saved_count"] == 1
    sid, rows = service.storage.append_client_timings.call_args.args
    assert sid == "test-session"
    assert rows[0]["session_id"] == sid and rows[0]["schema_version"] == 1
    assert rows[0]["batch_id"] == "batch-1" and rows[0]["frame_id"] == 1
    assert rows[0]["capture_to_overlay_ms"] == 110
    assert rows[0]["capture_backend"] == "worker"


# 이전 버전 로그와 호환 유지
def test_legacy_capture_log_still_accepted(service):
    """이전 브라우저가 처리 경로 없이 보낸 로그도 저장한다."""
    batch = sample_batch()
    del batch["records"][0]["capture_backend"]
    assert post(service, json=batch).status_code == 200
    assert service.storage.append_client_timings.call_args.args[1][0]["capture_backend"] is None


# 프런트엔드 버전 기록 확인
def test_client_version_is_preserved():
    """새 코드 적용 여부를 세션의 client 정보로 구분할 수 있게 한다."""
    assert ClientInfo(app_version="latency-v3").model_dump()["app_version"] == "latency-v3"
    assert ClientInfo().app_version == ""


# 새 프레임 전송 경로 기록
def test_video_frame_capture_log_accepted(service):
    """VideoFrame Worker 사용 여부를 실제 저장 데이터에서 확인한다."""
    batch = sample_batch()
    batch["records"][0]["capture_backend"] = "worker_video_frame"
    assert post(service, json=batch).status_code == 200
    assert service.storage.append_client_timings.call_args.args[1][0]["capture_backend"] == "worker_video_frame"


# 잘못된 측정값 차단
@pytest.mark.parametrize("field,value", [
    ("capture_ms", -1), ("capture_ms", "NaN"), ("frame_id", 0),
    ("status", "anything"), ("error_code", "x" * 65), ("unexpected", 1),
    ("capture_backend", "unknown"),
])
def test_invalid_record_rejected(service, field, value):
    """음수·비유한 시간, 잘못된 ID와 임의 필드를 저장하지 않는다."""
    batch = sample_batch()
    batch["records"][0][field] = value
    assert post(service, json=batch).status_code == 422
    service.storage.append_client_timings.assert_not_called()


# 로그 묶음 크기 제한
def test_batch_and_body_limits(service):
    """빈 묶음, 25건 초과 및 64KB 초과 요청을 거부한다."""
    batch = sample_batch()
    batch["records"] *= 26
    assert post(service, json=batch).status_code == 422
    batch["records"] = []
    assert post(service, json=batch).status_code == 422
    assert post(service, content=b"x" * (64 * 1024 + 1)).status_code == 413
    assert post(service, content=b"not-json").status_code == 422
    service.storage.append_client_timings.assert_not_called()


# 존재하지 않는 세션 차단
def test_unknown_session_rejected(service):
    """존재 검증에 실패하면 파일 저장을 시도하지 않는다."""
    service.get.side_effect = SessionError(404, "session_not_found", "세션 없음")
    assert post(service, json=sample_batch()).status_code == 404
    service.storage.append_client_timings.assert_not_called()


# 디스크 오류 전달
def test_storage_error_reported(service):
    """로그 저장에 실패했을 때 성공으로 응답하지 않는다."""
    service.storage.append_client_timings.side_effect = StorageError("disk full")
    response = post(service, json=sample_batch())
    assert response.status_code == 507
    assert response.json()["error"]["code"] == "storage_failed"


# 프레임별 JSONL 저장 형식
def test_storage_appends_rows_without_new_directory():
    """기존 세션 경로의 파일에 줄 단위로 쓰고 flush한다."""
    with patch.object(Path, "mkdir"):
        storage = SessionStorage(Path("existing-sessions"), save_frames=True)
    rows = [{"frame_id": 1}, {"frame_id": 2}]
    opened = mock_open()
    with patch("builtins.open", opened), patch.object(Path, "mkdir") as mkdir:
        storage.append_client_timings("test-session", rows)
    mkdir.assert_not_called()
    opened.assert_called_once_with(Path("existing-sessions/test-session/client_timings.jsonl"), "a", encoding="utf-8")
    written = opened().write.call_args.args[0]
    assert [json.loads(line) for line in written.splitlines()] == rows
    opened().flush.assert_called_once()


# 쓰기 오류 변환
def test_storage_wraps_io_error():
    """파일 오류가 API에서 처리할 수 있는 StorageError로 전달된다."""
    with patch.object(Path, "mkdir"):
        storage = SessionStorage(Path("existing-sessions"), save_frames=False)
    with patch("builtins.open", side_effect=OSError("disk full")), pytest.raises(StorageError):
        storage.append_client_timings("test-session", [{"frame_id": 1}])
