"""명세 14절의 위험 항목을 실제 API 호출로 검증한다."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app


def make_jpeg(w: int = 480, h: int = 640) -> bytes:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (300, 400), (0, 200, 0), -1)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models", min_free_disk_gb=0)


@pytest.fixture
def client(settings: Settings):
    with TestClient(create_app(settings)) as c:
        yield c


def start(client: TestClient, mode: str = "traffic", device: str = "iPhone 15 Pro") -> str:
    r = client.post("/api/sessions", json={"mode": mode, "model_id": f"{mode}-mock-v1", "device_type": device, "note": "test"})
    assert r.status_code == 201, r.text
    return r.json()["session_id"]


def upload(client: TestClient, sid: str, frame_id: int, data: bytes, ctype: str = "image/jpeg"):
    return client.post(
        f"/api/sessions/{sid}/frames",
        files={"image": (f"{frame_id}.jpg", data, ctype)},
        data={"frame_id": frame_id, "captured_at_ms": 1000 + frame_id, "client_sent_at_ms": 1001 + frame_id},
    )


def test_health_and_models(client: TestClient):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["active_session_id"] is None and h["storage_writable"]
    models = client.get("/api/models").json()["models"]
    ids = {m["id"] for m in models}
    assert {"traffic-mock-v1", "walking-mock-v1", "bus-mock-v1"} <= ids
    assert all(m["available"] for m in models if m["is_mock"])
    assert all(not m["available"] for m in models if not m["is_mock"])  # 가중치 없음


def test_session_id_contains_time_device_mode(client: TestClient):
    sid = start(client, "walking", "Galaxy S24+")
    assert sid.endswith("_galaxy-s24_walking")
    assert len(sid.split("_")[0]) == 8 and len(sid.split("_")[1]) == 6
    detail = client.get(f"/api/sessions/{sid}").json()
    assert detail["device_type"] == "Galaxy S24+" and detail["status"] == "running"


def test_reject_invalid_image_and_oversize(client: TestClient, settings: Settings):
    sid = start(client)
    r = upload(client, sid, 1, b"not-a-jpeg")
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_image"
    r = upload(client, sid, 2, b"x" * (settings.max_upload_bytes + 1))
    assert r.status_code == 413
    r = upload(client, sid, 3, make_jpeg(), ctype="image/png")
    assert r.status_code == 415


def test_reject_frame_after_stop(client: TestClient):
    sid = start(client)
    assert upload(client, sid, 1, make_jpeg()).status_code == 200
    assert client.post(f"/api/sessions/{sid}/stop").status_code == 200
    r = upload(client, sid, 2, make_jpeg())
    assert r.status_code == 409 and r.json()["error"]["code"] == "session_not_running"
    assert upload(client, "no-such-session", 1, make_jpeg()).status_code == 404


def test_recording_timings_and_generated_video_coexist(client: TestClient, settings: Settings):
    """실시간 녹화·지연 로그를 저장한 세션도 종료 후 결과 영상을 만든다."""
    sid = start(client)
    assert upload(client, sid, 1, make_jpeg()).status_code == 200
    recording = b"test-recording-upload"
    response = client.post(f"/api/sessions/{sid}/recording",
                           files={"video": ("recording.webm", recording, "video/webm")})
    assert response.status_code == 200, response.text
    batch = {"batch_id": "integration-1", "records": [{
        "frame_id": 1, "captured_at_ms": 1001, "capture_started_ms": 1,
        "capture_ms": 5, "jpeg_bytes": 100, "recording_active": True,
        "visibility": "visible", "status": "ok",
    }]}
    assert client.post(f"/api/sessions/{sid}/client-timings", json=batch).status_code == 200
    assert client.post(f"/api/sessions/{sid}/stop").status_code == 200
    detail = client.get(f"/api/sessions/{sid}").json()
    assert detail["status"] == "completed" and detail["video_status"] == "ready"
    assert client.get(f"/api/sessions/{sid}/video").status_code == 200
    directory = settings.sessions_dir / sid
    assert (directory / "realtime_overlay.webm").read_bytes() == recording
    logs = (directory / "client_timings.jsonl").read_text().splitlines()
    assert len(logs) == 1 and json.loads(logs[0])["frame_id"] == 1


def test_second_session_conflict_409(client: TestClient):
    sid = start(client)
    r = client.post("/api/sessions", json={"mode": "bus", "model_id": "bus-mock-v1", "device_type": "x"})
    assert r.status_code == 409
    assert r.json()["error"]["detail"]["active_session"]["id"] == sid
    assert client.get("/api/health").json()["active_session_id"] == sid


def test_mode_model_mismatch(client: TestClient):
    r = client.post("/api/sessions", json={"mode": "bus", "model_id": "traffic-mock-v1", "device_type": "x"})
    assert r.status_code == 400


def test_frame_id_roundtrip_and_normalized_boxes(client: TestClient):
    sid = start(client, "traffic")
    for fid in (7, 8, 9):
        r = upload(client, sid, fid, make_jpeg(480, 640))
        body = r.json()
        assert r.status_code == 200, r.text
        assert body["frame_id"] == fid and body["captured_at_ms"] == 1000 + fid
        assert body["image_width"] == 480 and body["image_height"] == 640
        assert body["saved"] is True
        assert body["event"]["type"] == "traffic_signal"
        for d in body["detections"]:
            b = d["box"]
            assert 0 <= b["x1"] <= b["x2"] <= 1 and 0 <= b["y1"] <= b["y2"] <= 1
        assert body["timing"]["server_ms"] >= body["timing"]["inference_ms"] > 0


def test_stop_summary_matches_saved_files(client: TestClient, settings: Settings):
    sid = start(client, "bus", "Galaxy Quantum 3")
    n_ok = 12
    for fid in range(1, n_ok + 1):
        assert upload(client, sid, fid, make_jpeg()).status_code == 200
    assert upload(client, sid, 99, b"bad").status_code == 400

    r = client.post(f"/api/sessions/{sid}/stop")
    assert r.status_code == 200
    body = r.json()
    assert body["already_stopped"] is False
    s = body["session"]
    assert s["status"] == "completed" and s["frame_count"] == n_ok and s["error_count"] == 1
    assert s["video_status"] == "pending"
    assert s["ended_at"] and s["average_inference_ms"] > 0 and s["p95_server_ms"] > 0

    sdir = settings.sessions_dir / sid
    frames = sorted((sdir / "frames").glob("*.jpg"))
    assert len(frames) == n_ok and frames[0].name == "00000001.jpg"
    lines = [json.loads(x) for x in (sdir / "results.jsonl").read_text().splitlines() if x.strip()]
    assert len(lines) == n_ok + 1
    assert sum(1 for x in lines if x["error"] is None) == n_ok
    assert lines[0]["image_path"] == "frames/00000001.jpg" and lines[0]["model_id"] == "bus-mock-v1"
    manifest = json.loads((sdir / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["frame_count"] == n_ok and manifest["device_type"] == "Galaxy Quantum 3"
    assert manifest["video_status"] == "ready"
    assert client.get(f"/api/sessions/{sid}").json()["video_status"] == "ready"
    video = cv2.VideoCapture(str(sdir / "annotated" / "results.mp4"))
    assert video.isOpened()
    assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == n_ok
    ok, image = video.read()
    assert ok and image.shape[:2] == (640, 480)
    video.release()
    response = client.get(f"/api/sessions/{sid}/video")
    assert response.status_code == 200 and response.headers["content-type"] == "video/mp4"

    # idempotent stop
    r2 = client.post(f"/api/sessions/{sid}/stop")
    assert r2.status_code == 200 and r2.json()["already_stopped"] is True

    # 기록 조회 API
    res = client.get(f"/api/sessions/{sid}/results", params={"after_frame_id": 10}).json()
    assert [x["frame_id"] for x in res["results"]] == [11, 12, 99]
    img = client.get(f"/api/sessions/{sid}/frames/3.jpg")
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    lst = client.get("/api/sessions", params={"mode": "bus"}).json()
    assert lst["total"] == 1 and lst["sessions"][0]["id"] == sid


def test_restart_marks_running_as_aborted(settings: Settings):
    with TestClient(create_app(settings)) as c:
        sid = start(c)
        assert upload(c, sid, 1, make_jpeg()).status_code == 200
        # 정상 종료되면 stop 이 호출되므로, 비정상 종료를 흉내 내기 위해 활성 세션만 버린다 (manifest 는 running 으로 남는다)
        c.app.state.service._active = None
    with TestClient(create_app(settings)) as c:
        row = c.get(f"/api/sessions/{sid}").json()
        assert row["status"] == "aborted" and row["ended_at"]
        assert row["frame_count"] == 1  # results.jsonl 에서 다시 센 값
        assert c.get("/api/health").json()["active_session_id"] is None
        # 새 세션을 시작할 수 있어야 한다
        assert start(c) != sid


def test_storage_failure_is_not_reported_as_success(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    from backend.app.services.storage_service import SessionStorage, StorageError

    sid = start(client)

    def boom(self, *a, **k):
        raise StorageError("disk write failed")

    monkeypatch.setattr(SessionStorage, "save_frame", boom)
    r = upload(client, sid, 1, make_jpeg())
    assert r.status_code == 507 and r.json()["error"]["code"] == "storage_failed"
    s = client.post(f"/api/sessions/{sid}/stop").json()["session"]
    assert s["frame_count"] == 0 and s["error_count"] == 1


def test_index_served(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200 and "Gildongmu" in r.text
    assert client.get("/static/js/app.js").status_code == 200


def test_folder_is_the_source_of_truth(client: TestClient, settings: Settings):
    import shutil

    sid = start(client)
    client.post(f"/api/sessions/{sid}/stop")
    assert not (settings.data_dir / "gildongmu.db").exists()
    assert client.get("/api/sessions").json()["total"] == 1
    shutil.rmtree(settings.sessions_dir / sid)
    assert client.get("/api/sessions").json()["total"] == 0
    assert client.get(f"/api/sessions/{sid}").status_code == 404
    assert client.get("/api/sessions/..%2F..").status_code == 404


def test_weights_files_are_discovered(client: TestClient, settings: Settings):
    folder = settings.model_dir / "traffic"
    folder.mkdir(parents=True)
    (folder / "best_v2.pt").write_bytes(b"fake-weights")
    (folder / "notes.txt").write_text("ignored")
    models = {m["id"]: m for m in client.get("/api/models").json()["models"]}
    assert models["traffic-best-v2"]["available"] and not models["traffic-best-v2"]["is_mock"]
    assert "traffic-none" not in models and "walking-none" in models
    # 템플릿 상태의 load() 는 아직 구현 전이므로 세션 시작이 명확한 오류로 실패해야 한다
    r = client.post("/api/sessions", json={"mode": "traffic", "model_id": "traffic-best-v2", "device_type": "x"})
    assert r.status_code == 500 and r.json()["error"]["code"] == "model_load_failed"
    assert client.get("/api/health").json()["active_session_id"] is None
