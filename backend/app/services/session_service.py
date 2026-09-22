"""세션 상태 관리와 프레임 처리.

세션 메타데이터는 DB 없이 각 세션 폴더의 manifest.json 에만 둔다.
폴더를 지우거나 옮기면 목록에서도 그대로 반영된다.
"""
from __future__ import annotations

import logging
import re
import statistics
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import cv2
import numpy as np

from ..config import APP_VERSION, Settings
from ..inference.base import InferenceContext, InferencePipeline
from ..inference.registry import PipelineLoadError, PipelineRegistry
from ..schemas import SessionCreate
from .storage_service import SessionStorage, StorageError
from .video_service import VIDEO_REL_PATH, export_session_video

log = logging.getLogger(__name__)


class SessionError(Exception):
    def __init__(self, status_code: int, code: str, message: str, detail: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.detail = detail


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def local_iso(dt: datetime) -> str:
    return dt.astimezone().isoformat(timespec="seconds")


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return s[:32] or "unknown"


@dataclass
class ActiveSession:
    id: str
    mode: str
    model_id: str
    confidence: float
    pipeline: InferencePipeline
    manifest: dict[str, Any]
    frame_count: int = 0
    error_count: int = 0
    inference_ms: list[float] = field(default_factory=list)
    server_ms: list[float] = field(default_factory=list)
    last_error: Optional[str] = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class SessionService:
    def __init__(self, settings: Settings, storage: SessionStorage, registry: PipelineRegistry) -> None:
        self.settings = settings
        self.storage = storage
        self.registry = registry
        self._active: Optional[ActiveSession] = None
        self._lock = threading.Lock()

    @property
    def active_session_id(self) -> Optional[str]:
        return self._active.id if self._active else None

    # ---- 서버 시작 시: 실행 중으로 남은 세션을 aborted 로 ----
    def recover_on_startup(self) -> int:
        n = 0
        for sid in self.storage.list_session_ids():
            m = self.storage.read_manifest(sid)
            if not m or m.get("status") != "running":
                continue
            ok, err = self.storage.count_results(sid)
            now = utc_now()
            m.update(
                status="aborted", ended_at=iso(now), ended_at_local=local_iso(now),
                frame_count=ok, error_count=err,
                last_error=m.get("last_error") or "server restarted while running",
            )
            try:
                self.storage.write_manifest(sid, m)
                n += 1
            except StorageError:
                log.exception("cannot mark %s as aborted", sid)
        if n:
            log.warning("marked %d stale running session(s) as aborted", n)
        return n

    # ---- 시작 ----
    def start(self, req: SessionCreate) -> dict[str, Any]:
        with self._lock:
            if self._active is not None:
                raise SessionError(409, "session_conflict", "이미 실행 중인 세션이 있습니다", {"active_session": self.get(self._active.id)})

            spec = self.registry.get_spec(req.model_id)
            if spec is None or spec.mode != req.mode:
                raise SessionError(400, "invalid_model", f"기능 {req.mode} 에 맞는 모델이 아닙니다: {req.model_id}")
            if not self.storage.writable():
                raise SessionError(507, "storage_unwritable", "저장 디렉터리에 쓸 수 없습니다")
            free = self.storage.free_gb()
            if free < self.settings.min_free_disk_gb:
                raise SessionError(507, "disk_full", f"디스크 여유 공간 부족: {free:.1f} GB", {"free_gb": free})
            try:
                pipeline = self.registry.get(req.model_id)
            except PipelineLoadError as exc:
                raise SessionError(500, "model_load_failed", str(exc)) from exc
            spec = pipeline.spec  # 로딩 후에는 가중치 해시가 채워져 있다

            started = utc_now()
            local = started.astimezone()  # 서버 PC 로컬 시각
            base = f"{local.strftime('%Y%m%d_%H%M%S')}_{slugify(req.device_type)}_{req.mode}"
            session_id = self.storage.unique_session_id(base)
            manifest: dict[str, Any] = {
                "session_id": session_id,
                "status": "running",
                "mode": req.mode,
                "device_type": req.device_type,
                "note": req.note,
                "model_id": spec.id,
                "model_name": spec.name,
                "model_version": spec.version,
                "is_mock": spec.is_mock,
                "weights_file": spec.extra.get("weights_file"),
                "weights_sha256": spec.extra.get("weights_sha256"),
                "code_version": APP_VERSION,
                "started_at": iso(started),
                "started_at_local": local_iso(started),
                "ended_at": None,
                "ended_at_local": None,
                "frame_count": 0,
                "error_count": 0,
                "average_inference_ms": None,
                "average_server_ms": None,
                "p95_server_ms": None,
                "last_error": None,
                "video_status": None,
                "video_path": None,
                "settings": req.settings.model_dump(),
                "client": req.client.model_dump(),
            }
            try:
                path = self.storage.create_session(session_id, manifest)
            except StorageError as exc:
                raise SessionError(507, "storage_failed", str(exc)) from exc

            pipeline.reset_session(session_id)
            self._active = ActiveSession(
                id=session_id, mode=req.mode, model_id=spec.id, confidence=req.settings.confidence,
                pipeline=pipeline, manifest=manifest,
            )
            log.info("session started %s", session_id)
            return {"session_id": session_id, "status": "running", "started_at": iso(started), "storage_path": str(path)}

    # ---- 프레임 ----
    def process_frame(
        self, session_id: str, image_bytes: bytes, content_type: str, frame_id: int, captured_at_ms: int, client_sent_at_ms: int
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        received = utc_now()
        active = self._active
        if active is None or active.id != session_id:
            row = self.get(session_id)  # 없으면 404
            raise SessionError(409, "session_not_running", f"세션이 실행 중이 아닙니다 (status={row['status']})")

        if len(image_bytes) > self.settings.max_upload_bytes:
            raise SessionError(413, "image_too_large", f"이미지 크기 초과 ({len(image_bytes)} bytes)")
        if content_type not in {"image/jpeg", "image/jpg"}:
            raise SessionError(415, "unsupported_media_type", f"JPEG 만 허용합니다: {content_type}")

        with active.lock:
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            t_decoded = time.perf_counter()
            if frame is None or frame.size == 0:
                self._record_error(active, frame_id, captured_at_ms, received, "invalid_image", "이미지를 디코딩할 수 없습니다")
                raise SessionError(400, "invalid_image", "이미지를 디코딩할 수 없습니다")
            h, w = frame.shape[:2]

            ctx = InferenceContext(session_id=session_id, frame_id=frame_id, captured_at_ms=captured_at_ms, confidence=active.confidence)
            try:
                with self.registry.gpu_lock:
                    result = active.pipeline.infer(frame, ctx)
            except Exception as exc:  # noqa: BLE001
                log.exception("inference failed")
                self._record_error(active, frame_id, captured_at_ms, received, "inference_failed", str(exc))
                raise SessionError(500, "inference_failed", f"추론 실패: {exc}") from exc
            t_inferred = time.perf_counter()

            try:
                rel = self.storage.save_frame(session_id, frame_id, image_bytes)
            except StorageError as exc:
                self._record_error(active, frame_id, captured_at_ms, received, "storage_failed", str(exc), best_effort=True)
                raise SessionError(507, "storage_failed", f"프레임 저장 실패: {exc}") from exc
            t_saved = time.perf_counter()

            timing = {
                "decode_ms": round((t_decoded - t0) * 1000, 2),
                "inference_ms": round((t_inferred - t_decoded) * 1000, 2),
                "save_ms": round((t_saved - t_inferred) * 1000, 2),
                "server_ms": round((t_saved - t0) * 1000, 2),
            }
            response = {
                "session_id": session_id,
                "frame_id": frame_id,
                "captured_at_ms": captured_at_ms,
                "server_received_at": iso(received),
                "image_width": w,
                "image_height": h,
                "detections": result.get("detections", []),
                "event": result.get("event", {}),
                "timing": timing,
                "saved": rel is not None,
            }
            record = {
                **response,
                "image_path": rel,
                "model_id": active.model_id,
                "confidence_threshold": active.confidence,
                "client_sent_at_ms": client_sent_at_ms,
                "error": None,
            }
            try:
                self.storage.append_result(session_id, record)
            except StorageError as exc:
                active.error_count += 1
                active.last_error = str(exc)
                raise SessionError(507, "storage_failed", f"결과 저장 실패: {exc}") from exc

            active.frame_count += 1
            active.inference_ms.append(timing["inference_ms"])
            active.server_ms.append(timing["server_ms"])
            if active.frame_count % 25 == 0:
                self._flush_manifest(active)
            return response

    def _record_error(self, active: ActiveSession, frame_id: int, captured_at_ms: int, received: datetime, code: str, message: str, best_effort: bool = False) -> None:
        active.error_count += 1
        active.last_error = f"{code}: {message}"
        try:
            self.storage.append_result(
                active.id,
                {"session_id": active.id, "frame_id": frame_id, "captured_at_ms": captured_at_ms, "server_received_at": iso(received),
                 "detections": [], "event": {}, "timing": None, "saved": False, "image_path": None, "model_id": active.model_id,
                 "error": {"code": code, "message": message}},
            )
        except StorageError:
            if not best_effort:
                raise

    # ---- 종료 ----
    def stop(self, session_id: str) -> tuple[dict[str, Any], bool]:
        with self._lock:
            active = self._active
            if active is None or active.id != session_id:
                return self.get(session_id), True  # 이미 종료됨 (없으면 404)
            with active.lock:
                ended = utc_now()
                try:
                    video_status = "pending" if self.settings.save_frames and active.frame_count else "no_frames"
                    self._flush_manifest(active, status="completed", ended_at=iso(ended),
                                         ended_at_local=local_iso(ended), video_status=video_status)
                except StorageError as exc:
                    raise SessionError(507, "storage_failed", f"세션 요약 저장 실패: {exc}") from exc
                try:
                    active.pipeline.close_session(session_id)
                except Exception:  # noqa: BLE001
                    log.exception("close_session failed")
                self._active = None
            log.info("session stopped %s frames=%d errors=%d", session_id, active.frame_count, active.error_count)
            return self.get(session_id), False

    def generate_video(self, session_id: str) -> None:
        """종료 응답 뒤 실행한다. 영상 실패는 세션 기록에 남긴다."""
        try:
            path = export_session_video(self.storage.session_dir(session_id))
            status = "ready" if path else "no_frames"
            error = None
        except Exception as exc:  # noqa: BLE001
            log.exception("video generation failed for %s", session_id)
            status = "failed"
            error = str(exc)
        manifest = self.storage.read_manifest(session_id)
        if manifest is not None:
            manifest.update(video_status=status, video_path=VIDEO_REL_PATH if status == "ready" else None,
                            video_error=error)
            try:
                self.storage.write_manifest(session_id, manifest)
            except StorageError:
                log.exception("cannot save video status for %s", session_id)

    def recover_pending_videos(self) -> None:
        """영상 생성 중 서버가 종료된 세션을 재시작 뒤 다시 처리한다."""
        for session_id in self.storage.list_session_ids():
            manifest = self.storage.read_manifest(session_id)
            if manifest and manifest.get("video_status") == "pending":
                self.generate_video(session_id)

    def _flush_manifest(self, active: ActiveSession, **extra: Any) -> None:
        active.manifest.update(
            frame_count=active.frame_count,
            error_count=active.error_count,
            average_inference_ms=_mean(active.inference_ms),
            average_server_ms=_mean(active.server_ms),
            p95_server_ms=_p95(active.server_ms),
            last_error=active.last_error,
            **extra,
        )
        self.storage.write_manifest(active.id, active.manifest)

    # ---- 조회 (폴더 기반) ----
    def _row(self, session_id: str, m: dict[str, Any]) -> dict[str, Any]:
        active = self._active
        live = active is not None and active.id == session_id
        video_path = m.get("video_path")
        if video_path is None and (self.storage.session_dir(session_id) / VIDEO_REL_PATH).is_file():
            video_path = VIDEO_REL_PATH  # 수동 생성한 기존 세션 영상도 조회 가능
        return {
            "id": session_id,  # 폴더 이름이 곧 세션 ID
            "mode": m.get("mode", "traffic"),
            "model_id": m.get("model_id", ""),
            "model_version": str(m.get("model_version", "")),
            "device_type": m.get("device_type", ""),
            "note": m.get("note", ""),
            "status": m.get("status", "failed"),
            "started_at": m.get("started_at", ""),
            "ended_at": m.get("ended_at"),
            "frame_count": active.frame_count if live else m.get("frame_count", 0),
            "error_count": active.error_count if live else m.get("error_count", 0),
            "average_inference_ms": m.get("average_inference_ms"),
            "average_server_ms": m.get("average_server_ms"),
            "p95_server_ms": m.get("p95_server_ms"),
            "storage_path": str(self.storage.session_dir(session_id)),
            "last_error": m.get("last_error"),
            "video_status": "ready" if video_path else m.get("video_status"),
            "video_path": video_path,
            "settings": m.get("settings", {}),
            "client": m.get("client", {}),
        }

    def get(self, session_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", session_id or ""):
            raise SessionError(404, "session_not_found", "세션이 없습니다")
        m = self.storage.read_manifest(session_id)
        if m is None:
            raise SessionError(404, "session_not_found", "세션이 없습니다")
        return self._row(session_id, m)

    def list(self, mode: Optional[str], status: Optional[str], limit: int, offset: int) -> tuple[list[dict[str, Any]], int]:
        rows = []
        for sid in self.storage.list_session_ids():
            m = self.storage.read_manifest(sid)
            if m is None:
                continue
            if mode and m.get("mode") != mode:
                continue
            if status and m.get("status") != status:
                continue
            rows.append(self._row(sid, m))
        rows.sort(key=lambda r: r["started_at"], reverse=True)
        return rows[offset : offset + limit], len(rows)


def _mean(xs: list[float]) -> Optional[float]:
    return round(statistics.fmean(xs), 2) if xs else None


def _p95(xs: list[float]) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(len(s) * 0.95))], 2)
