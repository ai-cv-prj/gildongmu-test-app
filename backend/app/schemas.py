"""API 요청/응답 스키마."""
from __future__ import annotations

from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

Mode = Literal["traffic", "walking", "bus"]
SessionStatus = Literal["running", "completed", "aborted", "failed"]


class ModelInfo(BaseModel):
    id: str
    mode: Mode
    name: str
    version: str
    available: bool
    is_mock: bool
    note: str = ""


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class HealthResponse(BaseModel):
    status: str
    version: str
    active_session_id: Optional[str]
    storage_writable: bool
    free_disk_gb: float


class ClientInfo(BaseModel):
    user_agent: str = ""
    screen_width: int = 0
    screen_height: int = 0
    platform: str = ""
    app_version: str = Field(default="", max_length=64)


class SessionSettings(BaseModel):
    confidence: float = 0.4
    image_max_side: int = 960
    target_fps: float = 5
    jpeg_quality: float = 0.8


class SessionCreate(BaseModel):
    mode: Mode
    model_id: str
    device_type: str = Field(min_length=1, max_length=64)
    note: str = Field(default="", max_length=500)
    settings: SessionSettings = SessionSettings()
    client: ClientInfo = ClientInfo()

    @model_validator(mode="after")
    def traffic_confidence_default(self) -> "SessionCreate":
        if self.mode == "traffic" and "confidence" not in self.settings.model_fields_set:
            self.settings.confidence = 0.25
        return self


class SessionCreated(BaseModel):
    session_id: str
    status: SessionStatus
    started_at: str
    storage_path: str


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    box: Box
    track_id: Optional[int] = None
    extra: dict[str, Any] = {}


class Timing(BaseModel):
    decode_ms: float
    inference_ms: float
    save_ms: float
    server_ms: float


class FrameResponse(BaseModel):
    session_id: str
    frame_id: int
    captured_at_ms: int
    server_received_at: str
    image_width: int
    image_height: int
    detections: list[Detection]
    event: dict[str, Any]
    timing: Timing
    saved: bool


class SessionSummary(BaseModel):
    raw_frame_count: int = 0
    export: Optional[dict[str, Any]] = None
    id: str
    mode: Mode
    model_id: str
    model_version: str
    device_type: str
    note: str
    status: SessionStatus
    started_at: str
    ended_at: Optional[str]
    frame_count: int
    error_count: int
    average_inference_ms: Optional[float]
    average_server_ms: Optional[float]
    p95_server_ms: Optional[float]
    storage_path: str
    last_error: Optional[str]
    video_status: Optional[str] = None
    video_path: Optional[str] = None


class SessionDetail(SessionSummary):
    settings: dict[str, Any]
    client: dict[str, Any]


class SessionList(BaseModel):
    sessions: list[SessionSummary]
    total: int


class StopResponse(BaseModel):
    session: SessionSummary
    already_stopped: bool


# 브라우저에서 측정한 프레임별 지연 기록
Milliseconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class ClientTiming(BaseModel):
    """동일한 브라우저 시계를 기준으로 캡처부터 오버레이 그리기까지 측정한다."""

    model_config = {"extra": "forbid"}
    frame_id: int = Field(ge=1)
    captured_at_ms: int = Field(ge=0)
    capture_started_ms: Milliseconds
    capture_ms: Milliseconds
    capture_backend: Literal["worker_video_frame", "worker", "canvas"] | None = None
    capture_interval_ms: Milliseconds | None = None
    request_started_ms: Milliseconds | None = None
    response_received_ms: Milliseconds | None = None
    request_ms: Milliseconds | None = None
    overlay_drawn_ms: Milliseconds | None = None
    response_to_overlay_ms: Milliseconds | None = None
    capture_to_overlay_ms: Milliseconds | None = None
    overlay_interval_ms: Milliseconds | None = None
    previous_overlay_age_ms: Milliseconds | None = None
    throttle_wait_ms: Milliseconds = 0
    retry_wait_ms: Milliseconds = 0
    jpeg_bytes: int = Field(ge=0)
    recording_active: bool
    visibility: Literal["visible", "hidden"]
    status: Literal["ok", "error", "cancelled"]
    overlay_status: Literal["drawn", "superseded", "error", "skipped"] = "skipped"
    error_code: str | None = Field(default=None, max_length=64)
    http_status: int | None = Field(default=None, ge=0, le=599)


class ClientTimingBatch(BaseModel):
    """로그 업로드량을 제한하고 재전송한 묶음을 식별한다."""

    model_config = {"extra": "forbid"}
    batch_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9-]+$")
    dropped_records: int = Field(default=0, ge=0)
    records: list[ClientTiming] = Field(min_length=1, max_length=25)
