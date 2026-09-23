"""API 요청/응답 스키마."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

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


class SessionDetail(SessionSummary):
    settings: dict[str, Any]
    client: dict[str, Any]


class SessionList(BaseModel):
    sessions: list[SessionSummary]
    total: int


class StopResponse(BaseModel):
    session: SessionSummary
    already_stopped: bool
