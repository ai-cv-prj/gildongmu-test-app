"""환경 설정. 저장소 루트의 .env 파일과 환경변수에서 읽는다."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

APP_VERSION = "0.1.0"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT_DIR / p).resolve()


@dataclass
class Settings:
    host: str = field(default_factory=lambda: os.getenv("APP_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("APP_PORT", "8000")))
    data_dir: Path = field(default_factory=lambda: _resolve(os.getenv("DATA_DIR", "backend/data")))
    model_dir: Path = field(default_factory=lambda: _resolve(os.getenv("MODEL_DIR", "backend/models")))
    save_frames: bool = field(default_factory=lambda: _env_bool("SAVE_FRAMES", True))
    max_upload_bytes: int = field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_BYTES", str(2 * 1024 * 1024))))
    min_free_disk_gb: float = field(default_factory=lambda: float(os.getenv("MIN_FREE_DISK_GB", "2")))

    walking_risk_enabled: bool = field(default_factory=lambda: _env_bool("WALKING_RISK_ENABLED", True))
    walking_risk_config: Path = field(default_factory=lambda: _resolve(os.getenv("WALKING_RISK_CONFIG", "config/walking_risk.yaml")))
    walking_mask_weights: Path = field(default_factory=lambda: _resolve(os.getenv("WALKING_MASK_WEIGHTS", "backend/models/walking/mask2former_w")))
    walking_precision: str = field(default_factory=lambda: os.getenv("WALKING_PRECISION", "fp32"))
    walking_font_path: str = field(default_factory=lambda: os.getenv("WALKING_FONT_PATH", ""))
    walking_export_enabled: bool = True

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def static_dir(self) -> Path:
        return ROOT_DIR / "backend" / "static"
