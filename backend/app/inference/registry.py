"""
file_path: backend/app/inference/registry.py

가중치를 찾아 화면에 등록하고 최초 사용 시 추론 모델을 로딩한다.

- mock 모델은 기능마다 항상 하나씩 있다.
- 실제 모델은 backend/models/<기능>/ 폴더의 가중치 파일을 찾아 파일마다 하나씩 등록한다.
- 파이프라인은 최초 사용 시 한 번 로딩하고 재사용한다.
"""
from __future__ import annotations

import hashlib
import logging
import re
import threading
from pathlib import Path
from typing import Any, Callable

from .base import InferencePipeline, ModelSpec
from .bus import BusPipeline
from .mock import MockPipeline
from .traffic import TrafficPipeline
from .walking import WalkingPipeline

log = logging.getLogger(__name__)

MODES = ("traffic", "walking", "bus")
MODE_KO = {"traffic": "신호등", "walking": "도보 장애물", "bus": "버스"}
REAL_PIPELINES: dict[str, Callable[[ModelSpec], Any]] = {"traffic": TrafficPipeline, "walking": WalkingPipeline, "bus": BusPipeline}
WEIGHT_EXTS = {".pt", ".pth", ".onnx", ".engine"}


class PipelineLoadError(RuntimeError):
    pass


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "weights"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class PipelineRegistry:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self._mock_specs = {m: MockPipeline(m).spec for m in MODES}
        self._loaded: dict[str, InferencePipeline] = {}
        self._lock = threading.Lock()
        self.gpu_lock = threading.Lock()  # GPU 추론 동시 실행 1개

    # ---- 목록: 호출할 때마다 폴더를 다시 읽으므로 가중치를 넣고 새로고침하면 바로 보인다 ----
    def _real_specs(self, mode: str) -> list[ModelSpec]:
        """기존 가중치 파일과 walking의 로컬 Mask2Former 모델 폴더를 등록한다."""
        folder = self.model_dir / mode
        files = sorted(p for p in folder.glob("*") if p.is_file() and p.suffix.lower() in WEIGHT_EXTS) if folder.exists() else []
        specs = [
            ModelSpec(id=f"{mode}-{_slug(p.stem)}{'' if p.suffix.lower() == '.pt' else '-' + p.suffix.lower().lstrip('.')}",
                      mode=mode, name=f"{MODE_KO[mode]} · {p.name}", version=p.stem, weights=p,
                      note=f"backend/models/{mode}/{p.name}")
            for p in files
        ]
        if mode == "walking":
            for weights in sorted(folder.glob("*/model.safetensors")):
                model_folder = weights.parent
                if not weights.is_file() or not all(
                    (model_folder / name).is_file() for name in ("config.json", "preprocessor_config.json")
                ):
                    continue
                specs.append(ModelSpec(
                    id=f"walking-{_slug(model_folder.name)}-safetensors", mode=mode,
                    name=f"보행가능·횡단보도 · {model_folder.name}", version=model_folder.name, weights=weights,
                    note=f"backend/models/walking/{model_folder.name}/ · 보행가능(초록) / 횡단보도(핑크) / 보행불가능(투명)",
                ))
        return specs or [ModelSpec(
            id=f"{mode}-none", mode=mode, name=f"{MODE_KO[mode]} 실제 모델", version="-", weights=None,
            note=f"backend/models/{mode}/ 에 가중치 파일을 넣어주세요",
        )]

    def list_models(self) -> list[ModelSpec]:
        out: list[ModelSpec] = []
        for mode in MODES:
            out.extend(self._real_specs(mode))
            out.append(self._mock_specs[mode])
        return out

    def get_spec(self, model_id: str) -> ModelSpec | None:
        return next((s for s in self.list_models() if s.id == model_id), None)

    def get(self, model_id: str) -> InferencePipeline:
        with self._lock:
            if model_id in self._loaded:
                return self._loaded[model_id]
            spec = self.get_spec(model_id)
            if spec is None:
                raise PipelineLoadError(f"unknown model_id: {model_id}")
            if not spec.available:
                raise PipelineLoadError(f"가중치 파일이 없습니다: {spec.note}")
            pipeline = MockPipeline(spec.mode) if spec.is_mock else REAL_PIPELINES[spec.mode](spec)
            try:
                log.info("loading pipeline %s", model_id)
                pipeline.load()
                if spec.weights is not None:
                    spec.extra["weights_file"] = spec.weights.name
                    spec.extra["weights_sha256"] = file_sha256(spec.weights)
                pipeline.spec = spec
            except Exception as exc:  # noqa: BLE001
                raise PipelineLoadError(f"{model_id} 로딩 실패: {exc}") from exc
            self._loaded[model_id] = pipeline
            return pipeline
