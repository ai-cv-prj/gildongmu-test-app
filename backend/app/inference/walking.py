"""
file_path: backend/app/inference/walking.py

3클래스 Mask2Former로 휴대폰 프레임의 보행가능 영역과 횡단보도를 추론한다.
test_video_inference.py와 같이 보행가능은 초록색, 횡단보도는 핑크색으로 표시한다.
보행불가능 영역은 투명하게 두어 원본 영상이 보이도록 한다.

모델 폴더: backend/models/walking/model/
필요 파일: config.json, preprocessor_config.json, model.safetensors
실행 패키지: torch==2.14.0, torchvision==0.29.0, transformers==5.17.0,
scipy==1.18.1, pillow==12.3.0 (앱의 .venv에 설치)
"""
from __future__ import annotations

import base64
import logging
from typing import Any

import cv2
import numpy as np

from .base import InferenceContext, ModelSpec

log = logging.getLogger(__name__)
LABEL_COLORS = {
    "walkable": (0, 255, 0),
    "crosswalk": (180, 105, 255),  # 참고 파일과 동일한 OpenCV BGR 순서
}


# 저장된 모델의 3클래스 라벨 확인
def get_segmentation_label_ids(model) -> dict[str, int]:
    """보행불가능·보행가능·횡단보도 세 클래스인지 확인하고 라벨 번호를 반환한다."""
    label_ids = {name: int(label_id) for label_id, name in model.config.id2label.items()}
    required = {"non_walkable", "walkable", "crosswalk"}
    if len(model.config.id2label) != 3 or set(label_ids) != required:
        raise ValueError("non_walkable, walkable, crosswalk 세 클래스의 가중치가 필요합니다.")
    return label_ids


# 보행가능 영역과 횡단보도를 휴대폰 표시용 PNG로 변환
def make_segmentation_event(class_map: np.ndarray, label_ids: dict[str, int]) -> dict[str, Any]:
    """초록색·핑크색 마스크와 전체 화면에서 각 클래스가 차지하는 비율을 반환한다."""
    overlay = np.zeros((*class_map.shape, 4), dtype=np.uint8)
    ratios = {}
    for name, color in LABEL_COLORS.items():
        mask = class_map == label_ids[name]
        overlay[mask] = (*color, 140)  # OpenCV BGRA: 불투명도 약 55%
        ratios[f"{name}_ratio"] = float(mask.mean())
    ok, encoded = cv2.imencode(".png", overlay)
    if not ok:
        raise RuntimeError("보행가능·횡단보도 영역 PNG를 생성하지 못했습니다.")
    return {
        "type": "walking_warning",
        "warning": False,  # 영역 분할만 수행하며 장애물 위험 여부는 판단하지 않는다.
        "warning_text": "",
        **ratios,
        "mask_png": base64.b64encode(encoded.tobytes()).decode("ascii"),
    }


class WalkingPipeline:
    """앱의 세션 인터페이스에 맞춰 3클래스 모델을 한 번 로딩하고 재사용한다."""

    mode = "walking"

    # 선택한 가중치와 모델 상태 초기화
    def __init__(self, spec: ModelSpec) -> None:
        """화면에서 선택한 가중치 경로를 보관한다."""
        self.spec = spec
        self.weights = spec.weights
        self.processor = None
        self.model = None
        self.device = None
        self.label_ids: dict[str, int] = {}

    # 첫 테스트 시작 시 모델 로딩
    def load(self) -> None:
        """선택된 로컬 모델을 읽고 CUDA가 사용 가능하면 GPU에 올린다."""
        # 선택적 패키지는 실제 모델을 사용할 때만 불러온다.
        import torch
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        if self.weights is None or self.weights.name != "model.safetensors":
            raise ValueError("Mask2Former 모델 폴더의 model.safetensors를 선택해 주세요.")
        model_dir = self.weights.parent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True, backend="pil")
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_dir, local_files_only=True)
        self.label_ids = get_segmentation_label_ids(model)
        self.model = model.to(self.device).eval()
        self.processor = processor
        log.info("walking model loaded: %s, device=%s", model_dir, self.device)

    # 테스트 시작 시 세션 상태 초기화
    def reset_session(self, session_id: str) -> None:
        """프레임 사이의 누적 상태가 없는 모델이므로 별도 초기화하지 않는다."""

    # 테스트 종료 시 세션 정리
    def close_session(self, session_id: str) -> None:
        """다음 테스트에서 재사용할 수 있도록 로딩된 모델을 유지한다."""

    # 휴대폰 프레임 한 장 추론
    def infer(self, frame_bgr: np.ndarray, context: InferenceContext) -> dict[str, Any]:
        """BGR 프레임을 분할하고 원본 크기의 초록색·핑크색 마스크를 JSON으로 반환한다."""
        import torch

        if self.model is None or self.processor is None:
            raise RuntimeError("load()로 모델을 먼저 불러와야 합니다.")
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb_frame, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
            # 의미 분할은 각 픽셀의 최상위 클래스를 사용한다. 박스 신뢰도 기준은 적용하지 않는다.
            class_map = self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=[frame_bgr.shape[:2]],
            )[0].cpu().numpy()
        return {"detections": [], "event": make_segmentation_event(class_map, self.label_ids)}
