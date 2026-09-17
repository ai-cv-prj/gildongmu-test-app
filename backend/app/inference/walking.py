"""도보 장애물 실제 모델 파이프라인. 도보 장애물 담당자가 수정하는 파일이다.

가중치 파일은 backend/models/walking/ 폴더에 넣는다. 확장자가 .pt .pth .onnx .engine 인 파일은
자동으로 UI 모델 목록에 뜬다. 여러 개를 넣으면 각각 선택지가 되어 가중치별 비교 테스트를 할 수 있다.

수정할 곳은 세 군데다.
1. CLASS_NAMES : 클래스 ID와 이름 표
2. load()      : self.weights 경로의 가중치를 읽어 self.model 에 넣는다 (세션 시작 시 한 번만 호출)
3. infer()     : OpenCV BGR 이미지 한 장을 추론해 아래 형식으로 돌려준다

점검: .venv/bin/python scripts/check_model.py --mode walking --image 사진.jpg
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import InferenceContext, ModelSpec, normalize_box  # noqa: F401

CLASS_NAMES: dict[int, str] = {
    # 0: "example_class",
}


class WalkingPipeline:
    mode = "walking"

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.weights = spec.weights  # 선택된 가중치 파일 경로 (pathlib.Path)
        self.model = None

    def load(self) -> None:
        # 예 (ultralytics):
        #   from ultralytics import YOLO
        #   self.model = YOLO(str(self.weights))
        raise NotImplementedError("backend/app/inference/walking.py 의 load() 에 모델 로딩 코드를 넣어주세요")

    def reset_session(self, session_id: str) -> None:
        """테스트 시작 시 호출. 추적기나 N프레임 누적 상태가 있으면 여기서 초기화한다."""

    def close_session(self, session_id: str) -> None:
        """테스트 종료 시 호출."""

    def infer(self, frame_bgr: np.ndarray, context: InferenceContext) -> dict[str, Any]:
        h, w = frame_bgr.shape[:2]
        detections: list[dict[str, Any]] = []
        # 예 (ultralytics):
        #   result = self.model.predict(frame_bgr, conf=context.confidence, verbose=False)[0]
        #   for b in result.boxes:
        #       x1, y1, x2, y2 = b.xyxy[0].tolist()
        #       cid = int(b.cls[0])
        #       detections.append({
        #           "class_id": cid,
        #           "class_name": CLASS_NAMES.get(cid, result.names.get(cid, str(cid))),
        #           "confidence": float(b.conf[0]),
        #           "box": normalize_box(x1, y1, x2, y2, w, h),   # 픽셀 → 0~1 좌표
        #           "track_id": None,
        #           "extra": {},
        #       })
        raise NotImplementedError("backend/app/inference/walking.py 의 infer() 에 추론 코드를 넣어주세요")
        return {"detections": detections, "event": {"type": "walking_warning", "warning": False, "warning_text": ""}}
