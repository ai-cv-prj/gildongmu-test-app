"""실제 가중치 없이 전체 흐름을 검증하기 위한 mock 파이프라인."""
from __future__ import annotations

import math
import random
import time
from typing import Any

import numpy as np

from .base import InferenceContext, ModelSpec

MOCK_CLASSES: dict[str, list[tuple[int, str]]] = {
    "traffic": [(0, "red_light"), (1, "green_light"), (2, "crosswalk")],
    "walking": [(0, "bollard"), (1, "pole"), (2, "bicycle"), (3, "person")],
    "bus": [(0, "bus"), (1, "bus_number")],
}


class MockPipeline:
    def __init__(self, mode: str) -> None:
        self.spec = ModelSpec(
            id=f"{mode}-mock-v1",
            mode=mode,
            name={"traffic": "신호등 Mock", "walking": "도보 장애물 Mock", "bus": "버스 Mock"}[mode] + " v1",
            version="1",
            is_mock=True,
            note="가짜 박스를 움직여 전체 흐름을 검증한다",
        )
        self.mode = mode
        self._rng = random.Random(42)

    def load(self) -> None:
        time.sleep(0.05)

    def reset_session(self, session_id: str) -> None:
        pass

    def close_session(self, session_id: str) -> None:
        pass

    def infer(self, frame_bgr: np.ndarray, context: InferenceContext) -> dict[str, Any]:
        time.sleep(self._rng.uniform(0.03, 0.08))
        t = context.frame_id * 0.15
        classes = MOCK_CLASSES[self.mode]

        detections = []
        # 주 박스: 원을 그리며 움직인다
        cx, cy = 0.5 + 0.25 * math.cos(t), 0.5 + 0.2 * math.sin(t)
        w, h = 0.22, 0.28
        cls_idx = (context.frame_id // 25) % len(classes)
        cid, cname = classes[cls_idx]
        detections.append(
            {
                "class_id": cid,
                "class_name": cname,
                "confidence": round(0.75 + 0.2 * abs(math.sin(t * 0.7)), 3),
                "box": {"x1": cx - w / 2, "y1": cy - h / 2, "x2": cx + w / 2, "y2": cy + h / 2},
                "track_id": 1,
                "extra": {},
            }
        )
        # 보조 박스: 가끔 등장
        if context.frame_id % 3 != 0:
            cid2, cname2 = classes[(cls_idx + 1) % len(classes)]
            detections.append(
                {
                    "class_id": cid2,
                    "class_name": cname2,
                    "confidence": round(0.45 + 0.1 * math.cos(t), 3),
                    "box": {"x1": 0.62, "y1": 0.66, "x2": 0.9, "y2": 0.92},
                    "track_id": 2,
                    "extra": {},
                }
            )

        detections = [d for d in detections if d["confidence"] >= context.confidence]
        event = self._event(cname, context.frame_id)
        if self.mode == "traffic":
            # 음성 정책도 실제 모델과 같은 선택 인덱스·대상 번호 계약으로 점검한다.
            selected = next((i for i, d in enumerate(detections)
                             if d["track_id"] == 1 and d["class_name"] in {"red_light", "green_light"}), None)
            event["selected_detection_index"] = selected
            event["detected_signal_count"] = sum(d["class_name"] != "crosswalk" for d in detections)
            if selected is None:
                event["signal_state"] = "unknown"
        return {"detections": detections, "event": event}

    def _event(self, main_class: str, frame_id: int) -> dict[str, Any]:
        if self.mode == "traffic":
            state = "green" if main_class == "green_light" else "red" if main_class == "red_light" else "unknown"
            return {"type": "traffic_signal", "signal_state": state}
        if self.mode == "walking":
            warn = (frame_id // 10) % 2 == 0
            return {"type": "walking_warning", "warning": warn, "warning_text": f"전방에 {main_class}" if warn else ""}
        return {
            "type": "bus_detection",
            "bus_number": "271" if (frame_id // 20) % 2 == 0 else None,
            "is_target": (frame_id // 20) % 2 == 0,
        }
