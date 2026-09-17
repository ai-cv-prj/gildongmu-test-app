"""보행자 신호등 검출, 횡단보도 연결, 신호 색상 분류.

모델 전용 패키지(학습 환경 기준): torch 2.11.0+cu128,
torchvision 0.26.0+cu128, ultralytics 8.4.150.
가중치: backend/models/traffic/best_YOLO.pt 와
backend/models/traffic/classifier/best_MobileNet.pt.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .base import InferenceContext, ModelSpec, normalize_box

CLASS_NAMES: dict[int, str] = {0: "pedestrian_signal", 1: "crosswalk"}
SIGNAL_CLASS = "pedestrian_signal"
CROSSWALK_CLASS = "crosswalk"
CLASSIFIER_FILENAME = "classifier/best_MobileNet.pt"
CLASSIFIER_CONFIDENCE = 0.60
CLASSIFIER_IMAGE_SIZE = 224
MIN_CROP_SIZE = 6
CROP_PADDING = 0.10


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def estimate_vanishing_point(frame, box, cv2):
    """Intersect two differently sloped, near-vertical crosswalk edge lines.

    A detector rectangle does not encode a vanishing point. Return None unless
    image edges provide a geometrically consistent estimate.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 80 or y2 - y1 < 80:
        return None
    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    segments = cv2.HoughLinesP(
        edges, 1, math.pi / 180, threshold=30,
        minLineLength=max(25, int((y2 - y1) * 0.12)), maxLineGap=15,
    )
    if segments is None:
        return None
    lines = []
    for (ax, ay, bx, by) in segments[:, 0]:
        dx, dy = bx - ax, by - ay
        if abs(dy) < 25 or abs(dx) > abs(dy) * 0.9:
            continue
        if ay > by:
            ax, ay, bx, by = bx, by, ax, ay
            dx, dy = -dx, -dy
        slope = dx / dy
        lines.append((x1 + ax, y1 + ay, x1 + bx, y1 + by, slope))
    candidates = []
    for i, a in enumerate(lines):
        for b in lines[i + 1:]:
            if abs(a[4] - b[4]) < 0.12:
                continue
            # x = slope*y + intercept
            intercept_a = a[0] - a[4] * a[1]
            intercept_b = b[0] - b[4] * b[1]
            py = (intercept_b - intercept_a) / (a[4] - b[4])
            px = a[4] * py + intercept_a
            if (y1 - 0.5 * height <= py <= y1 + 0.4 * (y2 - y1)
                    and x1 - 0.25 * width <= px <= x2 + 0.25 * width):
                candidates.append((px, py))
    if len(candidates) < 3:
        return None
    xs = sorted(point[0] for point in candidates)
    ys = sorted(point[1] for point in candidates)
    mid = len(candidates) // 2
    px, py = xs[mid], ys[mid]
    inliers = sum(math.hypot(x - px, y - py) < 0.08 * width for x, y in candidates)
    if inliers < 3 or inliers / len(candidates) < 0.6:
        return None
    return [float(px), float(py)]


def choose_near_crosswalk(crosswalks, width, height):
    """Prefer the crossing whose near edge is low and centered in the image."""
    candidates = []
    for index, item in enumerate(crosswalks):
        box = item["xyxy"]
        bottom = box[3] / height
        offset = abs(center(box)[0] - width / 2) / width
        if bottom >= 0.55 and offset <= 0.30:
            candidates.append((bottom - 0.5 * offset, index))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.08:
        return None
    return candidates[0][1]


def associate(frame, signals, crosswalks, cv2):
    """Return a frame decision with a signal index or an explicit unknown reason."""
    height, width = frame.shape[:2]
    decision = {"status": "unknown", "reason": None, "crosswalk_index": None,
                "signal_index": None, "vanishing_point": None, "candidates": []}
    if not signals:
        decision["reason"] = "no_signal_detected"
        return decision
    if len(signals) == 1:
        decision["status"] = "single_signal"
        decision["reason"] = "crosswalk_relation_unverified"
        decision["signal_index"] = 0
        return decision
    crosswalk_index = choose_near_crosswalk(crosswalks, width, height)
    if crosswalk_index is None:
        decision["reason"] = "no_unambiguous_near_crosswalk"
        return decision
    decision["crosswalk_index"] = crosswalk_index
    crosswalk = crosswalks[crosswalk_index]
    vp = estimate_vanishing_point(frame, crosswalk["xyxy"], cv2)
    if vp is None:
        decision["reason"] = "vanishing_point_unavailable"
        return decision
    decision["vanishing_point"] = vp
    ranked = []
    for index, signal in enumerate(signals):
        box = signal["xyxy"]
        sx, sy = center(box)
        # Pedestrian signal heads may sit beside the far end, but should be
        # above the crossing and reasonably close to its forward direction.
        horizontal = abs(sx - vp[0]) / width
        if sy >= crosswalk["xyxy"][1] or horizontal > 0.22:
            continue
        area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
        size_bonus = min(0.08, 0.08 * math.sqrt(area / (width * height)) / 0.04)
        score = 1 - horizontal / 0.22 + size_bonus
        ranked.append((score, index, horizontal, area))
        decision["candidates"].append({"signal_index": index, "score": round(score, 4),
                                       "horizontal_distance": round(horizontal, 4),
                                       "size_bonus": round(size_bonus, 4)})
    ranked.sort(reverse=True)
    if not ranked:
        decision["reason"] = "no_signal_in_crossing_direction"
    elif len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.12:
        first, second = ranked[:2]
        # Size may resolve a geometric tie, but cannot overturn a clear
        # directional difference. Require a substantial area difference.
        if abs(first[2] - second[2]) <= 0.04 and min(first[3], second[3]) > 0 \
                and max(first[3], second[3]) / min(first[3], second[3]) >= 2:
            decision["status"] = "candidate"
            decision["signal_index"] = first[1] if first[3] > second[3] else second[1]
        else:
            decision["reason"] = "ambiguous_signals"
    else:
        decision["status"] = "candidate"
        decision["signal_index"] = ranked[0][1]
    return decision


class TemporalSelector:
    def __init__(self, required_frames=3):
        self.required_frames = required_frames
        self.last_box = None
        self.last_crosswalk = None
        self.streak = 0

    def update(self, decision, signals, crosswalks):
        index = decision["signal_index"]
        if decision["status"] == "single_signal":
            self.last_box = None
            self.last_crosswalk = None
            self.streak = 0
            return decision
        if decision["status"] != "candidate" or index is None:
            self.last_box = None
            self.last_crosswalk = None
            self.streak = 0
            return decision
        box = signals[index]["xyxy"]
        crosswalk_box = crosswalks[decision["crosswalk_index"]]["xyxy"]
        consistent = (
            self.last_box is not None and self.last_crosswalk is not None
            and box_iou(box, self.last_box) >= 0.3
            and box_iou(crosswalk_box, self.last_crosswalk) >= 0.3
        )
        self.streak = self.streak + 1 if consistent else 1
        self.last_box = box
        self.last_crosswalk = crosswalk_box
        decision["stable_frames"] = self.streak
        if self.streak < self.required_frames:
            decision["status"] = "unknown"
            decision["reason"] = "waiting_for_temporal_consistency"
            decision["candidate_signal_index"] = index
            decision["signal_index"] = None
        else:
            decision["status"] = "matched"
        return decision


class TrafficPipeline:
    mode = "traffic"

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.weights = spec.weights
        self.model = None
        self.classifier = None
        self.class_names: list[str] = []
        self.device = "cpu"
        self._selectors: dict[str, TemporalSelector] = {}

    def load(self) -> None:
        if self.weights is None:
            raise ValueError("신호등 YOLO 가중치 경로가 없습니다")
        if self.weights.suffix.lower() != ".pt":
            raise ValueError("신호등 파이프라인은 Ultralytics YOLO .pt 가중치만 지원합니다")

        import torch
        from torchvision import models
        from ultralytics import YOLO

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(self.weights), task="detect")
        if self.model.task != "detect":
            raise ValueError("신호등 가중치는 YOLO 객체 검출 모델이어야 합니다")
        labels = {int(cid): str(name).strip().lower() for cid, name in self.model.names.items()}
        if labels != CLASS_NAMES:
            raise ValueError(f"YOLO 클래스는 {CLASS_NAMES}이어야 합니다: {self.model.names}")

        classifier_path = self.weights.parent / CLASSIFIER_FILENAME
        if not classifier_path.is_file():
            raise FileNotFoundError(f"신호등 색상 분류기 가중치가 없습니다: {classifier_path}")
        payload = torch.load(classifier_path, map_location="cpu", weights_only=True)
        if payload.get("model_name") != "mobilenet_v3_small":
            raise ValueError("분류기 체크포인트는 mobilenet_v3_small이어야 합니다")
        if payload.get("classifier_imgsz") != CLASSIFIER_IMAGE_SIZE:
            raise ValueError("분류기 체크포인트의 입력 크기는 224여야 합니다")
        self.class_names = [str(name) for name in payload["class_names"]]
        if set(self.class_names) != {"red", "green"}:
            raise ValueError(f"분류기 클래스는 red, green이어야 합니다: {self.class_names}")
        classifier = models.mobilenet_v3_small(weights=None)
        classifier.classifier[-1] = torch.nn.Linear(
            classifier.classifier[-1].in_features, len(self.class_names)
        )
        classifier.load_state_dict(payload["model_state_dict"], strict=True)
        self.classifier = classifier.to(self.device).eval()

    def reset_session(self, session_id: str) -> None:
        self._selectors[session_id] = TemporalSelector(required_frames=3)

    def close_session(self, session_id: str) -> None:
        self._selectors.pop(session_id, None)

    def _classify(self, frame_bgr: np.ndarray, box: list[float]) -> tuple[str, float | None]:
        import cv2
        import torch

        height, width = frame_bgr.shape[:2]
        x1, y1, x2, y2 = box
        pad_x = (x2 - x1) * CROP_PADDING
        pad_y = (y2 - y1) * CROP_PADDING
        left = max(0, int(np.floor(x1 - pad_x)))
        top = max(0, int(np.floor(y1 - pad_y)))
        right = min(width, int(np.ceil(x2 + pad_x)))
        bottom = min(height, int(np.ceil(y2 + pad_y)))
        if right - left < MIN_CROP_SIZE or bottom - top < MIN_CROP_SIZE:
            return "unknown", None

        crop = frame_bgr[top:bottom, left:right]
        crop_height, crop_width = crop.shape[:2]
        scale = min(CLASSIFIER_IMAGE_SIZE / crop_width, CLASSIFIER_IMAGE_SIZE / crop_height)
        resized_width = max(1, round(crop_width * scale))
        resized_height = max(1, round(crop_height * scale))
        resized = cv2.resize(crop, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        canvas = torch.zeros((3, CLASSIFIER_IMAGE_SIZE, CLASSIFIER_IMAGE_SIZE), dtype=torch.float32)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
        offset_y = (CLASSIFIER_IMAGE_SIZE - resized_height) // 2
        offset_x = (CLASSIFIER_IMAGE_SIZE - resized_width) // 2
        canvas[:, offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = tensor
        mean = canvas.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = canvas.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        batch = ((canvas - mean) / std).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probs = self.classifier(batch).softmax(dim=1)[0]
        score, index = probs.max(dim=0)
        confidence = float(score.item())
        color = self.class_names[int(index.item())]
        return (color if confidence >= CLASSIFIER_CONFIDENCE else "unknown"), confidence

    def infer(self, frame_bgr: np.ndarray, context: InferenceContext) -> dict[str, Any]:
        import cv2

        if self.model is None or self.classifier is None:
            raise RuntimeError("신호등 파이프라인이 로드되지 않았습니다")
        height, width = frame_bgr.shape[:2]
        result = self.model.predict(
            source=frame_bgr, imgsz=960, conf=context.confidence,
            device=self.device, verbose=False, save=False, stream=False,
        )[0]
        labels = {int(cid): str(name).strip().lower() for cid, name in result.names.items()}
        signals: list[dict[str, Any]] = []
        crosswalks: list[dict[str, Any]] = []
        if result.boxes is not None:
            boxes = result.boxes.cpu()
            for xyxy, score, class_id in zip(
                boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
            ):
                cid = int(class_id)
                item = {"class_id": cid, "class_name": labels[cid],
                        "confidence": float(score), "xyxy": [float(v) for v in xyxy]}
                if item["class_name"] == SIGNAL_CLASS:
                    signals.append(item)
                elif item["class_name"] == CROSSWALK_CLASS and score >= 0.50:
                    crosswalks.append(item)

        selector = self._selectors.setdefault(context.session_id, TemporalSelector(required_frames=3))
        association = selector.update(associate(frame_bgr, signals, crosswalks, cv2), signals, crosswalks)
        selected_index = association.get("signal_index")
        provisional = selected_index is None and association.get("candidate_signal_index") is not None
        if provisional:
            selected_index = association["candidate_signal_index"]
        detections: list[dict[str, Any]] = []
        signal_state = "unknown"
        if selected_index is not None:
            selected = signals[selected_index]
            if provisional:
                color, color_confidence = "unknown", None
            else:
                color, color_confidence = self._classify(frame_bgr, selected["xyxy"])
                signal_state = color
            x1, y1, x2, y2 = selected["xyxy"]
            detections.append({
                "class_id": selected["class_id"],
                "class_name": selected["class_name"],
                "confidence": selected["confidence"],
                "box": normalize_box(x1, y1, x2, y2, width, height),
                "track_id": None,
                "extra": {"signal_state": color, "color_confidence": color_confidence,
                          "association_status": association["status"]},
            })
        return {"detections": detections, "event": {
            "type": "traffic_signal", "signal_state": signal_state,
            "association_status": association["status"],
            "association_reason": association["reason"],
            "detected_signal_count": len(signals),
            "detected_crosswalk_count": len(crosswalks),
        }}
