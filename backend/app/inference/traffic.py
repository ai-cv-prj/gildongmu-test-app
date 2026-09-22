"""보행자 신호등 검출, 횡단보도 연결, 신호 색상 분류.

모델 전용 패키지(학습 환경 기준): torch 2.11.0+cu128,
torchvision 0.26.0+cu128, ultralytics 8.4.150, lap 0.5.13.
가중치: backend/models/traffic/best_YOLO.pt 와
backend/models/traffic/classifier/best_MobileNet.pt.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .base import InferenceContext, ModelSpec, normalize_box
from .traffic_geometry import estimate_stripe_direction
from .traffic_motion import estimate_camera_motion, motion_gray, transform_box
from .traffic_tracker import RECOVERY_CONFIDENCE, SignalTracker

CLASS_NAMES: dict[int, str] = {0: "pedestrian_signal", 1: "crosswalk"}
SIGNAL_CLASS = "pedestrian_signal"
CROSSWALK_CLASS = "crosswalk"
CROSSWALK_CONNECTION_CONFIDENCE = 0.50
SIGNAL_DUPLICATE_IOU = 0.60
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


def suppress_duplicate_signals(signals):
    """같은 위치에 겹친 신호등 검출은 높은 신뢰도 하나만 남긴다.

    트래킹·대상 개수 판단 전에 적용한다. 살아남은 박스의 원래 순서를 유지하며,
    떨어진 다른 신호등과 횡단보도는 제거하지 않는다.
    """
    kept = []
    for index in sorted(range(len(signals)), key=lambda i: signals[i]["confidence"], reverse=True):
        if all(box_iou(signals[index]["xyxy"], signals[other]["xyxy"]) < SIGNAL_DUPLICATE_IOU
               for other in kept):
            kept.append(index)
    return [signals[index] for index in sorted(kept)]


def estimate_vanishing_point(frame, box, cv2):
    return estimate_stripe_direction(frame, box, cv2)


def crosswalk_position(box, width, height):
    bottom = box[3] / height
    offset = abs(center(box)[0] - width / 2) / width
    reasons = []
    if bottom < 0.55:
        reasons.append("bottom_too_high")
    if offset > 0.30:
        reasons.append("off_center")
    return bottom, offset, reasons


def choose_near_crosswalk(crosswalks, width, height):
    """가까운 쪽 끝이 화면 아래에 있고 중앙에 놓인 횡단보도를 우선한다."""
    candidates = []
    for index, item in enumerate(crosswalks):
        bottom, offset, reasons = crosswalk_position(item["xyxy"], width, height)
        if not reasons:
            candidates.append((bottom - 0.5 * offset, index))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.08:
        return None
    return candidates[0][1]


def crosswalk_diagnostics(candidates, crosswalks, association, width, height, signal_count):
    """기존 연결 필터는 유지하면서 검출 후보를 별도로 제공한다."""
    boxes = []
    qualified_index = 0
    used_index = association.get("crosswalk_index")
    eligible_count = 0
    selected_detection_index = None
    for item in candidates:
        bottom, offset, reasons = crosswalk_position(item["xyxy"], width, height)
        qualified = item["confidence"] >= CROSSWALK_CONNECTION_CONFIDENCE
        used = qualified and qualified_index == used_index
        if qualified:
            qualified_index += 1
        if not qualified:
            reasons.insert(0, "below_confidence")
        eligible_count += not reasons
        status = ("below_confidence" if not qualified else "position_rejected" if reasons
                  else "used" if used else "eligible")
        if used:
            selected_detection_index = signal_count + len(boxes)
        boxes.append({
            "class_id": item["class_id"], "class_name": CROSSWALK_CLASS,
            "confidence": item["confidence"],
            "box": normalize_box(*item["xyxy"], width, height), "track_id": None,
            "extra": {"crosswalk_status": status, "exclusion_reasons": reasons,
                      "bottom_ratio": bottom, "center_offset_ratio": offset,
                      "connection_reason": association["reason"] if used else None},
        })
    detection_status = ("not_detected" if not candidates else "below_confidence" if not crosswalks
                        else "position_rejected" if not eligible_count else "eligible")
    connection_status = association["reason"] or association["status"]
    if connection_status == "no_unambiguous_near_crosswalk":
        connection_status = "ambiguous_crosswalks" if eligible_count else "not_attempted"
    return boxes, {
        "detection_status": detection_status, "connection_status": connection_status,
        "candidate_count": len(candidates), "qualified_count": len(crosswalks),
        "eligible_count": eligible_count, "selected_detection_index": selected_detection_index,
        "connection_confidence": CROSSWALK_CONNECTION_CONFIDENCE,
    }


def associate(frame, signals, crosswalks, cv2, *, require_geometry=False):
    """신호등 인덱스 또는 명시적인 판단 불가 사유를 담은 프레임별 판단을 반환한다."""
    height, width = frame.shape[:2]
    decision = {"status": "unknown", "reason": None, "crosswalk_index": None,
                "signal_index": None, "vanishing_point": None, "candidates": []}
    if not signals:
        decision["reason"] = "no_signal_detected"
        return decision
    if len(signals) == 1 and not require_geometry:
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
        # 보행자 신호등은 횡단보도 먼 쪽 끝의 옆에 있을 수 있지만,
        # 화면상 횡단보도보다 위에 있고 그 진행 방향에서 크게 벗어나지 않아야 한다.
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
        # 위치·방향 조건이 비슷하면 크기로 구분할 수 있지만, 뚜렷한 방향 차이를
        # 뒤집을 수는 없다. 크기로 구분하려면 면적 차이가 충분히 커야 한다.
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
    """BoT-SORT ID로 대상을 유지하고 횡단보도 연결의 연속성을 확인한다."""

    TRACK_MAX_GAP_MS = 1000

    def __init__(self, required_frames=3):
        self.required_frames = required_frames
        self.target_origin = None
        self.target_id = None
        self.target_requires_crosswalk = False
        self.previous_context = None
        self.previous_shape = None
        self.previous_gray = None
        self._clear_pending()

    def _clear_pending(self):
        self.last_track_id = None
        self.last_crosswalk = None
        self.streak = 0

    def _clear_target(self):
        self.target_origin = None
        self.target_id = None
        self.target_requires_crosswalk = False

    def _acquire_target(self, decision, signals, origin):
        self.target_origin = origin
        self.target_id = signals[decision["signal_index"]]["track_id"]
        self.target_requires_crosswalk = False
        decision.update(selection_origin=self.target_origin, track_id=self.target_id)

    def _match_target(self, signals):
        matches = [i for i, signal in enumerate(signals)
                   if signal.get("track_id") == self.target_id]
        if len(matches) == 1:
            return matches[0], {"reason": "matched", "tracker": "botsort"}
        return None, {"reason": "ambiguous_match" if matches else "target_missing",
                      "tracker": "botsort"}

    def select(self, frame, signals, crosswalks, cv2, context):
        """횡단보도로 확정한 대상은 유지하고, 임시 대상은 복수 검출 때 연결을 확인한다.

        신호등 하나는 바로 선택하고, 여러 개는 횡단보도 연결을 연속 확인한다.
        미검출 대상을 시간 기준으로 보관하거나 과거 박스·색상을 출력하지 않는다.
        """
        previous = self.previous_context
        continuous = previous is None or (
            context.frame_id == previous.frame_id + 1
            and 0 <= context.captured_at_ms - previous.captured_at_ms <= self.TRACK_MAX_GAP_MS
            and frame.shape[:2] == self.previous_shape
        )
        tracking = {"reason": "no_previous_target" if continuous else "discontinuous_frames"}
        tracking["tracker"] = "botsort"
        if not continuous:
            self._clear_target()
            self._clear_pending()
            self.previous_gray = None
        gray = motion_gray(frame, cv2)
        motion, motion_diagnostic = estimate_camera_motion(
            self.previous_gray, gray, frame.shape[:2], cv2,
        ) if self.last_crosswalk is not None else (
            None, {"reason": "no_previous_candidate"}
        )
        self.previous_gray = gray
        self.last_crosswalk = transform_box(self.last_crosswalk, motion)
        self.previous_context = context
        self.previous_shape = frame.shape[:2]

        if self.target_id is not None:
            index, tracking = self._match_target(signals)
            tracking.update(previous_frame_id=previous.frame_id, previous_track_id=self.target_id)
            tracking["camera_motion"] = motion_diagnostic
            if index is not None:
                if self.target_origin == "single_signal":
                    self.target_requires_crosswalk |= len(signals) > 1
                    if self.target_requires_crosswalk:
                        # 복수 검출 이후에는 하나만 남아도 시작한 연결 확인을 끝낸다.
                        # 확인 중에는 임시 대상의 색상을 안내하지 않는다.
                        decision = self.update(
                            associate(frame, signals, crosswalks, cv2, require_geometry=True),
                            signals, crosswalks,
                        )
                        decision["tracking"] = tracking
                        if decision["signal_index"] is not None:
                            if decision["signal_index"] == index:
                                self.target_origin = "crosswalk_matched"
                                self.target_requires_crosswalk = False
                                decision.update(selection_origin=self.target_origin,
                                                track_id=self.target_id)
                            else:
                                self._acquire_target(decision, signals, "crosswalk_matched")
                            self._clear_pending()
                        return decision
                self._clear_pending()
                return {"status": "tracked", "reason": "previous_target_retained",
                        "signal_index": index,
                        "crosswalk_index": None,
                        "selection_origin": self.target_origin, "track_id": self.target_id,
                        "tracking": tracking}
            self._clear_pending()
            self._clear_target()

        decision = associate(frame, signals, crosswalks, cv2)
        tracking["camera_motion"] = motion_diagnostic
        single = decision["status"] == "single_signal"
        decision = self.update(decision, signals, crosswalks)
        decision["tracking"] = tracking
        if decision["signal_index"] is not None:
            self._acquire_target(decision, signals, "single_signal" if single else "crosswalk_matched")
        return decision

    def update(self, decision, signals, crosswalks):
        index = decision["signal_index"]
        if index is not None and signals[index].get("track_id") is None:
            self._clear_pending()
            decision.update(status="unknown", reason="waiting_for_tracking",
                            candidate_signal_index=index, signal_index=None)
            return decision
        if decision["status"] == "single_signal":
            self._clear_pending()
            return decision
        if decision["status"] != "candidate" or index is None:
            self._clear_pending()
            return decision
        crosswalk_box = crosswalks[decision["crosswalk_index"]]["xyxy"]
        consistent = (
            self.last_track_id is not None and self.last_crosswalk is not None
            and signals[index]["track_id"] == self.last_track_id
            and box_iou(crosswalk_box, self.last_crosswalk) >= 0.3
        )
        self.streak = self.streak + 1 if consistent else 1
        self.last_track_id = signals[index].get("track_id")
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
        self._trackers: dict[str, SignalTracker] = {}

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
        self._trackers.pop(session_id, None)

    def close_session(self, session_id: str) -> None:
        self._selectors.pop(session_id, None)
        self._trackers.pop(session_id, None)

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
            source=frame_bgr, imgsz=960, conf=min(RECOVERY_CONFIDENCE, context.confidence),
            device=self.device, quantize=32, verbose=False, save=False, stream=False,
        )[0]
        labels = {int(cid): str(name).strip().lower() for cid, name in result.names.items()}
        signals: list[dict[str, Any]] = []
        crosswalks: list[dict[str, Any]] = []
        crosswalk_candidates: list[dict[str, Any]] = []
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
                elif item["class_name"] == CROSSWALK_CLASS and score >= context.confidence:
                    crosswalk_candidates.append(item)
                    if score >= CROSSWALK_CONNECTION_CONFIDENCE:
                        crosswalks.append(item)

        raw_signal_count = len(signals)
        signals = suppress_duplicate_signals(signals)
        suppressed_signal_count = raw_signal_count - len(signals)
        tracker = self._trackers.get(context.session_id)
        if tracker is None:
            tracker = self._trackers[context.session_id] = SignalTracker()
        tracker.update(frame_bgr, signals, context)
        # 낮은 신뢰도 검출 중 기존 객체와 연결되지 않은 것은 새 대상·복수 대상
        # 판단이나 화면 표시에 사용하지 않는다. 현재 검출 박스만 전달한다.
        unmatched_low_confidence_count = sum(
            signal["confidence"] < context.confidence and signal["track_id"] is None
            for signal in signals
        )
        signals = [signal for signal in signals
                   if signal["confidence"] >= context.confidence or signal["track_id"] is not None]
        selector = self._selectors.setdefault(
            context.session_id, TemporalSelector(required_frames=3),
        )
        association = selector.select(frame_bgr, signals, crosswalks, cv2, context)
        selected_index = association.get("signal_index")
        candidate_index = association.get("candidate_signal_index")
        detections: list[dict[str, Any]] = []
        signal_state = "unknown"
        # 검출 목록은 대상 선택 성공 여부와 무관하게 보존한다.
        # 색상 분류와 최종 안내 상태는 선택이 완료된 신호등에만 적용한다.
        for index, signal in enumerate(signals):
            color, color_confidence = "unknown", None
            selection_status = "unselected"
            if index == selected_index:
                selection_status = "selected"
                color, color_confidence = self._classify(frame_bgr, signal["xyxy"])
                signal_state = color
            elif index == candidate_index:
                selection_status = "candidate"
            x1, y1, x2, y2 = signal["xyxy"]
            detections.append({
                "class_id": signal["class_id"],
                "class_name": signal["class_name"],
                "confidence": signal["confidence"],
                "box": normalize_box(x1, y1, x2, y2, width, height),
                "track_id": signal.get("track_id"),
                "extra": {"signal_state": color, "color_confidence": color_confidence,
                          "selection_status": selection_status,
                          "association_status": association["status"]},
            })
        # 기존 신호등 인덱스가 유지되도록 횡단보도는 신호등 뒤에 추가한다.
        crosswalk_boxes, diagnostics = crosswalk_diagnostics(
            crosswalk_candidates, crosswalks, association, width, height, len(signals),
        )
        detections.extend(crosswalk_boxes)
        diagnostics["detector_confidence"] = context.confidence
        return {"detections": detections, "event": {
            "type": "traffic_signal", "signal_state": signal_state,
            "association_status": association["status"],
            "association_reason": association["reason"],
            "selection_origin": association.get("selection_origin"),
            "tracking": association.get("tracking"),
            # 현재 프레임 detections의 인덱스이며, 프레임 간 추적 ID는 아니다.
            "selected_detection_index": selected_index,
            "candidate_detection_index": candidate_index,
            "detected_signal_count": len(signals),
            "raw_detected_signal_count": raw_signal_count,
            "suppressed_signal_count": suppressed_signal_count,
            "unmatched_low_confidence_count": unmatched_low_confidence_count,
            "detected_crosswalk_count": len(crosswalks),
            "crosswalk_candidate_count": len(crosswalk_candidates),
            "crosswalk_diagnostics": diagnostics,
        }}
