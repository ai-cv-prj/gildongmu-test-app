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
from .traffic_geometry import estimate_stripe_direction
from .traffic_motion import estimate_camera_motion, motion_gray, transform_box

CLASS_NAMES: dict[int, str] = {0: "pedestrian_signal", 1: "crosswalk"}
SIGNAL_CLASS = "pedestrian_signal"
CROSSWALK_CLASS = "crosswalk"
CROSSWALK_CONNECTION_CONFIDENCE = 0.50
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
    """Prefer the crossing whose near edge is low and centered in the image."""
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
    """Expose detector candidates separately from the unchanged connection filter."""
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
    """Return a frame decision with a signal index or an explicit unknown reason."""
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
    # 초기 추적 기준. 검출 신뢰도 대신 연속 프레임의 위치와 크기를 비교한다.
    TRACK_MIN_IOU = 0.20
    TRACK_MAX_CENTER_DISTANCE = 0.50  # 이전 박스 대각선 길이에 대한 비율
    TRACK_MAX_SIZE_RATIO = 2.0
    TRACK_SCORE_MARGIN = 0.15
    TRACK_MAX_GAP_MS = 1000

    def __init__(self, required_frames=3):
        self.required_frames = required_frames
        self.target_box = None
        self.target_origin = None
        self.target_id = None
        self.next_target_id = 1
        self.previous_context = None
        self.previous_shape = None
        self.previous_gray = None
        self.target_needs_revalidation = False
        self._clear_pending()

    def _clear_pending(self):
        self.last_box = None
        self.last_crosswalk = None
        self.streak = 0

    def _clear_target(self):
        self.target_box = None
        self.target_origin = None
        self.target_id = None
        self.target_needs_revalidation = False

    def _acquire_target(self, decision, signals, origin):
        self.target_box = list(signals[decision["signal_index"]]["xyxy"])
        self.target_origin = origin
        self.target_id = self.next_target_id
        self.next_target_id += 1
        self.target_needs_revalidation = False
        decision.update(selection_origin=self.target_origin, track_id=self.target_id)

    def _reconsider_target(self, decision, index, signals, crosswalks, tracking):
        """A geometric challenger must persist; contradictions suspend color output.

        Keep the observed incumbent internally while confirming a challenger.
        Once contradicted, missing geometry or a single detection cannot silently
        restore its color: require three consistent geometric observations.
        """
        tracking["revalidation_status"] = decision["status"]
        tracking["revalidation_reason"] = decision["reason"]
        if decision["status"] == "candidate":
            challenger = decision["signal_index"]
            if challenger == index and not self.target_needs_revalidation:
                self._clear_pending()
                # Preserve the original acquisition evidence; one observation
                # does not upgrade a single-signal acquisition to a verified link.
                return None
            self.target_needs_revalidation = True
            decision = self.update(decision, signals, crosswalks)
            switching = challenger != index
            tracking["target_change"] = {
                "state": "confirmed" if decision["signal_index"] is not None else "pending",
                "previous_track_id": self.target_id,
                "candidate_signal_index": challenger,
                "stable_frames": decision["stable_frames"],
                "switching": switching,
            }
            if decision["signal_index"] is None:
                decision["reason"] = ("waiting_for_target_switch" if switching
                                      else "waiting_for_target_revalidation")
            elif switching:
                self._acquire_target(decision, signals, "crosswalk_matched")
                decision["reason"] = "target_switched"
                self._clear_pending()
            else:
                self.target_needs_revalidation = False
                self.target_origin = "crosswalk_matched"
                decision.update(status="tracked", reason="target_revalidated",
                                track_id=self.target_id, selection_origin=self.target_origin)
                self._clear_pending()
        else:
            self._clear_pending()
            # Failure to extract geometry alone is not evidence for a different
            # target. Explicit direction conflicts, however, must suppress output.
            conflict = decision["reason"] in {"ambiguous_signals", "no_signal_in_crossing_direction"}
            if not self.target_needs_revalidation and not conflict:
                return None
            self.target_needs_revalidation = True
            tracking["target_change"] = {"state": "blocked", "previous_track_id": self.target_id}
        decision["tracking"] = tracking
        return decision

    def _match_target(self, signals):
        box = self.target_box
        width, height = box[2] - box[0], box[3] - box[1]
        if min(width, height) <= 0:
            return None, {"reason": "target_missing"}
        ranked = []
        for index, signal in enumerate(signals):
            other = signal["xyxy"]
            ow, oh = other[2] - other[0], other[3] - other[1]
            if min(ow, oh) <= 0:
                continue
            size_ratio = max(width / ow, ow / width, height / oh, oh / height,
                             width * height / (ow * oh), ow * oh / (width * height))
            iou = box_iou(box, other)
            distance = math.dist(center(box), center(other)) / math.hypot(width, height)
            if (iou >= self.TRACK_MIN_IOU and distance <= self.TRACK_MAX_CENTER_DISTANCE
                    and size_ratio <= self.TRACK_MAX_SIZE_RATIO):
                ranked.append((iou - 0.25 * distance, index, iou, distance))
        ranked.sort(reverse=True)
        if not ranked:
            return None, {"reason": "target_missing"}
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < self.TRACK_SCORE_MARGIN:
            return None, {"reason": "ambiguous_match"}
        _, index, iou, distance = ranked[0]
        return index, {"reason": "matched", "iou": round(iou, 4),
                       "center_distance": round(distance, 4)}

    def select(self, frame, signals, crosswalks, cv2, context):
        """Track visible targets and recheck crossing geometry for alternatives.

        No box or color is emitted from history. Missing/ambiguous matches end
        the track and return to the original per-frame acquisition procedure.
        """
        previous = self.previous_context
        continuous = previous is None or (
            context.frame_id == previous.frame_id + 1
            and 0 <= context.captured_at_ms - previous.captured_at_ms <= self.TRACK_MAX_GAP_MS
            and frame.shape[:2] == self.previous_shape
        )
        tracking = {"reason": "no_previous_target" if continuous else "discontinuous_frames"}
        if not continuous:
            self._clear_target()
            self._clear_pending()
            self.previous_gray = None
        gray = motion_gray(frame, cv2)
        motion, motion_diagnostic = estimate_camera_motion(
            self.previous_gray, gray, frame.shape[:2], cv2,
        ) if self.target_box is not None or self.last_box is not None else (
            None, {"reason": "no_previous_candidate"}
        )
        self.previous_gray = gray
        self.target_box = transform_box(self.target_box, motion)
        self.last_box = transform_box(self.last_box, motion)
        self.last_crosswalk = transform_box(self.last_crosswalk, motion)
        self.previous_context = context
        self.previous_shape = frame.shape[:2]

        if self.target_box is not None:
            index, tracking = self._match_target(signals)
            tracking.update(previous_frame_id=previous.frame_id, previous_track_id=self.target_id)
            tracking["camera_motion"] = motion_diagnostic
            if index is not None:
                self.target_box = list(signals[index]["xyxy"])
                geometry = None
                if len(signals) > 1 or self.target_needs_revalidation:
                    geometry = associate(frame, signals, crosswalks, cv2, require_geometry=True)
                    reconsidered = self._reconsider_target(geometry, index, signals, crosswalks, tracking)
                    if reconsidered is not None:
                        return reconsidered
                self._clear_pending()
                return {"status": "tracked", "reason": "previous_target_retained",
                        "signal_index": index,
                        "crosswalk_index": geometry.get("crosswalk_index") if geometry else None,
                        "selection_origin": self.target_origin, "track_id": self.target_id,
                        "tracking": tracking}
            self._clear_target()
            self._clear_pending()

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
        if decision["status"] == "single_signal":
            self._clear_pending()
            return decision
        if decision["status"] != "candidate" or index is None:
            self._clear_pending()
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
                elif item["class_name"] == CROSSWALK_CLASS:
                    crosswalk_candidates.append(item)
                    if score >= CROSSWALK_CONNECTION_CONFIDENCE:
                        crosswalks.append(item)

        selector = self._selectors.setdefault(context.session_id, TemporalSelector(required_frames=3))
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
                "track_id": association.get("track_id") if index == selected_index else None,
                "extra": {"signal_state": color, "color_confidence": color_confidence,
                          "selection_status": selection_status,
                          "association_status": association["status"]},
            })
        # Append crosswalks after signals so existing signal indices stay valid.
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
            "detected_crosswalk_count": len(crosswalks),
            "crosswalk_candidate_count": len(crosswalk_candidates),
            "crosswalk_diagnostics": diagnostics,
        }}
