"""
file_path: tests/backend/traffic/test_traffic_selection.py

검출 성공과 안내 대상 선택 성공을 구분하는 회귀 테스트.
"""
from types import SimpleNamespace
from unittest.mock import Mock
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.inference.traffic import pipeline as traffic
from backend.app.inference.base import InferenceContext, ModelSpec
from backend.app.main import create_app


def make_pipeline(boxes):
    raw = SimpleNamespace(
        xyxy=np.array([b[:4] for b in boxes]),
        conf=np.array([b[4] for b in boxes]),
        cls=np.array([b[5] for b in boxes]),
    )
    raw.cpu = lambda: raw
    result = SimpleNamespace(boxes=raw, names=traffic.CLASS_NAMES)
    pipe = traffic.TrafficPipeline(ModelSpec("test", "traffic", "test", "test"))
    pipe.model = SimpleNamespace(predict=lambda **kwargs: [result])
    pipe.classifier = object()
    pipe._classify = Mock(return_value=("green", 0.95))
    return pipe


SIGNALS = [[100, 100, 120, 140, 0.8, 0], [300, 100, 320, 140, 0.7, 0]]
FRAME = np.zeros((640, 480, 3), dtype=np.uint8)


def infer(pipe, frame_id=1, captured_at_ms=None):
    timestamp = frame_id * 200 if captured_at_ms is None else captured_at_ms
    return pipe.infer(FRAME, InferenceContext("check", frame_id, timestamp, 0.25))


def test_two_signals_remain_visible_without_crosswalk():
    pipe = make_pipeline(SIGNALS)
    result = infer(pipe)
    assert len(result["detections"]) == 2
    assert result["event"]["signal_state"] == "unknown"
    assert result["event"]["association_reason"] == "no_unambiguous_near_crosswalk"
    assert result["event"]["selected_detection_index"] is None
    for d in result["detections"]:
        assert d["extra"]["selection_status"] == "unselected"
        assert d["extra"]["signal_state"] == "unknown"
        assert d["extra"]["color_confidence"] is None
        assert 0 <= d["box"]["x1"] < d["box"]["x2"] <= 1
    pipe._classify.assert_not_called()


def test_candidate_and_selected_target_are_distinct(monkeypatch):
    pipe = make_pipeline(SIGNALS + [[0, 200, 480, 640, 0.8, 1]])
    monkeypatch.setattr(traffic, "estimate_vanishing_point", lambda *args: [310, 180])
    for fid in (1, 2, 3):
        result = infer(pipe, fid)
        assert len(result["detections"]) == 3
        assert result["detections"][0]["extra"]["selection_status"] == "unselected"
        target = result["detections"][1]
        if fid < 3:
            assert target["extra"]["selection_status"] == "candidate"
            assert result["event"]["selected_detection_index"] is None
            assert result["event"]["candidate_detection_index"] == 1
            assert result["event"]["signal_state"] == "unknown"
            pipe._classify.assert_not_called()
        else:
            assert target["extra"]["selection_status"] == "selected"
            assert result["event"]["selected_detection_index"] == 1
            assert result["event"]["candidate_detection_index"] is None
            assert result["event"]["signal_state"] == "green"
            pipe._classify.assert_called_once()
    # 최초 횡단보도 연결로 선택한 대상도 횡단보도 미검출 때 유지한다.
    pipe.model = make_pipeline(SIGNALS).model
    result = infer(pipe, 4)
    assert result["event"]["selected_detection_index"] == 1
    assert result["event"]["association_status"] == "tracked"
    assert result["event"]["selection_origin"] == "crosswalk_matched"


def test_no_signal_has_no_target():
    pipe = make_pipeline([])
    result = infer(pipe)
    assert result["detections"] == []
    assert result["event"]["detected_signal_count"] == 0
    assert result["event"]["selected_detection_index"] is None
    assert result["event"]["association_reason"] == "no_signal_detected"
    pipe._classify.assert_not_called()


def test_weak_current_detection_keeps_selected_id_without_new_weak_candidates():
    pipe = make_pipeline(SIGNALS[:1])
    first = infer(pipe)
    original_id = first["detections"][0]["track_id"]
    weak = [[102, 100, 122, 140, 0.15, 0], [300, 100, 320, 140, 0.2, 0]]
    pipe.model = make_pipeline(weak).model
    for fid in (2, 3):
        result = infer(pipe, fid)
        assert len(result["detections"]) == 1
        target = result["detections"][0]
        assert target["track_id"] == original_id
        assert target["confidence"] == 0.15
        assert target["box"]["x1"] == 102 / FRAME.shape[1]
        assert result["event"]["association_status"] == "tracked"
        assert result["event"]["signal_state"] == "green"
        assert result["event"]["raw_detected_signal_count"] == 2
        assert result["event"]["suppressed_signal_count"] == 0
        assert result["event"]["unmatched_low_confidence_count"] == 1
    assert pipe._classify.call_count == 3


@pytest.mark.parametrize("threshold,score", [(0.25, 0.15), (0.4, 0.3)])
def test_lower_detector_threshold_does_not_create_weak_targets_or_crosswalks(threshold, score):
    pipe = make_pipeline([[100, 100, 120, 140, score, 0],
                          [0, 200, 480, 640, score, 1]])
    pipe.model.predict = Mock(wraps=pipe.model.predict)
    result = pipe.infer(FRAME, InferenceContext("check", 1, 200, threshold))
    assert pipe.model.predict.call_args.kwargs["conf"] == 0.1
    assert result["detections"] == []
    assert result["event"]["selected_detection_index"] is None
    assert result["event"]["unmatched_low_confidence_count"] == 1
    assert result["event"]["crosswalk_candidate_count"] == 0
    pipe._classify.assert_not_called()


def test_unselected_boxes_survive_api_and_session_storage(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models", min_free_disk_gb=0)
    with TestClient(create_app(settings)) as client:
        registry = client.app.state.service.registry
        pipe = make_pipeline(SIGNALS + [[0, 200, 480, 640, 0.4, 1]])
        pipe.spec = registry.get_spec("traffic-mock-v1")
        registry._loaded[pipe.spec.id] = pipe
        started = client.post("/api/sessions", json={
            "mode": "traffic", "model_id": pipe.spec.id, "device_type": "test",
        })
        assert started.status_code == 201
        sid = started.json()["session_id"]
        ok, jpeg = cv2.imencode(".jpg", FRAME)
        assert ok
        response = client.post(f"/api/sessions/{sid}/frames",
                               files={"image": ("frame.jpg", jpeg.tobytes(), "image/jpeg")},
                               data={"frame_id": 1, "captured_at_ms": 1000})
        assert response.status_code == 200
        body = response.json()
        assert len(body["detections"]) == 3
        assert body["detections"][2]["class_name"] == "crosswalk"
        assert body["detections"][2]["extra"]["crosswalk_status"] == "below_confidence"
        assert body["event"]["crosswalk_diagnostics"]["detection_status"] == "below_confidence"
        assert body["event"]["selected_detection_index"] is None
        saved = json.loads((settings.sessions_dir / sid / "results.jsonl").read_text().splitlines()[0])
        assert saved["detections"] == body["detections"]
        assert saved["event"] == body["event"]


@pytest.mark.parametrize("boxes,status", [
    ([], "not_detected"),
    ([[0, 200, 480, 640, 0.4, 1]], "below_confidence"),
    ([[0, 150, 480, 300, 0.646, 1]], "position_rejected"),
])
def test_crosswalk_detection_and_filter_failures_are_distinct(boxes, status):
    pipe = make_pipeline(SIGNALS + boxes)
    result = infer(pipe)
    diag = result["event"]["crosswalk_diagnostics"]
    assert diag["detection_status"] == status
    assert diag["connection_status"] == "not_attempted"
    assert diag["candidate_count"] == len(boxes)
    assert len(result["detections"]) == 2 + len(boxes)
    if boxes:
        d = result["detections"][2]
        assert d["confidence"] == boxes[0][4]
        assert d["extra"]["crosswalk_status"] == status
        assert d["box"]["y2"] == boxes[0][3] / FRAME.shape[0]
    pipe._classify.assert_not_called()


def test_direction_failure_keeps_crosswalk_box_and_signal_indices(monkeypatch):
    # 낮은 신뢰도 후보가 앞에 있어도 실제 연결에 사용된 박스의 인덱스가 맞아야 한다.
    pipe = make_pipeline([[0, 200, 480, 640, 0.4, 1], *SIGNALS,
                          [0, 200, 480, 640, 0.8, 1]])
    monkeypatch.setattr(traffic, "estimate_vanishing_point", lambda *args: None)
    result = infer(pipe)
    diag = result["event"]["crosswalk_diagnostics"]
    assert diag["detection_status"] == "eligible"
    assert diag["connection_status"] == "vanishing_point_unavailable"
    assert diag["selected_detection_index"] == 3
    assert result["detections"][3]["extra"]["crosswalk_status"] == "used"
    assert result["detections"][3]["extra"]["connection_reason"] == "vanishing_point_unavailable"
    assert result["event"]["selected_detection_index"] is None
    assert result["event"]["detected_crosswalk_count"] == 1
    assert result["event"]["crosswalk_candidate_count"] == 2
    pipe._classify.assert_not_called()


def test_ambiguous_crosswalks_are_not_reported_as_missing():
    pipe = make_pipeline(SIGNALS + [[0, 200, 480, 640, 0.8, 1], [0, 210, 480, 630, 0.7, 1]])
    result = infer(pipe)
    diag = result["event"]["crosswalk_diagnostics"]
    assert diag["detection_status"] == "eligible"
    assert diag["connection_status"] == "ambiguous_crosswalks"
    assert diag["selected_detection_index"] is None


def test_crosswalk_without_signal_still_has_a_box():
    pipe = make_pipeline([[0, 200, 480, 640, 0.8, 1]])
    result = infer(pipe)
    assert len(result["detections"]) == 1
    assert result["event"]["detected_signal_count"] == 0
    assert result["event"]["crosswalk_diagnostics"]["connection_status"] == "no_signal_detected"
    pipe._classify.assert_not_called()


CROSSING = [0, 200, 480, 640, 0.8, 1]


def acquire_left_target(pipe, monkeypatch):
    """실제 선택과 같이 횡단보도 연결을 세 번 확인한다."""
    monkeypatch.setattr(traffic, "estimate_vanishing_point", lambda *args: [110, 180])
    pipe.model = make_pipeline(SIGNALS + [CROSSING]).model
    for fid in (1, 2, 3):
        result = infer(pipe, fid)
        assert result["event"]["selected_detection_index"] == (0 if fid == 3 else None)
    return result["detections"][0]["track_id"]


@pytest.mark.parametrize("crossings,point,reason", [
    ([], [110, 180], "no_unambiguous_near_crosswalk"),
    ([[0, 200, 480, 640, 0.4, 1]], [110, 180], "no_unambiguous_near_crosswalk"),
    ([[0, 150, 480, 300, 0.8, 1]], [110, 180], "no_unambiguous_near_crosswalk"),
    ([CROSSING], None, "vanishing_point_unavailable"),
    ([CROSSING], [1000, 180], "no_signal_in_crossing_direction"),
])
def test_multiple_signals_need_crosswalk_direction(monkeypatch, crossings, point, reason):
    pipe = make_pipeline(SIGNALS + crossings)
    monkeypatch.setattr(traffic, "estimate_vanishing_point", lambda *args: point)
    for fid in range(1, 6):
        result = infer(pipe, fid)
        assert result["event"]["selected_detection_index"] is None
        assert result["event"]["signal_state"] == "unknown"
        assert result["event"]["association_reason"] == reason
    pipe._classify.assert_not_called()


def test_visible_target_stays_locked_beyond_2500ms_and_color_updates(monkeypatch):
    pipe = make_pipeline([])
    old_id = acquire_left_target(pipe, monkeypatch)
    # 다른 신호등이 횡단보도 방향과 더 잘 맞고 신뢰도가 높아도 교체하지 않는다.
    monkeypatch.setattr(traffic, "estimate_vanishing_point", lambda *args: [310, 180])
    for fid in range(4, 24):
        reordered = fid % 2 == 0
        other = [300, 100, 320, 140, 0.99, 0]
        boxes = [other, SIGNALS[0]] if reordered else [SIGNALS[0], other]
        pipe.model = make_pipeline(boxes + [CROSSING]).model
        color = "red" if fid < 10 else "green"
        pipe._classify.return_value = (color, 0.98)
        result = infer(pipe, fid)
        index = 1 if reordered else 0
        assert result["event"]["selected_detection_index"] == index
        assert result["detections"][index]["track_id"] == old_id
        assert result["event"]["selection_origin"] == "crosswalk_matched"
        assert result["event"]["signal_state"] == color
        assert result["event"]["association_reason"] == "previous_target_retained"
        assert result["detections"][1-index]["extra"]["signal_state"] == "unknown"
        assert "blink" not in result["event"]
        assert "blink" not in result["detections"][index]["extra"]
    assert pipe._classify.call_count == 21


@pytest.mark.parametrize("point", [None, [210, 180], [1000, 180]])
def test_locked_target_survives_changed_or_missing_crosswalk_direction(monkeypatch, point):
    pipe = make_pipeline([])
    old_id = acquire_left_target(pipe, monkeypatch)
    monkeypatch.setattr(traffic, "estimate_vanishing_point", lambda *args: point)
    pipe.model = make_pipeline(SIGNALS + [CROSSING]).model
    assert infer(pipe, 4)["detections"][0]["track_id"] == old_id
    # 횡단보도가 사라져도 현재 관측된 동일 대상의 색상은 계속 분류한다.
    pipe.model = make_pipeline(SIGNALS[:1]).model
    returned = infer(pipe, 5)
    assert returned["detections"][0]["track_id"] == old_id
    assert returned["event"]["signal_state"] == "green"


def test_small_motion_tracks_without_reselecting_crosswalk(monkeypatch):
    pipe = make_pipeline([])
    old_id = acquire_left_target(pipe, monkeypatch)
    for fid in (4, 5, 6):
        dx = (fid - 3) * 2
        pipe.model = make_pipeline([[100 + dx, 100, 120 + dx, 140, 0.6, 0], SIGNALS[1]]).model
        result = infer(pipe, fid)
        assert result["event"]["selected_detection_index"] == 0
        assert result["detections"][0]["track_id"] == old_id


@pytest.mark.parametrize("replacement", [
    [300, 100, 320, 140, 0.99, 0],
    [70, 40, 150, 200, 0.99, 0],
])
def test_lost_target_is_immediately_replaced_by_single_signal(monkeypatch, replacement):
    pipe = make_pipeline([])
    old_id = acquire_left_target(pipe, monkeypatch)
    pipe.model = make_pipeline([replacement]).model
    result = infer(pipe, 4)
    assert result["event"]["selected_detection_index"] == 0
    assert result["event"]["association_status"] == "single_signal"
    assert result["event"]["tracking"]["reason"] == "target_missing"
    assert result["detections"][0]["track_id"] != old_id
    assert pipe._classify.call_count == 2


def test_reselection_starts_immediately_and_requires_three_crosswalk_confirmations(monkeypatch):
    pipe = make_pipeline([])
    old_id = acquire_left_target(pipe, monkeypatch)
    pipe.model = make_pipeline([SIGNALS[1], [400, 100, 420, 140, 0.9, 0], CROSSING]).model
    monkeypatch.setattr(traffic, "estimate_vanishing_point", lambda *args: [310, 180])
    for fid in (4, 5):
        pending = infer(pipe, fid)
        assert pending["event"]["candidate_detection_index"] == 0
        assert pending["event"]["selected_detection_index"] is None
        assert pending["event"]["signal_state"] == "unknown"
        assert pending["event"]["association_reason"] == "waiting_for_temporal_consistency"
    selected = infer(pipe, 6)
    assert selected["event"]["selected_detection_index"] == 0
    assert selected["event"]["selection_origin"] == "crosswalk_matched"
    assert selected["detections"][0]["track_id"] != old_id
    assert pipe._classify.call_count == 2


def test_overlapping_detections_are_suppressed_before_botsort(monkeypatch):
    pipe = make_pipeline([])
    acquire_left_target(pipe, monkeypatch)
    pipe.model = make_pipeline([[98, 100, 118, 140, 0.99, 0], [102, 100, 122, 140, 0.4, 0]]).model
    result = infer(pipe, 4)
    assert result["event"]["tracking"]["tracker"] == "botsort"
    selected = result["event"]["selected_detection_index"]
    assert result["detections"][selected]["track_id"] == 1
    assert len(result["detections"]) == 1
    assert result["event"]["raw_detected_signal_count"] == 2
    assert result["event"]["suppressed_signal_count"] == 1
    assert pipe._classify.call_count == 2


def test_missing_target_is_cleared_immediately_without_historical_color(monkeypatch):
    pipe = make_pipeline([])
    old_id = acquire_left_target(pipe, monkeypatch)
    pipe.model = make_pipeline([]).model
    missing = infer(pipe, 4)
    assert missing["detections"] == []
    assert missing["event"]["signal_state"] == "unknown"
    assert missing["event"]["selected_detection_index"] is None
    assert pipe._selectors["check"].target_id is None
    pipe._classify.assert_called_once()
    pipe.model = make_pipeline(SIGNALS[:1]).model
    returned = infer(pipe, 5)
    assert returned["event"]["selected_detection_index"] == 0
    assert returned["detections"][0]["track_id"] != old_id


@pytest.mark.parametrize("fid,timestamp,shape", [
    (5, 1000, (640, 480, 3)),
    (4, 1601, (640, 480, 3)),
    (3, 800, (640, 480, 3)),
    (4, 500, (640, 480, 3)),
    (4, 800, (960, 540, 3)),
])
def test_discontinuous_frames_require_new_crosswalk_selection(monkeypatch, fid, timestamp, shape):
    pipe = make_pipeline([])
    acquire_left_target(pipe, monkeypatch)
    pipe.model = make_pipeline(SIGNALS).model
    result = pipe.infer(np.zeros(shape, dtype=np.uint8), InferenceContext("check", fid, timestamp, 0.25))
    assert result["event"]["tracking"]["reason"] == "discontinuous_frames"
    assert result["event"]["selected_detection_index"] is None
    pipe._classify.assert_called_once()


def test_target_state_is_isolated_by_session_and_cleared_on_reset(monkeypatch):
    pipe = make_pipeline([])
    acquire_left_target(pipe, monkeypatch)
    pipe.model = make_pipeline(SIGNALS).model
    other = pipe.infer(FRAME, InferenceContext("other", 4, 800, 0.25))
    assert other["event"]["selected_detection_index"] is None
    assert infer(pipe, 4)["event"]["selected_detection_index"] == 0
    pipe.reset_session("check")
    assert infer(pipe, 5)["event"]["selected_detection_index"] is None
    pipe.close_session("check")
    assert "check" not in pipe._selectors


def test_candidate_streak_resets_when_crosswalk_changes(monkeypatch):
    pipe = make_pipeline(SIGNALS + [CROSSING])
    monkeypatch.setattr(traffic, "estimate_vanishing_point", lambda *args: [110, 180])
    infer(pipe, 1)
    infer(pipe, 2)
    pipe.model = make_pipeline(SIGNALS + [[100, 200, 400, 360, 0.8, 1]]).model
    result = infer(pipe, 3)
    assert result["event"]["selected_detection_index"] is None
    infer(pipe, 4)
    assert infer(pipe, 5)["event"]["selected_detection_index"] == 0


def test_tracking_does_not_claim_crosswalk_link_failure(monkeypatch):
    pipe = make_pipeline([])
    acquire_left_target(pipe, monkeypatch)
    pipe.model = make_pipeline(SIGNALS + [[0, 150, 480, 300, 0.646, 1]]).model
    result = infer(pipe, 4)
    assert result["event"]["selected_detection_index"] == 0
    diag = result["event"]["crosswalk_diagnostics"]
    assert diag["detection_status"] == "position_rejected"
    assert diag["connection_status"] == "previous_target_retained"
    assert diag["selected_detection_index"] is None


def test_single_signal_is_immediate_but_multiple_signals_require_crosswalk():
    pipe = make_pipeline(SIGNALS[:1])
    first = infer(pipe)
    assert first["event"]["selected_detection_index"] == 0
    assert first["event"]["selection_origin"] == "single_signal"
    pipe.model = make_pipeline([SIGNALS[1], SIGNALS[0]]).model
    for fid in (2, 3, 4):
        pending = infer(pipe, fid)
        assert pending["event"]["selected_detection_index"] is None
        assert pending["event"]["signal_state"] == "unknown"
        assert pending["event"]["association_reason"] == "no_unambiguous_near_crosswalk"
    pipe._classify.assert_called_once()


@pytest.mark.parametrize("selected", [0, 1])
def test_provisional_target_confirms_crosswalk_then_locks(monkeypatch, selected):
    pipe = make_pipeline(SIGNALS[:1])
    first = infer(pipe)
    old_id = first["detections"][0]["track_id"]
    point = [110 if selected == 0 else 310, 180]
    monkeypatch.setattr(traffic, "estimate_vanishing_point", lambda *args: point)
    for fid in (2, 3, 4):
        # 검출 순서가 바뀌어도 같은 후보의 연속 확인은 유지한다.
        order = [1, 0] if fid % 2 else [0, 1]
        pipe.model = make_pipeline([SIGNALS[i] for i in order] + [CROSSING]).model
        result = infer(pipe, fid)
        if fid < 4:
            assert result["event"]["selected_detection_index"] is None
            assert result["event"]["candidate_detection_index"] == order.index(selected)
            assert result["event"]["signal_state"] == "unknown"
            assert all(d["extra"].get("color_confidence") is None for d in result["detections"])
            pipe._classify.assert_called_once()
        else:
            assert result["event"]["selected_detection_index"] == order.index(selected)
            assert result["event"]["selection_origin"] == "crosswalk_matched"
            new_id = result["detections"][order.index(selected)]["track_id"]
            assert (new_id == old_id) == (selected == 0)
    # 연결 확정 이후에는 방향 후보가 반대로 바뀌어도 기존 대상을 유지한다.
    point[0] = 310 if selected == 0 else 110
    for fid in range(5, 20):
        result = infer(pipe, fid)
        assert result["event"]["selected_detection_index"] == selected
        assert result["detections"][selected]["track_id"] == new_id
        assert result["event"]["association_reason"] == "previous_target_retained"


@pytest.mark.parametrize("failure", ["direction", "crosswalk", "candidate", "single"])
def test_provisional_confirmation_resets_on_interrupted_evidence(monkeypatch, failure):
    pipe = make_pipeline(SIGNALS[:1])
    infer(pipe)
    point = [310, 180]
    monkeypatch.setattr(traffic, "estimate_vanishing_point", lambda *args: point)
    pipe.model = make_pipeline(SIGNALS + [CROSSING]).model
    infer(pipe, 2)
    infer(pipe, 3)
    if failure == "direction":
        point = None
    elif failure == "crosswalk":
        pipe.model = make_pipeline(SIGNALS).model
    elif failure == "candidate":
        point = [110, 180]
    else:
        pipe.model = make_pipeline(SIGNALS[:1]).model
    interrupted = infer(pipe, 4)
    assert interrupted["event"]["selected_detection_index"] is None
    assert interrupted["event"]["signal_state"] == "unknown"
    pipe._classify.assert_called_once()
    point = [310, 180]
    pipe.model = make_pipeline(SIGNALS + [CROSSING]).model
    for fid in (5, 6):
        assert infer(pipe, fid)["event"]["selected_detection_index"] is None
    assert infer(pipe, 7)["event"]["selected_detection_index"] == 1


def test_provisional_confirmation_ends_on_target_loss(monkeypatch):
    pipe = make_pipeline(SIGNALS[:1])
    first = infer(pipe)
    monkeypatch.setattr(traffic, "estimate_vanishing_point", lambda *args: [310, 180])
    pipe.model = make_pipeline(SIGNALS + [CROSSING]).model
    assert infer(pipe, 2)["event"]["selected_detection_index"] is None
    pipe.model = make_pipeline([]).model
    assert infer(pipe, 3)["event"]["selected_detection_index"] is None
    pipe.model = make_pipeline(SIGNALS[:1]).model
    returned = infer(pipe, 4)
    assert returned["event"]["selected_detection_index"] == 0
    assert returned["event"]["selection_origin"] == "single_signal"
    assert returned["detections"][0]["track_id"] != first["detections"][0]["track_id"]


@pytest.mark.parametrize("reverse", [False, True])
def test_reported_frame75_duplicate_is_one_signal_before_selection(reverse):
    # 실제 신고 프레임의 정규화 좌표. 같은 신호등의 작은 박스가 큰 박스에 겹쳐 있다.
    boxes = [
        [.1993830928, .2077931404, .2296585931, .2352834702, .64086699, 0],
        [.1992494936, .2070301056, .2309090508, .2494961421, .28398138, 0],
    ]
    boxes = [[b[0]*480, b[1]*640, b[2]*480, b[3]*640, *b[4:]] for b in boxes]
    high = boxes[0]
    pipe = make_pipeline((list(reversed(boxes)) if reverse else boxes) + [CROSSING])
    result = infer(pipe)
    assert result["event"]["raw_detected_signal_count"] == 2
    assert result["event"]["suppressed_signal_count"] == 1
    assert result["event"]["detected_signal_count"] == 1
    assert result["event"]["selected_detection_index"] == 0
    assert result["event"]["selection_origin"] == "single_signal"
    assert result["event"]["signal_state"] == "green"
    assert len(result["detections"]) == 2
    assert result["detections"][1]["class_name"] == "crosswalk"
    assert result["detections"][0]["confidence"] == high[4]
    assert pipe._trackers["check"].next_id == 2
    pipe._classify.assert_called_once()


def test_separate_low_score_signals_are_not_suppressed():
    pipe = make_pipeline([SIGNALS[0], [123, 100, 143, 140, .28, 0],
                          [100, 145, 120, 185, .28, 0]])
    result = infer(pipe)
    assert result["event"]["detected_signal_count"] == 3
    assert result["event"]["suppressed_signal_count"] == 0
    assert len({d["track_id"] for d in result["detections"]}) == 3
    assert result["event"]["selected_detection_index"] is None
    pipe._classify.assert_not_called()


def test_duplicate_appearing_does_not_trigger_multiple_signal_reconfirmation():
    pipe = make_pipeline(SIGNALS[:1])
    first = infer(pipe)
    pipe.model = make_pipeline([SIGNALS[0], [99, 99, 121, 145, .3, 0]]).model
    result = infer(pipe, 2)
    assert result["event"]["selected_detection_index"] == 0
    assert result["event"]["association_reason"] == "previous_target_retained"
    assert result["detections"][0]["track_id"] == first["detections"][0]["track_id"]
    assert result["event"]["suppressed_signal_count"] == 1
    assert pipe._classify.call_count == 2
