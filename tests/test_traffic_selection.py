"""검출 성공과 안내 대상 선택 성공을 구분하는 회귀 테스트."""
from types import SimpleNamespace
from unittest.mock import Mock
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.inference import traffic
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


def infer(pipe, frame_id=1):
    return pipe.infer(FRAME, InferenceContext("check", frame_id, frame_id * 200, 0.25))


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


def test_one_to_two_keeps_target_despite_order_and_confidence_and_reclassifies():
    pipe = make_pipeline(SIGNALS[:1])
    first = infer(pipe)
    assert first["event"]["signal_state"] == "green"
    assert first["event"]["selected_detection_index"] == 0
    assert first["detections"][0]["extra"]["selection_status"] == "selected"
    # 새 신호등이 더 높은 확률로 먼저 나와도 이전 대상을 유지한다.
    pipe.model = make_pipeline([[300, 100, 320, 140, 0.99, 0], SIGNALS[0]]).model
    pipe._classify.return_value = ("red", 0.98)
    second = infer(pipe, 2)
    assert len(second["detections"]) == 2
    assert second["event"]["selected_detection_index"] == 1
    assert second["event"]["association_status"] == "tracked"
    assert second["event"]["selection_origin"] == "single_signal"
    assert second["event"]["signal_state"] == "red"
    assert second["detections"][1]["track_id"] == first["detections"][0]["track_id"]
    assert second["detections"][0]["extra"]["signal_state"] == "unknown"
    assert pipe._classify.call_count == 2


def test_small_motion_rechecks_geometry_but_can_track_when_crosswalk_is_missing(monkeypatch):
    pipe = make_pipeline(SIGNALS[:1])
    infer(pipe)
    recheck = Mock(wraps=traffic.associate)
    monkeypatch.setattr(traffic, "associate", recheck)
    for fid in (2, 3, 4):
        dx = fid * 2
        pipe.model = make_pipeline([[100 + dx, 100, 120 + dx, 140, 0.6, 0], SIGNALS[1]]).model
        result = infer(pipe, fid)
        assert result["event"]["selected_detection_index"] == 0
        assert result["detections"][0]["track_id"] == 1
    assert recheck.call_count == 3


@pytest.mark.parametrize("replacement", [
    [300, 100, 320, 140, 0.99, 0],  # 먼 곳의 다른 신호등
    [70, 40, 150, 200, 0.99, 0],  # 같은 중심이어도 크기가 크게 달라짐
])
def test_different_single_signal_uses_original_acquisition_with_new_track_id(replacement):
    pipe = make_pipeline(SIGNALS[:1])
    first = infer(pipe)
    pipe.model = make_pipeline([replacement]).model
    result = infer(pipe, 2)
    assert result["event"]["tracking"]["reason"] == "target_missing"
    assert result["event"]["association_status"] == "single_signal"
    assert result["event"]["selected_detection_index"] == 0
    assert result["detections"][0]["track_id"] != first["detections"][0]["track_id"]
    assert pipe._classify.call_count == 2


def test_ambiguous_overlap_ends_track_instead_of_picking_high_confidence():
    pipe = make_pipeline(SIGNALS[:1])
    infer(pipe)
    pipe.model = make_pipeline([[98, 100, 118, 140, 0.99, 0],
                                [102, 100, 122, 140, 0.4, 0]]).model
    result = infer(pipe, 2)
    assert result["event"]["tracking"]["reason"] == "ambiguous_match"
    assert result["event"]["selected_detection_index"] is None
    pipe._classify.assert_called_once()


def test_missing_target_emits_no_historical_box_or_color_and_ends_track():
    pipe = make_pipeline(SIGNALS[:1])
    infer(pipe)
    pipe.model = make_pipeline([]).model
    missing = infer(pipe, 2)
    assert missing["detections"] == []
    assert missing["event"]["signal_state"] == "unknown"
    pipe.model = make_pipeline(SIGNALS).model
    returned = infer(pipe, 3)
    assert returned["event"]["selected_detection_index"] is None
    pipe._classify.assert_called_once()


@pytest.mark.parametrize("fid,timestamp,shape", [
    (3, 600, (640, 480, 3)),  # 누락 프레임
    (2, 1201, (640, 480, 3)),  # 1초 초과
    (1, 400, (640, 480, 3)),  # 중복/역순 프레임
    (2, 100, (640, 480, 3)),  # 역순 촬영 시각
    (2, 400, (960, 540, 3)),  # 해상도 변경
])
def test_discontinuous_frames_cannot_keep_old_target(fid, timestamp, shape):
    pipe = make_pipeline(SIGNALS[:1])
    infer(pipe)
    pipe.model = make_pipeline(SIGNALS).model
    result = pipe.infer(np.zeros(shape, dtype=np.uint8), InferenceContext("check", fid, timestamp, 0.25))
    assert result["event"]["tracking"]["reason"] == "discontinuous_frames"
    assert result["event"]["selected_detection_index"] is None


def test_target_state_is_isolated_by_session_and_cleared_on_reset():
    pipe = make_pipeline(SIGNALS[:1])
    infer(pipe)
    pipe.model = make_pipeline(SIGNALS).model
    other = pipe.infer(FRAME, InferenceContext("other", 2, 400, 0.25))
    assert other["event"]["selected_detection_index"] is None
    pipe.reset_session("check")
    assert infer(pipe, 2)["event"]["selected_detection_index"] is None
    pipe.close_session("check")
    assert "check" not in pipe._selectors


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


def test_tracking_does_not_claim_crosswalk_link_failure():
    pipe = make_pipeline(SIGNALS[:1])
    infer(pipe)
    pipe.model = make_pipeline(SIGNALS + [[0, 150, 480, 300, 0.646, 1]]).model
    result = infer(pipe, 2)
    assert result["event"]["selected_detection_index"] == 0
    diag = result["event"]["crosswalk_diagnostics"]
    assert diag["detection_status"] == "position_rejected"
    assert diag["connection_status"] == "previous_target_retained"
    assert diag["selected_detection_index"] is None


def test_crosswalk_without_signal_still_has_a_box():
    pipe = make_pipeline([[0, 200, 480, 640, 0.8, 1]])
    result = infer(pipe)
    assert len(result["detections"]) == 1
    assert result["event"]["detected_signal_count"] == 0
    assert result["event"]["crosswalk_diagnostics"]["connection_status"] == "no_signal_detected"
    pipe._classify.assert_not_called()


CROSSING = [0, 200, 480, 640, 0.8, 1]


def seed_left_target(pipe):
    first = infer(pipe)
    assert first['event']['selected_detection_index'] == 0
    pipe.model = make_pipeline(SIGNALS + [CROSSING]).model
    return first['detections'][0]['track_id']


def test_visible_target_can_switch_after_three_geometric_confirmations(monkeypatch):
    pipe = make_pipeline(SIGNALS[:1])
    old_id = seed_left_target(pipe)
    monkeypatch.setattr(traffic, 'estimate_vanishing_point', lambda *args: [310, 180])
    for fid in (2, 3):
        result = infer(pipe, fid)
        assert result['event']['selected_detection_index'] is None
        assert result['event']['candidate_detection_index'] == 1
        assert result['event']['association_reason'] == 'waiting_for_target_switch'
        assert result['event']['signal_state'] == 'unknown'
        assert result['event']['tracking']['target_change']['stable_frames'] == fid - 1
    pipe._classify.assert_called_once()
    pipe._classify.return_value = ('red', 0.98)
    changed = infer(pipe, 4)
    assert changed['event']['selected_detection_index'] == 1
    assert changed['event']['association_reason'] == 'target_switched'
    assert changed['event']['signal_state'] == 'red'
    assert changed['event']['selection_origin'] == 'crosswalk_matched'
    assert changed['detections'][1]['track_id'] != old_id
    new_id = changed['detections'][1]['track_id']
    retained = infer(pipe, 5)
    assert retained['detections'][1]['track_id'] == new_id
    assert retained['event']['association_status'] == 'tracked'


def test_supporting_geometry_keeps_id_despite_detection_order_change(monkeypatch):
    pipe = make_pipeline(SIGNALS[:1])
    old_id = seed_left_target(pipe)
    monkeypatch.setattr(traffic, 'estimate_vanishing_point', lambda *args: [110, 180])
    pipe.model = make_pipeline([SIGNALS[1], SIGNALS[0], CROSSING]).model
    result = infer(pipe, 2)
    assert result['event']['selected_detection_index'] == 1
    assert result['detections'][1]['track_id'] == old_id
    assert result['event']['tracking']['revalidation_status'] == 'candidate'


def test_challenger_confirmation_survives_detection_order_change(monkeypatch):
    pipe = make_pipeline(SIGNALS[:1])
    seed_left_target(pipe)
    monkeypatch.setattr(traffic, 'estimate_vanishing_point', lambda *args: [310, 180])
    infer(pipe, 2)
    pipe.model = make_pipeline([SIGNALS[1], SIGNALS[0], CROSSING]).model
    assert infer(pipe, 3)['event']['selected_detection_index'] is None
    result = infer(pipe, 4)
    assert result['event']['selected_detection_index'] == 0
    assert result['event']['association_reason'] == 'target_switched'


def test_candidate_flicker_does_not_switch_or_restore_stale_color(monkeypatch):
    pipe = make_pipeline(SIGNALS[:1])
    old_id = seed_left_target(pipe)
    vp = [310, 180]
    monkeypatch.setattr(traffic, 'estimate_vanishing_point', lambda *args: vp)
    for fid, x in enumerate([310, 110, 310, 110, 110], 2):
        vp[0] = x
        result = infer(pipe, fid)
        assert result['event']['selected_detection_index'] is None
        assert result['event']['signal_state'] == 'unknown'
    restored = infer(pipe, 7)
    assert restored['event']['association_reason'] == 'target_revalidated'
    assert restored['event']['selected_detection_index'] == 0
    assert restored['detections'][0]['track_id'] == old_id


@pytest.mark.parametrize('missing', ['direction', 'crosswalk', 'other_signal'])
def test_conflict_cannot_be_bypassed_by_missing_geometry_or_single_signal(monkeypatch, missing):
    pipe = make_pipeline(SIGNALS[:1])
    seed_left_target(pipe)
    monkeypatch.setattr(traffic, 'estimate_vanishing_point', lambda *args: [310, 180])
    infer(pipe, 2)
    if missing == 'direction':
        monkeypatch.setattr(traffic, 'estimate_vanishing_point', lambda *args: None)
    elif missing == 'crosswalk':
        pipe.model = make_pipeline(SIGNALS).model
    else:
        pipe.model = make_pipeline(SIGNALS[:1]).model
    result = infer(pipe, 3)
    assert result['event']['selected_detection_index'] is None
    assert result['event']['signal_state'] == 'unknown'
    assert result['event']['tracking']['target_change']['state'] == 'blocked'
    pipe._classify.assert_called_once()
    # Missing evidence resets the challenger streak.
    pipe.model = make_pipeline(SIGNALS + [CROSSING]).model
    monkeypatch.setattr(traffic, 'estimate_vanishing_point', lambda *args: [310, 180])
    for fid in (4, 5):
        assert infer(pipe, fid)['event']['selected_detection_index'] is None
    assert infer(pipe, 6)['event']['association_reason'] == 'target_switched'


@pytest.mark.parametrize('vp', [[210, 180], [1000, 180]])
def test_geometric_ambiguity_or_direction_conflict_suppresses_tracked_color(monkeypatch, vp):
    pipe = make_pipeline(SIGNALS[:1])
    seed_left_target(pipe)
    monkeypatch.setattr(traffic, 'estimate_vanishing_point', lambda *args: vp)
    result = infer(pipe, 2)
    assert result['event']['selected_detection_index'] is None
    assert result['event']['association_reason'] in {'ambiguous_signals', 'no_signal_in_crossing_direction'}
    assert result['event']['signal_state'] == 'unknown'
    pipe._classify.assert_called_once()


def test_switch_streak_resets_when_crosswalk_changes(monkeypatch):
    pipe = make_pipeline(SIGNALS[:1])
    seed_left_target(pipe)
    monkeypatch.setattr(traffic, 'estimate_vanishing_point', lambda *args: [310, 180])
    infer(pipe, 2)
    infer(pipe, 3)
    # Different crossing has low overlap with the original despite same signal.
    pipe.model = make_pipeline(SIGNALS + [[100, 200, 400, 360, 0.8, 1]]).model
    result = infer(pipe, 4)
    assert result['event']['selected_detection_index'] is None
    assert result['event']['tracking']['target_change']['stable_frames'] == 1
