"""실제 BoT-SORT를 이용해 ID 연속성·세션 격리·검출 소실을 검증한다."""
import cv2
import numpy as np
import pytest

from backend.app.inference.base import InferenceContext
from backend.app.inference.traffic import TemporalSelector
from backend.app.inference.traffic_tracker import SignalTracker


def frame():
    image = np.zeros((640, 480, 3), np.uint8)
    for x, y in np.random.default_rng(38).integers([20, 20], [460, 620], (350, 2)):
        cv2.circle(image, (int(x), int(y)), 3, (180, 180, 180), -1)
    return image


def detections(xs=(100, 300), confidence=.8):
    return [dict(xyxy=[x, 100, x+20, 140], confidence=confidence,
                 class_id=0, class_name="pedestrian_signal") for x in xs]


def context(fid, *, timestamp=None, confidence=.25):
    return InferenceContext("test", fid, fid*200 if timestamp is None else timestamp, confidence)


def test_all_signal_ids_survive_camera_motion_and_detection_order_changes():
    tracker = SignalTracker()
    image = frame()
    first = detections()
    tracker.update(image, first, context(1))
    assert [s["track_id"] for s in first] == [1, 2]
    for fid, dx in enumerate([40, 80, 120], 2):
        current = detections((300+dx, 100+dx))
        moved = cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, 0]]), (480, 640))
        original_boxes = [list(s["xyxy"]) for s in current]
        tracker.update(moved, current, context(fid))
        assert [s["track_id"] for s in current] == [2, 1]
        assert [s["xyxy"] for s in current] == original_boxes


def test_new_signal_gets_an_id_on_first_detection_without_dropping_boxes():
    tracker = SignalTracker(); image = frame()
    first = detections((100,))
    tracker.update(image, first, context(1))
    second = detections()
    tracker.update(image, second, context(2))
    assert [s["track_id"] for s in second] == [1, 2]


def test_empty_frame_does_not_restore_prediction_or_preserve_lost_id():
    tracker = SignalTracker(); image = frame()
    first = detections()
    tracker.update(image, first, context(1))
    missing = []
    tracker.update(image, missing, context(2))
    assert missing == []
    assert tracker.tracker.lost_stracks == []
    returned = detections()
    tracker.update(image, returned, context(3))
    assert [s["track_id"] for s in returned] == [3, 4]


def test_sessions_have_independent_ids_and_new_session_cannot_reset_active_ids():
    image = frame(); first_tracker = SignalTracker()
    first_tracker.update(image, detections((100,)), context(1))
    second_tracker = SignalTracker()
    other = detections()
    second_tracker.update(image, other, context(1))
    assert [s["track_id"] for s in other] == [1, 2]
    second_tracker.tracker.reset()
    current = detections()
    first_tracker.update(image, current, context(2))
    assert [s["track_id"] for s in current] == [1, 2]
    assert len(set(s["track_id"] for s in current)) == len(current)


@pytest.mark.parametrize("fid,timestamp,shape", [
    (3, 600, (640, 480)), (2, 1201, (640, 480)),
    (2, 100, (640, 480)), (2, 400, (320, 240)),
])
def test_discontinuity_resets_motion_and_does_not_reuse_ids(fid, timestamp, shape):
    tracker = SignalTracker(); image = frame()
    tracker.update(image, detections((100,)), context(1))
    current = detections((100,))
    tracker.update(cv2.resize(image, shape[::-1]), current, context(fid, timestamp=timestamp))
    assert current[0]["track_id"] == 2


def test_user_detection_threshold_allows_single_low_score_signal():
    tracker = SignalTracker()
    for fid in range(1, 5):
        current = detections((100 + fid,), confidence=.15)
        tracker.update(frame(), current, context(fid, confidence=.1))
        assert current[0]["track_id"] == 1


@pytest.mark.parametrize("threshold,low_score", [(0.25, 0.15), (0.4, 0.3)])
def test_weak_detection_only_continues_existing_track(threshold, low_score):
    tracker = SignalTracker(); image = frame()
    first = detections((100,))
    tracker.update(image, first, context(1, confidence=threshold))
    for fid in (2, 3):
        current = detections((100 + fid, 300), confidence=low_score)
        tracker.update(image, current, context(fid, confidence=threshold))
        assert [s["track_id"] for s in current] == [first[0]["track_id"], None]
    recovered = detections((105,))
    tracker.update(image, recovered, context(4, confidence=threshold))
    assert recovered[0]["track_id"] == first[0]["track_id"]


def test_weak_detection_cannot_restore_track_after_empty_frame():
    tracker = SignalTracker(); image = frame()
    tracker.update(image, detections((100,)), context(1))
    tracker.update(image, [], context(2))
    current = detections((100,), confidence=.15)
    tracker.update(image, current, context(3))
    assert current[0]["track_id"] is None


def test_id_matching_does_not_fall_back_to_nearby_different_object():
    selector = TemporalSelector(); image = frame()
    first = detections((100,)); first[0]["track_id"] = 12
    selected = selector.select(image, first, [], cv2, context(1))
    assert selected["track_id"] == 12
    changed = detections(); changed[0]["track_id"] = 14; changed[1]["track_id"] = 15
    result = selector.select(image, changed, [], cv2, context(2))
    assert result["signal_index"] is None
    assert result["tracking"]["reason"] == "target_missing"
    assert selector.target_id is None


def test_candidate_confirmation_cannot_combine_different_tracker_ids():
    selector = TemporalSelector()
    crossing = [{"xyxy": [0, 200, 480, 640]}]
    for track_id in (1, 2, 3):
        signals = detections((100,)); signals[0]["track_id"] = track_id
        result = selector.update(dict(status="candidate", signal_index=0, crosswalk_index=0),
                                 signals, crossing)
        assert result["signal_index"] is None
        assert result["stable_frames"] == 1


def test_unassigned_detection_never_becomes_selected_color_target():
    selector = TemporalSelector()
    result = selector.select(frame(), detections((100,)), [], cv2, context(1))
    assert result["signal_index"] is None
    assert result["reason"] == "waiting_for_tracking"
