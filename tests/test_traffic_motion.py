"""카메라 이동 시 관측된 후보는 유지할 수 있지만 사라진 검출은 복원하지 않는지 검증한다."""
import cv2
import numpy as np
import pytest

from backend.app.inference import traffic
from backend.app.inference.base import InferenceContext
from backend.app.inference.traffic_motion import estimate_camera_motion, motion_gray, transform_box


def scene():
    image = np.zeros((640, 480, 3), np.uint8)
    rng = np.random.default_rng(38)
    for x, y in rng.integers([20, 20], [460, 620], (350, 2)):
        cv2.circle(image, (int(x), int(y)), 3, (180, 180, 180), -1)
    return image


def shifted(image, dx, dy):
    return cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, dy]]), (480, 640))


def context(fid):
    return InferenceContext('motion-test', fid, fid * 200, 0.25)


def signals(dx=0):
    return [{'xyxy': [100 + dx, 100, 120 + dx, 140], 'track_id': 1},
            {'xyxy': [300 + dx, 100, 320 + dx, 140], 'track_id': 2}]


def test_recovers_global_translation_from_images():
    image = scene()
    matrix, diagnostic = estimate_camera_motion(motion_gray(image, cv2),
                                               motion_gray(shifted(image, 40, 18), cv2), image.shape[:2], cv2)
    assert diagnostic['reason'] == 'compensated'
    assert transform_box([100, 100, 120, 140], matrix) == pytest.approx([140, 118, 160, 158], abs=1)


def test_local_motion_without_image_coverage_is_rejected():
    image = scene()
    image[240:] = 0
    matrix, _ = estimate_camera_motion(motion_gray(image, cv2), motion_gray(shifted(image, 40, 0), cv2), image.shape[:2], cv2)
    assert matrix is None


def test_large_camera_shift_keeps_current_visible_target_and_not_a_missing_one(monkeypatch):
    def unexpected_motion(*args):
        pytest.fail('횡단보도 후보가 없으면 선택기가 이동 보정을 실행하면 안 된다')

    monkeypatch.setattr(traffic, 'estimate_camera_motion', unexpected_motion)
    selector = traffic.TemporalSelector()
    image = scene()
    first = selector.select(image, signals()[:1], [], cv2, context(1))
    second = selector.select(shifted(image, 40, 0), signals(40)[:1], [], cv2, context(2))
    assert second['signal_index'] == 0
    assert second['track_id'] == first['track_id']
    assert second['status'] == 'tracked'
    missing = selector.select(shifted(image, 80, 0), [], [], cv2, context(3))
    assert missing['signal_index'] is None
    assert selector.target_id is None
    assert missing["tracking"]["previous_track_id"] == first["track_id"]


def test_pending_crosswalk_link_survives_camera_shift(monkeypatch):
    selector = traffic.TemporalSelector()
    image = scene()
    monkeypatch.setattr(traffic, 'estimate_vanishing_point', lambda frame, box, cv: [310 + box[0], 180])
    for fid, dx in enumerate([0, 40, 80], 1):
        crossing = [{'xyxy': [dx, 200, 400 + dx, 600]}]
        decision = selector.select(shifted(image, dx, 0), signals(dx), crossing, cv2, context(fid))
        assert decision['stable_frames'] == fid
        assert decision['signal_index'] == (1 if fid == 3 else None)
    assert decision['selection_origin'] == 'crosswalk_matched'


def test_duplicate_tracker_ids_cannot_select_a_target():
    selector = traffic.TemporalSelector()
    image = scene()
    selector.select(image, signals()[:1], [], cv2, context(1))
    ambiguous = [{'xyxy': [138, 100, 158, 140], 'track_id': 1}, {'xyxy': [142, 100, 162, 140], 'track_id': 1}]
    decision = selector.select(shifted(image, 40, 0), ambiguous, [], cv2, context(2))
    assert decision['signal_index'] is None
    assert decision['tracking']['reason'] == 'ambiguous_match'


def test_provisional_crosswalk_confirmation_survives_camera_motion(monkeypatch):
    selector = traffic.TemporalSelector()
    image = scene()
    first = selector.select(image, signals()[:1], [], cv2, context(1))
    monkeypatch.setattr(traffic, 'estimate_vanishing_point', lambda frame, box, cv: [310 + box[0], 180])
    for fid, dx in enumerate([40, 80, 120] + [120] * 10, 2):
        crossing = [{'xyxy': [dx, 200, 400 + dx, 600]}]
        decision = selector.select(shifted(image, dx, 0), signals(dx), crossing, cv2, context(fid))
        expected_motion = 'compensated' if fid in (3, 4) else 'no_previous_candidate'
        assert decision['tracking']['camera_motion']['reason'] == expected_motion
        if fid < 4:
            assert decision['signal_index'] is None
            assert decision['stable_frames'] == fid - 1
        else:
            assert decision['signal_index'] == 1
            assert decision['selection_origin'] == 'crosswalk_matched'
            assert decision['track_id'] != first['track_id']
            if fid > 4:
                assert decision['reason'] == 'previous_target_retained'
