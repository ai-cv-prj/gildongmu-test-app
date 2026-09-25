"""
file_path: tests/backend/walking/test_walking.py

보행 마스크의 RLE 전송, PNG 대체 경로와 색상·비율 보존을 검사한다.
모델이나 실제 파일 저장 없이 작은 배열만 사용한다.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest

from backend.app.inference.walking.pipeline import make_segmentation_event


# 전송 결과를 원래 색 번호로 복원
def decode_rle(event: dict) -> np.ndarray:
    """서버가 만든 little-endian 구간을 투명·초록·핑크 번호로 복원한다."""
    mask = event["mask_rle"]
    runs = np.frombuffer(base64.b64decode(mask["data"]), dtype="<u4")
    return np.repeat(runs & 3, runs >> 2).reshape(mask["height"], mask["width"])


# 모델 라벨 순서에 의존하지 않는 마스크
@pytest.mark.parametrize("ids", [
    {"non_walkable": 0, "walkable": 1, "crosswalk": 2},
    {"non_walkable": 2, "walkable": 0, "crosswalk": 1},
])
def test_rle_preserves_labels_and_ratios(ids):
    """줄 경계를 넘는 구간도 픽셀과 클래스 비율을 그대로 유지한다."""
    expected = np.array([[1, 1, 2], [2, 0, 0]], dtype=np.uint8)
    palette = np.array([ids["non_walkable"], ids["walkable"], ids["crosswalk"]])
    event = make_segmentation_event(palette[expected], ids)
    np.testing.assert_array_equal(decode_rle(event), expected)
    assert "mask_png" not in event
    assert event["walkable_ratio"] == pytest.approx(1 / 3)
    assert event["crosswalk_ratio"] == pytest.approx(1 / 3)
    assert event["warning"] is False


# 동일 색상이 이어지는 긴 구간
@pytest.mark.parametrize("code", [0, 1, 2])
def test_single_color_uses_one_run(code):
    """64K 이상 연속 픽셀도 잘리거나 넘치지 않고 복원한다."""
    labels = np.full((640, 360), code, dtype=np.uint8)
    event = make_segmentation_event(labels, {"non_walkable": 0, "walkable": 1, "crosswalk": 2})
    assert len(base64.b64decode(event["mask_rle"]["data"])) == 4
    np.testing.assert_array_equal(decode_rle(event), labels)


# 복잡한 마스크의 기존 경로 유지
def test_fragmented_mask_uses_identical_png():
    """4096개보다 많은 구간은 PNG로 보내고 BGRA 색과 투명도를 보존한다."""
    labels = (np.arange(10000).reshape(100, 100) % 3).astype(np.uint8)
    event = make_segmentation_event(labels, {"non_walkable": 0, "walkable": 1, "crosswalk": 2})
    assert "mask_rle" not in event
    png = np.frombuffer(base64.b64decode(event["mask_png"]), dtype=np.uint8)
    actual = cv2.imdecode(png, cv2.IMREAD_UNCHANGED)
    expected = np.array([[0, 0, 0, 0], [0, 255, 0, 140], [180, 105, 255, 140]], dtype=np.uint8)[labels]
    np.testing.assert_array_equal(actual, expected)


# 입력 형태 검증
@pytest.mark.parametrize("labels", [np.array([]), np.zeros((0, 3)), np.zeros((2, 2, 2))])
def test_invalid_mask_rejected(labels):
    """빈 배열이나 3차원 배열로 잘못된 마스크를 만들지 않는다."""
    with pytest.raises(ValueError, match="2차원"):
        make_segmentation_event(labels, {"non_walkable": 0, "walkable": 1, "crosswalk": 2})
