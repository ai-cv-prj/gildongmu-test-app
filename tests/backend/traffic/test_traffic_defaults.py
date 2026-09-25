"""
file_path: tests/backend/traffic/test_traffic_defaults.py

신호등 기본값이 다른 기능에 영향을 주지 않는지 확인한다.
"""
from __future__ import annotations

from backend.app.inference.traffic.pipeline import TemporalSelector
from backend.app.schemas import SessionCreate


def test_confidence_default_only_changes_for_traffic():
    traffic = SessionCreate(mode="traffic", model_id="traffic-mock-v1", device_type="phone")
    walking = SessionCreate(mode="walking", model_id="walking-mock-v1", device_type="phone")
    explicit = SessionCreate(mode="traffic", model_id="traffic-mock-v1", device_type="phone",
                             settings={"confidence": 0.4})
    assert traffic.settings.confidence == 0.25
    assert walking.settings.confidence == 0.4
    assert explicit.settings.confidence == 0.4


def test_candidate_matches_after_three_consistent_frames():
    selector = TemporalSelector()
    signals = [{"xyxy": [100, 100, 120, 140], "track_id": 1}]
    crosswalks = [{"xyxy": [0, 200, 500, 600]}]
    for frame in (1, 2, 3):
        decision = {"status": "candidate", "signal_index": 0, "crosswalk_index": 0, "reason": None}
        result = selector.update(decision, signals, crosswalks)
        assert result["stable_frames"] == frame
        assert result["status"] == ("matched" if frame == 3 else "unknown")
