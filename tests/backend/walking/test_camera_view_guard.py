"""
file_path: tests/backend/walking/test_camera_view_guard.py

Camera-view regressions for ground-filled, obscured and unstable scenes.
"""
import unittest
from types import SimpleNamespace

import numpy as np

from backend.app.inference.walking.risk.risk import RiskEngine
from backend.app.inference.walking.risk.risk_config import risk_config
from backend.app.inference.walking.visualization.risk_visualization import draw_risk
from .test_risk import FixedTracker, detection
from backend.app.inference.walking.response import make_response


CFG = {"camera_view_guard_enabled": True, "review_overlay": True,
       "hard_reset_gap_s": 2.0}
RNG = np.random.default_rng(7)
GRAY = np.clip(RNG.normal(120, 12, (320, 180)), 0, 255).astype(np.uint8)
GROUND = np.repeat(GRAY[:, :, None], 3, axis=2)
FORWARD = np.empty_like(GROUND)
FORWARD[:160] = np.clip(
    np.array([190, 110, 45]) + RNG.normal(0, 25, (160, 180, 3)),
    0, 255).astype(np.uint8)
FORWARD[160:] = np.clip(
    np.array([75, 145, 85]) + RNG.normal(0, 25, (160, 180, 3)),
    0, 255).astype(np.uint8)
BIG = detection((0, 0, 180, 320), "bird", 14)
NEAR = detection((72, 190, 108, 300), "person", 0)


def make_engine(stable=True):
    guard = SimpleNamespace(update=lambda frame: stable, reset=lambda: None)
    return RiskEngine(CFG, tracker=FixedTracker(), camera_guard=guard)


class CameraViewTests(unittest.TestCase):
    def test_ground_appearance_alone_does_not_mute_near_hazard(self):
        e = make_engine()
        for t in (0, .05, .11, .2):
            result = e.update(GROUND, [NEAR], t)
        self.assertEqual(result["camera_view"]["status"], "clear")
        self.assertEqual(result["detections"][0]["risk_level"], "danger")

    def test_large_false_box_on_ground_view_is_audited_not_warned(self):
        e = make_engine()
        e.update(GROUND, [], 0)
        e.update(GROUND, [], .05)
        result = e.update(GROUND, [BIG], .11)
        item = result["detections"][0]
        self.assertEqual(result["camera_view"]["status"], "unavailable")
        self.assertEqual(result["camera_view"]["reason"], "ground_dominated_view")
        self.assertEqual(result["warning"]["source"], "camera_view")
        self.assertEqual(result["warning"]["level"], "caution")
        self.assertEqual(item["untrusted_risk_level"], "danger")
        self.assertIsNone(item["risk_level"])
        self.assertIsNone(item["alert_level"])
        self.assertEqual(result["warning_groups"], [])
        self.assertTrue(any(x.get("source") == "camera_view" for x in result["events"]))
        shown = draw_risk(GROUND, result, risk_config(CFG))
        self.assertEqual(shown.shape, GROUND.shape)
        # The screen-sized red risk rectangle is absent from the upper corner.
        self.assertLess(int(shown[5, 5, 2]), 200)

    def test_prior_danger_is_bounded_then_camera_guidance_takes_over(self):
        e = make_engine()
        first = e.update(FORWARD, [NEAR], 0)
        self.assertEqual(first["warning"]["level"], "danger")
        e.update(GROUND, [], .1)
        e.update(GROUND, [], .15)
        hidden = e.update(GROUND, [BIG], .21)
        self.assertEqual(hidden["camera_view"]["status"], "unavailable")
        self.assertEqual(hidden["warning"]["level"], "danger")
        self.assertIn("previous hazard unverified", hidden["warning_text"])
        later = e.update(GROUND, [BIG], 1.0)
        self.assertEqual(later["warning"]["level"], "caution")
        self.assertEqual(later["warning"]["source"], "camera_view")

    def test_obscured_view_recovers_and_resets_stale_tracking(self):
        e = make_engine()
        dark = np.zeros_like(GROUND)
        e.update(dark, [BIG], 0)
        hidden = e.update(dark, [BIG], .2)
        self.assertEqual(hidden["camera_view"]["reason"], "obscured_view")
        self.assertEqual(hidden["warning"]["source"], "camera_view")
        e.update(FORWARD, [], .3)
        restored = e.update(FORWARD, [NEAR], .7)
        self.assertEqual(restored["camera_view"]["status"], "clear")
        self.assertTrue(restored["view_recovered"])
        self.assertTrue(restored["state_reset"])
        self.assertEqual(restored["warning"]["level"], "danger")

    def test_app_response_exposes_korean_camera_guidance(self):
        e = make_engine()
        e.update(GROUND, [], 0)
        e.update(GROUND, [], .05)
        result = e.update(GROUND, [BIG], .11)
        labels = {"non_walkable": 0, "walkable": 1, "crosswalk": 2}
        response = make_response(
            result, GROUND.shape, np.ones(GROUND.shape[:2], np.uint8), labels,
            {"risk_config": risk_config(CFG), "config_sha256": "test",
             "source_revision": "test"},
        )
        event = response["event"]
        self.assertEqual(event["camera_view"]["status"], "unavailable")
        self.assertEqual(event["level"], "caution")
        self.assertIn("카메라를 전방으로", event["warning_text"])
        self.assertEqual(response["detections"][0]["extra"]["untrusted_risk_level"], "danger")
        self.assertIsNone(response["detections"][0]["extra"]["alert_level"])
        self.assertEqual(event["counts"]["camera_view"], 1)
        self.assertEqual([x["source"] for x in event["risk_events"]
                          if x.get("source") == "camera_view"], ["camera_view"])

    def test_motion_uncertainty_does_not_hide_immediate_hazard(self):
        e = make_engine(stable=False)
        for t in (0, .1, .3):
            result = e.update(FORWARD, [NEAR], t)
        self.assertEqual(result["camera_view"]["status"], "uncertain")
        self.assertEqual(result["detections"][0]["risk_level"], "danger")
        self.assertEqual(result["warning"]["level"], "danger")


if __name__ == "__main__":
    unittest.main()
