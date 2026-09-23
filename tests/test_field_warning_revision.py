"""Field-session regressions for the wide ROI and honest path warnings."""
from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np
import yaml

from backend.app.inference.risk.risk_config import risk_config
from backend.app.inference.risk.risk import RiskEngine
from backend.app.inference.risk.surface_risk import SurfaceRisk
from backend.app.inference.risk.hazard_labels import LabelMemory
from backend.app.inference.risk.warning_summary import WarningSelector
from test_risk import FRAME, detection, engine

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] /
                     "backend/config/walking_risk.yaml").read_text())["risk"]
LABELS = {"non_walkable": 0, "walkable": 1, "crosswalk": 2}


class FieldWarningRevisionTests(unittest.TestCase):
    def test_wide_bottom_is_fixed_while_visible_ground_changes_top(self):
        mask_low = np.zeros((100, 100), np.uint8)
        mask_low[58:] = 1
        mask_high = np.zeros((100, 100), np.uint8)
        mask_high[30:] = 1
        low = engine(config=CFG).update(FRAME, [], 0, class_map=mask_low,
                                        label_ids=LABELS)["roi"]
        high = engine(config=CFG).update(FRAME, [], 0, class_map=mask_high,
                                         label_ids=LABELS)["roi"]
        for roi in (low, high):
            self.assertEqual(roi["corridor_polygon"][2:],
                             [[.98, .75], [.98, 1.], [.02, 1.], [.02, .75]])
            self.assertEqual(roi["immediate_polygon"], CFG["immediate_polygon"])
            self.assertLessEqual(roi["path_top_y"], .60)
        self.assertGreater(low["path_top_y"], high["path_top_y"])

    def test_wide_side_contact_is_caution_central_contact_is_danger(self):
        # SESAC-86의 보행불가 주변 필터가 켜져 있으므로 마스크 없이는 danger가 나올 수 없다.
        # 이 테스트가 보는 것은 ROI 폭이지 주변 보행 가능 여부가 아니므로 보행 가능 지면을 준다.
        walkable = np.ones((100, 100), np.uint8)

        def assess(box):
            return engine(config=CFG).update(FRAME, [detection(box)], 0,
                                             class_map=walkable,
                                             label_ids=LABELS)["detections"][0]

        side = assess((78, 50, 95, 85))
        center = assess((42, 50, 58, 85))
        self.assertEqual(side["risk_level"], "caution")
        self.assertIn("near_path_side_candidate", side["reasons"])
        self.assertEqual(center["risk_level"], "danger")
        self.assertGreaterEqual(side["geometry"]["immediate_overlap"], .2)
        self.assertEqual(assess((78, 35, 96, 91))["risk_level"], "danger")

    def test_occlusion_does_not_immediately_shrink_visible_roi(self):
        e = engine(config=CFG)
        visible = np.zeros((100, 100), np.uint8)
        visible[30:] = 1
        initial = e.update(FRAME, [], 0, class_map=visible,
                           label_ids=LABELS)["roi"]["path_top_y"]
        occluded = np.zeros((100, 100), np.uint8)
        held = e.update(FRAME, [], .1, class_map=occluded,
                        label_ids=LABELS)["roi"]
        self.assertAlmostEqual(held["path_top_y"], initial)
        self.assertEqual(held["ground_extent"]["reason"],
                         "held_unavailable_ground")

    def test_semantic_overlap_uses_nonwalkable_warning_not_forced_class(self):
        e = engine(config=CFG)
        mask = np.ones((100, 100), np.uint8)
        mask[47:68, 34:48] = 0
        for t in (0, .1, .2):
            result = e.update(FRAME, [detection((32, 44, 50, 71))], t,
                              class_map=mask, label_ids=LABELS)
        item = result["detections"][0]
        self.assertEqual(item["class_name"], "person")
        self.assertEqual(item["display_label"], "obstacle")
        self.assertIn("non_walkable_in_path", item["reasons"])
        self.assertIn("non-walkable area in path", result["warning_text"])

    def test_half_second_gap_keeps_path_warning_but_disables_motion(self):
        e = engine(config=CFG)
        mask = np.ones((100, 100), np.uint8)
        mask[45:70, 35:55] = 0
        for t in (0, .1, .2):
            before = e.update(FRAME, [], t, class_map=mask, label_ids=LABELS)
        self.assertEqual(before["surface"]["alert_level"], "caution")
        after = e.update(FRAME, [], .8, timestamp_valid=False,
                         class_map=mask, label_ids=LABELS)
        self.assertFalse(after["state_reset"])
        self.assertTrue(after["motion_gap"])
        self.assertEqual(after["surface"]["alert_level"], "caution")
        self.assertEqual(after["warning"]["level"], "caution")

    def test_conflicting_yolo_names_use_generic_display(self):
        memory = LabelMemory(risk_config(CFG))
        items = []
        for index, name in enumerate(("person", "person", "person", "tree_trunk")):
            item = {"class_name": name, "confidence": .9,
                    "track_id": 4, "event_id": 1, "risk_level": "caution"}
            memory.update([item], index * .1)
            items.append(deepcopy(item))
        self.assertEqual(items[2]["display_label"], "person")
        self.assertEqual(items[3]["display_label"], "obstacle")
        self.assertEqual(items[3]["label_status"], "conflicting")

    def test_new_central_bollard_beats_old_person_at_same_level(self):
        selector = WarningSelector(risk_config(CFG))
        def candidate(name, event, y, reason):
            return {"class_name": name, "display_label": name,
                    "label_status": "reliable", "risk_level": "caution",
                    "alert_level": "caution", "event_id": event,
                    "hazard_id": f"event:{event}", "warning_primary": True,
                    "reasons": [reason], "geometry": {
                        "point": [.5, y], "corridor_overlap": 1.,
                        "immediate_overlap": 0.}}
        old = candidate("person", 1, .45, "path_occupied")
        new = candidate("bollard", 97, .77, "approaching")
        chosen = selector.update([old, new], {}, [], 0)
        self.assertEqual(chosen["hazard_id"], "event:97")
        self.assertIn("bollard", chosen["text"])

    def test_unmatched_nonwalkable_region_survives_yolo_duplicate(self):
        cfg = risk_config(CFG)
        roi = {"corridor_polygon": cfg["corridor_polygon"],
               "immediate_polygon": cfg["immediate_polygon"]}
        mask = np.ones((100, 100), np.uint8)
        mask[47:68, 34:48] = 0
        mask[47:68, 58:72] = 0
        first = {"alert_level": "caution", "label_status": "reliable",
                 "detection_index": 0,
                 "geometry": {"box_norm": [.32, .44, .50, .71]}}
        surface = SurfaceRisk(cfg)
        for t in (0, .1, .2):
            result, events = surface.update(mask, LABELS, FRAME.shape, roi,
                                            t, True, True, [first])
        self.assertEqual(result["alert_level"], "caution")
        self.assertEqual(result["matched_region_count"], 1)
        self.assertEqual(result["unmatched_region_count"], 1)
        self.assertFalse(result["suppressed_duplicate"])
        self.assertTrue(any(x["source"] == "surface" for x in events))
        self.assertEqual(WarningSelector(cfg).update([], result, [], .2)["source"],
                         "surface")

    def test_unstable_class_cannot_suppress_semantic_warning(self):
        cfg = risk_config(CFG)
        roi = {"corridor_polygon": cfg["corridor_polygon"],
               "immediate_polygon": cfg["immediate_polygon"]}
        mask = np.ones((100, 100), np.uint8)
        mask[47:68, 34:48] = 0
        uncertain = {"alert_level": "caution", "label_status": "conflicting",
                     "detection_index": 0,
                     "geometry": {"box_norm": [.32, .44, .50, .71]}}
        surface = SurfaceRisk(cfg)
        for t in (0, .1, .2):
            result, _ = surface.update(mask, LABELS, FRAME.shape, roi,
                                       t, True, True, [uncertain])
        self.assertEqual(result["unmatched_region_count"], 1)
        self.assertFalse(result["suppressed_duplicate"])


if __name__ == "__main__":
    unittest.main()
