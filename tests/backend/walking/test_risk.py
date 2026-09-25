"""
file_path: tests/backend/walking/test_risk.py

Risk integration and failure cases without model weights or GPU.
"""
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from contextlib import redirect_stdout
import cv2
import numpy as np

from backend.app.inference.walking.risk.risk_config import risk_config, tracking_config
from backend.app.inference.walking.risk.risk import RiskEngine, VideoClock
from backend.app.inference.walking.risk.risk_geometry import geometry
from backend.app.inference.walking.risk.risk_motion import MotionHistory
from backend.app.inference.walking.risk.tracking import DetectionTracker
from backend.app.inference.walking.risk.alert_policy import AlertPolicy
from backend.app.inference.walking.visualization.risk_visualization import draw_risk

FRAME = np.zeros((100,100,3),np.uint8)

def detection(box=(40,55,60,90), name="person", cid=0):
    return {"xyxy":list(box),"class_name":name,"class_id":cid,"confidence":.9}

class FixedTracker:
    status = "active"
    def __init__(self, identity=1):
        self.identity, self.resets = identity, 0
    def reset(self):
        self.resets += 1
    def attach(self, detections, frame):
        return [{**deepcopy(d),"track_id":self.identity,"detection_index":i}
                for i,d in enumerate(detections)]

def engine(identity=1, stable=True, config=None):
    guard = SimpleNamespace(update=lambda frame:stable,reset=lambda:None)
    return RiskEngine(config,tracker=FixedTracker(identity),camera_guard=guard)

class RiskTests(unittest.TestCase):
    def test_default_polygons_valid_and_bad_config_rejected(self):
        risk_config()
        for cfg in ({"overlap_threshold":float("nan")},{"enabled":True},
                    {"corridor_polygon":[[0,0],[1,1],[0,1],[1,0]]},
                    {"immediate_polygon":[[0,0],[1,0],[1,1],[0,1]]},
                    {"history_window_s":.1,"min_history_s":.2},{"typo":1}):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                risk_config(cfg)
        for cfg in ({"backend":"auto"},{"track_buffer":True},{"match_thresh":float("inf")}):
            with self.assertRaises(ValueError):
                tracking_config(cfg)

    def test_signed_expansion_and_opt_in_ttc_risk(self):
        for enabled in (False,True):
            e=engine(config={"ttc_alerts":enabled})
            for t in (0,.1,.2):
                h=20/(1-t/.8)
                result=e.update(FRAME,[detection((40,65-h,60,65))],t)
            item=result["detections"][0]
            self.assertEqual(item["motion"]["approach_state"],"approaching")
            self.assertGreater(item["motion"]["relative_expansion_per_s"],0)
            self.assertEqual(item["risk_level"],"danger" if enabled else "caution")
        e=engine()
        for t in (0,.1,.2):
            h=30/(1+t)
            result=e.update(FRAME,[detection((40,65-h,60,65))],t)
        self.assertEqual(result["detections"][0]["motion"]["approach_state"],"receding")
        self.assertIsNone(result["detections"][0]["motion"]["ttc_scale_s"])

    def test_review_overlay_keeps_input_and_excludes_traffic_risk(self):
        e=engine(None)
        frame=np.zeros((480,270,3),np.uint8)
        result=e.update(frame,[detection((100,220,160,440)),
                               detection((20,40,40,80),name="traffic_light",cid=25)],0)
        saved=frame.copy()
        shown=draw_risk(frame,result,risk_config({"review_overlay":True}))
        np.testing.assert_array_equal(frame,saved)
        self.assertEqual(shown.shape,frame.shape)
        self.assertFalse(np.array_equal(shown,frame))
        self.assertIsNone(result["detections"][1]["risk_level"])

    def test_no_id_near_obstacle_warns_on_first_frame_and_preserves_raw(self):
        raw = [detection()]
        original = deepcopy(raw)
        result = engine(None).update(FRAME,raw,0)
        item = result["detections"][0]
        self.assertEqual(raw,original)
        self.assertEqual(item["xyxy"],raw[0]["xyxy"])
        self.assertEqual(item["risk_level"],"danger")
        self.assertIsNone(item["track_id"])
        self.assertEqual(result["events"][0]["type"],"raised")

    def test_far_and_edge_objects_are_not_automatically_danger(self):
        e=engine(None)
        result=e.update(FRAME,[detection((0,20,10,45)),detection((45,20,55,40))],0)
        self.assertEqual([x["risk_level"] for x in result["detections"]],["monitor","monitor"])
        self.assertIn("edge_candidate",result["detections"][0]["reasons"])
        self.assertEqual(result["events"],[])

    def test_ground_mask_is_context_not_veto(self):
        e=engine(None)
        result=e.update(FRAME,[detection()],0)
        e.add_sidewalk_context(result,np.zeros((100,100),int),
                               {"non_walkable":0,"walkable":1,"crosswalk":2},FRAME.shape)
        item=result["detections"][0]
        self.assertEqual(item["risk_level"],"danger")
        self.assertEqual(item["sidewalk"]["walkable_fraction"],0)
        e.add_sidewalk_context(result,None,None,FRAME.shape)
        self.assertEqual(item["risk_level"],"danger")

    def test_lateral_entry_outside_roi_without_scale_expansion(self):
        e=engine()
        for t in (0,.1,.2):
            result=e.update(FRAME,[detection((2+50*t,35,12+50*t,65))],t)
        item=result["detections"][0]
        self.assertLess(item["geometry"]["corridor_overlap"],.2)
        self.assertIn("lateral_entry",item["reasons"])
        self.assertEqual(item["risk_level"],"caution")
        self.assertIsNone(item["motion"]["ttc_scale_s"])

    def test_parallel_outside_path_does_not_raise_lateral_alert(self):
        e=engine()
        for t in (0,.1,.2):
            result=e.update(FRAME,[detection((2,35+10*t,12,65+10*t))],t)
        item=result["detections"][0]
        self.assertEqual(item["risk_level"],"monitor")
        self.assertIsNone(item["motion"]["time_to_corridor_s"])

    def test_ttc_logs_only_and_clipped_box_disables_ttc(self):
        e=engine()
        for t in (0,.1,.2):
            h=20/(1-t/2)
            result=e.update(FRAME,[detection((40,65-h,60,65))],t)
        item=result["detections"][0]
        self.assertIsNotNone(item["motion"]["ttc_scale_s"])
        self.assertNotIn("short_ttc",item["reasons"])
        result=e.update(FRAME,[detection((40,70,60,100))],.3)
        self.assertIsNone(result["detections"][0]["motion"]["ttc_scale_s"])

    def test_unstable_camera_does_not_cancel_immediate_risk(self):
        result=engine(stable=False).update(FRAME,[detection()],0)
        item=result["detections"][0]
        self.assertEqual(item["risk_level"],"danger")
        self.assertEqual(item["motion"]["quality"],"unstable")

    def test_gap_reverse_time_shape_and_reset_isolate_history(self):
        e=engine()
        e.update(FRAME,[detection()],1)
        for time,frame in ((2,FRAME),(1,FRAME),(1.1,np.zeros((120,100,3),np.uint8))):
            result=e.update(frame,[detection()],time)
            self.assertTrue(result["state_reset"])
            self.assertEqual(result["events"][0]["type"],"raised")
        self.assertEqual(e.tracker.resets,4)

    def test_class_change_discards_motion_history(self):
        e=engine()
        for t in (0,.1,.2):
            e.update(FRAME,[detection()],t)
        result=e.update(FRAME,[detection(name="car",cid=3)],.3)
        self.assertEqual(result["detections"][0]["motion"]["history_s"],0)

    def test_traffic_light_has_no_risk_overlay_or_events(self):
        raw=[detection(name="traffic_light",cid=25)]
        result=engine().update(FRAME,raw,0)
        self.assertIsNone(result["detections"][0]["risk_level"])
        self.assertEqual(result["events"],[])
        rendered=draw_risk(FRAME,result,risk_config({"draw_roi":False}))
        # Upper image (where traffic status is drawn) is left unchanged.
        np.testing.assert_array_equal(rendered[:40],FRAME[:40])

    def test_alert_cooldown_escalation_tentative_identity_and_release(self):
        e=engine(None)
        result=e.update(FRAME,[detection((40,30,60,65))],0)
        self.assertEqual(result["events"][0]["level"],"caution")
        e.tracker.identity=7
        result=e.update(FRAME,[detection((40,30,60,65))],.1)
        self.assertEqual(result["events"],[])
        result=e.update(FRAME,[detection((40,50,60,85))],.2)
        self.assertEqual(result["events"][0]["type"],"escalated")
        result=e.update(FRAME,[],.4)
        self.assertEqual(result["detections"],[])
        self.assertEqual(result["events"],[])
        result=e.update(FRAME,[],.8)
        self.assertEqual(result["events"][0]["type"],"lost")
        self.assertFalse(result["events"][0]["safety_confirmed"])

    def test_invalid_pts_disables_motion(self):
        clock=VideoClock(30)
        self.assertTrue(clock.read(0,0)[1])
        self.assertFalse(clock.read(0,1)[1])
        self.assertTrue(clock.read(66.7,2)[1])
        self.assertFalse(clock.read(float("nan"),3)[1])
        e=engine()
        for t in (0,.1,.2):
            result=e.update(FRAME,[detection()],t,False)
        self.assertEqual(result["detections"][0]["motion"]["quality"],"unstable")

class TrackerTests(unittest.TestCase):
    def test_actual_bytetrack_new_lost_and_reset(self):
        tracker=DetectionTracker({"backend":"bytetrack"})
        self.assertEqual(tracker.status,"active")
        a,b=detection((10,10,20,20)),detection((60,60,80,80),cid=1,name="bicycle")
        tracker.attach([a],FRAME)
        result=tracker.attach([a,b],FRAME)
        self.assertEqual(len(result),2)
        self.assertIsNone(result[1]["track_id"])
        result=tracker.attach([a,b],FRAME)
        identity=result[1]["track_id"]
        self.assertIsNotNone(identity)
        self.assertEqual(len(tracker.attach([a],FRAME)),1)
        self.assertEqual(tracker.attach([a,b],FRAME)[1]["track_id"],identity)
        self.assertEqual(tracker.attach([],FRAME),[])
        tracker.reset()
        self.assertEqual(tracker.backend.frame_id,0)

    def test_failure_preserves_detections_and_reports_degraded(self):
        tracker=DetectionTracker({"enabled":False})
        tracker.backend=SimpleNamespace(update=Mock(side_effect=RuntimeError("failure")),reset=Mock())
        raw=[detection()]
        with self.assertWarns(RuntimeWarning):
            result=tracker.attach(raw,FRAME)
        self.assertEqual(result[0]["xyxy"],raw[0]["xyxy"])
        self.assertIsNone(result[0]["track_id"])
        self.assertEqual(tracker.status,"failed")

