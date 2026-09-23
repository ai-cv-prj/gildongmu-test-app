import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.app.main import create_app
from backend.app.config import Settings
from backend.app.inference.walking_clock import FrameClock
from backend.app.inference.walking_response import danger_voice_target, make_response, warning_summary
from backend.app.inference.risk.risk_config import risk_config
from backend.app.services.walking_export import frame_timeline, export_video, read_status
from test_risk import engine, detection

LABELS={"non_walkable":0,"walkable":1,"crosswalk":2}

class FakeRisk:
    risk_enabled=True
    yolo_config={"conf":.25}
    metadata={"risk_config":risk_config({"review_overlay":True}),"label_ids":LABELS,"config_sha256":"test","source_revision":"test"}
    def __init__(self,spec):
        self.spec=spec
        self.states={}
        self.calls=0
        self.fail=False
    def reset_session(self,sid):self.states[sid]=engine()
    def close_session(self,sid):self.states.pop(sid,None)
    def infer(self,frame,ctx):
        self.calls+=1
        if self.fail:raise RuntimeError("test model failure")
        p=self.states[ctx.session_id].update(frame,[detection((40,55,80,150))],ctx.captured_at_ms/1000)
        mask=np.ones(frame.shape[:2],np.uint8)
        result=make_response(p,frame.shape,mask,LABELS,self.metadata)
        p.update(warning_text=result["event"]["warning_text"],level=result["event"]["level"])
        result["_walking_record"]={"prediction":p,"class_map":mask,"settings":self.metadata}
        return result

@pytest.fixture
def walking(tmp_path):
    cfg=Settings(data_dir=tmp_path/'data',model_dir=tmp_path/'models',save_frames=False,min_free_disk_gb=0,walking_export_enabled=False)
    with TestClient(create_app(cfg)) as c:
        svc=c.app.state.service
        spec=svc.registry.get_spec('walking-mock-v1')
        pipeline=FakeRisk(spec)
        svc.registry.get=lambda model_id:pipeline
        r=c.post('/api/sessions',json={"mode":"walking","model_id":spec.id,"device_type":"test"})
        assert r.status_code==201,r.text
        sid=r.json()["session_id"]
        yield c,svc,pipeline,sid,Path(r.json()["storage_path"])

def jpeg():
    return cv2.imencode('.jpg',np.zeros((200,120,3),np.uint8))[1].tobytes()

def upload(c,sid,i,data=None,timestamp=None):
    return c.post(f'/api/sessions/{sid}/frames',files={"image":("frame.jpg",data or jpeg(),"image/jpeg")},
                  data={"frame_id":i,"captured_at_ms":timestamp if timestamp is not None else 1000+i*200})

def test_clock_no_nominal_fps_and_discontinuities():
    c=FrameClock(.8)
    assert c.read(1000).valid
    second=c.read(1371)
    assert second.timestamp_s==pytest.approx(.371)
    assert not c.read(1371).valid
    assert c.read(1400).valid
    assert c.read(2600).reset_reason=="capture_gap"
    assert not c.read(None).valid

def test_input_is_saved_before_failure_and_duplicate_is_idempotent(walking):
    c,svc,p,sid,d=walking
    p.fail=True
    assert upload(c,sid,1).status_code==500
    assert (d/'frames/00000001.jpg').read_bytes()==jpeg()
    assert json.loads((d/'risk/00000001.json').read_text())["error"]
    assert upload(c,sid,1).status_code==500
    assert p.calls==1
    p.fail=False
    assert upload(c,sid,2).status_code==200
    assert upload(c,sid,2).status_code==200
    assert p.calls==2
    assert upload(c,sid,2,timestamp=10001).status_code==409
    assert len(list((d/'frames').glob('*.jpg')))==2
    assert c.get(f'/api/sessions/{sid}').json()["raw_frame_count"]==2

def test_risk_masks_and_video_keep_all_inputs(walking):
    c,svc,p,sid,d=walking
    a=upload(c,sid,1)
    assert a.status_code==200,a.text
    assert svc._active.confidence==.25
    p.fail=True
    assert upload(c,sid,2).status_code==500
    p.fail=False
    assert upload(c,sid,3).status_code==200
    c.post(f'/api/sessions/{sid}/stop')
    result=export_video(d)
    assert result["frame_count"]==3
    mapping=json.loads((d/'result_visualized.frames.json').read_text())
    assert [x['frame_id'] for x in mapping['frames']]==[1,2,3]
    assert mapping['frames'][1]['analysis_status']=='failed'
    assert [x['output_time_s'] for x in mapping['frames']]==pytest.approx([0,.2,.4])
    assert mapping['duration_s']==pytest.approx(.6)
    assert (d/'frames/00000001.jpg').read_bytes()==jpeg()
    with pytest.raises(RuntimeError,match='덮어쓸'):export_video(d)

def test_order_rejection_and_timestamp_reset(walking):
    c,svc,p,sid,d=walking
    assert upload(c,sid,2).status_code==200
    assert upload(c,sid,1).status_code==409
    assert not (d/'frames/00000001.jpg').exists()

def test_timeline_retains_large_gaps_and_falls_back_only_invalid():
    rows=[dict(captured_at_ms=t,received_at_ms=r) for t,r in [(1000,5000),(1300,5400),(6300,10300),(6200,11000)]]
    times,tail,sources=frame_timeline(rows)
    assert times==pytest.approx([0,.3,5.3,6.0])
    assert sources[-1]=='server_receive_fallback'

def test_held_warning_does_not_claim_new_motion():
    p=engine().update(np.zeros((100,100,3),np.uint8),[detection()],0)
    item=p["detections"][0]
    item.update(alert_level="danger",alert_status="held")
    assert '이전 경고 유지' in warning_summary(p)[1]

def test_danger_voice_target_groups_person_vehicle_and_obstacle():
    base={"alert_level":"danger","warning_primary":True,"event_id":7}
    for class_name,category in [("person","person"),("bus","vehicle"),("bollard","obstacle")]:
        assert danger_voice_target({"detections":[{**base,"class_name":class_name}]}) == {
            "category":category,"event_id":7}

def test_danger_voice_target_ignores_caution_and_uses_track_id_fallback():
    assert danger_voice_target({"detections":[
        {"alert_level":"caution","warning_primary":True,"event_id":1,"class_name":"person"},
        {"alert_level":"danger","warning_primary":True,"event_id":None,"track_id":9,"class_name":"car"},
    ]}) == {"category":"vehicle","event_id":9}
