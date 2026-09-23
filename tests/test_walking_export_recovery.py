import json
import time
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from backend.app.services.walking_export import WalkingExporter,read_status,export_video,validate_video
from backend.app.services.walking_storage import save_input,save_risk,write_json
from backend.app.services.session_service import SessionService
from test_walking_app import walking,upload,jpeg

def test_rotation_and_failed_frame_still_preserve_input_timing(walking):
    c,svc,p,sid,d=walking
    assert upload(c,sid,1).status_code==200
    landscape=cv2.imencode('.jpg',np.zeros((120,200,3),np.uint8))[1].tobytes()
    assert upload(c,sid,2,landscape,timestamp=1743).status_code==200
    c.post(f'/api/sessions/{sid}/stop')
    result=export_video(d)
    assert result['frame_count']==2
    m=json.loads((d/'result_visualized.frames.json').read_text())
    assert abs(m['frames'][1]['actual_output_pts_s']-.543)<.0015
    rect=m['frames'][1]['content_rect']
    assert rect[1]>0 and rect[2]==120  # portrait output, centered landscape content
    assert (d/'frames/00000002.jpg').read_bytes()==landscape

def test_encoder_failure_retry_and_published_output_recovery(walking):
    c,svc,p,sid,d=walking
    assert upload(c,sid,1).status_code==200
    c.post(f'/api/sessions/{sid}/stop')
    with patch('backend.app.services.walking_export.export_video',side_effect=RuntimeError('encoder unavailable')):
        e=WalkingExporter();e.submit(d);e.shutdown()
    assert read_status(d)['state']=='failed'
    e=WalkingExporter();e.submit(d);e.shutdown()
    assert read_status(d)['state']=='ready'
    before=(d/'result_visualized.mp4').read_bytes()
    write_json(d/'export.json',{'state':'running'})
    e=WalkingExporter();e.submit(d);e.shutdown()
    assert read_status(d)['state']=='ready'
    assert (d/'result_visualized.mp4').read_bytes()==before

def test_stale_session_recovery_counts_raw_inputs(walking):
    c,svc,p,sid,d=walking
    assert upload(c,sid,1).status_code==200
    p.fail=True
    assert upload(c,sid,2).status_code==500
    replacement=SessionService(svc.settings,svc.storage,svc.registry)
    assert replacement.recover_on_startup()==1
    row=replacement.get(sid)
    assert row['status']=='aborted'
    assert row['raw_frame_count']==2
    assert row['frame_count']==1 and row['error_count']==1

def test_raw_storage_failure_never_runs_inference(walking):
    c,svc,p,sid,d=walking
    with patch('backend.app.services.walking_frames.save_input',side_effect=OSError('disk full')):
        r=upload(c,sid,1)
    assert r.status_code==507
    assert p.calls==0
    assert svc._active.error_count==1
    assert not (d/'frames/00000001.jpg').exists()

def test_new_session_discards_tracking_state(walking):
    c,svc,p,sid,d=walking
    assert upload(c,sid,1).status_code==200
    first=p.states[sid]
    c.post(f'/api/sessions/{sid}/stop')
    assert sid not in p.states
    r=c.post('/api/sessions',json={'mode':'walking','model_id':p.spec.id,'device_type':'second'})
    next_id=r.json()['session_id']
    assert p.states[next_id] is not first
    assert upload(c,next_id,1).status_code==200

def test_modified_original_refuses_export(walking):
    import pytest
    c,svc,p,sid,d=walking
    assert upload(c,sid,1).status_code==200
    c.post(f'/api/sessions/{sid}/stop')
    (d/'frames/00000001.jpg').write_bytes(jpeg()+b'tampered')
    with pytest.raises(RuntimeError,match='해시'):export_video(d)
