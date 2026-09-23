"""Asynchronous H.264 export of every stored walking input, with capture-time pacing."""
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
import logging
import math
from pathlib import Path
import subprocess
import tempfile
import threading
import cv2
import numpy as np
from .walking_storage import write_json
from ..inference.walking_render import ensure_export_support, render_frame

log=logging.getLogger(__name__)

def read_status(directory):
    try: return json.loads((directory/"export.json").read_text())
    except (OSError,ValueError): return None

def frame_timeline(rows):
    times=[0.0]
    sources=["origin"]
    for before,after in zip(rows,rows[1:]):
        delta=(after["captured_at_ms"]-before["captured_at_ms"])/1000
        source="client_capture"
        if not math.isfinite(delta) or delta<=0 or (before["captured_at_ms"]<=0 or after["captured_at_ms"]<=0):
            delta=max(.001,(after["received_at_ms"]-before["received_at_ms"])/1000)
            source="server_receive_fallback"
        times.append(times[-1]+delta)
        sources.append(source)
    tail=(times[-1]-times[-2]) if len(times)>1 else .2
    return times,max(.001,tail),sources

def validate_video(path, expected_times):
    cap=cv2.VideoCapture(str(path))
    decoded=[]
    while True:
        ok,_=cap.read()
        if not ok:break
        decoded.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
    cap.release()
    if len(decoded)!=len(expected_times):
        raise RuntimeError(f"영상 프레임 검증 실패: {len(decoded)} / {len(expected_times)}")
    if any(abs(a-b)>.0015 for a,b in zip(decoded,expected_times)):
        raise RuntimeError("영상 촬영 시간축 검증 실패 (허용 오차 1.5ms)")
    return decoded

def recover_published_video(directory):
    path=directory/"result_visualized.mp4"
    mapping=json.loads((directory/"result_visualized.frames.json").read_text())
    expected=[f["output_time_s"] for f in mapping["frames"]]+[mapping["duration_s"]]
    decoded=validate_video(path,expected)
    return {"state":"ready","path":path.name,"frame_count":mapping["source_frame_count"],
            "encoded_frame_count":len(decoded),"duration_s":mapping["duration_s"],"size_bytes":path.stat().st_size}

def export_video(directory, font_path=""):
    encoder,_=ensure_export_support(font_path)
    rows=[json.loads(p.read_text()) for p in sorted((directory/"inputs").glob("*.json"))]
    if not rows: raise RuntimeError("저장된 원본 프레임이 없습니다")
    raw=list((directory/"frames").glob("*.jpg"))
    if len(raw)!=len(rows): raise RuntimeError("원본 프레임과 입력 기록 개수가 다릅니다. 복구 후 다시 출력하세요")
    final=directory/"result_visualized.mp4"
    if final.exists(): raise RuntimeError("기존 결과 영상을 덮어쓸 수 없습니다")
    times,tail,sources=frame_timeline(rows)
    mappings=[]
    with tempfile.TemporaryDirectory(prefix=".walking-export-",dir=directory) as tmp:
        tmp=Path(tmp)
        canvas_size=None
        lines=["ffconcat version 1.0"]
        for i,row in enumerate(rows):
            original=directory/row["image_path"]
            if hashlib.sha256(original.read_bytes()).hexdigest()!=row["sha256"]:
                raise RuntimeError(f"원본 해시 불일치: {row['frame_id']}")
            frame=cv2.imread(str(original))
            if frame is None: raise RuntimeError(f"원본 디코딩 실패: {row['image_path']}")
            rp=directory/"risk"/f"{row['frame_id']:08d}.json"
            record=json.loads(rp.read_text()) if rp.exists() else {"error":"incomplete inference"}
            mask=cv2.imread(str(directory/record["mask_path"]),cv2.IMREAD_UNCHANGED) if record.get("mask_path") else None
            if record.get("prediction") and mask is None: raise RuntimeError(f"보도 마스크 누락: {row['frame_id']}")
            visual=render_frame(frame,record,mask,font_path)
            if canvas_size is None:
                h,w=visual.shape[:2]
                canvas_size=(w+(w%2),h+(h%2))
            cw,ch=canvas_size
            h,w=visual.shape[:2]
            scale=min(cw/w,ch/h)
            nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
            padded=np.zeros((ch,cw,3),np.uint8)
            x,y=(cw-nw)//2,(ch-nh)//2
            padded[y:y+nh,x:x+nw]=cv2.resize(visual,(nw,nh))
            name=f"{i:08d}.png"
            if not cv2.imwrite(str(tmp/name),padded): raise RuntimeError("결과 프레임 저장 실패")
            duration=times[i+1]-times[i] if i+1<len(rows) else tail
            lines.extend([f"file '{name}'","option framerate 1000",f"duration {duration:.6f}"])
            mappings.append({**row,"output_index":i,"output_time_s":times[i],"duration_s":duration,"time_source":sources[i],
                             "analysis_status":"failed" if record.get("error") else "ok",
                             "source_shape":[w,h],"content_rect":[x,y,nw,nh]})
        # Endpoint duplicate carries the final frame's duration; it is explicitly mapped.
        lines.extend([f"file '{len(rows)-1:08d}.png'","option framerate 1000"])
        (tmp/"input.ffconcat").write_text("\n".join(lines)+"\n")
        command=[encoder,"-hide_banner","-loglevel","error","-nostdin","-f","concat","-safe","0",
                 "-i",str(tmp/"input.ffconcat"),"-fps_mode","vfr","-c:v","libx264","-threads","2",
                 "-preset","fast","-crf","20","-pix_fmt","yuv420p","-video_track_timescale","1000000",
                 "-movflags","+faststart",str(tmp/"video.mp4")]
        result=subprocess.run(command,capture_output=True,text=True,timeout=3600)
        if result.returncode: raise RuntimeError(result.stderr[-2000:])
        actual_times=validate_video(tmp/"video.mp4",times+[times[-1]+tail])
        decoded=len(actual_times)
        for row,pts in zip(mappings,actual_times):row["actual_output_pts_s"]=pts
        mapping={"schema_version":1,"source_frame_count":len(rows),"encoded_frame_count":decoded,
                 "duration_s":times[-1]+tail,"frames":mappings,"endpoint_duplicate_of":rows[-1]["frame_id"]}
        write_json(directory/"result_visualized.frames.json",mapping)
        # Hard-link publication refuses overwrite and is atomic on the session filesystem.
        import os
        os.link(tmp/"video.mp4",final)
    return {"state":"ready","path":final.name,"frame_count":len(rows),"encoded_frame_count":decoded,
            "duration_s":times[-1]+tail,"size_bytes":final.stat().st_size}

class WalkingExporter:
    def __init__(self,font_path=""):
        self.font_path=font_path
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix="walking-export")
        self.lock=threading.Lock()
        self.pending=set()

    def submit(self,directory):
        directory=Path(directory)
        with self.lock:
            status=read_status(directory)
            if directory in self.pending or (status and status["state"]=="ready" and (directory/"result_visualized.mp4").is_file()): return
            write_json(directory/"export.json",{"state":"pending"})
            self.pending.add(directory)
            self.pool.submit(self._run,directory)

    def _run(self,directory):
        try:
            write_json(directory/"export.json",{"state":"running"})
            result=(recover_published_video(directory) if (directory/"result_visualized.mp4").exists()
                    else export_video(directory,self.font_path))
            write_json(directory/"export.json",result)
        except Exception as exc:
            log.exception("walking export failed: %s",directory.name)
            write_json(directory/"export.json",{"state":"failed","error":str(exc)})
        finally:
            with self.lock:self.pending.discard(directory)

    def shutdown(self):
        self.pool.shutdown(wait=True)
