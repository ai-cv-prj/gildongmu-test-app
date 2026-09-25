"""
file_path: backend/app/services/walking/walking_storage.py

Durable, append-only walking frame records, separate from traffic storage."""
import hashlib
import json
import os
from pathlib import Path
import cv2
import numpy as np

def json_default(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value).__name__)

def write_json(path, value):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("w",encoding="utf-8") as f:
        json.dump(value,f,ensure_ascii=False,default=json_default,allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp,path)

def append_json(path, value):
    with Path(path).open("a",encoding="utf-8") as f:
        f.write(json.dumps(value,ensure_ascii=False,default=json_default,allow_nan=False)+"\n")
        f.flush()
        os.fsync(f.fileno())

def save_input(directory, frame_id, jpeg, captured_at_ms, received_ms):
    rel=f"frames/{frame_id:08d}.jpg"
    with (directory/rel).open("xb") as f:
        f.write(jpeg)
        f.flush()
        os.fsync(f.fileno())
    row={"frame_id":frame_id,"image_path":rel,"captured_at_ms":captured_at_ms,
         "received_at_ms":received_ms,"size_bytes":len(jpeg),"sha256":hashlib.sha256(jpeg).hexdigest()}
    (directory/"inputs").mkdir(exist_ok=True)
    write_json(directory/"inputs"/f"{frame_id:08d}.json",row)
    append_json(directory/"frames.jsonl",row)
    return row

def save_risk(directory, frame_id, record):
    (directory/"risk").mkdir(exist_ok=True)
    record=dict(record)
    class_map=record.pop("class_map",None)
    if class_map is not None:
        (directory/"masks").mkdir(exist_ok=True)
        rel=f"masks/{frame_id:08d}.png"
        ok,png=cv2.imencode(".png",class_map.astype(np.uint8))
        if not ok: raise OSError("lossless class-map PNG encoding failed")
        with (directory/rel).open("xb") as f:
            f.write(png.tobytes()); f.flush(); os.fsync(f.fileno())
        record["mask_path"]=rel
        record["mask_shape"]=list(class_map.shape)
    record["frame_id"]=frame_id
    write_json(directory/"risk"/f"{frame_id:08d}.json",record)
    append_json(directory/"risk.jsonl",record)
