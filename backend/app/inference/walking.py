"""
file_path: backend/app/inference/walking.py

32클래스 YOLO로 휴대폰 프레임의 도보 장애물을 검출한다.
같은 프레임의 보도 마스크와 integration 위험 엔진으로 경고를 판단한다.

가중치: backend/models/walking/finetune_v2_exp02_stage2_best.pt (gildongmu exp02 stage2)
학습 설정: imgsz 640, NMS head
실행 패키지: torch==2.14.0+cu130, torchvision==0.29.0+cu130, ultralytics==8.4.152
(앱의 .venv에 설치, backend/requirements.txt에는 넣지 않는다)
"""
from __future__ import annotations

import base64
import logging
from typing import Any

import cv2
import numpy as np

from .base import InferenceContext, ModelSpec, normalize_box

log = logging.getLogger(__name__)
LABEL_COLORS = {
    "walkable": (0, 255, 0),
    "crosswalk": (180, 105, 255),  # 참고 파일과 동일한 OpenCV BGR 순서
}

# gildongmu configs/schema/classes.yaml 의 canonical 순서와 같아야 한다.
CLASS_NAMES: dict[int, str] = {
    0: "person", 1: "bicycle", 2: "bus", 3: "car", 4: "handcart", 5: "cat", 6: "dog",
    7: "motorcycle", 8: "kick_scooter", 9: "stroller", 10: "truck", 11: "wheelchair",
    12: "bird", 13: "barricade", 14: "bench", 15: "bollard", 16: "chair",
    17: "fire_hydrant", 18: "kiosk", 19: "parking_meter", 20: "pole",
    21: "potted_plant", 22: "utility_box", 23: "transit_stop", 24: "table",
    25: "traffic_light", 26: "traffic_sign", 27: "tree_trunk", 28: "movable_obstacle",
    29: "suitcase", 30: "skateboard", 31: "trash_bin",
}
IMAGE_SIZE = 640
# 임시 경고 규칙: 박스가 화면 가로 중앙 40% 구간과 겹치고 아래쪽 끝이 화면 55% 아래에 있으면 경고
PATH_X_RANGE = (0.30, 0.70)
PATH_MIN_BOTTOM = 0.55


# 진행 경로에 걸친 검출인지 판단
def in_walking_path(box: dict[str, float]) -> bool:
    """정규화 박스가 화면 하단 중앙의 진행 경로 구간과 겹치는지 반환한다."""
    return box["x2"] >= PATH_X_RANGE[0] and box["x1"] <= PATH_X_RANGE[1] and box["y2"] >= PATH_MIN_BOTTOM


# 보행가능 영역과 횡단보도를 휴대폰 표시용 데이터로 변환
def make_segmentation_event(class_map: np.ndarray, label_ids: dict[str, int]) -> dict[str, Any]:
    """픽셀을 바꾸지 않는 RLE 마스크를 만들고 복잡한 마스크만 PNG로 반환한다."""
    if class_map.ndim != 2 or not class_map.size:
        raise ValueError("마스크는 비어 있지 않은 2차원 배열이어야 합니다.")
    # 전송용 번호는 모델 라벨 순서와 무관하게 투명=0, 초록=1, 핑크=2다.
    labels = np.zeros(class_map.shape, dtype=np.uint8)
    ratios = {}
    for code, name in enumerate(LABEL_COLORS, start=1):
        mask = class_map == label_ids[name]
        labels[mask] = code
        ratios[f"{name}_ratio"] = float(mask.mean())
    event = {
        "type": "walking_warning",
        "warning": False,  # 영역 분할만 수행하며 장애물 위험 여부는 판단하지 않는다.
        "warning_text": "",
        **ratios,
    }
    flat = labels.reshape(-1)
    starts = np.r_[0, np.flatnonzero(flat[1:] != flat[:-1]) + 1]
    # 같은 색이 이어지는 길이와 색 번호를 little-endian uint32 한 개에 담는다.
    # 지나치게 많은 구간은 브라우저 루프·응답 크기를 늘리므로 PNG로 보낸다.
    if starts.size <= 4096 and flat.size <= 4194304:
        lengths = np.diff(np.r_[starts, flat.size]).astype(np.uint32)
        runs = ((lengths << 2) | flat[starts]).astype("<u4")
        event["mask_rle"] = {
            "width": int(labels.shape[1]), "height": int(labels.shape[0]),
            "data": base64.b64encode(runs.tobytes()).decode("ascii"),
        }
    else:
        palette = np.array([(0, 0, 0, 0), *[(*color, 140) for color in LABEL_COLORS.values()]], dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", palette[labels])
        if not ok:
            raise RuntimeError("보행가능·횡단보도 영역 PNG를 생성하지 못했습니다.")
        event["mask_png"] = base64.b64encode(encoded.tobytes()).decode("ascii")
    return event


class WalkingPipeline:
    """One detector and segmenter; separate risk state for every session."""
    mode = "walking"

    def __init__(self, spec: ModelSpec, settings=None) -> None:
        if settings is None:
            from ..config import Settings
            settings = Settings()
        self.spec, self.weights = spec, spec.weights
        self.settings = settings
        self.risk_enabled = self.settings.walking_risk_enabled
        self.model = self.segmenter = None
        self.device = "cpu"
        self.sessions = {}
        self.metadata = {}

    def load(self) -> None:
        import torch
        from ultralytics import YOLO
        if self.weights is None or self.weights.suffix.lower() != ".pt":
            raise ValueError("도보 장애물 파이프라인은 YOLO .pt 가중치만 지원합니다")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = YOLO(str(self.weights), task="detect")
        if model.task != "detect" or {int(k):str(v) for k,v in model.names.items()} != CLASS_NAMES:
            raise ValueError("YOLO 가중치가 32클래스 canonical 순서와 다릅니다")
        self.yolo_config = {"conf":.25,"imgsz":640,"head":"nms","iou":.7,"max_det":300,"rect":True}
        if self.risk_enabled:
            import hashlib
            import json
            import yaml
            from .risk.risk_config import risk_config, tracking_config
            from .risk.risk import RiskEngine
            from .walking_sidewalk import SidewalkSegmenter
            from .walking_render import ensure_export_support
            raw=self.settings.walking_risk_config.read_bytes()
            cfg=yaml.safe_load(raw)
            if not isinstance(cfg,dict) or not all(k in cfg for k in ("risk","tracking","yolo")):
                raise ValueError("walking 설정에 risk/tracking/yolo가 필요합니다")
            self.risk_config=risk_config(cfg["risk"])
            self.tracking_config=tracking_config(cfg["tracking"])
            self.yolo_config=cfg["yolo"]
            yc=self.yolo_config
            if (yc.get("head")!="nms" or not 0<=yc.get("conf",-1)<=1 or
                    not isinstance(yc.get("imgsz"),int) or yc["imgsz"]<=0):
                raise ValueError("walking YOLO conf/imgsz/head 설정 오류")
            _, font_path = ensure_export_support(self.settings.walking_font_path)
            self.segmenter=SidewalkSegmenter(self.settings.walking_mask_weights,self.device,self.settings.walking_precision)
            probe=RiskEngine(self.risk_config,self.tracking_config)
            if self.tracking_config["enabled"] and probe.tracker.status!="active":
                raise RuntimeError("walking 추적기를 초기화하지 못했습니다. requirements-walking.txt를 확인하세요")
            snapshot={"risk_config":self.risk_config,"tracking_config":self.tracking_config,
                      "yolo_config":yc,"label_ids":self.segmenter.label_ids,
                      "precision":self.settings.walking_precision,"overlay_alpha":.55,
                      "render":{"font_path":font_path,"font_size_width_divisor":27,"warning_position":"top_left"},
                      "mask_processor":self.segmenter.processor.to_dict(),
                      "device":self.device,"yolo_precision":"fp32",
                      "source_revision":"81c4d640c3b673e487d86a1dabdf1d8b03c54572"}
            import importlib.metadata
            snapshot["package_versions"]={name:importlib.metadata.version(name) for name in
                ("torch","ultralytics","transformers","numpy","scipy","lap","Pillow","imageio-ffmpeg")}
            snapshot["config_sha256"]=hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest()
            snapshot["mask_weights_sha256"]=hashlib.sha256((self.settings.walking_mask_weights/"model.safetensors").read_bytes()).hexdigest()
            self.metadata=snapshot
        self.model=model
        log.info("walking loaded: risk=%s device=%s",self.risk_enabled,self.device)

    def reset_session(self, session_id: str) -> None:
        if self.risk_enabled:
            from .risk.risk import RiskEngine
            from .walking_clock import FrameClock
            self.sessions[session_id]={"engine":RiskEngine(self.risk_config,self.tracking_config),
                "clock":FrameClock(self.risk_config["reset_gap_s"], self.risk_config["hard_reset_gap_s"]),"last_frame":0,"count":0}

    def close_session(self, session_id: str) -> None:
        self.sessions.pop(session_id,None)

    def infer(self, frame_bgr: np.ndarray, context: InferenceContext) -> dict[str, Any]:
        import time
        if self.model is None:
            raise RuntimeError("load()로 모델을 먼저 불러와야 합니다")
        h,w=frame_bgr.shape[:2]
        state=self.sessions.get(context.session_id)
        if self.risk_enabled and (state is None or context.frame_id<=state["last_frame"]):
            raise ValueError("walking 세션 또는 프레임 순서가 올바르지 않습니다")
        started=time.perf_counter()
        cfg=self.yolo_config
        result=self.model.predict(source=frame_bgr,imgsz=cfg["imgsz"],conf=context.confidence,
            device=self.device,nms=None,iou=cfg["iou"],max_det=cfg["max_det"],rect=cfg["rect"],
            verbose=False,save=False,save_txt=False,save_crop=False,stream=False)[0]
        if tuple(result.orig_shape)!=frame_bgr.shape[:2]:
            raise ValueError("YOLO 결과와 원본 프레임 크기가 다릅니다")
        raw=[]
        if result.boxes is not None:
            boxes=result.boxes.cpu()
            raw=[{"xyxy":xyxy,"class_id":int(cid),"class_name":CLASS_NAMES[int(cid)],"confidence":float(score)}
                 for xyxy,score,cid in zip(boxes.xyxy.tolist(),boxes.conf.tolist(),boxes.cls.tolist())]
        detected=time.perf_counter()
        if not self.risk_enabled:
            detections=[{**{k:d[k] for k in ("class_id","class_name","confidence")},
                "box":normalize_box(*d["xyxy"],w,h),"track_id":None} for d in raw]
            for d in detections:d["extra"]={"in_path":in_walking_path(d["box"])}
            in_path=[d for d in detections if d["extra"]["in_path"]]
            nearest=max(in_path,key=lambda d:d["box"]["y2"],default=None)
            return {"detections":detections,"event":{"type":"walking_warning","warning":nearest is not None,
                "warning_text":f"전방 {nearest['class_name']}" if nearest else "",
                "detected_count":len(detections),"in_path_count":len(in_path)}}
        from .walking_response import make_response
        class_map=self.segmenter.predict(frame_bgr)
        segmented=time.perf_counter()
        reading=state["clock"].read(context.captured_at_ms)
        if reading.reset_reason:state["engine"].reset()
        prediction=state["engine"].update(frame_bgr,raw,reading.timestamp_s,reading.valid,
                                          class_map,self.segmenter.label_ids)
        state["engine"].add_sidewalk_context(prediction,class_map,self.segmenter.label_ids,frame_bgr.shape)
        prediction.update(reset_reason=reading.reset_reason,timestamp_source="client_capture",
                          frame_gap_s=reading.gap_s,state_reset=prediction["state_reset"] or bool(reading.reset_reason))
        assessed=time.perf_counter()
        response=make_response(prediction,frame_bgr.shape,class_map,self.segmenter.label_ids,self.metadata)
        response["event"]["stage_timing"]={"yolo_ms":(detected-started)*1000,
            "sidewalk_ms":(segmented-detected)*1000,"risk_ms":(assessed-segmented)*1000}
        if state["count"]==0:
            response["event"]["settings_snapshot"]={**self.metadata,"actual_confidence":context.confidence,
                "predictor_options":{key:getattr(self.model.predictor.args,key,None) for key in
                    ("conf","iou","max_det","rect","imgsz","nms","quantize")}}
        prediction.update(warning_text=response["event"]["warning_text"],level=response["event"]["level"])
        response["_walking_record"]={"prediction":prediction,"class_map":class_map,
            "label_ids":self.segmenter.label_ids,"settings":self.metadata}
        state["last_frame"]=context.frame_id
        state["count"]+=1
        return response
