"""One warning meaning for the live canvas and recorded result video."""
import base64
import cv2
import numpy as np
from .base import normalize_box

LEVELS = {"monitor": 0, "caution": 1, "danger": 2}
NAMES = {"person":"보행자", "bicycle":"자전거", "car":"차량", "bus":"버스", "truck":"트럭",
         "pole":"기둥", "bollard":"볼라드", "tree_trunk":"나무", "motorcycle":"오토바이",
         "kick_scooter":"킥보드", "potted_plant":"화분", "barricade":"차단물",
         "handcart":"손수레", "cat":"고양이", "dog":"개", "stroller":"유모차", "wheelchair":"휠체어",
         "bird":"새", "bench":"벤치", "chair":"의자", "fire_hydrant":"소화전", "kiosk":"키오스크",
         "parking_meter":"주차 계량기", "utility_box":"시설물", "transit_stop":"정류장",
         "table":"탁자", "traffic_light":"신호등", "traffic_sign":"표지판", "movable_obstacle":"이동식 장애물",
         "suitcase":"여행 가방", "skateboard":"스케이트보드", "trash_bin":"쓰레기통"}
DIRECTIONS = {"left":"왼쪽", "right":"오른쪽", "front":"전방"}

def warning_summary(prediction):
    candidates = []
    for item in prediction["detections"]:
        level = item.get("alert_level", item.get("risk_level"))
        if level not in ("caution", "danger") or not item.get("warning_primary", True):
            continue
        reasons = item.get("reasons", [])
        direction = DIRECTIONS.get((item.get("geometry") or {}).get("side_direction"), "") if "side_close_candidate" in reasons else ""
        label = NAMES.get(item["class_name"], "장애물")
        if item.get("alert_status") == "held":
            detail = "이전 경고 유지 · 관측 불확실"
        elif "lateral_entry" in reasons or "relative_path_entry" in reasons:
            detail = f"{label} 경로 진입 주의"
        elif any(r in reasons for r in ("side_close_candidate","static_near_contact","near_path_occupied","large_static_candidate")):
            detail = f"{direction} {label} 근접".strip()
        elif "short_ttc" in reasons or "approaching" in reasons:
            detail = f"{label} 접근 주의"
        else:
            detail = f"{label} 진행 경로 확인"
        candidates.append((level, item.get("event_id", 0), detail))
    surface = prediction.get("surface") or {}
    if surface.get("alert_level"):
        candidates.append(("caution", 10**9, "진행 경로 확인" + (" · 관측 불확실" if surface.get("status")=="uncertain" else "")))
    for a in prediction.get("advisories", []):
        candidates.append(("caution", a.get("event_id", 0), f"{DIRECTIONS.get(a.get('direction'), '')} 근접 물체 시야 이탈".strip()))
    if not candidates:
        return "monitor", ""
    level, _, message = min(candidates, key=lambda x: (-LEVELS[x[0]], x[1], x[2]))
    return level, f"{'위험' if level == 'danger' else '주의'} · {message}"

def mask_preview(class_map, label_ids, max_side=384, alpha=.55):
    h,w = class_map.shape
    ratio=min(1.0,max_side/max(h,w))
    labels=cv2.resize(class_map.astype(np.uint8),(max(1,round(w*ratio)),max(1,round(h*ratio))),interpolation=cv2.INTER_NEAREST)
    rgba=np.zeros((*labels.shape,4),np.uint8)
    for label,color in (("walkable",(0,255,0)),("crosswalk",(180,105,255))):
        selected=labels==label_ids[label]
        rgba[selected,:3]=color
        rgba[selected,3]=round(255*alpha)
    ok,png=cv2.imencode(".png",rgba)
    if not ok: raise RuntimeError("보도 마스크 표시 인코딩 실패")
    return base64.b64encode(png.tobytes()).decode("ascii")

def make_response(prediction, shape, class_map, label_ids, metadata):
    h,w=shape[:2]
    detections=[]
    for item in prediction["detections"]:
        g,m,p=item.get("geometry") or {},item.get("motion") or {},item.get("proximity") or {}
        extra={key:item.get(key) for key in ("risk_level","alert_level","alert_status","assessment_quality","reasons","hold_reason","release_reason","event_id","warning_group_id","warning_group_size","warning_primary","sidewalk")}
        extra.update(band=p.get("band"),in_path=max(g.get("corridor_overlap",0),g.get("immediate_overlap",0))>=metadata["risk_config"]["overlap_threshold"],
                     ttc_s=m.get("ttc_scale_s"),approach_state=m.get("approach_state"),motion_quality=m.get("quality"),
                     ttc_invalid_reason=m.get("ttc_invalid_reason"),time_to_corridor_s=m.get("time_to_corridor_s"),
                     relative_expansion_per_s=m.get("relative_expansion_per_s"))
        detections.append({"class_id":item["class_id"],"class_name":item["class_name"],"confidence":item["confidence"],
                           "box":normalize_box(*item["xyxy"],w,h),"track_id":item.get("track_id"),"extra":extra})
    level,message=warning_summary(prediction)
    counts={k:0 for k in LEVELS}
    for d in prediction["detections"]:
        lev=d.get("alert_level",d.get("risk_level"))
        if lev in counts and (lev=="monitor" or d.get("warning_primary",True)):counts[lev]+=1
    counts["surface"]=int(bool(prediction["surface"].get("alert_level")))
    counts["advisories"]=len(prediction["advisories"])
    counts["tracked"]=sum(d["track_id"] is not None for d in detections)
    event={key:prediction.get(key) for key in ("roi","surface","advisories","warning_groups","timestamp_s","timestamp_valid",
             "state_epoch","state_reset","tracker_status","camera_motion_stable","reset_reason","timestamp_source","frame_gap_s")}
    event.update(type="walking_warning",risk_schema_version=1,warning=bool(message),warning_text=message,level=level,
                 counts=counts,detected_count=len(detections),in_path_count=sum(d["extra"]["in_path"] for d in detections),
                 risk_events=prediction["events"],config_sha256=metadata["config_sha256"],
                 source_revision=metadata["source_revision"],image_width=w,image_height=h,
                 mask_png=mask_preview(class_map,label_ids))
    return {"detections":detections,"event":event}
