"""One warning meaning for the live canvas and recorded result video."""
from .base import normalize_box
from .walking import make_segmentation_event

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
    selected = prediction.get("warning") or {}
    level = selected.get("level", "monitor")
    if level not in ("caution", "danger"):
        return "monitor", ""
    source = selected.get("source")
    if source == "camera_view":
        camera_view = prediction.get("camera_view") or {}
        if "previous_hazard_unverified" in (selected.get("reasons") or []):
            detail = "이전 위험 확인 불가 · 카메라를 전방으로 들어 주세요"
        elif camera_view.get("status") == "unavailable":
            detail = "촬영 불가 · 카메라를 전방으로 들어 주세요"
        else:
            detail = "카메라 흔들림 · 화면을 안정적으로 촬영해 주세요"
    elif source == "surface":
        surface = prediction.get("surface") or {}
        detail = ("진행 경로 확인 · 관측 불확실" if surface.get("status") == "uncertain"
                  else "진행 경로의 비보행 영역 확인")
    elif source == "advisory":
        hazard_id = selected.get("hazard_id")
        advisory = next((a for a in prediction.get("advisories", [])
                         if f"advisory:{a.get('event_id')}" == hazard_id), {})
        direction = DIRECTIONS.get(advisory.get("direction"), "")
        detail = f"{direction} 근접 물체 시야 이탈".strip()
    else:
        index = selected.get("detection_index")
        item = next((d for d in prediction.get("detections", [])
                     if d.get("detection_index") == index), {})
        reasons = set(item.get("reasons") or selected.get("reasons") or [])
        if source == "surface_object" or item.get("semantic_path_overlap"):
            detail = "진행 경로의 비보행 영역 확인"
        else:
            label = (NAMES.get(item.get("display_label"), "장애물")
                     if item.get("label_status") == "reliable" else "장애물")
            direction = (DIRECTIONS.get((item.get("geometry") or {}).get("side_direction"), "")
                         if "side_close_candidate" in reasons else "")
            if item.get("alert_status") == "held":
                detail = "이전 경고 유지 · 관측 불확실"
            elif "lateral_entry" in reasons or "relative_path_entry" in reasons:
                detail = f"{label} 경로 진입 주의"
            elif reasons.intersection(("side_close_candidate", "static_near_contact",
                                       "near_path_occupied", "large_static_candidate")):
                detail = f"{direction} {label} 근접".strip()
            elif "short_ttc" in reasons or "approaching" in reasons:
                detail = f"{label} 접근 주의"
            else:
                detail = f"{label} 진행 경로 확인"
    return level, f"{'위험' if level == 'danger' else '주의'} · {detail}"

def make_response(prediction, shape, class_map, label_ids, metadata):
    h,w=shape[:2]
    detections=[]
    for item in prediction["detections"]:
        g,m,p=item.get("geometry") or {},item.get("motion") or {},item.get("proximity") or {}
        extra={key:item.get(key) for key in ("risk_level","untrusted_risk_level","alert_level","alert_status","assessment_quality","reasons","hold_reason","release_reason","event_id","warning_group_id","warning_group_size","warning_primary","sidewalk","display_label","label_status","hazard_id","semantic_path_overlap")}
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
    counts["surface"]=int(bool(prediction["surface"].get("alert_level")) and not prediction["surface"].get("suppressed_duplicate"))
    counts["advisories"]=len(prediction["advisories"])
    counts["tracked"]=sum(d["track_id"] is not None for d in detections)
    counts["camera_view"]=int((prediction.get("camera_view") or {}).get("status") in ("uncertain","unavailable"))
    event={key:prediction.get(key) for key in ("roi","surface","advisories","warning_groups","motion_gap","timestamp_s","timestamp_valid",
             "state_epoch","state_reset","tracker_status","camera_motion_stable","camera_view","view_recovered",
             "reset_reason","timestamp_source","frame_gap_s")}
    event["selected_warning"] = prediction.get("warning")
    event.update(type="walking_warning",risk_schema_version=1,warning=bool(message),warning_text=message,level=level,
                 counts=counts,detected_count=len(detections),in_path_count=sum(d["extra"]["in_path"] for d in detections),
                 risk_events=prediction["events"],config_sha256=metadata["config_sha256"],
                 source_revision=metadata["source_revision"],image_width=w,image_height=h)
    # 마스크 전송은 기존 보행 영역 기능의 RLE 경로를 그대로 쓴다. 위험 필드가 우선한다.
    segmentation = make_segmentation_event(class_map,label_ids)
    for key in ("walkable_ratio","crosswalk_ratio","mask_rle","mask_png"):
        if key in segmentation:
            event.setdefault(key,segmentation[key])
    return {"detections":detections,"event":event}
