"""Walking-only ingress: persist input before inference and reject replay mutations."""
import hashlib
import json
import time
import cv2
import numpy as np
from ..inference.base import InferenceContext
from .storage_service import StorageError
from .walking_storage import save_input, save_risk, write_json

def process_walking_frame(svc,active,jpeg,frame_id,captured_at_ms,client_sent_at_ms,received,t0):
    from .session_service import SessionError, iso
    directory=svc.storage.session_dir(active.id)
    with active.lock:
        if svc._active is not active:
            raise SessionError(409,"session_not_running","세션이 종료되었습니다")
        if frame_id<1:
            raise SessionError(400,"invalid_frame_id","frame_id는 1 이상이어야 합니다")
        digest=hashlib.sha256(jpeg).hexdigest()
        ip=directory/"inputs"/f"{frame_id:08d}.json"
        op=directory/"outcomes"/f"{frame_id:08d}.json"
        if ip.exists():
            previous=json.loads(ip.read_text())
            if previous["sha256"]!=digest or previous["captured_at_ms"]!=captured_at_ms:
                raise SessionError(409,"frame_conflict","같은 frame_id의 원본 또는 촬영 시각이 다릅니다")
            if op.exists():
                outcome=json.loads(op.read_text())
                if "error" in outcome:
                    e=outcome["error"]
                    raise SessionError(e["status"],e["code"],e["message"])
                return outcome["response"]
            raise SessionError(409,"frame_incomplete","원본은 저장됐지만 처리가 완료되지 않은 프레임입니다")
        if frame_id<=active.walking_last_frame:
            raise SessionError(409,"frame_out_of_order","이전 프레임 번호는 재사용할 수 없습니다")
        frame=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
        if not jpeg.startswith(b"\xff\xd8") or frame is None or frame.size==0:
            svc._record_error(active,frame_id,captured_at_ms,received,"invalid_image","유효한 JPEG가 아닙니다")
            raise SessionError(400,"invalid_image","유효한 JPEG가 아닙니다")
        decoded=time.perf_counter()
        try:
            row=save_input(directory,frame_id,jpeg,captured_at_ms,received.timestamp()*1000)
        except OSError as exc:
            active.error_count += 1
            active.last_error = f"원본 저장 실패: {exc}"
            try: svc._flush_manifest(active)
            except StorageError: pass
            raise SessionError(507,"storage_failed",f"원본 저장 실패: {exc}") from exc
        active.walking_last_frame=frame_id
        active.manifest["raw_frame_count"]=active.manifest.get("raw_frame_count",0)+1
        saved_input=time.perf_counter()
        (directory/"outcomes").mkdir(exist_ok=True)
        try:
            ctx=InferenceContext(session_id=active.id,frame_id=frame_id,captured_at_ms=captured_at_ms,confidence=active.confidence)
            with svc.registry.gpu_lock:
                result=active.pipeline.infer(frame,ctx)
            inferred=time.perf_counter()
            record=result.pop("_walking_record")
            save_risk(directory,frame_id,record)
            saved=time.perf_counter()
            timing={"decode_ms":round((decoded-t0)*1000,2),"inference_ms":round((inferred-saved_input)*1000,2),
                    "save_ms":round(((saved_input-decoded)+(saved-inferred))*1000,2),"server_ms":round((saved-t0)*1000,2)}
            response={"session_id":active.id,"frame_id":frame_id,"captured_at_ms":captured_at_ms,
                      "server_received_at":iso(received),"image_width":frame.shape[1],"image_height":frame.shape[0],
                      "detections":result["detections"],"event":result["event"],"timing":timing,"saved":True}
            public={**response,"image_path":row["image_path"],"model_id":active.model_id,
                    "confidence_threshold":active.confidence,"client_sent_at_ms":client_sent_at_ms,"error":None}
            svc.storage.append_result(active.id,public)
            write_json(op,{"response":response})
        except Exception as exc:
            storage_error=isinstance(exc,(OSError,StorageError))
            status,code=(507,"storage_failed") if storage_error else (500,"inference_failed")
            message=str(exc)
            active.error_count+=1
            active.last_error=f"{code}: {message}"
            # Failed inference must not seed the next frame's motion/tracker history.
            active.pipeline.reset_session(active.id)
            failure={"code":code,"message":message,"status":status}
            try:
                if not (directory/"risk"/f"{frame_id:08d}.json").exists():
                    save_risk(directory,frame_id,{"error":failure})
                svc.storage.append_result(active.id,{"session_id":active.id,**row,"saved":True,"error":failure,
                    "detections":[],"event":{},"timing":None,"model_id":active.model_id})
                write_json(op,{"error":failure})
                svc._flush_manifest(active)
            except (OSError,StorageError):
                pass  # The durable raw JPEG and input record remain available for export/recovery.
            raise SessionError(status,code,f"처리 실패 (원본 보존됨): {message}") from exc
        active.frame_count+=1
        active.inference_ms.append(timing["inference_ms"])
        active.server_ms.append(timing["server_ms"])
        if active.frame_count%25==0:svc._flush_manifest(active)
        return response
