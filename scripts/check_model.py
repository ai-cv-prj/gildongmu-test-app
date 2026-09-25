"""휴대폰 없이 이미지 한 장으로 본인 파이프라인을 점검한다.

사용:
  .venv/bin/python scripts/check_model.py --mode traffic --image 사진.jpg
  .venv/bin/python scripts/check_model.py --mode traffic --image 사진.jpg --model traffic-best-v2
  .venv/bin/python scripts/check_model.py --list

하는 일: 가중치 찾기 → load() → infer() 5회 → 반환 형식 검사 → 박스를 그린 이미지를 check_output/ 에 저장
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.config import Settings  # noqa: E402
from backend.app.inference.base import InferenceContext  # noqa: E402
from backend.app.inference.registry import PipelineLoadError, PipelineRegistry  # noqa: E402

EVENT_TYPES = {"traffic": "traffic_signal", "walking": "walking_warning", "bus": "bus_detection"}


def validate(result: object, mode: str) -> list[str]:
    errs: list[str] = []
    if not isinstance(result, dict):
        return [f"infer() 는 dict 를 반환해야 합니다 (현재 {type(result).__name__})"]
    dets = result.get("detections")
    if not isinstance(dets, list):
        errs.append('"detections" 는 list 여야 합니다')
        dets = []
    for i, d in enumerate(dets):
        for key, typ in (("class_id", int), ("class_name", str), ("confidence", (int, float)), ("box", dict)):
            if not isinstance(d.get(key), typ):
                errs.append(f"detections[{i}].{key} 가 없거나 타입이 다릅니다: {d.get(key)!r}")
        box = d.get("box") if isinstance(d.get("box"), dict) else {}
        vals = [box.get(k) for k in ("x1", "y1", "x2", "y2")]
        if any(not isinstance(v, (int, float)) for v in vals):
            errs.append(f"detections[{i}].box 에 x1,y1,x2,y2 숫자가 필요합니다: {box}")
        elif not all(0.0 <= float(v) <= 1.0 for v in vals):
            errs.append(f"detections[{i}].box 가 0~1 범위를 벗어났습니다 (픽셀 좌표면 normalize_box 사용): {box}")
        elif vals[0] > vals[2] or vals[1] > vals[3]:
            errs.append(f"detections[{i}].box 는 x1<=x2, y1<=y2 여야 합니다: {box}")
        if isinstance(d.get("confidence"), (int, float)) and not 0 <= d["confidence"] <= 1:
            errs.append(f"detections[{i}].confidence 는 0~1 이어야 합니다: {d['confidence']}")
    ev = result.get("event")
    if not isinstance(ev, dict) or ev.get("type") != EVENT_TYPES[mode]:
        errs.append(f'"event" 는 {{"type": "{EVENT_TYPES[mode]}", ...}} 형식이어야 합니다: {ev!r}')
    elif mode == "traffic" and ev.get("signal_state") not in {"red", "green", "unknown"}:
        errs.append(f"event.signal_state 는 red/green/unknown 중 하나여야 합니다: {ev.get('signal_state')!r}")
    try:
        import json

        # 밑줄로 시작하는 키는 앱 내부 전달용이라 응답에 담기지 않는다(보행 위험의 마스크 등)
        json.dumps({key: value for key, value in result.items() if not key.startswith("_")})
    except TypeError as exc:
        errs.append(f"JSON 으로 저장할 수 없는 값이 있습니다 (numpy/tensor 는 float(), int() 로 변환): {exc}")
    return errs


def detection_style(d: dict) -> tuple[str, tuple[int, int, int]]:
    if d.get("class_name") == "crosswalk":
        from backend.app.services.video_service import crosswalk_style

        return crosswalk_style(d)
    if d.get("class_name") == "pedestrian_signal":
        extra = d.get("extra") or {}
        selection = extra.get("selection_status")
        if selection in {"unselected", "candidate"}:
            color = (32, 176, 255) if selection == "candidate" else (255, 140, 79)
            return f"{selection.upper()} det {d['confidence']:.2f}", color
        state = extra.get("signal_state")
        colors = {"red": (59, 57, 229), "green": (74, 168, 31), "unknown": (136, 136, 136)}
        if state in colors:
            score = extra.get("color_confidence")
            label = state.upper()
            if selection == "selected":
                label = f"TARGET {label}"
            if state != "unknown" and isinstance(score, (int, float)):
                label += f" {score * 100:.1f}%"
            return label, colors[state]
    return f"{d['class_name']} {d['confidence']:.2f}", (255, 140, 60)


def draw(frame: np.ndarray, result: dict) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    for d in sorted(result.get("detections", []), key=lambda d: d["class_name"] != "crosswalk"):
        b = d["box"]
        p1, p2 = (int(b["x1"] * w), int(b["y1"] * h)), (int(b["x2"] * w), int(b["y2"] * h))
        label, color = detection_style(d)
        cv2.rectangle(out, p1, p2, color, 2)
        cv2.putText(out, label, (p1[0], max(16, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=list(EVENT_TYPES))
    ap.add_argument("--image", type=Path)
    ap.add_argument("--model", help="모델 ID. 생략하면 해당 기능의 첫 번째 실제 가중치 (없으면 mock)")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--list", action="store_true", help="인식된 모델 목록만 출력")
    args = ap.parse_args()

    registry = PipelineRegistry(Settings().model_dir)
    if args.list:
        for s in registry.list_models():
            print(f"{'O' if s.available else 'X'}  {s.id:28s} {s.name}   [{s.note}]")
        return 0
    if not args.mode or not args.image:
        ap.error("--mode 와 --image 가 필요합니다")
    confidence = args.conf if args.conf is not None else (0.25 if args.mode in ("traffic", "walking") else 0.4)

    frame = cv2.imread(str(args.image))
    if frame is None:
        print(f"[실패] 이미지를 읽을 수 없습니다: {args.image}")
        return 1

    specs = [s for s in registry.list_models() if s.mode == args.mode and s.available]
    spec = next((s for s in specs if s.id == args.model), None) if args.model else next((s for s in specs if not s.is_mock), specs[0])
    if spec is None:
        print(f"[실패] 모델 ID 를 찾을 수 없습니다: {args.model}. --list 로 확인하세요")
        return 1
    print(f"모델: {spec.id}  ({spec.note})")
    if spec.is_mock:
        print("  ※ 실제 가중치가 없어서 mock 으로 실행합니다. backend/models/%s/ 에 가중치를 넣으세요." % args.mode)

    t = time.perf_counter()
    try:
        pipe = registry.get(spec.id)
    except PipelineLoadError as exc:
        print(f"[실패] load(): {exc}")
        return 1
    print(f"[통과] load()  {1000 * (time.perf_counter() - t):.0f} ms")

    pipe.reset_session("check")
    times, result = [], None
    for i in range(1, 6):
        t = time.perf_counter()
        try:
            result = pipe.infer(frame, InferenceContext(
                session_id="check", frame_id=i,
                # 보행 위험 판단은 시간 간격을 쓰므로 단조 증가 시각을 준다
                captured_at_ms=int(time.monotonic() * 1000) if args.mode == "walking" else 0,
                confidence=confidence))
        except Exception as exc:  # noqa: BLE001
            print(f"[실패] infer(): {type(exc).__name__}: {exc}")
            return 1
        times.append(1000 * (time.perf_counter() - t))
    pipe.close_session("check")
    print(f"[통과] infer() 5회  첫 회 {times[0]:.0f} ms, 이후 평균 {sum(times[1:]) / 4:.0f} ms")

    errs = validate(result, args.mode)
    if errs:
        print("[실패] 반환 형식")
        for e in errs:
            print("   -", e)
        return 1
    event_summary = ({k:v for k,v in result["event"].items() if k not in ("mask_png","mask_rle","settings_snapshot")}
                     if args.mode == "walking" else result["event"])
    print(f"[통과] 반환 형식  검출 {len(result['detections'])}개, event={event_summary}")
    for d in result["detections"]:
        print(f"   - {d['class_name']} {d['confidence']:.2f} {d['box']}")

    out_dir = ROOT / "check_output"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{args.mode}_{args.image.stem}.jpg"
    if "_walking_record" in result:
        from backend.app.inference.walking.visualization.render import render_frame
        record = result["_walking_record"]
        shown = render_frame(frame, record, record["class_map"])
    else:
        shown = draw(frame, result)
    cv2.imwrite(str(out), shown)
    print(f"박스를 그린 이미지: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
