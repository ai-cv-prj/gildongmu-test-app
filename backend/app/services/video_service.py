"""저장된 프레임과 추론 기록으로 탐지 결과 MP4를 만든다."""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np

VIDEO_REL_PATH = "annotated/results.mp4"


def render(frame: np.ndarray, result: dict) -> np.ndarray:
    out = frame.copy()
    height, width = out.shape[:2]
    detections = result.get("detections") or []
    state = (result.get("event") or {}).get("signal_state")
    status = f"DETECTED {len(detections)}" if detections else "NO DETECTION"
    if state:
        status += f" | signal: {state.upper()}"
    if result.get("error"):
        status = f"ERROR: {result['error'].get('code', 'unknown')}"

    cv2.rectangle(out, (0, 0), (width, 38), (24, 24, 24), -1)
    cv2.putText(out, f"#{result['frame_id']:08d}  {status}", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.57, (255, 255, 255), 2, cv2.LINE_AA)

    for detection in detections:
        box = detection["box"]
        x1 = max(0, min(width - 1, round(box["x1"] * width)))
        y1 = max(0, min(height - 1, round(box["y1"] * height)))
        x2 = max(0, min(width - 1, round(box["x2"] * width)))
        y2 = max(0, min(height - 1, round(box["y2"] * height)))
        color_state = (detection.get("extra") or {}).get("signal_state")
        color = {"red": (40, 40, 240), "green": (50, 210, 60)}.get(color_state, (0, 200, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        if color_state:
            label += f" {color_state.upper()}"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = y1 - 5 if y1 - text_height - baseline - 8 >= 38 else y2 + text_height + 8
        label_y = min(height - baseline - 2, label_y)
        label_x = min(x1, max(0, width - text_width - 8))
        cv2.rectangle(out, (label_x, label_y - text_height - 5),
                      (label_x + text_width + 8, label_y + baseline + 3), color, -1)
        cv2.putText(out, label, (label_x + 4, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def session_fps(session: Path) -> float:
    manifest = session / "manifest.json"
    if not manifest.is_file():
        return 5.0
    settings = json.loads(manifest.read_text(encoding="utf-8")).get("settings") or {}
    fps = float(settings.get("target_fps") or 5.0)
    if fps <= 0:
        raise ValueError(f"잘못된 target_fps: {manifest}")
    return fps


def open_video(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise OSError(f"MP4 영상을 만들 수 없습니다: {path}")
    return writer


def export_session_video(session: Path) -> Path | None:
    """종료된 세션의 사진을 기록 순서대로 영상화한다. 프레임이 없으면 None."""
    output = session / VIDEO_REL_PATH
    temporary = output.with_name("results.tmp.mp4")
    writer: cv2.VideoWriter | None = None
    size: tuple[int, int] | None = None
    try:
        with (session / "results.jsonl").open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                frame_id = int(record["frame_id"])
                frame_path = session / "frames" / f"{frame_id:08d}.jpg"
                if not frame_path.is_file():
                    continue
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    continue
                annotated = render(frame, record)
                height, width = annotated.shape[:2]
                if writer is None:
                    size = (width, height)
                    writer = open_video(temporary, session_fps(session), size)
                if size != (width, height):
                    raise ValueError(f"세션 안의 프레임 크기가 다릅니다: {frame_path}")
                writer.write(annotated)
    except Exception:
        if writer is not None:
            writer.release()
        temporary.unlink(missing_ok=True)
        raise
    if writer is None:
        return None
    writer.release()
    os.replace(temporary, output)
    return output
