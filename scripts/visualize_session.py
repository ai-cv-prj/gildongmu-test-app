"""저장된 세션의 results.jsonl을 원본 프레임에 그려 JPEG로 저장한다.

사용: .venv/bin/python scripts/visualize_session.py backend/data/sessions/<세션 ID>
결과: <세션>/annotated/00000001.jpg ... 와 contact_sheet.jpg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def render(frame: np.ndarray, result: dict) -> np.ndarray:
    """0~1 정규화 박스를 이미지 좌표로 바꾸고, 탐지 상태를 표시한다."""
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
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = y1 - 5 if y1 - text_height - baseline - 8 >= 38 else y2 + text_height + 8
        label_y = min(height - baseline - 2, label_y)
        label_x = min(x1, max(0, width - text_width - 8))
        cv2.rectangle(out, (label_x, label_y - text_height - 5),
                      (label_x + text_width + 8, label_y + baseline + 3), color, -1)
        cv2.putText(out, label, (label_x + 4, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def make_contact_sheet(images: list[np.ndarray], output: Path) -> None:
    columns = 5
    thumb_width, thumb_height = 216, 384
    padding = 8
    rows = (len(images) + columns - 1) // columns
    sheet = np.full((rows * (thumb_height + padding) + padding,
                     columns * (thumb_width + padding) + padding, 3), 235, dtype=np.uint8)
    for index, frame in enumerate(images):
        x = padding + index % columns * (thumb_width + padding)
        y = padding + index // columns * (thumb_height + padding)
        sheet[y:y + thumb_height, x:x + thumb_width] = frame
    if not cv2.imwrite(str(output), sheet):
        raise OSError(f"모음 사진을 저장할 수 없습니다: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="manifest.json, results.jsonl, frames/가 있는 폴더")
    args = parser.parse_args()
    session = args.session.resolve()
    results_path = session / "results.jsonl"
    if not results_path.is_file():
        parser.error(f"결과 파일이 없습니다: {results_path}")

    output_dir = session / "annotated"
    output_dir.mkdir(exist_ok=True)
    images: list[np.ndarray] = []
    detected = 0
    missing = 0
    with results_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                result = json.loads(line)
                frame_id = int(result["frame_id"])
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{results_path}:{line_number}: 잘못된 결과: {exc}") from exc
            frame_path = session / "frames" / f"{frame_id:08d}.jpg"
            frame = cv2.imread(str(frame_path))
            if frame is None:
                print(f"[건너뜀] 프레임을 읽을 수 없습니다: {frame_path}")
                missing += 1
                continue
            annotated = render(frame, result)
            output = output_dir / frame_path.name
            if not cv2.imwrite(str(output), annotated):
                raise OSError(f"사진을 저장할 수 없습니다: {output}")
            images.append(cv2.resize(annotated, (216, 384), interpolation=cv2.INTER_AREA))
            detected += bool(result.get("detections"))

    if images:
        make_contact_sheet(images, output_dir / "contact_sheet.jpg")
    print(f"프레임 {len(images)}장 저장 (탐지 {detected}장, 프레임 없음 {missing}장): {output_dir}")
    if images:
        print(f"전체 보기: {output_dir / 'contact_sheet.jpg'}")


if __name__ == "__main__":
    main()
