"""저장된 세션의 results.jsonl을 원본 프레임에 그려 사진과 영상으로 저장한다.

사용: .venv/bin/python scripts/visualize_session.py backend/data/sessions/<세션 ID>
여러 세션: .venv/bin/python scripts/visualize_session.py backend/data/sessions/20260918_*_traffic --combine backend/data/sessions/20260918_results.mp4
결과: <세션>/annotated/00000001.jpg ... 와 results.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.services.video_service import open_video, render, session_fps  # noqa: E402


def session_title(session: Path, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    card = np.full((height, width, 3), 28, dtype=np.uint8)
    parts = session.name.split("_", 2)
    lines = ["SESSION", " ".join(parts[:2]), parts[2] if len(parts) > 2 else ""]
    for index, line in enumerate(lines):
        cv2.putText(card, line, (24, height // 2 - 45 + index * 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return card


# 보행 위험 세션 판별
def walking_record(session: Path, frame_id: int, result: dict) -> bool:
    """위험 판단 기록이 있으면 실시간 화면과 같은 렌더러를 쓴다."""
    return bool((result.get("event") or {}).get("risk_schema_version")
                or (session / "inputs" / f"{frame_id:08d}.json").is_file())


# 저장된 위험 판단으로 한 장 그리기
def render_walking(session: Path, frame: np.ndarray, frame_id: int) -> np.ndarray:
    """마스크와 ROI·경고를 결과 영상과 같은 방식으로 합성한다."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend.app.inference.walking.visualization.render import render_frame

    record_path = session / "risk" / f"{frame_id:08d}.json"
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {"error": "missing risk record"}
    mask = cv2.imread(str(session / record["mask_path"]), cv2.IMREAD_UNCHANGED) if record.get("mask_path") else None
    return render_frame(frame, record, mask)


def process_session(session: Path, combined: cv2.VideoWriter | None,
                    combined_size: tuple[int, int] | None, combined_fps: float | None
                    ) -> None:
    session = session.resolve()
    results_path = session / "results.jsonl"
    if not results_path.is_file():
        raise FileNotFoundError(f"결과 파일이 없습니다: {results_path}")

    output_dir = session / "annotated"
    output_dir.mkdir(exist_ok=True)
    fps = session_fps(session)
    frame_count = 0
    detected = 0
    missing = 0
    video: cv2.VideoWriter | None = None
    video_size: tuple[int, int] | None = None
    try:
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
                if not frame_path.is_file():
                    print(f"[건너뜀] 프레임이 없습니다: {frame_path}")
                    missing += 1
                    continue
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    print(f"[건너뜀] 프레임을 읽을 수 없습니다: {frame_path}")
                    missing += 1
                    continue
                annotated = (render_walking(session, frame, frame_id)
                             if walking_record(session, frame_id, result) else render(frame, result))
                height, width = annotated.shape[:2]
                if video is None:
                    video_size = (width, height)
                    video = open_video(output_dir / "results.mp4", fps, video_size)
                    if combined is not None and combined_size is not None:
                        card = session_title(session, combined_size)
                        for _ in range(round(combined_fps or fps)):
                            combined.write(card)
                if video_size != (width, height):
                    raise ValueError(f"세션 안의 프레임 크기가 다릅니다: {frame_path}")
                output = output_dir / frame_path.name
                if not cv2.imwrite(str(output), annotated):
                    raise OSError(f"사진을 저장할 수 없습니다: {output}")
                video.write(annotated)
                if combined is not None and combined_size is not None:
                    joined = cv2.resize(annotated, combined_size) if (width, height) != combined_size else annotated
                    combined.write(joined)
                frame_count += 1
                detected += bool(result.get("detections"))
    finally:
        if video is not None:
            video.release()

    print(f"프레임 {frame_count}장 저장 (탐지 {detected}장, 프레임 없음 {missing}장): {output_dir}")
    if frame_count:
        print(f"영상: {output_dir / 'results.mp4'} ({fps:g} fps)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", type=Path,
                        help="manifest.json, results.jsonl, frames/가 있는 폴더")
    parser.add_argument("--combine", type=Path, help="모든 세션을 이어 붙일 MP4 파일")
    args = parser.parse_args()

    combined: cv2.VideoWriter | None = None
    combined_size: tuple[int, int] | None = None
    combined_fps: float | None = None
    try:
        for session in args.sessions:
            # 통합 영상의 크기와 재생 속도는 첫 번째 세션을 따른다.
            if args.combine and combined is None:
                first_frame = next(iter(sorted((session / "frames").glob("*.jpg"))), None)
                if first_frame is not None:
                    sample = cv2.imread(str(first_frame))
                    if sample is not None:
                        height, width = sample.shape[:2]
                        combined_size = (width, height)
                        combined_fps = session_fps(session)
                        combined = open_video(args.combine.resolve(), combined_fps, combined_size)
            process_session(session, combined, combined_size, combined_fps)
    finally:
        if combined is not None:
            combined.release()
    if combined is not None:
        print(f"통합 영상: {args.combine.resolve()} ({combined_fps:g} fps)")


if __name__ == "__main__":
    main()
