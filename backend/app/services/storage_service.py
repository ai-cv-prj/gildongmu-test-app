"""
file_path: backend/app/services/storage_service.py

세션의 원본 프레임, 추론 결과, 요약과 실시간 탐지 영상을 저장한다.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


class StorageError(RuntimeError):
    pass


class SessionStorage:
    def __init__(self, sessions_dir: Path, save_frames: bool) -> None:
        self.sessions_dir = sessions_dir
        self.save_frames = save_frames
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    # ---- 디스크 ----
    def free_gb(self) -> float:
        return shutil.disk_usage(self.sessions_dir).free / (1024**3)

    def writable(self) -> bool:
        try:
            probe = self.sessions_dir / ".write_probe"
            probe.write_text("ok")
            probe.unlink()
            return True
        except OSError:
            return False

    # ---- 세션 ----
    def unique_session_id(self, base: str) -> str:
        sid, n = base, 1
        while (self.sessions_dir / sid).exists():
            n += 1
            sid = f"{base}_{n}"
        return sid

    def session_dir(self, session_id: str) -> Path:
        return self.sessions_dir / session_id

    def create_session(self, session_id: str, manifest: dict[str, Any]) -> Path:
        d = self.session_dir(session_id)
        try:
            (d / "frames").mkdir(parents=True, exist_ok=False)
            self.write_manifest(session_id, manifest)
            (d / "results.jsonl").touch()
        except OSError as exc:
            raise StorageError(f"cannot create session directory: {exc}") from exc
        return d

    def write_manifest(self, session_id: str, manifest: dict[str, Any]) -> None:
        path = self.session_dir(session_id) / "manifest.json"
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as exc:
            raise StorageError(f"cannot write manifest: {exc}") from exc

    def read_manifest(self, session_id: str) -> dict[str, Any] | None:
        path = self.session_dir(session_id) / "manifest.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def list_session_ids(self) -> list[str]:
        """manifest.json 이 있는 세션 폴더 이름들."""
        if not self.sessions_dir.exists():
            return []
        return sorted(d.name for d in self.sessions_dir.iterdir() if d.is_dir() and (d / "manifest.json").exists())

    def count_results(self, session_id: str) -> tuple[int, int]:
        """results.jsonl 에서 (성공 프레임 수, 오류 수)를 센다. 비정상 종료 복구에 쓴다."""
        path = self.session_dir(session_id) / "results.jsonl"
        ok = err = 0
        if not path.exists():
            return ok, err
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue  # 쓰다가 끊긴 마지막 줄
                if rec.get("error"):
                    err += 1
                else:
                    ok += 1
        return ok, err

    def frame_rel_path(self, frame_id: int) -> str:
        return f"frames/{frame_id:08d}.jpg"

    def save_frame(self, session_id: str, frame_id: int, jpeg_bytes: bytes) -> str | None:
        """저장된 상대 경로를 반환. save_frames=False 면 None."""
        if not self.save_frames:
            return None
        rel = self.frame_rel_path(frame_id)
        path = self.session_dir(session_id) / rel
        try:
            with open(path, "wb") as f:
                f.write(jpeg_bytes)
                f.flush()
                os.fsync(f.fileno())
        except OSError as exc:
            raise StorageError(f"cannot save frame: {exc}") from exc
        return rel

    # 실시간 탐지 녹화 영상 저장
    def save_recording(self, session_id: str, video_bytes: bytes) -> str:
        """
        카메라와 탐지 오버레이가 합성된 WebM 영상을 세션 폴더에 저장한다.
        """
        rel = "realtime_overlay.webm"
        path = self.session_dir(session_id) / rel
        try:
            with open(path, "wb") as file:
                file.write(video_bytes)
                file.flush()
                os.fsync(file.fileno())
        except OSError as exc:
            raise StorageError(f"cannot save recording: {exc}") from exc
        return rel

    def append_result(self, session_id: str, record: dict[str, Any]) -> None:
        path = self.session_dir(session_id) / "results.jsonl"
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
        except OSError as exc:
            raise StorageError(f"cannot append result: {exc}") from exc

    def read_results(self, session_id: str, after_frame_id: int, limit: int) -> list[dict[str, Any]]:
        path = self.session_dir(session_id) / "results.jsonl"
        if not path.exists():
            return []
        out: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("frame_id", 0) > after_frame_id:
                    out.append(rec)
                    if len(out) >= limit:
                        break
        return out

    def frame_path(self, session_id: str, frame_id: int) -> Path:
        return self.session_dir(session_id) / self.frame_rel_path(frame_id)
