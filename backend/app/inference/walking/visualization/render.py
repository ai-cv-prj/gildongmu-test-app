"""
file_path: backend/app/inference/walking/visualization/render.py

Integration overlays plus the same Korean warning used by the live canvas.
"""
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .scene_visualization import overlay_segmentation, draw_detections
from .risk_visualization import draw_risk

def ensure_export_support(font_path=""):
    import imageio_ffmpeg
    encoder = imageio_ffmpeg.get_ffmpeg_exe()
    candidates = [font_path] if font_path else [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/mnt/c/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgun.ttf"]
    font = next((p for p in candidates if Path(p).is_file()), None)
    if not font:
        raise RuntimeError("한글 폰트가 필요합니다. WALKING_FONT_PATH에 TTF/TTC 경로를 지정하세요")
    ImageFont.truetype(font, 18)
    return encoder, font

@lru_cache(maxsize=16)
def _font(path, size):
    return ImageFont.truetype(path, size)

def warning_overlay(frame, text, level, font_path=""):
    if not text:
        return frame
    _, path = ensure_export_support(font_path)
    rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    layer = Image.new("RGBA", rgb.size)
    draw = ImageDraw.Draw(layer)
    size = max(12, round(frame.shape[1] / 27))
    font = _font(path, size)
    while draw.textbbox((0,0),text,font=font)[2] > rgb.width-28 and size > 9:
        size -= 1
        font = _font(path, size)
    bounds = draw.textbbox((0,0),text,font=font)
    height = bounds[3]-bounds[1]
    color = (255,90,90,255) if level=="danger" else (255,210,65,255)
    draw.rounded_rectangle((8,8,min(rgb.width-8,bounds[2]+24),height+24),radius=4,fill=(12,16,22,220))
    draw.text((16,16-bounds[1]),text,font=font,fill=color)
    return cv2.cvtColor(np.asarray(Image.alpha_composite(rgb,layer).convert("RGB")),cv2.COLOR_RGB2BGR)

def render_frame(frame, record, class_map=None, font_path=""):
    if record.get("error") or not record.get("prediction"):
        return warning_overlay(frame.copy(),"분석 실패 · 원본 프레임 보존","caution",font_path)
    prediction, settings = record["prediction"], record["settings"]
    result = overlay_segmentation(frame,class_map,settings["label_ids"],settings.get("overlay_alpha",.55))
    config = settings["risk_config"]
    view_unavailable=(prediction.get("camera_view") or {}).get("status")=="unavailable"
    if not config.get("review_overlay",False) and not view_unavailable:
        result = draw_detections(result,prediction["detections"])
    result = draw_risk(result,prediction,config)
    # The integration renderer draws the main corridor; retain all extra candidates.
    roi = prediction.get("roi") or {}
    for polygon in ([] if view_unavailable else roi.get("corridor_polygons",[])[1:]):
        points = np.rint(np.asarray(polygon)*[frame.shape[1]-1,frame.shape[0]-1]).astype(np.int32)
        cv2.polylines(result,[points],True,(255,220,20),2,cv2.LINE_AA)
    return warning_overlay(result,prediction.get("warning_text",""),prediction.get("level","monitor"),font_path)
