"""
file_path: backend/app/inference/walking/sidewalk.py

Same-frame local Mask2Former adapter; imported only for real walking.
"""
from contextlib import nullcontext
from pathlib import Path
import cv2
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

class SidewalkSegmenter:
    def __init__(self, weights, device="cpu", precision="fp32"):
        weights = Path(weights)
        for name in ("model.safetensors", "config.json", "preprocessor_config.json"):
            if not (weights/name).is_file():
                raise FileNotFoundError(f"보도 모델 파일이 없습니다: {weights/name}")
        if precision not in ("fp32", "fp16"):
            raise ValueError("WALKING_PRECISION은 fp32 또는 fp16이어야 합니다")
        if precision == "fp16" and not str(device).startswith("cuda"):
            raise ValueError("walking FP16은 CUDA 장치에서만 지원합니다")
        self.device, self.precision = torch.device(device), precision
        self.processor = AutoImageProcessor.from_pretrained(weights, local_files_only=True)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(weights, local_files_only=True)
        self.label_ids = {name: int(cid) for cid, name in self.model.config.id2label.items()}
        if set(self.label_ids) != {"non_walkable", "walkable", "crosswalk"} or set(self.label_ids.values()) != {0, 1, 2}:
            raise ValueError("보도 모델은 non_walkable/walkable/crosswalk 3클래스여야 합니다")
        self.model = self.model.to(self.device).eval()

    def predict(self, frame):
        inputs = self.processor(images=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        precision = torch.autocast("cuda", dtype=torch.float16) if self.precision == "fp16" else nullcontext()
        with torch.inference_mode():
            with precision:
                outputs = self.model(**inputs)
            labels = self.processor.post_process_semantic_segmentation(outputs, target_sizes=[frame.shape[:2]])[0]
        return labels.cpu().numpy()
