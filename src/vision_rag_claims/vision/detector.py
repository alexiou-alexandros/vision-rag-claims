from pathlib import Path
from typing import Optional

import mmcv
import numpy as np
from mmdet.apis import inference_detector, init_detector

from vision_rag_claims.config import settings
from vision_rag_claims.schemas import Detection, DetectionResult


class DamageDetector:
    def __init__(
        self,
        score_threshold: Optional[float] = None,
        device: Optional[str] = None,
    ) -> None:
        self.score_threshold = score_threshold or settings.score_threshold
        self.device = device or settings.device
        self.model = init_detector(
            str(settings.mmdet_config_path),
            str(settings.mmdet_checkpoint_path),
            device=self.device,
        )
        self.classes: tuple[str, ...] = self.model.dataset_meta["classes"]

    def detect(self, image: str | Path | np.ndarray) -> DetectionResult:
        if isinstance(image, (str, Path)):
            image = mmcv.imread(str(image))
        raw = inference_detector(self.model, image)
        return self._postprocess(raw, image.shape)

    def _postprocess(self, raw_result, image_shape: tuple) -> DetectionResult:
        pred = raw_result.pred_instances
        scores = pred.scores.cpu().numpy()
        labels = pred.labels.cpu().numpy()
        bboxes = pred.bboxes.cpu().numpy()
        masks = pred.masks.cpu().numpy()

        detections = []
        for i, score in enumerate(scores):
            if score < self.score_threshold:
                continue
            x1, y1, x2, y2 = (int(v) for v in bboxes[i])
            detections.append(Detection(
                damage_type=self.classes[labels[i]],
                confidence=float(score),
                bbox=(x1, y1, x2, y2),
                mask_area_pixels=int(masks[i].sum()),
                bbox_area_pixels=(x2 - x1) * (y2 - y1),
                mask=masks[i],
            ))

        return DetectionResult(
            detections=detections,
            image_height=image_shape[0],
            image_width=image_shape[1],
        )
