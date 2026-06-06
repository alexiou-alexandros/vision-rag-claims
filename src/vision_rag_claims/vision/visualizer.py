import cv2
import numpy as np

from vision_rag_claims.schemas import DetectionResult


def draw_detections(
    image: np.ndarray,
    result: DetectionResult,
    palette: list[tuple[int, int, int]],
    class_names: list[str],
    draw_masks: bool = True,
    draw_boxes: bool = True,
    draw_labels: bool = True,
) -> np.ndarray:
    """Return a copy of *image* with bounding boxes, masks, and labels drawn."""
    annotated = image.copy()
    palette_bgr = [(b, g, r) for (r, g, b) in palette]

    for det in result.detections:
        class_idx = class_names.index(det.damage_type)
        color = palette_bgr[class_idx]

        if draw_masks and det.mask is not None:
            overlay = np.zeros_like(annotated)
            overlay[det.mask] = color
            annotated = cv2.addWeighted(annotated, 1.0, overlay, 0.5, 0)

        x1, y1, x2, y2 = det.bbox

        if draw_boxes:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)

        if draw_labels:
            label = f"{det.damage_type} {det.confidence:.0%}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - text_h - baseline - 4),
                (x1 + text_w, y1),
                color,
                thickness=-1,
            )
            cv2.putText(
                annotated,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return annotated
