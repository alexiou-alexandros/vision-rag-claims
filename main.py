"""CLI entry point for running damage detection on a single image."""

import argparse

import mmcv

from vision_rag_claims.vision.detector import DamageDetector
from vision_rag_claims.vision.visualizer import draw_detections


def run(image_path: str, output_path: str) -> None:
    detector = DamageDetector()
    image = mmcv.imread(image_path)
    result = detector.detect(image)

    annotated = draw_detections(
        image=image,
        result=result,
        palette=detector.model.dataset_meta["palette"],
        class_names=list(detector.classes),
    )

    mmcv.imwrite(annotated, output_path)
    print(f"Saved annotated image to {output_path}")
    print(result.model_dump(exclude={"detections": {"__all__": {"mask"}}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect car damage in an image.")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--output", default="annotated.jpg", help="Path for the annotated output image")
    args = parser.parse_args()
    run(args.image, args.output)
