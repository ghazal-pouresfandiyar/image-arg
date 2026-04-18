#!/usr/bin/env python3
"""Visualize object detections by drawing bounding boxes on images.

This script loads detected objects and their bounding boxes, draws them on images,
and saves annotated images for analysis and debugging.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "dataset" / "annotated.csv"
IMAGES_DIR = ROOT_DIR / "dataset" / "images_for_annotation"
OBJECTS_JSON_PATH = ROOT_DIR / "dataset" / "features" / "detected_objects.json"
FEATURES_DIR = ROOT_DIR / "dataset" / "features"

VIZ_OUTPUT_DIR = FEATURES_DIR / "visualizations"
IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".webp"]

# Color palette for different object classes
COLORS = {
	"person": (0, 255, 0),  # Green
	"car": (0, 165, 255),  # Orange
	"truck": (0, 100, 255),  # Red
	"bear": (0, 0, 255),  # Blue
	"dog": (255, 0, 0),  # Cyan
	"cat": (255, 255, 0),  # Yellow
	"bird": (255, 0, 255),  # Magenta
	"horse": (128, 0, 128),  # Purple
	"sheep": (165, 42, 42),  # Brown
}


def find_image_path(row_id: str) -> Path | None:
	"""Return the first matching local image path for a row id."""
	row_id = str(row_id).strip()
	if not row_id:
		return None

	for suffix in IMAGE_SUFFIXES:
		candidate = IMAGES_DIR / f"{row_id}{suffix}"
		if candidate.exists():
			return candidate

	matches = sorted(IMAGES_DIR.glob(f"{row_id}.*"))
	for candidate in matches:
		if candidate.is_file():
			return candidate

	return None


def get_color(class_name: str) -> tuple[int, int, int]:
	"""Get BGR color for object class."""
	if class_name in COLORS:
		return COLORS[class_name]
	# Hash-based color for unknown classes
	hash_val = hash(class_name) % 256
	return (hash_val, (hash_val + 85) % 256, (hash_val + 170) % 256)


def draw_bboxes(
	image: np.ndarray,
	bboxes: list[dict],
	thickness: int = 2,
) -> np.ndarray:
	"""Draw bounding boxes on image.
	
	Args:
		image: BGR image array
		bboxes: list of dicts with keys: class, confidence, bbox (x1, y1, x2, y2)
		thickness: line thickness for boxes
	
	Returns:
		Annotated image
	"""
	annotated = image.copy()

	for bbox_data in bboxes:
		class_name = bbox_data["class"]
		confidence = bbox_data["confidence"]
		x1, y1, x2, y2 = [int(x) for x in bbox_data["bbox"]]

		color = get_color(class_name)

		# Draw rectangle
		cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

		# Draw label with confidence
		label = f"{class_name} {confidence:.2f}"
		font = cv2.FONT_HERSHEY_SIMPLEX
		font_scale = 0.5
		font_thickness = 1
		text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]

		# Background for text
		cv2.rectangle(
			annotated,
			(x1, y1 - text_size[1] - 4),
			(x1 + text_size[0], y1),
			color,
			-1,
		)

		# Text
		cv2.putText(
			annotated,
			label,
			(x1, y1 - 2),
			font,
			font_scale,
			(255, 255, 255),
			font_thickness,
		)

	return annotated


def main() -> None:
	if not DATASET_PATH.exists():
		raise FileNotFoundError(f"CSV not found: {DATASET_PATH}")
	if not IMAGES_DIR.exists():
		raise NotADirectoryError(f"Image folder not found: {IMAGES_DIR}")
	if not OBJECTS_JSON_PATH.exists():
		raise FileNotFoundError(f"Objects JSON not found: {OBJECTS_JSON_PATH}")

	# Load detections
	with open(OBJECTS_JSON_PATH, "r", encoding="utf-8") as f:
		all_detections = json.load(f)

	# Create output directory
	VIZ_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	# Visualize each image
	print("Generating visualizations...")
	successful = 0
	failed = 0

	for idx, detection_data in enumerate(all_detections):
		row_id = detection_data["id"]
		bboxes = detection_data["bboxes"]

		if not bboxes:
			print(f"  {row_id}: No detections")
			continue

		image_path = find_image_path(row_id)
		if image_path is None:
			print(f"  {row_id}: Image not found")
			failed += 1
			continue

		try:
			image = cv2.imread(str(image_path))
			if image is None:
				print(f"  {row_id}: Failed to read image")
				failed += 1
				continue

			annotated = draw_bboxes(image, bboxes, thickness=2)

			output_path = VIZ_OUTPUT_DIR / f"{row_id}_annotated.jpg"
			cv2.imwrite(str(output_path), annotated)
			successful += 1

			if (idx + 1) % 10 == 0:
				print(f"  Processed {idx + 1}/{len(all_detections)}")

		except Exception as e:
			print(f"  {row_id}: Error - {e}")
			failed += 1

	print(f"\nVisualizations saved:")
	print(f"  Successful: {successful}")
	print(f"  Failed: {failed}")
	print(f"  Output directory: {VIZ_OUTPUT_DIR}")


if __name__ == "__main__":
	main()
