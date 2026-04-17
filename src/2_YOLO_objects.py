#!/usr/bin/env python3
"""Extract object-level features using YOLO object detection.

This script reads dataset/annotated.csv, detects objects in each image using YOLO,
extracts object classes, counts, and bounding boxes, and saves features for model grounding.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "dataset" / "annotated.csv"
IMAGES_DIR = ROOT_DIR / "dataset" / "images_for_annotation"
FEATURES_DIR = ROOT_DIR / "dataset" / "features"
MODEL_NAME = "yolov8m.pt"  # Medium YOLO model; use yolov8n.pt for faster, yolov8l.pt for more accurate

IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".webp"]

# Object detection outputs
OBJECTS_JSON_PATH = FEATURES_DIR / "detected_objects.json"
OBJECTS_INDEX_PATH = FEATURES_DIR / "objects_index.csv"
OBJECT_COUNTS_PATH = FEATURES_DIR / "object_counts.npy"
CONFIG_PATH = FEATURES_DIR / "object_detection_config.json"


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


def extract_objects(
	image_path: Path,
	model: YOLO,
	confidence_threshold: float = 0.5,
) -> dict[str, int | list]:
	"""Extract detected objects, counts, and bounding boxes from image.
	
	Returns:
		dict with keys:
		- "objects": list of object class names detected
		- "counts": dict mapping class name to count
		- "bboxes": list of dicts with class, confidence, and bbox coords
	"""
	results = model.predict(source=str(image_path), verbose=False, conf=confidence_threshold)
	
	if not results or len(results) == 0:
		return {"objects": [], "counts": {}, "bboxes": []}
	
	result = results[0]
	detected = {"objects": [], "counts": {}, "bboxes": []}
	
	if result.boxes is not None:
		for box in result.boxes:
			class_id = int(box.cls[0])
			class_name = result.names[class_id]
			confidence = float(box.conf[0])
			bbox_xyxy = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
			
			detected["objects"].append(class_name)
			detected["counts"][class_name] = detected["counts"].get(class_name, 0) + 1
			detected["bboxes"].append({
				"class": class_name,
				"confidence": round(confidence, 3),
				"bbox": [round(x, 2) for x in bbox_xyxy],  # [x1, y1, x2, y2]
			})
	
	detected["objects"] = list(set(detected["objects"]))  # Unique classes
	return detected


def main() -> None:
	if not DATASET_PATH.exists():
		raise FileNotFoundError(f"CSV not found: {DATASET_PATH}")
	if not IMAGES_DIR.exists():
		raise NotADirectoryError(f"Image folder not found: {IMAGES_DIR}")

	# Load YOLO model
	print(f"Loading YOLO model: {MODEL_NAME}...")
	model = YOLO(MODEL_NAME)

	df = pd.read_csv(DATASET_PATH, dtype=str, keep_default_na=False)
	
	kept_rows: list[dict[str, str]] = []
	all_objects_data: list[dict] = []
	object_counts_list: list[list] = []
	skipped_missing_image = 0
	skipped_detection_error = 0

	# Global tracking for consistent feature vectors
	all_classes_set = set()

	# First pass: collect all detected classes
	print("First pass: scanning for all object classes...")
	for idx, (_, row) in enumerate(df.iterrows()):
		if (idx + 1) % 10 == 0:
			print(f"  Scanned {idx + 1}/{len(df)}")
		
		row_dict = row.to_dict()
		row_id = str(row_dict.get("id", "")).strip()
		image_path = find_image_path(row_id)

		if image_path is None:
			skipped_missing_image += 1
			continue

		try:
			detected = extract_objects(image_path, model, confidence_threshold=0.5)
			all_classes_set.update(detected["counts"].keys())
		except Exception:
			skipped_detection_error += 1
			continue

	sorted_classes = sorted(list(all_classes_set))
	print(f"Found {len(sorted_classes)} unique object classes: {sorted_classes}\n")

	# Second pass: extract counts for all images
	print("Second pass: extracting object counts...")
	skipped_missing_image = 0
	skipped_detection_error = 0

	for idx, (_, row) in enumerate(df.iterrows()):
		if (idx + 1) % 10 == 0:
			print(f"  Processed {idx + 1}/{len(df)}")

		row_dict = row.to_dict()
		row_id = str(row_dict.get("id", "")).strip()
		image_path = find_image_path(row_id)

		if image_path is None:
			skipped_missing_image += 1
			continue

		try:
			detected = extract_objects(image_path, model, confidence_threshold=0.5)
		except Exception:
			skipped_detection_error += 1
			continue

		# Build count vector for this image
		count_vector = [detected["counts"].get(cls, 0) for cls in sorted_classes]
		object_counts_list.append(count_vector)

		# Store object data for JSON export
		all_objects_data.append({
			"id": row_id,
			"detected_objects": detected["objects"],
			"object_counts": detected["counts"],
			"bboxes": detected["bboxes"],
		})

		kept_rows.append(row_dict)

	if len(df) > 0 and not kept_rows:
		raise RuntimeError(
			"No rows were processed successfully. Abort writing to avoid data loss."
		)

	# Convert to numpy array for consistency with CLIP features
	object_counts_array = np.array(object_counts_list, dtype=np.int32)

	FEATURES_DIR.mkdir(parents=True, exist_ok=True)

	# Save object counts as .npy
	np.save(OBJECT_COUNTS_PATH, object_counts_array)

	# Save detailed object data as JSON
	with open(OBJECTS_JSON_PATH, "w", encoding="utf-8") as f:
		json.dump(all_objects_data, f, indent=2)

	# Save index with ids for alignment with other features
	index_df = pd.DataFrame(kept_rows)
	index_columns = ["id", "human_generated_argument"]
	for col in index_columns:
		if col not in index_df.columns:
			index_df[col] = ""
	index_df[index_columns].to_csv(OBJECTS_INDEX_PATH, index=False)

	# Save config
	config = {
		"model": MODEL_NAME,
		"object_classes": sorted_classes,
		"num_classes": len(sorted_classes),
		"object_counts_shape": object_counts_array.shape,
		"confidence_threshold": 0.5,
	}
	CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

	print(f"\nProcessed rows: {len(df)}")
	print(f"Rows kept: {len(kept_rows)}")
	print(f"Rows dropped (missing image): {skipped_missing_image}")
	print(f"Rows dropped (detection error): {skipped_detection_error}")
	print(f"Unique object classes detected: {len(sorted_classes)}")
	print(f"Object classes: {sorted_classes}\n")
	print(f"Saved object counts: {OBJECT_COUNTS_PATH} (shape: {object_counts_array.shape})")
	print(f"Saved object data (JSON): {OBJECTS_JSON_PATH}")
	print(f"Saved row index: {OBJECTS_INDEX_PATH}")
	print(f"Saved config: {CONFIG_PATH}")


if __name__ == "__main__":
	main()
