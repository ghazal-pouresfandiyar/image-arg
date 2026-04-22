#!/usr/bin/env python3
"""Simple YOLO object detection and storage."""

import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "dataset" / "annotated.csv"
IMAGES_DIR = ROOT_DIR / "dataset" / "images_for_annotation"
FEATURES_DIR = ROOT_DIR / "dataset" / "features"
YOLO_DETECTION_DIR = ROOT_DIR / "output" / "yolo_detection"

# Output files
OBJECTS_JSON_PATH = FEATURES_DIR / "detected_objects.json"
YOLO_MODEL_PATH = ROOT_DIR / "yolov8l.pt"
CONFIDENCE_THRESHOLD = 0.5


def extract_objects(image_path, model, confidence_threshold=0.5):
	"""Extract objects detected in image."""
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
			x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
			
			detected["objects"].append(class_name)
			detected["counts"][class_name] = detected["counts"].get(class_name, 0) + 1
			
			detected["bboxes"].append({
				"class": class_name,
				"confidence": round(confidence, 3),
				"bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
			})
	
	# Remove duplicates from objects list
	detected["objects"] = list(dict.fromkeys(detected["objects"]))
	return detected


def draw_boxes(image_path, detected, output_path):
	"""Draw bounding boxes on image and save."""
	image = cv2.imread(str(image_path))
	if image is None:
		return False

	output_path.parent.mkdir(parents=True, exist_ok=True)

	for bbox_info in detected["bboxes"]:
		x1, y1, x2, y2 = bbox_info["bbox"]
		class_name = bbox_info["class"]
		confidence = bbox_info["confidence"]

		cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

		label = f"{class_name} {confidence:.2f}"
		cv2.putText(image, label, (int(x1), int(y1) - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

	cv2.imwrite(str(output_path), image)
	return True


def main():
	"""Run YOLO detection on all images."""
	print("\n" + "="*60)
	print("YOLO Object Detection")
	print("="*60)
	
	if not DATASET_PATH.exists():
		print(f"Error: {DATASET_PATH} not found")
		return
	
	if not IMAGES_DIR.exists():
		print(f"Error: {IMAGES_DIR} not found")
		return

	print(f"Loading YOLO model...")
	model = YOLO(YOLO_MODEL_PATH)

	print(f"Reading dataset...")
	df = pd.read_csv(DATASET_PATH, dtype=str, keep_default_na=False)
	
	FEATURES_DIR.mkdir(parents=True, exist_ok=True)
	YOLO_DETECTION_DIR.mkdir(parents=True, exist_ok=True)

	print(f"Processing {len(df)} images...")
	
	all_results = []
	processed = 0
	skipped = 0

	for idx, (_, row) in enumerate(df.iterrows()):
		row_id = str(row.get("id", "")).strip()
		image_path = IMAGES_DIR / f"{row_id}.jpg"

		if not image_path.exists():
			skipped += 1
			continue

		# Run detection
		detected = extract_objects(image_path, model, CONFIDENCE_THRESHOLD)
		
		# Save image with boxes
		output_image_path = YOLO_DETECTION_DIR / f"{row_id}.jpg"
		draw_boxes(image_path, detected, output_image_path)

		# Store results
		all_results.append({
			"id": row_id,
			"objects": detected["objects"],
			"counts": detected["counts"],
			"bboxes": detected["bboxes"]
		})

		processed += 1
		if (processed) % 10 == 0:
			print(f"  Processed: {processed}")

	# Save results to JSON
	with open(OBJECTS_JSON_PATH, "w") as f:
		json.dump(all_results, f, indent=2)

	print(f"\n✓ Detection Complete")
	print(f"  Processed: {processed}")
	print(f"  Skipped: {skipped}")
	print(f"  Images saved: {YOLO_DETECTION_DIR}")
	print(f"  JSON saved: {OBJECTS_JSON_PATH}\n")


if __name__ == "__main__":
	main()
