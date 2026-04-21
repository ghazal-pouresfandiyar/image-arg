#!/usr/bin/env python3
"""YOLO object detection with CLIP validation.

This script performs two-stage object detection:
1. Stage 1: Extract object-level features using YOLO object detection
2. Stage 2: Validate YOLO detections against CLIP embeddings to catch mislabeling

Run with: python src/3_YOLO_objects.py
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "dataset" / "annotated.csv"
IMAGES_DIR = ROOT_DIR / "dataset" / "images_for_annotation"
FEATURES_DIR = ROOT_DIR / "dataset" / "features"
YOLO_DETECTION_DIR = ROOT_DIR / "yolo_detection"

# YOLO model and outputs
YOLO_MODEL_NAME = "yolov8l.pt"
OBJECTS_JSON_PATH = FEATURES_DIR / "detected_objects.json"
OBJECTS_INDEX_PATH = FEATURES_DIR / "objects_index.csv"
OBJECT_COUNTS_PATH = FEATURES_DIR / "object_counts.npy"
CONFIG_PATH = FEATURES_DIR / "object_detection_config.json"

# CLIP validation outputs
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
VALIDATION_RESULTS_PATH = FEATURES_DIR / "yolo_clip_validation.json"
VALIDATION_REPORT_PATH = FEATURES_DIR / "yolo_clip_validation_report.csv"
MISMATCHES_PATH = FEATURES_DIR / "yolo_clip_mismatches.json"

# Configuration
YOLO_CONFIDENCE_THRESHOLD = 0.5
CLIP_SIMILARITY_THRESHOLD = 0.5

# Common COCO classes for candidate labels  (COCO: Common Objects in Context)
COMMON_LABELS = [
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
	"boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
	"cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
	"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
	"snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
	"surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
	"spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
	"hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
	"dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "microwave",
	"oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
	"teddy bear", "hair drier", "toothbrush", "tree", "flower", "rock", "bush", "bird",
	"fish", "snake", "lizard", "insect", "spider", "building", "house", "bridge"
]


# ============================================================================
# STAGE 1: YOLO OBJECT DETECTION
# ============================================================================


def extract_objects(image_path, model, confidence_threshold=0.5):
	"""Extract detected objects, counts, and bounding boxes from image."""
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
			bbox_xyxy = box.xyxy[0].cpu().numpy().tolist()
			
			detected["objects"].append(class_name)
			
			if class_name in detected["counts"]:
				detected["counts"][class_name] = detected["counts"][class_name] + 1
			else:
				detected["counts"][class_name] = 1
			
			bbox_rounded = [round(x, 2) for x in bbox_xyxy]
			
			detected["bboxes"].append({
				"class": class_name,
				"confidence": round(confidence, 3),
				"bbox": bbox_rounded,
			})
	
	unique_objects = []
	for obj in detected["objects"]:
		if obj not in unique_objects:
			unique_objects.append(obj)
	detected["objects"] = unique_objects
	
	return detected


def draw_and_save_detections(image_path, detected, output_path):
	"""Draw bounding boxes on image and save to output path."""
	image = cv2.imread(str(image_path))
	if image is None:
		return False

	output_path.parent.mkdir(parents=True, exist_ok=True)

	for bbox_info in detected["bboxes"]:
		x1, y1, x2, y2 = bbox_info["bbox"]
		class_name = bbox_info["class"]
		confidence = bbox_info["confidence"]

		cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

		label = f"{class_name} ({confidence:.2f})"
		font = cv2.FONT_HERSHEY_SIMPLEX
		font_scale = 0.8
		thickness = 2
		
		text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
		
		label_x = int(x1)
		label_y = int(y1) - 8
		cv2.rectangle(image, 
			(label_x - 2, label_y - text_size[1] - 4),
			(label_x + text_size[0] + 2, label_y + 4),
			(0, 255, 0), -1)
		
		cv2.putText(image, label, (label_x, label_y),
				font, font_scale, (0, 0, 0), thickness)

	cv2.imwrite(str(output_path), image)
	return True


def run_yolo_detection():
	"""Run Stage 1: YOLO object detection."""
	print("\n" + "="*70)
	print("STAGE 1: YOLO OBJECT DETECTION")
	print("="*70)
	
	if not DATASET_PATH.exists():
		raise FileNotFoundError(f"CSV not found: {DATASET_PATH}")
	if not IMAGES_DIR.exists():
		raise NotADirectoryError(f"Image folder not found: {IMAGES_DIR}")

	print(f"Loading YOLO model: {YOLO_MODEL_NAME}...")
	model = YOLO(YOLO_MODEL_NAME)

	df = pd.read_csv(DATASET_PATH, dtype=str, keep_default_na=False)
	
	kept_rows = []
	all_objects_data = []
	object_counts_list = []
	skipped_missing_image = 0
	skipped_detection_error = 0
	all_classes_set = []

	# First pass: collect all detected classes
	print("Scanning for all object classes...")
	for idx, (_, row) in enumerate(df.iterrows()):
		if (idx + 1) % 10 == 0:
			print(f"  Scanned {idx + 1}/{len(df)}")
		
		row_dict = row.to_dict()
		row_id = str(row_dict.get("id", "")).strip()
		image_path = IMAGES_DIR / f"{row_id}.jpg"

		if not image_path.exists():
			skipped_missing_image += 1
			continue

		try:
			detected = extract_objects(image_path, model, confidence_threshold=0.25)
			for class_name in detected["counts"].keys():
				if class_name not in all_classes_set:
					all_classes_set.append(class_name)
		except Exception:
			skipped_detection_error += 1
			continue

	all_classes_set.sort()
	sorted_classes = all_classes_set
	print(f"Found {len(sorted_classes)} unique object classes\n")

	FEATURES_DIR.mkdir(parents=True, exist_ok=True)
	YOLO_DETECTION_DIR.mkdir(parents=True, exist_ok=True)

	# Second pass: extract counts for all images
	print("Extracting object counts and drawing detections...")
	skipped_missing_image = 0
	skipped_detection_error = 0

	for idx, (_, row) in enumerate(df.iterrows()):
		if (idx + 1) % 10 == 0:
			print(f"  Processed {idx + 1}/{len(df)}")

		row_dict = row.to_dict()
		row_id = str(row_dict.get("id", "")).strip()
		image_path = IMAGES_DIR / f"{row_id}.jpg"

		if not image_path.exists():
			skipped_missing_image += 1
			continue

		try:
			detected = extract_objects(image_path, model, confidence_threshold=YOLO_CONFIDENCE_THRESHOLD)
		except Exception:
			skipped_detection_error += 1
			continue

		count_vector = []
		for cls in sorted_classes:
			if cls in detected["counts"]:
				count_vector.append(detected["counts"][cls])
			else:
				count_vector.append(0)
		object_counts_list.append(count_vector)

		all_objects_data.append({
			"id": row_id,
			"detected_objects": detected["objects"],
			"object_counts": detected["counts"],
			"bboxes": detected["bboxes"],
		})

		output_image_path = YOLO_DETECTION_DIR / f"{row_id}.jpg"
		draw_and_save_detections(image_path, detected, output_image_path)

		kept_rows.append(row_dict)

	if len(df) > 0 and not kept_rows:
		raise RuntimeError("No rows were processed successfully.")

	object_counts_array = np.array(object_counts_list, dtype=np.int32)
	np.save(OBJECT_COUNTS_PATH, object_counts_array)

	with open(OBJECTS_JSON_PATH, "w", encoding="utf-8") as f:
		json.dump(all_objects_data, f, indent=2)

	index_df = pd.DataFrame(kept_rows)
	index_columns = ["id", "human_generated_argument"]
	for col in index_columns:
		if col not in index_df.columns:
			index_df[col] = ""
	index_df[index_columns].to_csv(OBJECTS_INDEX_PATH, index=False)

	config = {
		"model": YOLO_MODEL_NAME,
		"object_classes": sorted_classes,
		"num_classes": len(sorted_classes),
		"object_counts_shape": object_counts_array.shape,
		"confidence_threshold": YOLO_CONFIDENCE_THRESHOLD,
	}
	CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

	print(f"\n✓ YOLO Detection Complete")
	print(f"  Processed: {len(df)}, Kept: {len(kept_rows)}, Classes: {len(sorted_classes)}")


# ============================================================================
# STAGE 2: CLIP VALIDATION
# ============================================================================

def load_clip_model():
	"""Load CLIP model and processor."""
	device = "cuda" if torch.cuda.is_available() else "cpu"
	processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
	model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
	model.to(device)
	model.eval()
	return processor, model, device


def extract_patch(image_path, bbox, margin=0.05):
	"""Extract patch from image given bbox coordinates."""
	try:
		image = Image.open(image_path).convert("RGB")
		img_array = np.array(image)
		h, w = img_array.shape[:2]
		
		x1, y1, x2, y2 = bbox
		margin_x = (x2 - x1) * margin
		margin_y = (y2 - y1) * margin
		x1 = max(0, int(x1 - margin_x))
		y1 = max(0, int(y1 - margin_y))
		x2 = min(w, int(x2 + margin_x))
		y2 = min(h, int(y2 + margin_y))
		
		patch = img_array[y1:y2, x1:x2]
		if patch.size == 0:
			return None
		return Image.fromarray(patch)
	except Exception as e:
		print(f"Error extracting patch: {e}")
		return None


def get_image_embedding(image, processor, model, device):
	"""Get CLIP embedding for an image."""
	try:
		inputs = processor(images=image, return_tensors="pt")
		inputs = {key: value.to(device) for key, value in inputs.items()}
		
		with torch.no_grad():
			image_features = model.get_image_features(**inputs)
		
		if hasattr(image_features, "pooler_output"):
			image_features = image_features.pooler_output
		elif isinstance(image_features, (tuple, list)) and image_features:
			image_features = image_features[0]
		
		image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
		return image_features[0].detach().cpu().numpy().astype(np.float32)
	except Exception as e:
		print(f"Error getting image embedding: {e}")
		return None


def get_text_embeddings(labels, processor, model, device):
	"""Get CLIP embeddings for text labels."""
	embeddings = {}
	try:
		inputs = processor(text=labels, return_tensors="pt", padding=True)
		inputs = {key: value.to(device) for key, value in inputs.items()}
		
		with torch.no_grad():
			text_features = model.get_text_features(**inputs)
		
		# Extract tensor from BaseModelOutputWithPooling object
		if hasattr(text_features, "pooler_output"):
			text_features = text_features.pooler_output
		elif isinstance(text_features, (tuple, list)) and text_features:
			text_features = text_features[0]
		
		# Normalize
		text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
		
		for i, label in enumerate(labels):
			embeddings[label] = text_features[i].detach().cpu().numpy().astype(np.float32)
	except Exception as e:
		print(f"Error getting text embeddings: {e}")
	
	return embeddings
def cosine_similarity(vec1, vec2):
	"""Compute cosine similarity between two vectors."""
	if vec1 is None or vec2 is None:
		return 0.0
	return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8))


def get_candidate_labels(detected_label):
	"""Get candidate labels to compare against."""
	candidates = COMMON_LABELS.copy()
	if detected_label not in candidates:
		candidates.insert(0, detected_label)
	return candidates


def validate_detections(processor, model, device):
	"""Validate all YOLO detections with local bbox and global image CLIP analysis."""
	if not OBJECTS_JSON_PATH.exists():
		raise FileNotFoundError(f"YOLO detections not found: {OBJECTS_JSON_PATH}")
	
	with open(OBJECTS_JSON_PATH, "r") as f:
		detections = json.load(f)
	
	validation_results = []
	total_bboxes = 0
	skipped = {"patch": 0, "embedding": 0, "text": 0}
	
	for detection in detections:
		image_id = detection.get("id")
		image_path = IMAGES_DIR / f"{image_id}.jpg"
		
		if not image_path.exists():
			continue
		
		bboxes = detection.get("bboxes", [])
		if not bboxes:
			continue
		
		# Get full image embedding
		full_image = Image.open(image_path).convert("RGB")
		full_image_embedding = get_image_embedding(full_image, processor, model, device)
		
		# Get full-image CLIP scores for all candidates
		candidates = get_candidate_labels("")
		text_embeddings = get_text_embeddings(candidates, processor, model, device)
		
		full_image_similarities = {label: cosine_similarity(full_image_embedding, emb) 
		                            for label, emb in text_embeddings.items()}
		top_global_labels = sorted(full_image_similarities.items(), key=lambda x: x[1], reverse=True)[:5]
		
		# Process each bbox
		for bbox_idx, bbox_info in enumerate(bboxes):
			total_bboxes += 1
			detected_label = bbox_info.get("class")
			yolo_confidence = bbox_info.get("confidence", 0)
			bbox = bbox_info.get("bbox")
			
			# Extract bbox patch
			patch = extract_patch(image_path, bbox)
			if patch is None:
				skipped["patch"] += 1
				continue
			
			# Get bbox patch embedding
			patch_embedding = get_image_embedding(patch, processor, model, device)
			if patch_embedding is None:
				skipped["embedding"] += 1
				continue
			
			# Get candidates and text embeddings
			candidates = get_candidate_labels(detected_label)
			text_embeddings = get_text_embeddings(candidates, processor, model, device)
			
			if not text_embeddings:
				skipped["text"] += 1
				continue
			
			# Compute bbox CLIP scores
			bbox_similarities = {label: cosine_similarity(patch_embedding, emb) 
			                     for label, emb in text_embeddings.items()}
			
			# Get top-1 and top-2
			sorted_scores = sorted(bbox_similarities.items(), key=lambda x: x[1], reverse=True)
			bbox_clip_top_label = sorted_scores[0][0]
			bbox_clip_top_score = float(sorted_scores[0][1])
			bbox_clip_second_score = float(sorted_scores[1][1]) if len(sorted_scores) > 1 else 0.0
			bbox_clip_margin = bbox_clip_top_score - bbox_clip_second_score
			
			# Decision logic
			if detected_label == bbox_clip_top_label and bbox_clip_margin > 0.15:
				analysis_flag = "agree"
			elif bbox_clip_top_score < 0.3 or bbox_clip_margin < 0.05:
				analysis_flag = "uncertain"
			elif detected_label != bbox_clip_top_label and bbox_clip_margin > 0.15:
				analysis_flag = "conflict"
			else:
				analysis_flag = "uncertain"
			
			result = {
				"bbox": bbox,
				"yolo_label": detected_label,
				"yolo_confidence": float(yolo_confidence),
				
				"bbox_clip_top_label": bbox_clip_top_label,
				"bbox_clip_top_score": bbox_clip_top_score,
				"bbox_clip_second_score": bbox_clip_second_score,
				"bbox_clip_margin": bbox_clip_margin,
				
				"full_image_clip_top_labels": [
					{"label": label, "score": float(score)} 
					for label, score in top_global_labels
				],
				
				"analysis_flag": analysis_flag,
				
				# Additional metadata
				"image_id": image_id,
				"bbox_index": bbox_idx,
				"all_bbox_similarities": {k: float(v) for k, v in bbox_similarities.items()},
			}
			
			validation_results.append(result)
	
	return validation_results, total_bboxes, skipped


def save_validation_results(validation_results, total_bboxes, skipped):
	"""Save validation results with rich analysis."""
	FEATURES_DIR.mkdir(parents=True, exist_ok=True)
	
	# Save full validation results
	with open(VALIDATION_RESULTS_PATH, "w") as f:
		json.dump(validation_results, f, indent=2)
	
	# Group by image_id for structured output
	results_by_image = {}
	for result in validation_results:
		image_id = result["image_id"]
		if image_id not in results_by_image:
			results_by_image[image_id] = {
				"image_id": image_id,
				"detections": []
			}
		
		# Clean detection object (remove metadata fields for final output)
		detection = {
			"bbox": result["bbox"],
			"yolo_label": result["yolo_label"],
			"yolo_confidence": result["yolo_confidence"],
			
			"bbox_clip_top_label": result["bbox_clip_top_label"],
			"bbox_clip_top_score": result["bbox_clip_top_score"],
			"bbox_clip_second_score": result["bbox_clip_second_score"],
			"bbox_clip_margin": result["bbox_clip_margin"],
			
			"full_image_clip_top_labels": result["full_image_clip_top_labels"],
			
			"analysis_flag": result["analysis_flag"],
		}
		
		results_by_image[image_id]["detections"].append(detection)
	
	# Save structured output
	with open(FEATURES_DIR / "yolo_clip_analysis.json", "w") as f:
		json.dump(list(results_by_image.values()), f, indent=2)
	
	# Create backward-compatible mismatches for UI (flag != "agree")
	mismatch_data = []
	for result in validation_results:
		if result["analysis_flag"] != "agree":
			mismatch_data.append({
				"image_id": result["image_id"],
				"bbox_index": result["bbox_index"],
				"yolo_label": result["yolo_label"],
				"yolo_confidence": result["yolo_confidence"],
				"clip_suggestion": result["bbox_clip_top_label"],
				"clip_similarity_to_yolo_label": result["bbox_clip_top_score"] if result["yolo_label"] == result["bbox_clip_top_label"] else 0.0,
				"clip_similarity_to_suggestion": result["bbox_clip_top_score"],
			})
	
	with open(MISMATCHES_PATH, "w") as f:
		json.dump(mismatch_data, f, indent=2)
	
	# Create CSV summary with analysis flags
	summary_data = []
	for result in validation_results:
		summary_data.append({
			"image_id": result["image_id"],
			"bbox_index": result["bbox_index"],
			"yolo_label": result["yolo_label"],
			"yolo_confidence": round(result["yolo_confidence"], 3),
			"bbox_clip_top_label": result["bbox_clip_top_label"],
			"bbox_clip_top_score": round(result["bbox_clip_top_score"], 4),
			"bbox_clip_margin": round(result["bbox_clip_margin"], 4),
			"analysis_flag": result["analysis_flag"],
			"global_top_label": result["full_image_clip_top_labels"][0]["label"] if result["full_image_clip_top_labels"] else "-",
		})
	
	df = pd.DataFrame(summary_data)
	df.to_csv(VALIDATION_REPORT_PATH, index=False)
	
	# Flag-based analysis
	flag_counts = {}
	for result in validation_results:
		flag = result["analysis_flag"]
		flag_counts[flag] = flag_counts.get(flag, 0) + 1
	
	print(f"\n✓ CLIP Analysis Complete")
	print(f"  Total bboxes: {total_bboxes}")
	print(f"  Validated: {len(validation_results)}")
	print(f"  Analysis flags:")
	for flag, count in sorted(flag_counts.items()):
		pct = count / len(validation_results) * 100 if validation_results else 0
		print(f"    - {flag}: {count} ({pct:.1f}%)")
	print(f"  Skipped (patch): {skipped['patch']}")
	print(f"  Skipped (embedding): {skipped['embedding']}")
	print(f"  Skipped (text): {skipped['text']}")


def run_clip_validation():
	"""Run Stage 2: CLIP validation."""
	print("\n" + "="*70)
	print("STAGE 2: CLIP VALIDATION")
	print("="*70)
	
	if not OBJECTS_JSON_PATH.exists():
		raise FileNotFoundError(f"YOLO detections not found: {OBJECTS_JSON_PATH}")
	if not IMAGES_DIR.exists():
		raise NotADirectoryError(f"Images folder not found: {IMAGES_DIR}")
	
	print(f"Loading CLIP model...")
	processor, model, device = load_clip_model()
	
	print("Validating YOLO detections...")
	validation_results, total_bboxes, skipped = validate_detections(processor, model, device)
	
	save_validation_results(validation_results, total_bboxes, skipped)


def main():
	"""Run both stages."""
	try:
		run_yolo_detection()
		run_clip_validation()
		
		print("\n" + "="*70)
		print("✓ Pipeline completed successfully!")
		print("="*70 + "\n")
		
	except Exception as e:
		print(f"\n❌ Error: {e}")
		import traceback
		traceback.print_exc()


if __name__ == "__main__":
	main()
