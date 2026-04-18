#!/usr/bin/env python3
"""Ground premises to detected objects.

This script aligns text premises with object detections, verifying whether
premises are grounded in image evidence. E.g., premise "Polar bears are losing habitat"
is grounded if bears are detected in the image.
"""

import json
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "dataset" / "annotated.csv"
OBJECTS_JSON_PATH = ROOT_DIR / "dataset" / "features" / "detected_objects.json"
FEATURES_DIR = ROOT_DIR / "dataset" / "features"

GROUNDING_OUTPUT_PATH = FEATURES_DIR / "premise_groundings.json"

# Simple mapping: premise keywords → object classes to match
# Expand this as needed for your domain
GROUNDING_KEYWORDS = {
	"polar bear": ["bear"],
	"bear": ["bear"],
	"fish": ["person"],  # Placeholder; refine based on your data
	"tree": ["horse"],  # Placeholder
	"car": ["car"],
	"truck": ["truck"],
	"person": ["person"],
	"animal": ["dog", "cat", "bird", "bear", "horse", "sheep"],
	"bird": ["bird"],
	"dog": ["dog"],
	"cat": ["cat"],
	"horse": ["horse"],
	"sheep": ["sheep"],
}


def extract_premise_keywords(premise):
	"""Extract grounding keywords from a premise."""
	premise_lower = premise.lower()
	matched = []
	for keyword in GROUNDING_KEYWORDS.keys():
		if keyword in premise_lower:
			matched.append(keyword)
	return matched


def ground_premise(premise, detected_objects):
	"""Check if premise is grounded in detected objects.
	
	Returns:
		dict with:
		- "premise": original premise
		- "grounded": bool (any object match found)
		- "matched_keywords": list of keywords found in premise
		- "matched_objects": list of detected object classes
		- "object_counts": dict of matched object counts
	"""
	keywords = extract_premise_keywords(premise)
	matched_objects = []
	matched_counts = {}

	for keyword in keywords:
		target_classes = GROUNDING_KEYWORDS[keyword]
		for obj_class in target_classes:
			if obj_class in detected_objects and detected_objects[obj_class] > 0:
				matched_objects.append(obj_class)
				matched_counts[obj_class] = detected_objects[obj_class]

	unique_objects = []
	for obj in matched_objects:
		if obj not in unique_objects:
			unique_objects.append(obj)
	matched_objects = unique_objects

	return {
		"premise": premise,
		"grounded": len(matched_objects) > 0,
		"matched_keywords": keywords,
		"matched_objects": matched_objects,
		"object_counts": matched_counts,
	}


def main():
	if not DATASET_PATH.exists():
		raise FileNotFoundError(f"CSV not found: {DATASET_PATH}")
	if not OBJECTS_JSON_PATH.exists():
		raise FileNotFoundError(f"Objects JSON not found: {OBJECTS_JSON_PATH}")

	# Load annotated dataset
	df = pd.read_csv(DATASET_PATH, dtype=str, keep_default_na=False)

	# Load detected objects
	with open(OBJECTS_JSON_PATH, "r", encoding="utf-8") as f:
		all_detections = json.load(f)

	# Build id → detections map
	detections_by_id = {obj["id"]: obj for obj in all_detections}

	# Ground premises
	groundings = []
	grounded_count = 0
	ungrounded_count = 0

	for idx, (_, row) in enumerate(df.iterrows()):
		row_id = str(row.get("id", "")).strip()
		premises_str = str(row.get("premises", "")).strip()

		if row_id not in detections_by_id or not premises_str:
			continue

		detection_data = detections_by_id[row_id]
		detected_objects = detection_data["object_counts"]

		# Parse premises (stored as JSON array string)
		try:
			premises_list = json.loads(premises_str)
		except json.JSONDecodeError:
			premises_list = [premises_str]

		row_groundings = {
			"id": row_id,
			"premises": premises_list,
			"detected_objects": detection_data["detected_objects"],
			"object_counts": detected_objects,
			"premise_groundings": [],
		}

		for premise in premises_list:
			grounding = ground_premise(premise, detected_objects)
			row_groundings["premise_groundings"].append(grounding)

			if grounding["grounded"]:
				grounded_count += 1
			else:
				ungrounded_count += 1

		groundings.append(row_groundings)

	# Save groundings
	FEATURES_DIR.mkdir(parents=True, exist_ok=True)
	with open(GROUNDING_OUTPUT_PATH, "w", encoding="utf-8") as f:
		json.dump(groundings, f, indent=2)

	print(f"Processed {len(groundings)} images")
	print(f"Total premises grounded: {grounded_count}")
	print(f"Total premises ungrounded: {ungrounded_count}")
	print(f"Grounding rate: {100 * grounded_count / (grounded_count + ungrounded_count):.1f}%\n")
	print(f"Saved premise groundings: {GROUNDING_OUTPUT_PATH}")

	# Print sample
	print("\n--- Sample grounding (first 3 images) ---")
	for grounding_data in groundings[:3]:
		print(f"\nImage ID: {grounding_data['id']}")
		print(f"Detected objects: {grounding_data['detected_objects']}")
		for pg in grounding_data["premise_groundings"]:
			status = "✓ GROUNDED" if pg["grounded"] else "✗ NOT GROUNDED"
			print(f"  {status}: {pg['premise']}")
			if pg["matched_objects"]:
				print(f"    → Matched via: {pg['matched_objects']} {pg['object_counts']}")


if __name__ == "__main__":
	main()
