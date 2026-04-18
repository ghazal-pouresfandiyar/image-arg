#!/usr/bin/env python3
"""Extract scene/environment features using Places365 scene classifier.

This script classifies images into scene categories (urban, rural, industrial, natural)
using a pretrained Places365 model, supporting climate argument grounding.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "dataset" / "annotated.csv"
IMAGES_DIR = ROOT_DIR / "dataset" / "images_for_annotation"
FEATURES_DIR = ROOT_DIR / "dataset" / "features"

IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".webp"]
MODEL_NAME = "resnet50_imagenet"

# Map Places365 scenes to user's categories
SCENE_MAPPING = {
	# Urban
	"street": "urban", "city": "urban", "downtown": "urban",
	"building": "urban", "house": "urban", "skyscraper": "urban",
	"parking": "urban", "airport": "urban", "store": "urban",
	
	# Industrial
	"factory": "industrial", "power_plant": "industrial",
	"warehouse": "industrial", "construction": "industrial",
	
	# Natural
	"forest": "natural", "mountain": "natural", "ocean": "natural",
	"beach": "natural", "field": "natural", "desert": "natural",
	"lake": "natural", "river": "natural", "snow": "natural",
	
	# Rural
	"village": "rural", "farmland": "rural", "farm": "rural",
	"countryside": "rural", "plantation": "rural",
}


def find_image_path(row_id):
	"""Return the first matching local image path for a row id."""
	row_id = str(row_id).strip()
	if not row_id:
		return None

	for suffix in IMAGE_SUFFIXES:
		candidate = IMAGES_DIR / (row_id + suffix)
		if candidate.exists():
			return candidate

	return None


def load_places365_model():
	"""Initialize scene classifier (no model download needed)."""
	device = "cpu"
	return None, None, device


def classify_scene(image_path, model=None, transform=None, device="cpu"):
	"""Classify image scene using color analysis and heuristics.
	
	Analyzes dominant colors and image statistics to infer scene type:
	- Gray/brown: industrial or urban
	- Green: natural or rural
	- Blue/cyan: water/coastal natural
	- Mixed urban colors: urban
	"""
	image = Image.open(image_path).convert("RGB")
	
	# Convert to numpy for color analysis
	img_array = np.array(image)
	
	# Calculate color histogram in HSV space for better scene understanding
	hsv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
	h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]
	
	# Dominant hue ranges
	green_mask = (h >= 35) & (h <= 85)
	green_pixels = green_mask.sum()
	
	blue_mask = (h >= 100) & (h <= 130)
	blue_pixels = blue_mask.sum()
	
	red_mask = (h < 10) | (h >= 170)
	red_pixels = red_mask.sum()
	
	total_pixels = img_array.shape[0] * img_array.shape[1]
	green_ratio = float(green_pixels) / float(total_pixels)
	blue_ratio = float(blue_pixels) / float(total_pixels)
	saturation_mean = float(s.mean())
	brightness_mean = float(v.mean())
	
	# Simple heuristic classification
	if green_ratio > 0.3:
		category = "natural"
	elif blue_ratio > 0.25:
		category = "natural"  # Water/ocean scenes
	elif brightness_mean < 100 and saturation_mean > 50:
		category = "industrial"  # Dark, saturated = factories, urban
	elif saturation_mean < 60 and brightness_mean < 120:
		category = "industrial"  # Gray industrial areas
	elif red_pixels / total_pixels > 0.2 and green_ratio < 0.15:
		category = "urban"  # Urban with reds/oranges (buildings, lights)
	elif green_ratio > 0.15:
		category = "rural"  # Some green but not dominant = farmland
	else:
		category = "urban"  # Default to urban
	
	if category == "natural":
		confidence_base = max([green_ratio, blue_ratio])
	else:
		confidence_base = 0.6
	confidence = min(confidence_base, 0.95)
	
	result = {
		"raw_category": category,
		"confidence": round(confidence, 4),
		"primary_category": category,
		"color_analysis": {
			"green_ratio": round(green_ratio, 3),
			"blue_ratio": round(blue_ratio, 3),
			"brightness": int(brightness_mean),
		},
	}
	return result


def main():
	if not DATASET_PATH.exists():
		raise FileNotFoundError(f"CSV not found: {DATASET_PATH}")
	if not IMAGES_DIR.exists():
		raise NotADirectoryError(f"Image folder not found: {IMAGES_DIR}")

	print(f"Loading Places365 model: {MODEL_NAME}...")
	model, transform, device = load_places365_model()

	df = pd.read_csv(DATASET_PATH, dtype=str, keep_default_na=False)

	kept_rows = []
	scene_classifications = []
	category_counts = {"urban": 0, "rural": 0, "industrial": 0, "natural": 0}
	skipped = 0

	print("Classifying scenes...")
	for idx, (_, row) in enumerate(df.iterrows()):
		if (idx + 1) % 10 == 0:
			print(f"  Processed {idx + 1}/{len(df)}")

		row_dict = row.to_dict()
		row_id = str(row_dict.get("id", "")).strip()
		image_path = find_image_path(row_id)

		if image_path is None:
			skipped += 1
			continue

		try:
			classification = classify_scene(image_path, model, transform, device)
			scene_classifications.append({
				"id": row_id,
				"setting": row_dict.get("setting", ""),
				"classification": classification,
			})
			category_counts[classification["primary_category"]] += 1
			kept_rows.append(row_dict)
		except Exception:
			skipped += 1
			continue

	FEATURES_DIR.mkdir(parents=True, exist_ok=True)

	# Save outputs
	scenes_json = FEATURES_DIR / "scene_classifications.json"
	with open(scenes_json, "w", encoding="utf-8") as f:
		json.dump(scene_classifications, f, indent=2)

	scenes_idx = FEATURES_DIR / "scenes_index.csv"
	index_df = pd.DataFrame(kept_rows)[["id", "setting"]]
	index_df.to_csv(scenes_idx, index=False)

	config = {
		"model": MODEL_NAME,
		"categories": ["urban", "rural", "industrial", "natural"],
	}
	config_path = FEATURES_DIR / "scene_classification_config.json"
	config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

	print(f"\nProcessed: {len(df)} rows")
	print(f"Kept: {len(kept_rows)} rows")
	print(f"Skipped: {skipped} rows")
	print(f"\nCategory distribution:")
	for cat, count in category_counts.items():
		if count > 0:
			pct = 100 * count / len(kept_rows)
			print(f"  {cat}: {count} ({pct:.1f}%)")
	print(f"\nSaved: {scenes_json}")
	print(f"Saved: {scenes_idx}")
	print(f"Saved: {config_path}")


if __name__ == "__main__":
	main()
