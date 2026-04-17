#!/usr/bin/env python3
"""Extract climate-relevant visual attributes from images using CLIP and image analysis.

This script analyzes each image to detect environmental and climate-related visual signals
such as smoke, fire, water, floods, ice, deforestation, pollution, etc.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "dataset" / "annotated.csv"
IMAGES_DIR = ROOT_DIR / "dataset" / "images_for_annotation"
FEATURES_DIR = ROOT_DIR / "dataset" / "features"

IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".webp"]
MODEL_NAME = "openai/clip-vit-base-patch32"

ATTRIBUTES_JSON_PATH = FEATURES_DIR / "climate_attributes.json"
ATTRIBUTES_INDEX_PATH = FEATURES_DIR / "attributes_index.csv"
CONFIG_PATH = FEATURES_DIR / "climate_attributes_config.json"

# Climate attributes to detect
CLIMATE_ATTRIBUTES = [
	"smoke",
	"fire",
	"water",
	"flood",
	"ice",
	"ice_melting",
	"deforestation",
	"dry_land",
	"greenery",
	"pollution",
	"urbanization",
	"industrial_activity",
]

# Text descriptions for CLIP attribute detection
ATTRIBUTE_PROMPTS = {
	"smoke": "smoke, industrial pollution, air pollution, burning emissions, smoky air",
	"fire": "wildfire, burning forests, flames, forest fire, burning",
	"water": "rivers, oceans, lakes, water, water bodies, aquatic",
	"flood": "water overflow, submerged areas, flooding, flood disaster, inundated",
	"ice": "snow, glaciers, frozen surfaces, icy, snowcapped",
	"ice_melting": "melting glaciers, disappearing ice, ice melt, glacier retreat",
	"deforestation": "cut trees, cleared forests, logging, deforestation, tree removal",
	"dry_land": "drought, desertification, cracked soil, dry land, arid",
	"greenery": "healthy forests, vegetation, lush, natural environment, green trees",
	"pollution": "smog, dirty air, waste, contaminated, pollution, litter",
	"urbanization": "buildings, roads, cities, infrastructure, urban development",
	"industrial_activity": "factories, chimneys, power plants, industrial emissions, industrial",
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


def load_clip_model() -> tuple[CLIPProcessor, CLIPModel, str]:
	"""Load CLIP model and processor."""
	device = "cuda" if torch.cuda.is_available() else "cpu"
	processor = CLIPProcessor.from_pretrained(MODEL_NAME)
	model = CLIPModel.from_pretrained(MODEL_NAME)
	model.to(device)
	model.eval()
	return processor, model, device


def detect_attributes_clip(
	image_path: Path,
	processor: CLIPProcessor,
	model: CLIPModel,
	device: str,
	threshold: float = 0.5,
) -> dict[str, int]:
	"""Detect climate attributes using CLIP text-image matching."""
	image = Image.open(image_path).convert("RGB")
	
	attributes = {}
	
	with torch.no_grad():
		# Prepare image
		image_inputs = processor(images=image, return_tensors="pt")
		image_inputs = {key: value.to(device) for key, value in image_inputs.items()}
		image_features = model.get_image_features(**image_inputs)
		# Handle both tensor and output object
		if hasattr(image_features, 'pooler_output'):
			image_features = image_features.pooler_output
		image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
		
		# Check each attribute
		for attr, description in ATTRIBUTE_PROMPTS.items():
			text_inputs = processor(text=description, return_tensors="pt", padding=True)
			text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
			text_features = model.get_text_features(**text_inputs)
			# Handle both tensor and output object
			if hasattr(text_features, 'pooler_output'):
				text_features = text_features.pooler_output
			text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
			
			# Compute similarity
			similarity = (image_features @ text_features.T).item()
			
			# Threshold-based detection
			attributes[attr] = 1 if similarity > threshold else 0
	
	return attributes


def analyze_brightness_and_color(image_path: Path) -> tuple[str, str]:
	"""Analyze brightness level and color dominance."""
	image = cv2.imread(str(image_path))
	if image is None:
		return "medium", "natural"
	
	# Convert to HSV for better color analysis
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
	
	# Brightness level
	brightness = v.mean()
	if brightness < 85:
		brightness_level = "low"
	elif brightness < 170:
		brightness_level = "medium"
	else:
		brightness_level = "high"
	
	# Color dominance
	green_pixels = ((h >= 35) & (h <= 85)).sum()
	gray_pixels = (s < 30).sum()  # Low saturation = grayish
	dark_pixels = (v < 85).sum()  # Dark pixels
	total_pixels = h.shape[0] * h.shape[1]
	
	green_ratio = green_pixels / total_pixels
	gray_ratio = gray_pixels / total_pixels
	dark_ratio = dark_pixels / total_pixels
	
	# Determine dominance
	if green_ratio > 0.3:
		color_dominance = "greenish"
	elif gray_ratio > 0.4:
		color_dominance = "grayish"
	elif dark_ratio > 0.4:
		color_dominance = "dark"
	elif (s > 80).sum() / total_pixels > 0.3:  # High saturation with pollutants
		color_dominance = "polluted"
	else:
		color_dominance = "natural"
	
	return brightness_level, color_dominance


def main() -> None:
	if not DATASET_PATH.exists():
		raise FileNotFoundError(f"CSV not found: {DATASET_PATH}")
	if not IMAGES_DIR.exists():
		raise NotADirectoryError(f"Image folder not found: {IMAGES_DIR}")

	print("Loading CLIP model...")
	processor, model, device = load_clip_model()

	df = pd.read_csv(DATASET_PATH, dtype=str, keep_default_na=False)

	kept_rows: list[dict] = []
	all_attributes: list[dict] = []
	skipped = 0

	print("Extracting climate attributes...")
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
			# Detect attributes using CLIP
			attributes = detect_attributes_clip(image_path, processor, model, device, threshold=0.4)
			
			# Analyze brightness and color
			brightness_level, color_dominance = analyze_brightness_and_color(image_path)
			
			# Compile output
			attribute_record = {
				"id": row_id,
				**attributes,
				"brightness_level": brightness_level,
				"color_dominance": color_dominance,
			}
			
			all_attributes.append(attribute_record)
			kept_rows.append(row_dict)
		except Exception as e:
			print(f"  Error processing {row_id}: {e}")
			skipped += 1
			continue

	if not kept_rows:
		raise RuntimeError("No rows processed successfully.")

	FEATURES_DIR.mkdir(parents=True, exist_ok=True)

	# Save attributes as JSON
	with open(ATTRIBUTES_JSON_PATH, "w", encoding="utf-8") as f:
		json.dump(all_attributes, f, indent=2)

	# Save index
	index_df = pd.DataFrame(kept_rows)[["id"]]
	index_df.to_csv(ATTRIBUTES_INDEX_PATH, index=False)

	# Save config
	config = {
		"model": MODEL_NAME,
		"attributes": CLIMATE_ATTRIBUTES,
		"brightness_levels": ["low", "medium", "high"],
		"color_dominances": ["natural", "grayish", "dark", "greenish", "polluted"],
		"similarity_threshold": 0.4,
	}
	CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

	print(f"\nProcessed: {len(df)} rows")
	print(f"Kept: {len(kept_rows)} rows")
	print(f"Skipped: {skipped} rows")
	print(f"\nSaved: {ATTRIBUTES_JSON_PATH}")
	print(f"Saved: {ATTRIBUTES_INDEX_PATH}")
	print(f"Saved: {CONFIG_PATH}")

	# Print sample
	print("\n--- Sample attributes (first 2 images) ---")
	for attr_data in all_attributes[:2]:
		print(f"\nImage {attr_data['id']}:")
		print(f"  Brightness: {attr_data['brightness_level']}")
		print(f"  Color: {attr_data['color_dominance']}")
		detected = [k for k in CLIMATE_ATTRIBUTES if attr_data.get(k) == 1]
		if detected:
			print(f"  Detected: {', '.join(detected)}")
		else:
			print(f"  Detected: none")


if __name__ == "__main__":
	main()
