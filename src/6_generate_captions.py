#!/usr/bin/env python3
"""Generate factual, descriptive captions for climate dataset images.

This script creates one-sentence captions per image describing what is visually present,
focusing on main objects, visible actions, and environment type.
"""

import json
from pathlib import Path

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

CAPTIONS_JSON_PATH = FEATURES_DIR / "image_captions.json"
CAPTIONS_INDEX_PATH = FEATURES_DIR / "captions_index.csv"
CONFIG_PATH = FEATURES_DIR / "captions_config.json"

# Load pre-extracted features
OBJECTS_JSON_PATH = FEATURES_DIR / "detected_objects.json"
SCENE_JSON_PATH = FEATURES_DIR / "scene_classifications.json"
ATTRIBUTES_JSON_PATH = FEATURES_DIR / "climate_attributes.json"

# Caption templates based on detected features
CAPTION_TEMPLATES = {
	"smoke_industrial": "A {scene_type} area with {object_count} emitting thick smoke.",
	"smoke_natural": "A {scene_type} landscape with smoke rising into the air.",
	"fire": "A {scene_type} area affected by fire with visible flames and smoke.",
	"flood_urban": "A {scene_type} street and buildings submerged under floodwater.",
	"flood_natural": "A {scene_type} area covered with floodwater.",
	"ice_melting": "A glacier-covered mountain with large areas of melting ice.",
	"ice_snow": "A {scene_type} landscape covered with snow and ice.",
	"deforestation": "A {scene_type} area with felled trees and cleared forest land.",
	"greenery": "A {scene_type} landscape with lush vegetation and healthy green trees.",
	"dry_land": "A {scene_type} area of dry, arid land with visible drought conditions.",
	"water_natural": "A {scene_type} landscape with visible water bodies, rivers, or ocean.",
	"urbanization": "A {scene_type} area with buildings, roads, and urban infrastructure.",
	"industrial": "An industrial area with factories and infrastructure.",
	"generic": "A {scene_type} area with {objects}.",
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


def load_clip_model():
	"""Load CLIP model."""
	device = "cuda" if torch.cuda.is_available() else "cpu"
	processor = CLIPProcessor.from_pretrained(MODEL_NAME)
	model = CLIPModel.from_pretrained(MODEL_NAME)
	model.to(device)
	model.eval()
	return processor, model, device


def classify_image_state(
	processor: CLIPProcessor,
	model: CLIPModel,
	device: str,
	image_path: Path,
) -> str:
	"""Classify high-level image state using CLIP."""
	image = Image.open(image_path).convert("RGB")
	
	states = {
		"pristine_natural": "untouched nature, healthy forest, clean water, green landscape",
		"degraded": "damaged environment, barren land, polluted area, destruction",
		"industrial_active": "factory, power plant, industrial activity, infrastructure",
		"urban_developed": "city, urban development, buildings, roads, civilization",
		"disaster": "disaster, extreme weather, flooding, fire, emergency",
	}
	
	max_sim = -1
	best_state = "generic"
	
	with torch.no_grad():
		image_inputs = processor(images=image, return_tensors="pt")
		image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
		image_features = model.get_image_features(**image_inputs)
		if hasattr(image_features, 'pooler_output'):
			image_features = image_features.pooler_output
		image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
		
		for state, desc in states.items():
			text_inputs = processor(text=desc, return_tensors="pt", padding=True)
			text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
			text_features = model.get_text_features(**text_inputs)
			if hasattr(text_features, 'pooler_output'):
				text_features = text_features.pooler_output
			text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
			
			sim = (image_features @ text_features.T).item()
			if sim > max_sim:
				max_sim = sim
				best_state = state
	
	return best_state


def generate_caption(
	row_id: str,
	objects_data: dict,
	scene_data: dict,
	attributes_data: dict,
	image_state: str,
) -> str:
	"""Generate a factual caption from detected features."""
	
	# Extract data
	detected_objects = objects_data.get("detected_objects", [])
	object_counts = objects_data.get("object_counts", {})
	scene_cat = scene_data.get("classification", {}).get("primary_category", "natural")
	attrs = attributes_data
	
	# Determine main subject and action
	if attrs.get("fire") == 1:
		template = CAPTION_TEMPLATES["fire"]
		return f"A {scene_cat} area affected by fire with visible flames and smoke."
	
	if attrs.get("smoke") == 1:
		if attrs.get("industrial_activity") == 1:
			# Find what's emitting smoke
			if detected_objects:
				obj_str = ", ".join(detected_objects[:2])
				return f"An industrial area with {obj_str} emitting thick smoke."
			return "An industrial facility emitting thick smoke into the air."
		else:
			return f"A {scene_cat} landscape with smoke rising into the air."
	
	if attrs.get("flood") == 1:
		if scene_cat == "urban":
			return "A city street and buildings submerged under floodwater."
		else:
			return f"A {scene_cat} area covered with floodwater."
	
	if attrs.get("ice_melting") == 1:
		return "A glacier-covered mountain with large areas of melting ice."
	
	if attrs.get("ice") == 1:
		return f"A {scene_cat} landscape covered with snow and ice."
	
	if attrs.get("deforestation") == 1:
		return "A forest area with felled trees and cleared land."
	
	if attrs.get("greenery") == 1:
		return f"A {scene_cat} landscape with lush vegetation and healthy green trees."
	
	if attrs.get("dry_land") == 1:
		return f"A {scene_cat} area of dry, arid land with visible drought conditions."
	
	if attrs.get("water") == 1:
		return f"A {scene_cat} landscape with visible water bodies and aquatic environments."
	
	if attrs.get("urbanization") == 1:
		return f"A {scene_cat} area with buildings, roads, and urban infrastructure."
	
	if attrs.get("industrial_activity") == 1:
		return "An industrial area with factories, chimneys, and infrastructure."
	
	# Generic captions based on scene and objects
	if detected_objects:
		obj_str = ", ".join(detected_objects[:2])
		return f"A {scene_cat} area featuring {obj_str}."
	
	return f"A {scene_cat} landscape."


def main() -> None:
	if not DATASET_PATH.exists():
		raise FileNotFoundError(f"CSV not found: {DATASET_PATH}")
	if not IMAGES_DIR.exists():
		raise NotADirectoryError(f"Image folder not found: {IMAGES_DIR}")
	if not all(p.exists() for p in [OBJECTS_JSON_PATH, SCENE_JSON_PATH, ATTRIBUTES_JSON_PATH]):
		raise FileNotFoundError("Required feature files not found. Run other extraction scripts first.")

	# Load pre-extracted features
	with open(OBJECTS_JSON_PATH) as f:
		objects_list = json.load(f)
	objects_by_id = {obj["id"]: obj for obj in objects_list}

	with open(SCENE_JSON_PATH) as f:
		scenes_list = json.load(f)
	scenes_by_id = {s["id"]: s for s in scenes_list}

	with open(ATTRIBUTES_JSON_PATH) as f:
		attrs_list = json.load(f)
	attrs_by_id = {a["id"]: a for a in attrs_list}

	df = pd.read_csv(DATASET_PATH, dtype=str, keep_default_na=False)

	print("Loading CLIP model...")
	processor, model, device = load_clip_model()

	kept_rows = []
	all_captions = []
	skipped = 0

	print("Generating captions...")
	for idx, (_, row) in enumerate(df.iterrows()):
		if (idx + 1) % 10 == 0:
			print(f"  Processed {idx + 1}/{len(df)}")

		row_id = str(row.get("id", "")).strip()
		image_path = find_image_path(row_id)

		if image_path is None or row_id not in objects_by_id:
			skipped += 1
			continue

		try:
			# Get pre-extracted features
			objects_data = objects_by_id[row_id]
			scene_data = scenes_by_id.get(row_id, {})
			attributes_data = attrs_by_id.get(row_id, {})

			# Generate caption
			caption = generate_caption(row_id, objects_data, scene_data, attributes_data, None)

			all_captions.append({
				"id": row_id,
				"caption": caption,
			})
			kept_rows.append(row)
		except Exception as e:
			print(f"  Error {row_id}: {e}")
			skipped += 1
			continue

	if not kept_rows:
		raise RuntimeError("No captions generated.")

	FEATURES_DIR.mkdir(parents=True, exist_ok=True)

	# Save captions
	with open(CAPTIONS_JSON_PATH, "w", encoding="utf-8") as f:
		json.dump(all_captions, f, indent=2)

	# Save index
	index_df = pd.DataFrame(kept_rows)[["id"]]
	index_df.to_csv(CAPTIONS_INDEX_PATH, index=False)

	# Save config
	config = {
		"model": "rule-based + CLIP ensemble",
		"output_format": "single factual sentence",
		"strategy": "priority-based: fire > smoke > flood > ice > deforestation > greenery > dry_land > water > urbanization > generic",
	}
	CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

	print(f"\nProcessed: {len(df)} rows")
	print(f"Captions generated: {len(all_captions)}")
	print(f"Skipped: {skipped}")
	print(f"\nSaved: {CAPTIONS_JSON_PATH}")
	print(f"Saved: {CAPTIONS_INDEX_PATH}")
	print(f"Saved: {CONFIG_PATH}")

	# Print sample
	print("\n--- Sample captions (first 5 images) ---")
	for cap_data in all_captions[:5]:
		print(f"ID {cap_data['id']}: {cap_data['caption']}")


if __name__ == "__main__":
	main()
