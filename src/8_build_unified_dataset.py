#!/usr/bin/env python3
"""Build unified multimodal dataset combining all extracted features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "dataset" / "annotated.csv"
FEATURES_DIR = ROOT_DIR / "dataset" / "features"

# Input feature files
CLIP_EMB_PATH = FEATURES_DIR / "clip_image_embeddings.npy"
OBJ_COUNTS_PATH = FEATURES_DIR / "object_counts.npy"
SCENE_CLASS_PATH = FEATURES_DIR / "scene_classifications.json"
ATTRS_PATH = FEATURES_DIR / "climate_attributes.json"
PREMISE_EMB_PATH = FEATURES_DIR / "premise_embeddings.npy"
FACT_RETR_PATH = FEATURES_DIR / "fact_retrievals.json"
CAPTIONS_PATH = FEATURES_DIR / "image_captions.json"

CLIP_INDEX_PATH = FEATURES_DIR / "feature_index.csv"

# Output
OUTPUT_DIR = ROOT_DIR / "dataset" / "unified_features"
UNIFIED_FEATURES_PATH = OUTPUT_DIR / "unified_features.npz"
METADATA_PATH = OUTPUT_DIR / "metadata.json"


def load_all_features() -> dict:
	"""Load all pre-extracted features."""
	features = {}
	
	print("Loading features...")
	
	clip_emb = np.load(CLIP_EMB_PATH)
	obj_counts = np.load(OBJ_COUNTS_PATH)
	premise_emb = np.load(PREMISE_EMB_PATH)
	
	features["clip_embeddings"] = clip_emb
	features["object_counts"] = obj_counts
	features["premise_embeddings"] = premise_emb
	
	print(f"  CLIP embeddings: {clip_emb.shape}")
	print(f"  Object counts: {obj_counts.shape}")
	print(f"  Premise embeddings: {premise_emb.shape}")
	
	clip_idx = pd.read_csv(CLIP_INDEX_PATH)
	
	with open(SCENE_CLASS_PATH) as f:
		scenes = json.load(f)
	with open(ATTRS_PATH) as f:
		attrs = json.load(f)
	with open(FACT_RETR_PATH) as f:
		fact_retr = json.load(f)
	with open(CAPTIONS_PATH) as f:
		captions = json.load(f)
	
	scenes_by_id = {s["id"]: s for s in scenes}
	attrs_by_id = {a["id"]: a for a in attrs}
	facts_by_id = {f["id"]: f for f in fact_retr}
	captions_by_id = {c["id"]: c for c in captions}
	
	# Scene one-hot
	scene_categories = ["urban", "rural", "industrial", "natural"]
	scene_onehot = np.zeros((len(clip_idx), len(scene_categories)), dtype=np.float32)
	
	for idx, row in clip_idx.iterrows():
		image_id = str(row["id"])
		if image_id in scenes_by_id:
			category = scenes_by_id[image_id]["classification"]["primary_category"]
			if category in scene_categories:
				cat_idx = scene_categories.index(category)
				scene_onehot[idx, cat_idx] = 1.0
	
	features["scene_onehot"] = scene_onehot
	print(f"  Scene one-hot: {scene_onehot.shape}")
	
	# Climate attributes
	climate_attrs = []
	brightness_map = {"low": 0, "medium": 1, "high": 2}
	color_map = {"natural": 0, "grayish": 1, "dark": 2, "greenish": 3, "polluted": 4}
	
	for idx, row in clip_idx.iterrows():
		image_id = str(row["id"])
		attr_vec = []
		
		if image_id in attrs_by_id:
			attr_data = attrs_by_id[image_id]
			for attr in ["smoke", "fire", "water", "flood", "ice", "ice_melting",
						 "deforestation", "dry_land", "greenery", "pollution",
						 "urbanization", "industrial_activity"]:
				attr_vec.append(float(attr_data.get(attr, 0)))
			
			brightness = brightness_map.get(attr_data.get("brightness_level", "medium"), 1)
			color = color_map.get(attr_data.get("color_dominance", "natural"), 0)
			attr_vec.extend([brightness, color])
		else:
			attr_vec = [0] * 14
		
		climate_attrs.append(attr_vec)
	
	climate_attrs = np.array(climate_attrs, dtype=np.float32)
	features["climate_attributes"] = climate_attrs
	print(f"  Climate attributes: {climate_attrs.shape}")
	
	features["image_ids"] = clip_idx["id"].values
	features["scenes_by_id"] = scenes_by_id
	features["facts_by_id"] = facts_by_id
	features["captions_by_id"] = captions_by_id
	
	return features


def create_unified_tensor(features: dict) -> dict:
	"""Concatenate all features into unified tensor."""
	
	print("\nCreating unified feature tensor...")
	
	unified = np.concatenate([
		features["clip_embeddings"],
		features["object_counts"],
		features["scene_onehot"],
		features["climate_attributes"],
		features["premise_embeddings"],
	], axis=1)
	
	unified = unified.astype(np.float32)
	
	feature_dims = {
		"clip_embeddings": 512,
		"object_counts": 18,
		"scene_onehot": 4,
		"climate_attributes": 14,
		"premise_embeddings": 384,
	}
	
	print(f"Unified feature shape: {unified.shape}")
	print(f"Total dimension: {sum(feature_dims.values())}")
	
	return {
		"unified": unified,
		"feature_dims": feature_dims,
		"image_ids": features["image_ids"],
	}


def load_targets(features: dict) -> dict:
	"""Load target conclusions and premises."""
	
	print("\nLoading targets...")
	
	df = pd.read_csv(DATASET_PATH, dtype=str, keep_default_na=False)
	df_by_id = {str(row["id"]): row for _, row in df.iterrows()}
	
	targets = {
		"image_ids": features["image_ids"],
		"premises": [],
		"conclusions": [],
	}
	
	for image_id in features["image_ids"]:
		if str(image_id) in df_by_id:
			row = df_by_id[str(image_id)]
			
			try:
				premises = json.loads(row.get("premises", "[]"))
			except:
				premises = []
			targets["premises"].append(premises)
			
			try:
				conclusions = json.loads(row.get("conclusions", "[]"))
			except:
				conclusions = []
			targets["conclusions"].append(conclusions)
		else:
			targets["premises"].append([])
			targets["conclusions"].append([])
	
	return targets


def save_unified_dataset(unified_data: dict, targets: dict, features: dict) -> None:
	"""Save unified dataset."""
	
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	
	print("\nSaving unified dataset...")
	
	np.savez_compressed(
		UNIFIED_FEATURES_PATH,
		unified_features=unified_data["unified"],
		image_ids=unified_data["image_ids"],
	)
	print(f"  Saved: {UNIFIED_FEATURES_PATH}")
	
	metadata = {
		"num_images": len(unified_data["image_ids"]),
		"total_feature_dim": int(unified_data["unified"].shape[1]),
		"feature_dims": unified_data["feature_dims"],
		"scene_categories": ["urban", "rural", "industrial", "natural"],
		"climate_attributes": [
			"smoke", "fire", "water", "flood", "ice", "ice_melting",
			"deforestation", "dry_land", "greenery", "pollution",
			"urbanization", "industrial_activity", "brightness_level", "color_dominance"
		],
	}
	
	with open(METADATA_PATH, "w") as f:
		json.dump(metadata, f, indent=2)
	print(f"  Saved: {METADATA_PATH}")
	
	targets_path = OUTPUT_DIR / "targets.json"
	targets_export = {
		"image_ids": [str(x) for x in targets["image_ids"]],
		"premises": targets["premises"],
		"conclusions": targets["conclusions"],
	}
	with open(targets_path, "w") as f:
		json.dump(targets_export, f, indent=2)
	print(f"  Saved: {targets_path}")
	
	captions_path = OUTPUT_DIR / "captions.json"
	captions_export = []
	for image_id in targets["image_ids"]:
		if str(image_id) in features["captions_by_id"]:
			cap = features["captions_by_id"][str(image_id)]
			captions_export.append({"id": str(image_id), "caption": cap.get("caption", "")})
	with open(captions_path, "w") as f:
		json.dump(captions_export, f, indent=2)
	print(f"  Saved: {captions_path}")
	
	facts_path = OUTPUT_DIR / "retrieved_facts.json"
	facts_export = []
	for image_id in targets["image_ids"]:
		if str(image_id) in features["facts_by_id"]:
			fact = features["facts_by_id"][str(image_id)]
			facts_export.append({
				"id": str(image_id),
				"retrieved_facts": fact.get("retrieved_facts", [])
			})
	with open(facts_path, "w") as f:
		json.dump(facts_export, f, indent=2)
	print(f"  Saved: {facts_path}")


def main() -> None:
	features = load_all_features()
	unified_data = create_unified_tensor(features)
	targets = load_targets(features)
	save_unified_dataset(unified_data, targets, features)
	
	print("\n✅ Unified dataset created successfully!")
	print(f"\nOutput: {OUTPUT_DIR}")
	print(f"Samples: {unified_data['unified'].shape[0]}")
	print(f"Feature dimension: {unified_data['unified'].shape[1]}")


if __name__ == "__main__":
	main()
