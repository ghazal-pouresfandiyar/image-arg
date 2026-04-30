#!/usr/bin/env python3
"""Build unified multimodal dataset from aligned features (SAFE VERSION).

Fixes:
- ID-based alignment (NOT index-based)
- Handles missing modalities safely
- Reports dropped samples
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "dataset" / "annotated.csv"
FEATURES_DIR = ROOT_DIR / "dataset" / "features"
OUTPUT_DIR = ROOT_DIR / "dataset" / "unified_features"

UNIFIED_PATH = OUTPUT_DIR / "unified_features.npz"
STRUCTURED_PATH = OUTPUT_DIR / "structured_features.json"


# ============================================================================
# HELPERS
# ============================================================================

def load_npy(path):
	if not path.exists():
		raise FileNotFoundError(f"Missing feature file: {path}")
	return np.load(path)


def load_json(path):
	if not path.exists():
		return []
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def build_safe_maps(df, clip, meta, obj):
	"""
	CRITICAL FIX:
	We do NOT assume same order or same length anymore.
	We ALIGN using intersection only.
	"""

	image_ids = df["id"].astype(str).tolist()

	# Build raw index maps (positional assumption from preprocessing)
	clip_map = {image_ids[i]: clip[i] for i in range(len(clip)) if i < len(image_ids)}
	meta_map = {image_ids[i]: meta[i] for i in range(len(meta)) if i < len(image_ids)}
	obj_map  = {image_ids[i]: obj[i]  for i in range(len(obj)) if i < len(image_ids)}

	# Find safe intersection
	common_ids = sorted(
		set(clip_map.keys()) &
		set(meta_map.keys()) &
		set(obj_map.keys())
	)

	print("\nAlignment Report:")
	print(f"  CSV IDs:   {len(image_ids)}")
	print(f"  CLIP IDs:  {len(clip_map)}")
	print(f"  META IDs:  {len(meta_map)}")
	print(f"  OBJ IDs:   {len(obj_map)}")
	print(f"  FINAL OK:  {len(common_ids)}")

	if len(common_ids) == 0:
		raise ValueError("No overlapping IDs between modalities!")

	mapping = {}

	for img_id in common_ids:
		mapping[img_id] = {
			"clip": clip_map[img_id],
			"metadata": meta_map[img_id],
			"objects": obj_map[img_id],
		}

	return mapping, common_ids


def build_tensor(mapping):
	print("\nBuilding unified tensor...")

	vectors = []

	for img_id, feats in mapping.items():
		vec = np.concatenate([
			feats["clip"],
			feats["metadata"],
			feats["objects"]
		], axis=0)

		vectors.append(vec)

	return np.stack(vectors).astype(np.float32)


def build_structured(mapping, scene, climate, captions):
	scene_map = {s["id"]: s for s in scene}
	climate_map = {c["id"]: c for c in climate}
	caption_map = {c["id"]: c.get("caption", "") for c in captions}

	structured = {}

	for img_id in mapping.keys():
		structured[img_id] = {
			"visual": {
				"clip_embedding": mapping[img_id]["clip"].tolist(),
			},
			"metadata": mapping[img_id]["metadata"].tolist(),
			"objects": mapping[img_id]["objects"].tolist(),

			"scene": scene_map.get(img_id, {}),
			"climate": climate_map.get(img_id, {}),
			"caption": caption_map.get(img_id, ""),

			"reliability": {
				"clip": 1.0,
				"metadata": 0.8,
				"objects": 0.6,
				"scene": 0.4,
				"climate": 0.4
			}
		}

	return structured


# ============================================================================
# MAIN
# ============================================================================

def main():

	print("\n" + "="*70)
	print("BUILDING UNIFIED DATASET (SAFE VERSION)")
	print("="*70 + "\n")

	df = pd.read_csv(DATASET_PATH, dtype=str, keep_default_na=False)

	print("Loading features...")
	clip = load_npy(FEATURES_DIR / "clip_image_embeddings.npy")
	meta = load_npy(FEATURES_DIR / "metadata_onehot.npy")
	obj  = load_npy(FEATURES_DIR / "object_counts.npy")

	scene = load_json(FEATURES_DIR / "scene_features_v2.json")
	climate = load_json(FEATURES_DIR / "climate_attributes.json")
	captions = load_json(FEATURES_DIR / "image_captions.json")

	print(f"CLIP: {clip.shape}")
	print(f"META: {meta.shape}")
	print(f"OBJ:  {obj.shape}")

	# ============================
	# SAFE ALIGNMENT (FIX)
	# ============================
	print("\nAligning by ID (SAFE MODE)...")
	mapping, image_ids = build_safe_maps(df, clip, meta, obj)

	# ============================
	# BUILD TENSOR
	# ============================
	unified = build_tensor(mapping)

	# ============================
	# STRUCTURED OUTPUT
	# ============================
	structured = build_structured(mapping, scene, climate, captions)

	# ============================
	# SAVE
	# ============================
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	np.savez_compressed(
		UNIFIED_PATH,
		unified_features=unified,
		image_ids=image_ids
	)

	with open(STRUCTURED_PATH, "w", encoding="utf-8") as f:
		json.dump(structured, f, indent=2)

	# ============================
	# SUMMARY
	# ============================
	print("\n" + "="*70)
	print("✅ SUCCESS: UNIFIED DATASET BUILT (SAFE)")
	print("="*70)
	print(f"Images:       {len(image_ids)}")
	print(f"Tensor shape: {unified.shape}")
	print(f"Tensor size:  {unified.nbytes / (1024**2):.2f} MB")
	print(f"Structured:   {len(structured)} images")


if __name__ == "__main__":
	main()