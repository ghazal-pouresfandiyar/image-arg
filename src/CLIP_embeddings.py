#!/usr/bin/env python3
"""Extract CLIP image features and export them as .npy files.

This script reads dataset/final.csv, finds each image by id in
dataset/images_for_annotation, extracts a CLIP embedding, builds one-hot
metadata features, concatenates them, and saves everything under
dataset/features.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "dataset" / "final.csv"
IMAGES_DIR = ROOT_DIR / "dataset" / "images_for_annotation"
FEATURES_DIR = ROOT_DIR / "dataset" / "features"
MODEL_NAME = "openai/clip-vit-base-patch32"

METADATA_COLUMNS = ["animals", "consequences", "climateaction", "type", "setting"]
IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".webp"]

IMAGE_FEATURES_PATH = FEATURES_DIR / "clip_image_embeddings.npy"
METADATA_FEATURES_PATH = FEATURES_DIR / "metadata_onehot.npy"
COMBINED_FEATURES_PATH = FEATURES_DIR / "multimodal_features.npy"
INDEX_PATH = FEATURES_DIR / "feature_index.csv"
CONFIG_PATH = FEATURES_DIR / "feature_config.json"


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


def build_metadata_onehot(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
	"""Convert metadata columns to one-hot features."""
	onehot_df = pd.get_dummies(df[METADATA_COLUMNS], prefix=METADATA_COLUMNS, dtype=float)
	matrix = onehot_df.to_numpy(dtype=np.float32)
	return matrix, list(onehot_df.columns)


def load_clip_model() -> tuple[CLIPProcessor, CLIPModel, str]:
	"""Load CLIP model and processor once."""
	device = "cuda" if torch.cuda.is_available() else "cpu"
	processor = CLIPProcessor.from_pretrained(MODEL_NAME)
	model = CLIPModel.from_pretrained(MODEL_NAME)
	model.to(device)
	model.eval()
	return processor, model, device


def extract_clip_embedding(
	image_path: Path,
	processor: CLIPProcessor,
	model: CLIPModel,
	device: str,
) -> np.ndarray:
	"""Extract one normalized CLIP image embedding."""
	image = Image.open(image_path).convert("RGB")
	inputs = processor(images=image, return_tensors="pt")
	inputs = {key: value.to(device) for key, value in inputs.items()}

	with torch.no_grad():
		image_features = model.get_image_features(**inputs)

	if hasattr(image_features, "pooler_output"):
		image_features = image_features.pooler_output
	elif isinstance(image_features, (tuple, list)) and image_features:
		image_features = image_features[0]

	if not torch.is_tensor(image_features):
		raise TypeError("CLIP output is not a tensor-like feature vector.")

	image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
	vector = image_features[0].detach().cpu().numpy().astype(np.float32)
	return vector


def main() -> None:
	if not DATASET_PATH.exists():
		raise FileNotFoundError(f"CSV not found: {DATASET_PATH}")
	if not IMAGES_DIR.exists():
		raise NotADirectoryError(f"Image folder not found: {IMAGES_DIR}")

	df = pd.read_csv(DATASET_PATH, dtype=str, keep_default_na=False)
	processor, model, device = load_clip_model()

	kept_rows: list[dict[str, str]] = []
	image_embeddings: list[np.ndarray] = []
	skipped_missing_image = 0
	skipped_bad_image = 0

	for _, row in df.iterrows():
		row_dict = row.to_dict()
		row_id = str(row_dict.get("id", "")).strip()
		image_path = find_image_path(row_id)

		if image_path is None:
			skipped_missing_image += 1
			continue

		try:
			embedding = extract_clip_embedding(image_path, processor, model, device)
		except Exception:
			skipped_bad_image += 1
			continue

		image_embeddings.append(embedding)
		kept_rows.append(row_dict)

	if len(df) > 0 and not kept_rows:
		raise RuntimeError(
			"No rows were encoded successfully. Abort writing to avoid data loss."
		)

	kept_df = pd.DataFrame(kept_rows)
	image_matrix = np.vstack(image_embeddings).astype(np.float32)
	metadata_matrix, metadata_feature_names = build_metadata_onehot(kept_df)
	combined_matrix = np.concatenate([image_matrix, metadata_matrix], axis=1).astype(np.float32)

	FEATURES_DIR.mkdir(parents=True, exist_ok=True)
	np.save(IMAGE_FEATURES_PATH, image_matrix)
	np.save(METADATA_FEATURES_PATH, metadata_matrix)
	np.save(COMBINED_FEATURES_PATH, combined_matrix)

	index_columns = ["id", "human_generated_argument"] + METADATA_COLUMNS
	for col in index_columns:
		if col not in kept_df.columns:
			kept_df[col] = ""
	kept_df[index_columns].to_csv(INDEX_PATH, index=False)

	config = {
		"model": MODEL_NAME,
		"image_embedding_dim": int(image_matrix.shape[1]),
		"metadata_feature_dim": int(metadata_matrix.shape[1]),
		"combined_feature_dim": int(combined_matrix.shape[1]),
		"metadata_columns": METADATA_COLUMNS,
		"metadata_onehot_feature_names": metadata_feature_names,
	}
	CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

	print(f"Processed rows: {len(df)}")
	print(f"Rows kept: {len(kept_df)}")
	print(f"Rows dropped (missing image): {skipped_missing_image}")
	print(f"Rows dropped (unreadable image): {skipped_bad_image}")
	print(f"Saved image features: {IMAGE_FEATURES_PATH}")
	print(f"Saved metadata features: {METADATA_FEATURES_PATH}")
	print(f"Saved combined features: {COMBINED_FEATURES_PATH}")
	print(f"Saved row index: {INDEX_PATH}")
	print(f"Saved config: {CONFIG_PATH}")


if __name__ == "__main__":
	main()
