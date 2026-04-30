#!/usr/bin/env python3
"""Generate one-hot encoded metadata features from the dataset.

This script reads dataset/annotated.csv and converts categorical metadata columns
(animals, consequences, climateaction, type, setting) into one-hot encoded features.
Each metadata column gets its own one-hot vector.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "dataset" / "annotated.csv"
FEATURES_DIR = ROOT_DIR / "dataset" / "features"

METADATA_COLUMNS = ["animals", "consequences", "climateaction", "type", "setting"]

METADATA_FEATURES_PATH = FEATURES_DIR / "metadata_onehot.npy"
METADATA_INDEX_PATH = FEATURES_DIR / "metadata_index.csv"
CONFIG_PATH = FEATURES_DIR / "metadata_onehot_config.json"


def build_metadata_onehot(df):
	"""Convert each metadata column to its own one-hot features.
	
	Each column gets its own one-hot vector. For example:
	- animals=[1,0,0,0,0] if Fish
	- consequences=[0,1,0,0,0,0,0,0] if Drought
	All vectors are concatenated together.
	"""
	metadata_matrices = []
	metadata_feature_names = []
	
	for col in METADATA_COLUMNS:
		# Get unique values for this column only
		unique_values = sorted(list(df[col].unique()))
		
		# Create one-hot encoding for this column
		onehot_vectors = []
		for idx, row in df.iterrows():
			value = row[col]
			# Find position of this value in unique values
			position = unique_values.index(value)
			# Create one-hot vector
			vector = [0.0] * len(unique_values)
			vector[position] = 1.0
			onehot_vectors.append(vector)
		
		# Convert to numpy array
		col_matrix = np.array(onehot_vectors, dtype=np.float32)
		metadata_matrices.append(col_matrix)
		
		# Store feature names for this column
		for i, val in enumerate(unique_values):
			metadata_feature_names.append(f"{col}_{val}")
	
	# Concatenate all column vectors horizontally
	matrix = np.concatenate(metadata_matrices, axis=1).astype(np.float32)
	return matrix, metadata_feature_names


def main():
	if not DATASET_PATH.exists():
		raise FileNotFoundError(f"CSV not found: {DATASET_PATH}")

	df = pd.read_csv(DATASET_PATH, dtype=str, keep_default_na=False)

	print(f"Processing {len(df)} rows...")
	metadata_matrix, metadata_feature_names = build_metadata_onehot(df)
	
	FEATURES_DIR.mkdir(parents=True, exist_ok=True)
	np.save(METADATA_FEATURES_PATH, metadata_matrix)
	
	# Save index with ids
	index_columns = ["id"] + METADATA_COLUMNS
	index_df = df[index_columns].copy()
	index_df.to_csv(METADATA_INDEX_PATH, index=False)
	
	# Save config
	config = {
		"metadata_columns": METADATA_COLUMNS,
		"metadata_feature_names": metadata_feature_names,
		"total_features": len(metadata_feature_names),
		"total_samples": len(df),
		"feature_dim": int(metadata_matrix.shape[1]),
	}
	CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

	print(f"✓ Processed rows: {len(df)}")
	print(f"✓ Metadata features created: {metadata_matrix.shape[1]}")
	print(f"✓ Saved metadata features: {METADATA_FEATURES_PATH} (shape: {metadata_matrix.shape})")
	print(f"✓ Saved metadata index: {METADATA_INDEX_PATH}")
	print(f"✓ Saved config: {CONFIG_PATH}")


if __name__ == "__main__":
	main()
