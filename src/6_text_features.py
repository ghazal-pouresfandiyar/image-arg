#!/usr/bin/env python3
"""Extract text features for premises and retrieve relevant facts.

This script:
1. Encodes premises using a sentence transformer (semantic embeddings)
2. Loads climate fact bank from facts.json
3. Retrieves relevant facts per image using semantic similarity
4. Enables retrieval-augmented argument generation
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "dataset" / "annotated.csv"
FACTS_PATH = ROOT_DIR / "dataset" / "facts.json"
FEATURES_DIR = ROOT_DIR / "dataset" / "features"

# Sentence transformer model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

PREMISE_EMBEDDINGS_PATH = FEATURES_DIR / "premise_embeddings.npy"
PREMISE_INDEX_PATH = FEATURES_DIR / "premise_index.csv"
FACT_EMBEDDINGS_PATH = FEATURES_DIR / "fact_embeddings.npy"
FACT_RETRIEVAL_PATH = FEATURES_DIR / "fact_retrievals.json"
CONFIG_PATH = FEATURES_DIR / "text_features_config.json"


def load_sentence_transformer() -> SentenceTransformer:
	"""Load pretrained sentence transformer."""
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model = SentenceTransformer(MODEL_NAME, device=device)
	return model


def parse_premises(premises_str: str) -> list[str]:
	"""Parse premises from JSON array string in CSV."""
	if not premises_str or not premises_str.strip():
		return []
	
	try:
		return json.loads(premises_str)
	except json.JSONDecodeError:
		# If single string, return as list
		return [premises_str]


def parse_conclusions(conclusions_str: str) -> list[str]:
	"""Parse conclusions from JSON array string in CSV."""
	if not conclusions_str or not conclusions_str.strip():
		return []
	
	try:
		return json.loads(conclusions_str)
	except json.JSONDecodeError:
		return [conclusions_str]


def load_facts(facts_path: Path) -> dict:
	"""Load facts from facts.json."""
	if not facts_path.exists():
		return {"global": [], "climate_specific": []}
	
	with open(facts_path, "r", encoding="utf-8") as f:
		facts = json.load(f)
	
	# Flatten all facts into a single list
	all_facts = []
	for category, fact_list in facts.items():
		if isinstance(fact_list, list):
			all_facts.extend(fact_list)
	
	return {"global": all_facts, "by_category": facts}


def retrieve_relevant_facts(
	premises: list[str],
	fact_embeddings: np.ndarray,
	all_facts: list[str],
	model: SentenceTransformer,
	top_k: int = 5,
) -> list[dict]:
	"""Retrieve top-k most relevant facts for given premises."""
	
	if not premises or not all_facts:
		return []
	
	# Encode premises
	premise_embeddings = model.encode(premises, convert_to_tensor=True)
	
	# Compute similarity between premises and facts
	max_relevance_scores = None
	
	for premise_emb in premise_embeddings:
		similarities = util.pytorch_cos_sim(premise_emb, fact_embeddings)[0]
		
		if max_relevance_scores is None:
			max_relevance_scores = similarities.cpu().numpy()
		else:
			# Take max similarity across all premises
			curr_scores = similarities.cpu().numpy()
			max_relevance_scores = np.maximum(max_relevance_scores, curr_scores)
	
	# Get top-k facts
	top_k = min(top_k, len(all_facts))
	top_indices = np.argsort(max_relevance_scores)[-top_k:][::-1]
	
	retrievals = []
	for idx in top_indices:
		retrievals.append({
			"fact": all_facts[idx],
			"relevance_score": float(max_relevance_scores[idx]),
		})
	
	return retrievals


def main() -> None:
	if not DATASET_PATH.exists():
		raise FileNotFoundError(f"CSV not found: {DATASET_PATH}")

	print("Loading sentence transformer...")
	model = load_sentence_transformer()

	df = pd.read_csv(DATASET_PATH, dtype=str, keep_default_na=False)

	print("Loading facts...")
	facts_data = load_facts(FACTS_PATH)
	all_facts = facts_data["global"]

	if not all_facts:
		print("Warning: No facts loaded from facts.json")
		all_facts = ["Climate change is a global phenomenon.", "Environmental action is necessary."]

	print(f"Total facts available: {len(all_facts)}")

	# Encode all facts once
	print("Encoding facts...")
	fact_embeddings = model.encode(all_facts, convert_to_tensor=False)
	fact_embeddings = np.array(fact_embeddings, dtype=np.float32)

	# Process premises
	all_premise_records = []
	all_premise_embeddings = []
	all_retrievals = []
	skipped = 0

	print("Processing premises and facts...")
	for idx, (_, row) in enumerate(df.iterrows()):
		if (idx + 1) % 10 == 0:
			print(f"  Processed {idx + 1}/{len(df)}")

		row_id = str(row.get("id", "")).strip()
		premises_str = str(row.get("premises", "")).strip()
		conclusions_str = str(row.get("conclusions", "")).strip()

		premises = parse_premises(premises_str)
		conclusions = parse_conclusions(conclusions_str)

		if not premises:
			skipped += 1
			continue

		# Encode premises
		premise_embeddings = model.encode(premises, convert_to_tensor=False)
		premise_embeddings = np.array(premise_embeddings, dtype=np.float32)

		# Retrieve relevant facts
		retrievals = retrieve_relevant_facts(
			premises,
			torch.tensor(fact_embeddings, dtype=torch.float32),
			all_facts,
			model,
			top_k=5,
		)

		# Store record
		all_premise_records.append({
			"id": row_id,
			"num_premises": len(premises),
			"num_conclusions": len(conclusions),
			"premise_texts": premises,
			"conclusion_texts": conclusions,
			"num_retrieved_facts": len(retrievals),
		})

		# Average premise embeddings per image (if multiple premises)
		avg_embedding = premise_embeddings.mean(axis=0)
		all_premise_embeddings.append(avg_embedding)

		# Store retrievals
		all_retrievals.append({
			"id": row_id,
			"num_premises": len(premises),
			"retrieved_facts": retrievals,
		})

	if not all_premise_records:
		raise RuntimeError("No premise records created.")

	FEATURES_DIR.mkdir(parents=True, exist_ok=True)

	# Save premise embeddings
	premise_embeddings_array = np.array(all_premise_embeddings, dtype=np.float32)
	np.save(PREMISE_EMBEDDINGS_PATH, premise_embeddings_array)

	# Save premise index
	premise_df = pd.DataFrame(all_premise_records)[["id", "num_premises", "num_conclusions"]]
	premise_df.to_csv(PREMISE_INDEX_PATH, index=False)

	# Save fact embeddings
	np.save(FACT_EMBEDDINGS_PATH, fact_embeddings)

	# Save fact retrievals
	with open(FACT_RETRIEVAL_PATH, "w", encoding="utf-8") as f:
		json.dump(all_retrievals, f, indent=2)

	# Save config
	config = {
		"model": MODEL_NAME,
		"embedding_dim": int(premise_embeddings_array.shape[1]),
		"num_facts": len(all_facts),
		"retrieval_top_k": 5,
		"strategy": "retrieval-augmented: premises → cost_sim → facts",
	}
	CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

	print(f"\nProcessed: {len(df)} rows")
	print(f"Premise records: {len(all_premise_records)}")
	print(f"Skipped: {skipped}")
	print(f"\nPremise embeddings shape: {premise_embeddings_array.shape}")
	print(f"Fact embeddings shape: {fact_embeddings.shape}")
	print(f"\nSaved: {PREMISE_EMBEDDINGS_PATH}")
	print(f"Saved: {PREMISE_INDEX_PATH}")
	print(f"Saved: {FACT_EMBEDDINGS_PATH}")
	print(f"Saved: {FACT_RETRIEVAL_PATH}")
	print(f"Saved: {CONFIG_PATH}")

	# Print samples
	print("\n--- Sample premise records (first 2 images) ---")
	for rec in all_premise_records[:2]:
		print(f"\nID {rec['id']}:")
		print(f"  Premises: {rec['num_premises']}")
		print(f"  Conclusions: {rec['num_conclusions']}")
		print(f"  Retrieved facts: {rec['num_retrieved_facts']}")

	print("\n--- Sample fact retrievals (first 2 images) ---")
	for ret in all_retrievals[:2]:
		print(f"\nID {ret['id']}:")
		print(f"  Number of premises: {ret['num_premises']}")
		print(f"  Top retrieved facts:")
		for fact_data in ret["retrieved_facts"][:3]:
			print(f"    • {fact_data['fact'][:60]}... (relevance: {fact_data['relevance_score']:.3f})")


if __name__ == "__main__":
	main()
