"""
NLI Premise-Conclusion Alignment Evaluation Script
===================================================
Measures if generated conclusions logically follow from their claimed premises.

Key Metric:
  - alignment_score: percentage of conclusions entailed by their premises (confidence > 0.5)

This addresses the 95% neutral NLI issue by checking premise-conclusion pairs
specifically, rather than premise-set to conclusion pairs.
"""

import json
import logging
import numpy as np
import pandas as pd
import torch
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_OUTPUT_DIR = PROJECT_ROOT / "models" / "output_model"
EVALUATION_DIR = Path(__file__).parent

NLI_MODEL_NAME = "microsoft/deberta-large-mnli"
BATCH_SIZE = 16
MAX_TOKEN_LENGTH = 512
ENTAILMENT_THRESHOLD = 0.5
RANDOM_SEED = 42

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# NLI MODEL
# ============================================================================

def load_nli_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading {NLI_MODEL_NAME} on {device.upper()}")
    
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
    model = model.to(device).eval()
    
    id2label = {k: v.lower() for k, v in model.config.id2label.items()}
    log.info(f"Label mapping: {id2label}")
    return tokenizer, model, device, id2label


def score_premise_conclusion_pairs(
    tokenizer, model, device: str, id2label: Dict,
    pairs: List[Tuple[str, str]],
) -> List[Dict]:
    """Score (premise, conclusion) pairs using NLI model."""
    results = []
    
    for start in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[start:start + BATCH_SIZE]
        if not batch:
            continue
            
        try:
            premises = [p for p, _ in batch]
            conclusions = [c for _, c in batch]
            
            enc = tokenizer(
                premises, conclusions,
                truncation=True, max_length=MAX_TOKEN_LENGTH,
                padding=True, return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            
            with torch.no_grad():
                logits = model(**enc).logits.cpu()
            
            probs = torch.softmax(logits, dim=-1).numpy()
            
            for prob_row in probs:
                score = {id2label[i]: float(prob_row[i]) for i in range(len(prob_row))}
                score["predicted_label"] = id2label[int(np.argmax(prob_row))]
                results.append(score)
                
        except Exception as e:
            log.warning(f"Batch error: {e}")
            for _ in batch:
                results.append({
                    "entailment": 0.0, "neutral": 1.0, "contradiction": 0.0,
                    "predicted_label": "error"
                })
    
    return results


# ============================================================================
# DATA LOADING
# ============================================================================

def load_model_outputs() -> Dict[str, List[Dict]]:
    """Load all model JSON files from output directory."""
    model_data = {}
    
    json_files = list(MODELS_OUTPUT_DIR.glob("*.json"))
    if not json_files:
        log.warning(f"No JSON files found in {MODELS_OUTPUT_DIR}")
        return {}
    
    for json_file in json_files:
        model_name = json_file.stem.replace("_outputs", "")
        log.info(f"Loading {model_name} from {json_file.name}")
        
        try:
            with open(json_file, 'r') as f:
                records = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            log.error(f"Failed to load {json_file}: {e}")
            continue
        
        if isinstance(records, dict):
            records = [records]
        
        model_data[model_name] = records
        log.info(f"Loaded {len(records)} records for {model_name}")
    
    return model_data


def parse_based_on(based_on) -> List[int]:
    """Parse the 'based_on' field from conclusion entry."""
    if isinstance(based_on, list):
        return [int(x) for x in based_on if str(x).isdigit()]
    if isinstance(based_on, str):
        nums = re.findall(r'\d+', based_on)
        return [int(n) for n in nums]
    return []


def extract_premise_conclusion_pairs(record: Dict) -> List[Tuple[str, str, str]]:
    """
    Extract (premise, conclusion, image_id) pairs from a record.
    
    If the new format with 'based_on' is used, only pairs the conclusion
    with its claimed supporting premises.
    
    If the old format is used, pairs each conclusion with each premise.
    """
    parsed = record.get("parsed_output", {})
    premises = parsed.get("premises", [])
    conclusions = parsed.get("conclusions", [])
    image_id = str(record.get("image_id", "unknown"))
    
    pairs = []
    
    if not premises or not conclusions:
        return pairs
    
    # Check if new format (conclusions have 'text' and 'based_on')
    is_new_format = False
    if conclusions and isinstance(conclusions[0], dict):
        if "text" in conclusions[0] or "based_on" in conclusions[0]:
            is_new_format = True
    
    if is_new_format:
        # New format: pair conclusion with its claimed premises
        for concl_entry in conclusions:
            if isinstance(concl_entry, dict):
                concl_text = concl_entry.get("text", "")
                based_on = parse_based_on(concl_entry.get("based_on", []))
            elif isinstance(concl_entry, str):
                concl_text = concl_entry
                based_on = list(range(1, len(premises) + 1))  # all premises
            else:
                continue
            
            if not concl_text:
                continue
            
            # Pair with claimed premises (1-indexed)
            for p_idx in based_on:
                if 1 <= p_idx <= len(premises):
                    premise_text = premises[p_idx - 1]
                    if premise_text:
                        pairs.append((premise_text, concl_text, image_id))
            
            # If no based_on specified, pair with all premises
            if not based_on:
                for prem in premises:
                    if prem:
                        pairs.append((prem, concl_text, image_id))
    else:
        # Old format: pair each conclusion with each premise
        for concl in conclusions:
            concl_text = concl if isinstance(concl, str) else str(concl)
            for prem in premises:
                prem_text = prem if isinstance(prem, str) else str(prem)
                if prem_text and concl_text:
                    pairs.append((prem_text, concl_text, image_id))
    
    return pairs


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(
    model_name: str,
    records: List[Dict],
    tokenizer, model, device: str, id2label: Dict,
) -> Dict:
    """Evaluate premise-conclusion alignment for one model."""
    log.info(f"\nEvaluating {model_name}...")
    
    # Extract all pairs
    all_pairs = []
    pair_metadata = []  # Track which image each pair belongs to
    
    for record in records:
        pairs = extract_premise_conclusion_pairs(record)
        for prem, concl, img_id in pairs:
            all_pairs.append((prem, concl))
            pair_metadata.append({
                "image_id": img_id,
                "premise": prem,
                "conclusion": concl,
            })
    
    if not all_pairs:
        log.warning(f"No valid premise-conclusion pairs found for {model_name}")
        return {"model": model_name, "total_pairs": 0}
    
    log.info(f"Scoring {len(all_pairs)} premise-conclusion pairs...")
    
    # Score all pairs
    nli_results = score_premise_conclusion_pairs(tokenizer, model, device, id2label, all_pairs)
    
    # Compute alignment metrics
    entailment_count = sum(1 for r in nli_results if r["predicted_label"] == "entailment")
    high_confidence_entailment = sum(
        1 for r in nli_results 
        if r["entailment"] > ENTAILMENT_THRESHOLD
    )
    
    total_pairs = len(nli_results)
    alignment_score = entailment_count / total_pairs if total_pairs > 0 else 0.0
    high_confidence_alignment = high_confidence_entailment / total_pairs if total_pairs > 0 else 0.0
    
    # Per-image alignment
    image_results = {}
    for meta, nli in zip(pair_metadata, nli_results):
        img_id = meta["image_id"]
        if img_id not in image_results:
            image_results[img_id] = {"pairs": 0, "entailed": 0, "high_conf_entailed": 0}
        image_results[img_id]["pairs"] += 1
        if nli["predicted_label"] == "entailment":
            image_results[img_id]["entailed"] += 1
        if nli["entailment"] > ENTAILMENT_THRESHOLD:
            image_results[img_id]["high_conf_entailed"] += 1
    
    per_image_alignment = {
        img_id: data["entailed"] / data["pairs"] 
        for img_id, data in image_results.items()
        if data["pairs"] > 0
    }
    
    # Aggregate statistics
    stats = {
        "model": model_name,
        "total_pairs": total_pairs,
        "total_images": len(image_results),
        "alignment_score": alignment_score,
        "high_confidence_alignment": high_confidence_alignment,
        "mean_entailment": float(np.mean([r["entailment"] for r in nli_results])),
        "mean_neutral": float(np.mean([r["neutral"] for r in nli_results])),
        "mean_contradiction": float(np.mean([r["contradiction"] for r in nli_results])),
        "per_image_alignment_mean": float(np.mean(list(per_image_alignment.values()))),
        "per_image_alignment_std": float(np.std(list(per_image_alignment.values()))) if per_image_alignment else 0.0,
    }
    
    # Build per-pair results
    per_pair_results = []
    for meta, nli in zip(pair_metadata, nli_results):
        per_pair_results.append({
            "image_id": meta["image_id"],
            "premise": meta["premise"],
            "conclusion": meta["conclusion"],
            "entailment": nli["entailment"],
            "neutral": nli["neutral"],
            "contradiction": nli["contradiction"],
            "predicted_label": nli["predicted_label"],
            "is_aligned": nli["entailment"] > ENTAILMENT_THRESHOLD,
        })
    
    return {
        "model": model_name,
        "stats": stats,
        "per_pair_results": per_pair_results,
    }


# ============================================================================
# OUTPUT
# ============================================================================

def save_results(all_results: Dict[str, Dict]):
    """Save alignment evaluation results."""
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save per-model CSVs
    for model_name, result in all_results.items():
        if not result.get("per_pair_results"):
            continue
        
        df = pd.DataFrame(result["per_pair_results"])
        df = df.round(4)
        output_path = EVALUATION_DIR / f"{model_name}_alignment.csv"
        df.to_csv(output_path, index=False)
        log.info(f"Saved: {output_path}")
    
    # Save summary
    summary_rows = []
    for model_name, result in all_results.items():
        stats = result.get("stats", {})
        summary_rows.append({
            "Model": stats.get("model", model_name),
            "Total_Pairs": stats.get("total_pairs", 0),
            "Total_Images": stats.get("total_images", 0),
            "Alignment_Score": round(stats.get("alignment_score", 0), 4),
            "High_Confidence_Alignment": round(stats.get("high_confidence_alignment", 0), 4),
            "Mean_Entailment": round(stats.get("mean_entailment", 0), 4),
            "Mean_Neutral": round(stats.get("mean_neutral", 0), 4),
            "Mean_Contradiction": round(stats.get("mean_contradiction", 0), 4),
            "Per_Image_Alignment_Mean": round(stats.get("per_image_alignment_mean", 0), 4),
            "Per_Image_Alignment_Std": round(stats.get("per_image_alignment_std", 0), 4),
        })
    
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        summary_path = EVALUATION_DIR / "argument_quality_summary.csv"
        df_summary.to_csv(summary_path, index=False)
        log.info(f"Saved summary: {summary_path}")
    
    # Print summary
    log.info("\n" + "=" * 70)
    log.info("ARGUMENT QUALITY SUMMARY")
    log.info("=" * 70)
    for row in summary_rows:
        log.info(f"\n{row['Model']}:")
        log.info(f"  Alignment Score: {row['Alignment_Score']:.2%}")
        log.info(f"  High-Conf Alignment: {row['High_Confidence_Alignment']:.2%}")
        log.info(f"  Mean Entailment: {row['Mean_Entailment']:.4f}")
        log.info(f"  Per-Image Alignment: {row['Per_Image_Alignment_Mean']:.2%} ± {row['Per_Image_Alignment_Std']:.2%}")
    log.info("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    set_seed(RANDOM_SEED)
    
    log.info("=" * 70)
    log.info("NLI Premise-Conclusion Alignment Evaluation")
    log.info("=" * 70)
    
    # Load model outputs
    model_data = load_model_outputs()
    if not model_data:
        log.error("No model outputs found. Exiting.")
        return False
    
    # Load NLI model
    tokenizer, nli_model, device, id2label = load_nli_model()
    
    # Evaluate each model
    all_results = {}
    for model_name, records in model_data.items():
        result = evaluate_model(model_name, records, tokenizer, nli_model, device, id2label)
        all_results[model_name] = result
    
    # Save results
    save_results(all_results)
    
    return True


if __name__ == "__main__":
    success = main()
    raise SystemExit(0 if success else 1)
