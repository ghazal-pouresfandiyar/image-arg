"""
Output Diversity Metrics Script
===============================
Detect mode collapse and measure output variance across generated arguments.

Metrics:
  - Semantic Diversity: Average pairwise cosine distance between SBERT embeddings
  - Lexical Diversity: Type-Token Ratio (TTR) of generated text
  - Unique Output Rate: Percentage of unique conclusions across all images
  - Inter-Image Variance: Variance of embedding centroids per image

High diversity = model generates different arguments for different images
Low diversity = potential mode collapse (same generic output for all images)
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
from collections import Counter

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_OUTPUT_DIR = PROJECT_ROOT / "models" / "output_model"
EVALUATION_DIR = Path(__file__).parent

SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
RANDOM_SEED = 42

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def set_seed(seed: int):
    np.random.seed(seed)


# ============================================================================
# TEXT PROCESSING
# ============================================================================

def extract_all_conclusions(records: List[Dict]) -> List[str]:
    """Extract all conclusion texts from records."""
    conclusions = []
    
    for record in records:
        parsed = record.get("parsed_output", {})
        concl_list = parsed.get("conclusions", [])
        
        for concl in concl_list:
            if isinstance(concl, dict):
                text = concl.get("text", "")
            elif isinstance(concl, str):
                text = concl
            else:
                continue
            
            text = text.strip()
            if text:
                conclusions.append(text)
    
    return conclusions


def extract_all_premises(records: List[Dict]) -> List[str]:
    """Extract all premise texts from records."""
    premises = []
    
    for record in records:
        parsed = record.get("parsed_output", {})
        prem_list = parsed.get("premises", [])
        
        for prem in prem_list:
            if isinstance(prem, str):
                text = prem.strip()
            else:
                continue
            
            if text:
                premises.append(text)
    
    return premises


def compute_lexical_diversity(texts: List[str]) -> Dict[str, float]:
    """Compute lexical diversity metrics."""
    if not texts:
        return {"ttr": 0.0, "avg_unique_words": 0.0}
    
    all_words = []
    unique_words_per_text = []
    
    for text in texts:
        words = text.lower().split()
        all_words.extend(words)
        unique_words_per_text.append(len(set(words)))
    
    # Type-Token Ratio (unique words / total words)
    total_words = len(all_words)
    unique_words = len(set(all_words))
    ttr = unique_words / total_words if total_words > 0 else 0.0
    
    # Average unique words per text
    avg_unique = np.mean(unique_words_per_text) if unique_words_per_text else 0.0
    
    return {
        "ttr": ttr,
        "avg_unique_words": float(avg_unique),
        "total_words": total_words,
        "unique_words": unique_words,
    }


# ============================================================================
# SEMANTIC DIVERSITY
# ============================================================================

def compute_semantic_diversity(
    texts: List[str],
    sbert_model: SentenceTransformer,
) -> Dict[str, float]:
    """
    Compute semantic diversity using SBERT embeddings.
    
    Metrics:
    - avg_pairwise_distance: Mean cosine distance between all text pairs
    - min_pairwise_distance: Minimum distance (most similar pair)
    - max_pairwise_distance: Maximum distance (most different pair)
    - embedding_variance: Variance of embedding dimensions
    """
    if len(texts) < 2:
        return {
            "avg_pairwise_distance": 0.0,
            "min_pairwise_distance": 0.0,
            "max_pairwise_distance": 0.0,
            "embedding_variance": 0.0,
            "n_texts": len(texts),
        }
    
    # Embed all texts
    embeddings = sbert_model.encode(texts, convert_to_numpy=True)
    
    # Compute pairwise cosine distances
    dist_matrix = cosine_distances(embeddings)
    
    # Extract upper triangle (excluding diagonal)
    n = len(texts)
    upper_triangle = dist_matrix[np.triu_indices(n, k=1)]
    
    # Compute metrics
    avg_distance = float(np.mean(upper_triangle))
    min_distance = float(np.min(upper_triangle))
    max_distance = float(np.max(upper_triangle))
    
    # Embedding variance (how spread out are the embeddings)
    embedding_variance = float(np.mean(np.var(embeddings, axis=0)))
    
    return {
        "avg_pairwise_distance": avg_distance,
        "min_pairwise_distance": min_distance,
        "max_pairwise_distance": max_distance,
        "embedding_variance": embedding_variance,
        "n_texts": n_texts,
    }


def compute_per_image_diversity(
    records: List[Dict],
    sbert_model: SentenceTransformer,
) -> Dict[str, float]:
    """
    Compute diversity of conclusions within each image.
    
    For each image, embed its conclusions and compute pairwise distance.
    Then average across all images.
    """
    per_image_distances = []
    
    for record in records:
        parsed = record.get("parsed_output", {})
        concl_list = parsed.get("conclusions", [])
        
        texts = []
        for concl in concl_list:
            if isinstance(concl, dict):
                text = concl.get("text", "").strip()
            elif isinstance(concl, str):
                text = concl.strip()
            else:
                continue
            if text:
                texts.append(text)
        
        if len(texts) < 2:
            continue
        
        embeddings = sbert_model.encode(texts, convert_to_numpy=True)
        dist_matrix = cosine_distances(embeddings)
        n = len(texts)
        upper_triangle = dist_matrix[np.triu_indices(n, k=1)]
        per_image_distances.append(float(np.mean(upper_triangle)))
    
    if not per_image_distances:
        return {
            "mean_intra_image_distance": 0.0,
            "std_intra_image_distance": 0.0,
            "n_images_with_multiple_conclusions": 0,
        }
    
    return {
        "mean_intra_image_distance": float(np.mean(per_image_distances)),
        "std_intra_image_distance": float(np.std(per_image_distances)),
        "n_images_with_multiple_conclusions": len(per_image_distances),
    }


# ============================================================================
# UNIQUENESS
# ============================================================================

def compute_uniqueness(conclusions: List[str]) -> Dict[str, float]:
    """Compute uniqueness metrics for conclusions."""
    if not conclusions:
        return {
            "unique_rate": 0.0,
            "total_conclusions": 0,
            "unique_conclusions": 0,
            "most_common_count": 0,
        }
    
    # Normalize for comparison
    normalized = [c.lower().strip() for c in conclusions]
    counter = Counter(normalized)
    
    total = len(normalized)
    unique = len(counter)
    unique_rate = unique / total if total > 0 else 0.0
    
    # Most common conclusion
    most_common = counter.most_common(1)[0] if counter else ("", 0)
    
    return {
        "unique_rate": unique_rate,
        "total_conclusions": total,
        "unique_conclusions": unique,
        "most_common_text": most_common[0],
        "most_common_count": most_common[1],
    }


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


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model_diversity(
    model_name: str,
    records: List[Dict],
    sbert_model: SentenceTransformer,
) -> Dict:
    """Evaluate diversity for one model."""
    log.info(f"\nEvaluating diversity for {model_name}...")
    
    # Extract texts
    all_conclusions = extract_all_conclusions(records)
    all_premises = extract_all_premises(records)
    
    log.info(f"  Total conclusions: {len(all_conclusions)}")
    log.info(f"  Total premises: {len(all_premises)}")
    
    # Lexical diversity
    concl_lexical = compute_lexical_diversity(all_conclusions)
    prem_lexical = compute_lexical_diversity(all_premises)
    
    # Semantic diversity
    concl_semantic = compute_semantic_diversity(all_conclusions, sbert_model)
    prem_semantic = compute_semantic_diversity(all_premises, sbert_model)
    
    # Per-image diversity
    per_image = compute_per_image_diversity(records, sbert_model)
    
    # Uniqueness
    uniqueness = compute_uniqueness(all_conclusions)
    
    stats = {
        "model": model_name,
        "n_records": len(records),
        # Conclusion metrics
        "concl_lexical_ttr": concl_lexical["ttr"],
        "concl_avg_unique_words": concl_lexical["avg_unique_words"],
        "concl_semantic_diversity": concl_semantic["avg_pairwise_distance"],
        "concl_min_distance": concl_semantic["min_pairwise_distance"],
        "concl_max_distance": concl_semantic["max_pairwise_distance"],
        "concl_embedding_variance": concl_semantic["embedding_variance"],
        "concl_unique_rate": uniqueness["unique_rate"],
        "concl_unique_count": uniqueness["unique_conclusions"],
        "concl_most_common_count": uniqueness["most_common_count"],
        # Premise metrics
        "prem_lexical_ttr": prem_lexical["ttr"],
        "prem_semantic_diversity": prem_semantic["avg_pairwise_distance"],
        # Per-image metrics
        "mean_intra_image_distance": per_image["mean_intra_image_distance"],
        "n_images_multi_concl": per_image["n_images_with_multiple_conclusions"],
    }
    
    # Log summary
    log.info(f"  Conclusion Lexical TTR: {concl_lexical['ttr']:.4f}")
    log.info(f"  Conclusion Semantic Diversity: {concl_semantic['avg_pairwise_distance']:.4f}")
    log.info(f"  Conclusion Unique Rate: {uniqueness['unique_rate']:.2%}")
    log.info(f"  Most Common Conclusion: {uniqueness['most_common_count']}x")
    
    return {
        "model": model_name,
        "stats": stats,
        "conclusions": all_conclusions,
    }


# ============================================================================
# OUTPUT
# ============================================================================

def save_results(all_results: Dict[str, Dict]):
    """Save diversity evaluation results."""
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save summary
    summary_rows = []
    for model_name, result in all_results.items():
        stats = result.get("stats", {})
        summary_rows.append({
            "Model": stats.get("model", model_name),
            "N_Records": stats.get("n_records", 0),
            "Conclusion_Lexical_TTR": round(stats.get("concl_lexical_ttr", 0), 4),
            "Conclusion_Semantic_Diversity": round(stats.get("concl_semantic_diversity", 0), 4),
            "Conclusion_Unique_Rate": round(stats.get("concl_unique_rate", 0), 4),
            "Conclusion_Most_Common_Count": stats.get("concl_most_common_count", 0),
            "Premise_Lexical_TTR": round(stats.get("prem_lexical_ttr", 0), 4),
            "Premise_Semantic_Diversity": round(stats.get("prem_semantic_diversity", 0), 4),
            "Mean_Intra_Image_Distance": round(stats.get("mean_intra_image_distance", 0), 4),
        })
    
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        summary_path = EVALUATION_DIR / "diversity_summary.csv"
        df_summary.to_csv(summary_path, index=False)
        log.info(f"Saved summary: {summary_path}")
    
    # Print summary
    log.info("\n" + "=" * 70)
    log.info("OUTPUT DIVERSITY SUMMARY")
    log.info("=" * 70)
    for row in summary_rows:
        log.info(f"\n{row['Model']}:")
        log.info(f"  Conclusion Lexical TTR: {row['Conclusion_Lexical_TTR']:.4f}")
        log.info(f"  Conclusion Semantic Diversity: {row['Conclusion_Semantic_Diversity']:.4f}")
        log.info(f"  Conclusion Unique Rate: {row['Conclusion_Unique_Rate']:.2%}")
        log.info(f"  Most Common Conclusion: {row['Conclusion_Most_Common_Count']}x")
        log.info(f"  Premise Semantic Diversity: {row['Premise_Semantic_Diversity']:.4f}")
        log.info(f"  Mean Intra-Image Distance: {row['Mean_Intra_Image_Distance']:.4f}")
    log.info("=" * 70)
    
    # Mode collapse warning
    for row in summary_rows:
        if row["Conclusion_Unique_Rate"] < 0.5:
            log.warning(f"⚠️  {row['Model']}: Low unique rate ({row['Conclusion_Unique_Rate']:.2%}) — possible mode collapse!")
        if row["Conclusion_Most_Common_Count"] > len(all_results[row['Model']]['conclusions']) * 0.3:
            log.warning(f"⚠️  {row['Model']}: One conclusion dominates — possible mode collapse!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    set_seed(RANDOM_SEED)
    
    log.info("=" * 70)
    log.info("Output Diversity Metrics Evaluation")
    log.info("=" * 70)
    
    # Load model outputs
    model_data = load_model_outputs()
    if not model_data:
        log.error("No model outputs found. Exiting.")
        return False
    
    # Load SBERT model
    log.info(f"\nLoading SBERT model: {SBERT_MODEL_NAME}")
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
    
    # Evaluate each model
    all_results = {}
    for model_name, records in model_data.items():
        result = evaluate_model_diversity(model_name, records, sbert_model)
        all_results[model_name] = result
    
    # Save results
    save_results(all_results)
    
    return True


if __name__ == "__main__":
    success = main()
    raise SystemExit(0 if success else 1)
