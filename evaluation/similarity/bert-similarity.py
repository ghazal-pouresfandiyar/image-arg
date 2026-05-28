"""
SBERT Embedding Similarity Evaluation Script
============================================
Compare human premises/conclusions vs model outputs using sentence-transformers.
Implements max-overlap matching strategy for set-to-set similarity.

Output:
  - Per-model CSVs: {model_name}_similarity.csv with detailed metrics per image
  - Summary CSV: summary_comparison.csv with model-level statistics
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "dataset"
MODELS_OUTPUT_DIR = PROJECT_ROOT / "models" / "output_model"
EVAL_OUTPUT_DIR = Path(__file__).parent
ANNOTATED_CSV = DATA_DIR / "annotated.csv"

# SBERT model selection
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# TEXT PREPROCESSING & EMBEDDING
# ============================================================================

def normalize_text(text: str) -> str:
    """Normalize text: lowercase, strip whitespace."""
    if isinstance(text, str):
        return text.lower().strip()
    return ""


def normalize_sentence_list(sentences: List[str]) -> List[str]:
    """Normalize a list of sentences."""
    if not sentences:
        return []
    return [normalize_text(s) for s in sentences if isinstance(s, str) and s.strip()]


def load_sbert_model() -> SentenceTransformer:
    """Load SBERT model once for reuse."""
    logger.info(f"Loading SBERT model: {SBERT_MODEL_NAME}")
    model = SentenceTransformer(SBERT_MODEL_NAME)
    return model


def embed_sentences(sentences: List[str], model: SentenceTransformer) -> np.ndarray:
    """
    Embed list of sentences to numpy array.
    
    Args:
        sentences: List of strings to embed
        model: SentenceTransformer model instance
    
    Returns:
        numpy array of shape (len(sentences), EMBEDDING_DIM)
        or (0, EMBEDDING_DIM) if sentences is empty
    """
    if not sentences or len(sentences) == 0:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    
    embeddings = model.encode(sentences, convert_to_numpy=True)
    return embeddings


def compute_similarity_matrix(human_emb: np.ndarray, model_emb: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix between human and model embeddings.
    
    Args:
        human_emb: Shape (H, 384)
        model_emb: Shape (M, 384)
    
    Returns:
        Similarity matrix of shape (H, M)
    """
    if human_emb.shape[0] == 0 or model_emb.shape[0] == 0:
        return np.empty((human_emb.shape[0], model_emb.shape[0]), dtype=np.float32)
    
    sim = cosine_similarity(human_emb, model_emb)
    return sim


def compute_max_overlap_similarity(human_sentences: List[str], 
                                   model_sentences: List[str],
                                   model: SentenceTransformer) -> Dict[str, float]:
    """
    Compute max-overlap similarity between human and model sentence sets.
    
    Strategy: For each human sentence, find max similarity to any model sentence.
    Then average all max values.
    
    Args:
        human_sentences: List of human sentences
        model_sentences: List of model sentences
        model: SentenceTransformer instance
    
    Returns:
        Dict with keys:
            - 'similarity': Main score (mean of max similarities)
            - 'coverage_ratio': Proportion of human sentences with model match
            - 'min_sim': Minimum max-similarity across human sentences
            - 'max_sim': Maximum max-similarity across human sentences
    """
    # Normalize texts
    human_norm = normalize_sentence_list(human_sentences)
    model_norm = normalize_sentence_list(model_sentences)
    
    # Edge cases: empty lists
    if len(human_norm) == 0 and len(model_norm) == 0:
        # Both empty: perfect agreement
        return {
            'similarity': 1.0,
            'coverage_ratio': 1.0,
            'min_sim': 1.0,
            'max_sim': 1.0
        }
    elif len(human_norm) == 0 or len(model_norm) == 0:
        # One empty: no similarity
        return {
            'similarity': 0.0,
            'coverage_ratio': 0.0,
            'min_sim': 0.0,
            'max_sim': 0.0
        }
    
    # Embed sentences
    human_emb = embed_sentences(human_norm, model)
    model_emb = embed_sentences(model_norm, model)
    
    # Compute similarity matrix (H x M)
    sim_matrix = compute_similarity_matrix(human_emb, model_emb)
    
    # Max-overlap: for each human sentence, find max similarity to any model sentence
    max_similarities = np.max(sim_matrix, axis=1)  # Shape (H,)
    
    # Compute metrics
    mean_similarity = float(np.mean(max_similarities))
    min_similarity = float(np.min(max_similarities))
    max_similarity = float(np.max(max_similarities))
    
    # Coverage ratio: how many human sentences have a reasonable match (>0.5)
    coverage_count = np.sum(max_similarities > 0.5)
    coverage_ratio = float(coverage_count / len(human_norm))
    
    return {
        'similarity': mean_similarity,
        'coverage_ratio': coverage_ratio,
        'min_sim': min_similarity,
        'max_sim': max_similarity
    }


# ============================================================================
# DATA LOADING
# ============================================================================

def load_human_data() -> Dict[str, Dict[str, List[str]]]:
    """
    Load annotated.csv and extract premises/conclusions per image_id.
    
    Returns:
        Dict: {image_id: {"premises": [...], "conclusions": [...]}}
    """
    logger.info(f"Loading human annotations from {ANNOTATED_CSV}")
    
    try:
        df = pd.read_csv(ANNOTATED_CSV)
    except FileNotFoundError:
        logger.error(f"File not found: {ANNOTATED_CSV}")
        return {}
    
    human_data = {}
    for _, row in df.iterrows():
        image_id = str(row['id'])
        
        # Parse JSON arrays
        try:
            premises = json.loads(row['premises']) if isinstance(row['premises'], str) else []
            conclusions = json.loads(row['conclusions']) if isinstance(row['conclusions'], str) else []
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON for image_id {image_id}: {e}")
            premises = []
            conclusions = []
        
        human_data[image_id] = {
            'premises': premises,
            'conclusions': conclusions
        }
    
    logger.info(f"Loaded human data for {len(human_data)} images")
    return human_data


def load_model_outputs() -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    """
    Load all model JSON files from output directory.
    
    Returns:
        Dict: {model_name: {image_id: {"premises": [...], "conclusions": [...]}}}
    """
    logger.info(f"Loading model outputs from {MODELS_OUTPUT_DIR}")
    
    model_data = {}
    
    # Find all JSON files matching *_outputs.json pattern
    #json_files = list(MODELS_OUTPUT_DIR.glob("*_outputs.json"))
    json_files = list(MODELS_OUTPUT_DIR.glob("*.json"))
    if not json_files:
        logger.warning(f"No JSON files found in {MODELS_OUTPUT_DIR}")
        return {}
    
    for json_file in json_files:
        # Extract model name from filename (e.g., "llava_outputs.json" -> "llava")
        model_name = json_file.stem.replace("_outputs", "")
        
        logger.info(f"Loading {model_name} from {json_file.name}")
        
        try:
            with open(json_file, 'r') as f:
                records = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Failed to load {json_file}: {e}")
            continue
        
        # Handle case where JSON is a single dict or a list
        if isinstance(records, dict):
            records = [records]
        
        model_images = {}
        for record in records:
            image_id = str(record.get('image_id', ''))
            parsed_output = record.get('parsed_output', {})
            
            premises = parsed_output.get('premises', [])
            conclusions = parsed_output.get('conclusions', [])
            
            # Ensure they are lists
            if not isinstance(premises, list):
                premises = [premises] if premises else []
            if not isinstance(conclusions, list):
                conclusions = [conclusions] if conclusions else []
            
            model_images[image_id] = {
                'premises': premises,
                'conclusions': conclusions
            }
        
        model_data[model_name] = model_images
        logger.info(f"Loaded {len(model_images)} image records for {model_name}")
    
    return model_data


# ============================================================================
# EVALUATION PIPELINE
# ============================================================================

def evaluate_image_pair(image_id: str,
                       human_data: Dict[str, List[str]],
                       model_data: Dict[str, List[str]],
                       sbert_model: SentenceTransformer) -> Dict[str, float]:
    """
    Compute similarity metrics for one image (human vs one model).
    
    Args:
        image_id: Image identifier
        human_data: {"premises": [...], "conclusions": [...]}
        model_data: {"premises": [...], "conclusions": [...]}
        sbert_model: Loaded SBERT model
    
    Returns:
        Dict with 10 metrics:
            - premise_similarity
            - premise_coverage_ratio
            - premise_min_sim
            - premise_max_sim
            - conclusion_similarity
            - conclusion_coverage_ratio
            - conclusion_min_sim
            - conclusion_max_sim
    """
    # Get texts
    human_premises = human_data.get('premises', [])
    human_conclusions = human_data.get('conclusions', [])
    model_premises = model_data.get('premises', [])
    model_conclusions = model_data.get('conclusions', [])
    
    # Compute similarities
    premise_scores = compute_max_overlap_similarity(human_premises, model_premises, sbert_model)
    conclusion_scores = compute_max_overlap_similarity(human_conclusions, model_conclusions, sbert_model)
    
    return {
        'image_id': image_id,
        'premise_similarity': premise_scores['similarity'],
        'premise_coverage_ratio': premise_scores['coverage_ratio'],
        'premise_min_sim': premise_scores['min_sim'],
        'premise_max_sim': premise_scores['max_sim'],
        'conclusion_similarity': conclusion_scores['similarity'],
        'conclusion_coverage_ratio': conclusion_scores['coverage_ratio'],
        'conclusion_min_sim': conclusion_scores['min_sim'],
        'conclusion_max_sim': conclusion_scores['max_sim']
    }


def evaluate_all_images(human_data: Dict,
                       model_data: Dict,
                       sbert_model: SentenceTransformer) -> Tuple[List[Dict], Dict]:
    """
    Evaluate all images for a single model.
    
    Returns:
        - List of per-image dicts
        - Dict with aggregate statistics
    """
    per_image_results = []
    
    for image_id in sorted(human_data.keys()):
        # Get data or use empty if not present (handles missing data)
        human = human_data.get(image_id, {'premises': [], 'conclusions': []})
        model = model_data.get(image_id, {'premises': [], 'conclusions': []})
        
        scores = evaluate_image_pair(image_id, human, model, sbert_model)
        per_image_results.append(scores)
    
    # Compute aggregate statistics
    df_results = pd.DataFrame(per_image_results)
    
    aggregate_stats = {
        'n_images': len(per_image_results),
        'premise_similarity_mean': float(df_results['premise_similarity'].mean()),
        'premise_similarity_std': float(df_results['premise_similarity'].std()),
        'premise_similarity_min': float(df_results['premise_similarity'].min()),
        'premise_similarity_max': float(df_results['premise_similarity'].max()),
        'premise_similarity_median': float(df_results['premise_similarity'].median()),
        'conclusion_similarity_mean': float(df_results['conclusion_similarity'].mean()),
        'conclusion_similarity_std': float(df_results['conclusion_similarity'].std()),
        'conclusion_similarity_min': float(df_results['conclusion_similarity'].min()),
        'conclusion_similarity_max': float(df_results['conclusion_similarity'].max()),
        'conclusion_similarity_median': float(df_results['conclusion_similarity'].median()),
    }
    
    return per_image_results, aggregate_stats


# ============================================================================
# OUTPUT GENERATION
# ============================================================================

def save_per_model_csv(results: List[Dict], model_name: str, output_dir: Path) -> Path:
    """Save per-image results CSV for one model."""
    df = pd.DataFrame(results)
    
    # Round to 4 decimal places
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(4)
    
    output_path = output_dir / f"{model_name}_similarity.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved per-model results: {output_path}")
    
    return output_path


def save_summary_csv(all_stats: Dict[str, Dict], output_dir: Path) -> Path:
    """Save summary comparison CSV across all models."""
    summary_data = []
    
    for model_name, stats in all_stats.items():
        summary_data.append({
            'Model': model_name,
            'N_Images': stats['n_images'],
            'Premise_Similarity_Mean': round(stats['premise_similarity_mean'], 4),
            'Premise_Similarity_Std': round(stats['premise_similarity_std'], 4),
            'Premise_Similarity_Min': round(stats['premise_similarity_min'], 4),
            'Premise_Similarity_Max': round(stats['premise_similarity_max'], 4),
            'Premise_Similarity_Median': round(stats['premise_similarity_median'], 4),
            'Conclusion_Similarity_Mean': round(stats['conclusion_similarity_mean'], 4),
            'Conclusion_Similarity_Std': round(stats['conclusion_similarity_std'], 4),
            'Conclusion_Similarity_Min': round(stats['conclusion_similarity_min'], 4),
            'Conclusion_Similarity_Max': round(stats['conclusion_similarity_max'], 4),
            'Conclusion_Similarity_Median': round(stats['conclusion_similarity_median'], 4),
        })
    
    df_summary = pd.DataFrame(summary_data)
    output_path = output_dir / "summary_comparison.csv"
    df_summary.to_csv(output_path, index=False)
    logger.info(f"Saved summary comparison: {output_path}")
    
    return output_path


# ============================================================================
# VALIDATION
# ============================================================================

def validate_results(results: List[Dict], model_name: str) -> bool:
    """Validate similarity scores are in valid range."""
    df = pd.DataFrame(results)
    
    # Check for NaN values
    if df.isnull().any().any():
        logger.error(f"NaN values found in {model_name} results")
        return False
    
    # Check similarity scores in [0, 1]
    sim_cols = ['premise_similarity', 'conclusion_similarity']
    for col in sim_cols:
        if (df[col] < 0).any() or (df[col] > 1).any():
            logger.error(f"Invalid similarity values in {model_name}: {col}")
            return False
    
    logger.info(f"Validation passed for {model_name}: {len(results)} images")
    return True


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main evaluation pipeline."""
    logger.info("=" * 70)
    logger.info("SBERT Embedding Similarity Evaluation")
    logger.info("=" * 70)
    
    # Ensure output directory exists
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Phase 1: Load data
    logger.info("\n[Phase 1] Loading data...")
    human_data = load_human_data()
    model_data = load_model_outputs()
    
    if not human_data:
        logger.error("No human data loaded. Exiting.")
        return False
    
    if not model_data:
        logger.error("No model data loaded. Exiting.")
        logger.error(f"Searched in: {MODELS_OUTPUT_DIR}")
        logger.error(f"Directory exists: {MODELS_OUTPUT_DIR.exists()}")
        if MODELS_OUTPUT_DIR.exists():
            logger.error(f"Files found: {list(MODELS_OUTPUT_DIR.glob('*.json'))}")
        return False
    
    # Phase 2: Load SBERT model
    logger.info("\n[Phase 2] Loading SBERT model...")
    sbert_model = load_sbert_model()
    logger.info(f"SBERT model loaded successfully")
    
    # Phase 3: Evaluate all models
    logger.info("\n[Phase 3] Computing similarity scores...")
    all_model_results = {}
    all_model_stats = {}
    
    for model_name in sorted(model_data.keys()):
        logger.info(f"\nEvaluating {model_name}...")
        model_imgs = model_data[model_name]
        
        per_image_results, aggregate_stats = evaluate_all_images(
            human_data, model_imgs, sbert_model
        )
        
        # Validate
        if not validate_results(per_image_results, model_name):
            logger.warning(f"Validation failed for {model_name}, skipping output")
            continue
        
        all_model_results[model_name] = per_image_results
        all_model_stats[model_name] = aggregate_stats
        
        logger.info(f"{model_name} - Premise Sim: {aggregate_stats['premise_similarity_mean']:.4f} "
                   f"± {aggregate_stats['premise_similarity_std']:.4f}")
        logger.info(f"{model_name} - Conclusion Sim: {aggregate_stats['conclusion_similarity_mean']:.4f} "
                   f"± {aggregate_stats['conclusion_similarity_std']:.4f}")
    
    # Phase 4: Save outputs
    logger.info("\n[Phase 4] Saving outputs...")
    saved_files = []
    
    for model_name, results in all_model_results.items():
        csv_path = save_per_model_csv(results, model_name, EVAL_OUTPUT_DIR)
        saved_files.append(csv_path)
    
    summary_path = save_summary_csv(all_model_stats, EVAL_OUTPUT_DIR)
    saved_files.append(summary_path)
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("Evaluation Complete!")
    logger.info("=" * 70)
    logger.info(f"Models evaluated: {len(all_model_stats)}")
    logger.info(f"Images per model: {all_model_stats[list(all_model_stats.keys())[0]]['n_images']}")
    logger.info(f"\nOutput files saved:")
    for file_path in saved_files:
        logger.info(f"  - {file_path}")
    
    # Print summary table
    logger.info("\nSummary Statistics:")
    logger.info("-" * 70)
    for model_name, stats in all_model_stats.items():
        logger.info(f"\n{model_name}:")
        logger.info(f"  Premise Similarity:     {stats['premise_similarity_mean']:.4f} "
                   f"± {stats['premise_similarity_std']:.4f}")
        logger.info(f"  Conclusion Similarity:  {stats['conclusion_similarity_mean']:.4f} "
                   f"± {stats['conclusion_similarity_std']:.4f}")
    
    logger.info("\n" + "=" * 70)
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)