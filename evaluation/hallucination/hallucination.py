"""
Hallucination Detection Script
===============================
Detect unsupported claims in model outputs using:
1. Embedding similarity (model text vs metadata support)
2. Metadata entity matching (is entity in image metadata?)

No NLI scoring — pure embedding + metadata grounding.

Output:
  - Per-model CSVs: {model_name}_hallucination.csv
  - Summary CSV: summary_hallucination.csv
  - Details JSON: hallucination_details.json
"""

import os
import json
import logging
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
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

# SBERT model
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Hallucination thresholds
PREMISE_THRESHOLD = 0.60
CONCLUSION_THRESHOLD = 0.45

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# ENTITY EXTRACTION & MATCHING
# ============================================================================

# Common climate/environment keywords
CLIMATE_KEYWORDS = {
    'climate': 0,
    'warming': 0,
    'greenhouse': 0,
    'carbon': 0,
    'emissions': 0,
    'fossil': 0,
    'renewable': 0,
    'solar': 0,
    'wind': 0,
    'temperature': 0,
    'heat': 0,
    'melting': 0,
    'drought': 0,
    'flood': 0,
    'wildfire': 0,
    'hurricane': 0,
    'storm': 0,
    'pollution': 0,
    'acidification': 0,
    'bleaching': 0,
    'extinction': 0,
    'habitat': 0,
    'biodiversity': 0,
    'ecosystem': 0,
    'conservation': 0,
    'sustainability': 0,
    'renewable': 0,
}

# Specificity markers (dates, numbers, percentages)
SPECIFICITY_PATTERNS = [
    r'\d{4}',  # Years
    r'\d+%',  # Percentages
    r'\d+°',  # Temperature
    r'\d+\s*(km|miles|meters|feet)',  # Distances
    r'january|february|march|april|may|june|july|august|september|october|november|december',
]

def normalize_text(text: str) -> str:
    """Normalize text for matching."""
    if isinstance(text, str):
        return text.lower().strip()
    return ""


def extract_metadata_entities(metadata: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Extract entities from metadata fields.
    
    Returns:
        Dict with keys: animals, consequences, climate_actions, settings
    """
    entities = {
        'animals': [],
        'consequences': [],
        'climate_actions': [],
        'settings': [],
    }
    
    # Animals
    if 'animals' in metadata and metadata['animals']:
        animals = str(metadata['animals']).lower()
        if animals != 'no animals':
            entities['animals'] = [a.strip() for a in animals.split(',')]
    
    # Consequences
    if 'consequences' in metadata and metadata['consequences']:
        consequences = str(metadata['consequences']).lower()
        entities['consequences'] = [c.strip() for c in consequences.split(',')]
    
    # Climate action
    if 'climateaction' in metadata and metadata['climateaction']:
        climateaction = str(metadata['climateaction']).lower()
        if climateaction != 'no climate action':
            entities['climate_actions'] = [climateaction.strip()]
    
    # Setting
    if 'setting' in metadata and metadata['setting']:
        setting = str(metadata['setting']).lower()
        entities['settings'] = [s.strip() for s in setting.split(',')]
    
    return entities


def extract_text_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract potential entities from model-generated text.
    
    Returns:
        Dict with keys: animals, consequences, climate_actions, settings, specific_claims
    """
    text_lower = text.lower()
    entities = {
        'animals': [],
        'consequences': [],
        'climate_actions': [],
        'settings': [],
        'specific_claims': [],
    }
    
    # Extract climate keywords found
    for keyword in CLIMATE_KEYWORDS.keys():
        if keyword in text_lower:
            entities['climate_actions'].append(keyword)
    
    # Detect specific claims (dates, numbers, percentages)
    for pattern in SPECIFICITY_PATTERNS:
        matches = re.findall(pattern, text_lower)
        if matches:
            entities['specific_claims'].extend(matches)
    
    # Extract noun phrases (simple heuristic: capitalized words or known animals)
    animal_keywords = [
        'fish', 'bear', 'whale', 'dolphin', 'seal', 'bird', 'deer', 'elk',
        'cow', 'sheep', 'chicken', 'insect', 'bee', 'butterfly', 'lion',
        'tiger', 'elephant', 'penguin', 'owl', 'eagle', 'ant', 'spider'
    ]
    for animal in animal_keywords:
        if animal in text_lower:
            entities['animals'].append(animal)
    
    return entities


def compute_entity_mismatch_penalty(text_entities: Dict, metadata_entities: Dict) -> float:
    """
    Compute penalty for entity mismatches.
    
    Returns:
        Penalty score in [0, 1]
    """
    penalty = 0.0
    
    # Penalty for animals mentioned but not in metadata
    if text_entities['animals']:
        metadata_animals = set(metadata_entities['animals'])
        text_animals = set(text_entities['animals'])
        unmatched_animals = text_animals - metadata_animals
        if unmatched_animals and metadata_animals:
            # Only penalize if we have metadata animals to compare against
            penalty += 0.15 * len(unmatched_animals) / len(text_animals)
    
    # Penalty for consequences mentioned but not in metadata
    if text_entities['consequences']:
        metadata_cons = set(metadata_entities['consequences'])
        text_cons = set(text_entities['consequences'])
        unmatched_cons = text_cons - metadata_cons
        if unmatched_cons and metadata_cons:
            penalty += 0.15 * len(unmatched_cons) / len(text_cons)
    
    # Penalty for specific claims without context in metadata
    if text_entities['specific_claims']:
        # Heavy penalty for dates/numbers/percentages
        penalty += 0.20 * min(len(text_entities['specific_claims']) / 2, 1.0)
    
    return min(penalty, 1.0)


# ============================================================================
# TEXT PREPROCESSING & EMBEDDING
# ============================================================================

def load_sbert_model() -> SentenceTransformer:
    """Load SBERT model once for reuse."""
    logger.info(f"Loading SBERT model: {SBERT_MODEL_NAME}")
    model = SentenceTransformer(SBERT_MODEL_NAME)
    return model


def embed_text(text: str, model: SentenceTransformer) -> np.ndarray:
    """Embed single text or list of texts."""
    if isinstance(text, str):
        text = [text]
    
    if not text or len(text) == 0:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    
    embeddings = model.encode(text, convert_to_numpy=True)
    return embeddings


def compute_text_similarity(text1: str, text2: str, model: SentenceTransformer) -> float:
    """Compute cosine similarity between two texts."""
    if not text1 or not text2:
        return 0.0
    
    emb1 = embed_text(text1, model)
    emb2 = embed_text(text2, model)
    
    if emb1.shape[0] == 0 or emb2.shape[0] == 0:
        return 0.0
    
    sim = cosine_similarity(emb1, emb2)[0][0]
    return float(sim)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_human_data() -> Dict[str, Dict]:
    """
    Load annotated.csv and extract all relevant fields.
    
    Returns:
        Dict: {image_id: {metadata fields + premises + conclusions}}
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
        
        # Extract metadata
        metadata = {
            'animals': row.get('animals', ''),
            'consequences': row.get('consequences', ''),
            'climateaction': row.get('climateaction', ''),
            'setting': row.get('setting', ''),
            'type': row.get('type', ''),
            'blip2_caption': row.get('blip2_caption', ''),
        }
        
        human_data[image_id] = {
            'metadata': metadata,
            'premises': premises,
            'conclusions': conclusions,
        }
    
    logger.info(f"Loaded human data for {len(human_data)} images")
    return human_data


def load_model_outputs() -> Dict[str, Dict[str, Dict]]:
    """
    Load all model JSON files from output directory.
    
    Returns:
        Dict: {model_name: {image_id: {"premises": [...], "conclusions": [...]}}}
    """
    logger.info(f"Loading model outputs from {MODELS_OUTPUT_DIR}")
    
    model_data = {}
    
    # Find all JSON files matching *_outputs.json pattern
    json_files = list(MODELS_OUTPUT_DIR.glob("*_outputs.json"))
    
    if not json_files:
        logger.warning(f"No JSON files found in {MODELS_OUTPUT_DIR}")
        return {}
    
    for json_file in json_files:
        # Extract model name from filename
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
                'conclusions': conclusions,
            }
        
        model_data[model_name] = model_images
        logger.info(f"Loaded {len(model_images)} image records for {model_name}")
    
    return model_data


# ============================================================================
# HALLUCINATION DETECTION
# ============================================================================

def build_support_text(metadata: Dict[str, str]) -> str:
    """Build combined metadata support text."""
    parts = [
        metadata.get('animals', ''),
        metadata.get('consequences', ''),
        metadata.get('climateaction', ''),
        metadata.get('setting', ''),
        metadata.get('type', ''),
        metadata.get('blip2_caption', ''),
    ]
    support_text = " ".join([p for p in parts if p])
    return support_text if support_text else "image"


def compute_hallucination_score(sentence: str,
                               support_text: str,
                               metadata_entities: Dict,
                               sbert_model: SentenceTransformer,
                               sentence_type: str = 'conclusion') -> Dict[str, float]:
    """
    Compute hallucination score for a single sentence.
    
    Args:
        sentence: Model-generated text
        support_text: Metadata support context
        metadata_entities: Extracted metadata entities
        sbert_model: SBERT model instance
        sentence_type: 'premise' or 'conclusion'
    
    Returns:
        Dict with scores: similarity, entity_penalty, hallucination_score, is_hallucination
    """
    if not sentence or not support_text:
        return {
            'similarity': 0.0,
            'entity_penalty': 0.0,
            'hallucination_score': 1.0,
            'is_hallucination': True,
        }
    
    # Signal 1: Embedding similarity
    similarity = compute_text_similarity(sentence, support_text, sbert_model)
    
    # Signal 2: Entity matching penalty
    text_entities = extract_text_entities(sentence)
    entity_penalty = compute_entity_mismatch_penalty(text_entities, metadata_entities)
    
    # Combine signals: 60% embedding + 40% entity penalty
    hallucination_score = 0.6 * (1.0 - similarity) + 0.4 * entity_penalty
    
    # Apply threshold based on sentence type
    threshold = PREMISE_THRESHOLD if sentence_type == 'premise' else CONCLUSION_THRESHOLD
    is_hallucination = similarity < threshold or hallucination_score > 0.5
    
    return {
        'similarity': float(similarity),
        'entity_penalty': float(entity_penalty),
        'hallucination_score': float(hallucination_score),
        'is_hallucination': is_hallucination,
    }


def evaluate_image_pair(image_id: str,
                       model_output: Dict,
                       metadata: Dict,
                       sbert_model: SentenceTransformer) -> Dict[str, float]:
    """
    Evaluate hallucination for one image-model pair.
    
    Returns:
        Dict with per-image metrics
    """
    # Build support text and extract metadata entities
    support_text = build_support_text(metadata)
    metadata_entities = extract_metadata_entities(metadata)
    
    # Evaluate premises
    premise_scores = []
    premise_hallucinations = []
    for premise in model_output.get('premises', []):
        score_dict = compute_hallucination_score(
            premise, support_text, metadata_entities, sbert_model, 'premise'
        )
        premise_scores.append(score_dict['similarity'])
        premise_hallucinations.append(score_dict['is_hallucination'])
    
    # Evaluate conclusions
    conclusion_scores = []
    conclusion_hallucinations = []
    for conclusion in model_output.get('conclusions', []):
        score_dict = compute_hallucination_score(
            conclusion, support_text, metadata_entities, sbert_model, 'conclusion'
        )
        conclusion_scores.append(score_dict['similarity'])
        conclusion_hallucinations.append(score_dict['is_hallucination'])
    
    # Compute metrics
    premise_halluc_rate = (sum(premise_hallucinations) / len(premise_hallucinations)
                          if premise_hallucinations else 0.0)
    conclusion_halluc_rate = (sum(conclusion_hallucinations) / len(conclusion_hallucinations)
                             if conclusion_hallucinations else 0.0)
    
    premise_avg_severity = (1.0 - np.mean(premise_scores)) if premise_scores else 0.0
    conclusion_avg_severity = (1.0 - np.mean(conclusion_scores)) if conclusion_scores else 0.0
    
    # Metadata mismatch rate
    all_entities = extract_text_entities(" ".join(model_output.get('premises', []) + 
                                                  model_output.get('conclusions', [])))
    entity_mismatch = compute_entity_mismatch_penalty(all_entities, metadata_entities)
    
    return {
        'image_id': image_id,
        'premise_hallucination_rate': float(premise_halluc_rate),
        'premise_avg_severity': float(premise_avg_severity),
        'conclusion_hallucination_rate': float(conclusion_halluc_rate),
        'conclusion_avg_severity': float(conclusion_avg_severity),
        'metadata_mismatch_rate': float(entity_mismatch),
    }


def evaluate_all_images(model_name: str,
                       model_output: Dict,
                       human_data: Dict,
                       sbert_model: SentenceTransformer) -> Tuple[List[Dict], Dict]:
    """
    Evaluate all images for a single model.
    
    Returns:
        - List of per-image dicts
        - Dict with aggregate statistics
    """
    per_image_results = []
    
    for image_id in sorted(human_data.keys()):
        # Get data
        human = human_data.get(image_id, {'metadata': {}, 'premises': [], 'conclusions': []})
        model = model_output.get(image_id, {'premises': [], 'conclusions': []})
        
        metrics = evaluate_image_pair(image_id, model, human['metadata'], sbert_model)
        per_image_results.append(metrics)
    
    # Compute aggregate statistics
    df_results = pd.DataFrame(per_image_results)
    
    aggregate_stats = {
        'n_images': len(per_image_results),
        'premise_hallucination_rate_mean': float(df_results['premise_hallucination_rate'].mean()),
        'premise_hallucination_rate_std': float(df_results['premise_hallucination_rate'].std()),
        'premise_avg_severity_mean': float(df_results['premise_avg_severity'].mean()),
        'premise_avg_severity_std': float(df_results['premise_avg_severity'].std()),
        'conclusion_hallucination_rate_mean': float(df_results['conclusion_hallucination_rate'].mean()),
        'conclusion_hallucination_rate_std': float(df_results['conclusion_hallucination_rate'].std()),
        'conclusion_avg_severity_mean': float(df_results['conclusion_avg_severity'].mean()),
        'conclusion_avg_severity_std': float(df_results['conclusion_avg_severity'].std()),
        'metadata_mismatch_rate_mean': float(df_results['metadata_mismatch_rate'].mean()),
        'metadata_mismatch_rate_std': float(df_results['metadata_mismatch_rate'].std()),
    }
    
    return per_image_results, aggregate_stats


# ============================================================================
# OUTPUT GENERATION
# ============================================================================

def save_per_model_csv(results: List[Dict], model_name: str, output_dir: Path) -> Path:
    """Save per-image hallucination results CSV for one model."""
    df = pd.DataFrame(results)
    
    # Round to 4 decimal places
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(4)
    
    output_path = output_dir / f"{model_name}_hallucination.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved per-model hallucination results: {output_path}")
    
    return output_path


def save_summary_csv(all_stats: Dict[str, Dict], output_dir: Path) -> Path:
    """Save summary hallucination comparison CSV across all models."""
    summary_data = []
    
    for model_name, stats in all_stats.items():
        summary_data.append({
            'Model': model_name,
            'N_Images': stats['n_images'],
            'Premise_Halluc_Rate_Mean': round(stats['premise_hallucination_rate_mean'], 4),
            'Premise_Halluc_Rate_Std': round(stats['premise_hallucination_rate_std'], 4),
            'Premise_Severity_Mean': round(stats['premise_avg_severity_mean'], 4),
            'Premise_Severity_Std': round(stats['premise_avg_severity_std'], 4),
            'Conclusion_Halluc_Rate_Mean': round(stats['conclusion_hallucination_rate_mean'], 4),
            'Conclusion_Halluc_Rate_Std': round(stats['conclusion_hallucination_rate_std'], 4),
            'Conclusion_Severity_Mean': round(stats['conclusion_avg_severity_mean'], 4),
            'Conclusion_Severity_Std': round(stats['conclusion_avg_severity_std'], 4),
            'Metadata_Mismatch_Mean': round(stats['metadata_mismatch_rate_mean'], 4),
        })
    
    df_summary = pd.DataFrame(summary_data)
    output_path = output_dir / "summary_hallucination.csv"
    df_summary.to_csv(output_path, index=False)
    logger.info(f"Saved summary hallucination comparison: {output_path}")
    
    return output_path


def save_details_json(all_results: Dict[str, List[Dict]], output_dir: Path) -> Path:
    """Save detailed hallucination information as JSON."""
    details = {}
    
    for model_name, results in all_results.items():
        # Sort by hallucination severity
        sorted_results = sorted(
            results,
            key=lambda x: (x['premise_hallucination_rate'] + x['conclusion_hallucination_rate']) / 2,
            reverse=True
        )
        
        # Keep top 10 most problematic images
        details[model_name] = sorted_results[:10]
    
    output_path = output_dir / "hallucination_details.json"
    with open(output_path, 'w') as f:
        json.dump(details, f, indent=2)
    logger.info(f"Saved hallucination details: {output_path}")
    
    return output_path


# ============================================================================
# VALIDATION
# ============================================================================

def validate_results(results: List[Dict], model_name: str) -> bool:
    """Validate hallucination scores are in valid range."""
    df = pd.DataFrame(results)
    
    # Check for NaN values
    if df.isnull().any().any():
        logger.error(f"NaN values found in {model_name} results")
        return False
    
    # Check scores in [0, 1]
    score_cols = [
        'premise_hallucination_rate', 'premise_avg_severity',
        'conclusion_hallucination_rate', 'conclusion_avg_severity',
        'metadata_mismatch_rate'
    ]
    for col in score_cols:
        if (df[col] < 0).any() or (df[col] > 1).any():
            logger.error(f"Invalid scores in {model_name}: {col}")
            return False
    
    logger.info(f"Validation passed for {model_name}: {len(results)} images")
    return True


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main hallucination detection pipeline."""
    logger.info("=" * 70)
    logger.info("Hallucination Detection Pipeline")
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
    logger.info("\n[Phase 3] Computing hallucination scores...")
    all_model_results = {}
    all_model_stats = {}
    
    for model_name in sorted(model_data.keys()):
        logger.info(f"\nEvaluating {model_name}...")
        model_imgs = model_data[model_name]
        
        per_image_results, aggregate_stats = evaluate_all_images(
            model_name, model_imgs, human_data, sbert_model
        )
        
        # Validate
        if not validate_results(per_image_results, model_name):
            logger.warning(f"Validation failed for {model_name}, skipping output")
            continue
        
        all_model_results[model_name] = per_image_results
        all_model_stats[model_name] = aggregate_stats
        
        logger.info(f"{model_name} - Premise Halluc Rate: {aggregate_stats['premise_hallucination_rate_mean']:.4f} "
                   f"± {aggregate_stats['premise_hallucination_rate_std']:.4f}")
        logger.info(f"{model_name} - Conclusion Halluc Rate: {aggregate_stats['conclusion_hallucination_rate_mean']:.4f} "
                   f"± {aggregate_stats['conclusion_hallucination_rate_std']:.4f}")
    
    # Phase 4: Save outputs
    logger.info("\n[Phase 4] Saving outputs...")
    saved_files = []
    
    for model_name, results in all_model_results.items():
        csv_path = save_per_model_csv(results, model_name, EVAL_OUTPUT_DIR)
        saved_files.append(csv_path)
    
    summary_path = save_summary_csv(all_model_stats, EVAL_OUTPUT_DIR)
    saved_files.append(summary_path)
    
    details_path = save_details_json(all_model_results, EVAL_OUTPUT_DIR)
    saved_files.append(details_path)
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("Hallucination Detection Complete!")
    logger.info("=" * 70)
    logger.info(f"Models evaluated: {len(all_model_stats)}")
    logger.info(f"Images per model: {all_model_stats[list(all_model_stats.keys())[0]]['n_images']}")
    logger.info(f"\nOutput files saved:")
    for file_path in saved_files:
        logger.info(f"  - {file_path}")
    
    # Print summary table
    logger.info("\nHallucination Summary Statistics:")
    logger.info("-" * 70)
    for model_name, stats in all_model_stats.items():
        logger.info(f"\n{model_name}:")
        logger.info(f"  Premise Hallucination Rate:    {stats['premise_hallucination_rate_mean']:.4f} "
                   f"± {stats['premise_hallucination_rate_std']:.4f}")
        logger.info(f"  Premise Avg Severity:          {stats['premise_avg_severity_mean']:.4f} "
                   f"± {stats['premise_avg_severity_std']:.4f}")
        logger.info(f"  Conclusion Hallucination Rate: {stats['conclusion_hallucination_rate_mean']:.4f} "
                   f"± {stats['conclusion_hallucination_rate_std']:.4f}")
        logger.info(f"  Conclusion Avg Severity:       {stats['conclusion_avg_severity_mean']:.4f} "
                   f"± {stats['conclusion_avg_severity_std']:.4f}")
        logger.info(f"  Metadata Mismatch Rate:        {stats['metadata_mismatch_rate_mean']:.4f} "
                   f"± {stats['metadata_mismatch_rate_std']:.4f}")
    
    logger.info("\n" + "=" * 70)
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
