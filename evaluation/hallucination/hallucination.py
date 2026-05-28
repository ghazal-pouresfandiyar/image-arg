"""
Hallucination Detection Script (Deep Grounding)
==============================================
Detect unsupported claims in model outputs using:
1. SBERT similarity to metadata support
2. SBERT similarity to BLIP2 captions
3. CLIP text-image similarity (optional)
4. Rule-based entity mismatch penalties

Outputs (new names - legacy outputs untouched):
  - Per-model CSVs: {model_name}_hallucination_deep.csv
  - Summary CSV: summary_hallucination_deep.csv
  - Details JSON: hallucination_details_deep.json
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

try:
    import torch
    import torch.nn.functional as F
    from transformers import CLIPModel, CLIPProcessor
    HAS_CLIP = True
except Exception as exc:  # pragma: no cover - environment dependent
    HAS_CLIP = False
    CLIP_IMPORT_ERROR = str(exc)


# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "dataset"
MODELS_OUTPUT_DIR = PROJECT_ROOT / "models" / "output_model"
EVAL_OUTPUT_DIR = Path(__file__).parent
ANNOTATED_CSV = DATA_DIR / "annotated.csv"

# CLIP assets
CLIP_FEATURES_DIR = DATA_DIR / "features"
CLIP_EMBEDDINGS_PATH = CLIP_FEATURES_DIR / "clip_image_embeddings.npy"
CLIP_INDEX_PATH = CLIP_FEATURES_DIR / "clip_index.csv"
CLIP_CONFIG_PATH = CLIP_FEATURES_DIR / "clip_config.json"
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"

# SBERT model
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

# Scoring weights
WEIGHT_METADATA = 0.50
WEIGHT_SIGNAL_B = 0.30
WEIGHT_ENTITY = 0.20

# Label thresholds
GROUNDED_MAX = 0.30
WEAKLY_GROUNDED_MAX = 0.60

# Output names (avoid overwriting legacy outputs)
PER_MODEL_SUFFIX = "hallucination_deep.csv"
SUMMARY_FILE = "summary_hallucination_deep.csv"
DETAILS_FILE = "hallucination_details_deep.json"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

NO_VALUE_MARKERS = {"no animals", "no climate action", "none", "n/a", "na"}


def normalize_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip().lower()


def clean_support_parts(parts: List[Any]) -> List[str]:
    cleaned: List[str] = []
    seen = set()
    for part in parts:
        value = normalize_text(part)
        if not value or value in NO_VALUE_MARKERS:
            continue
        if value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    return cleaned


def build_support_context(metadata: Dict[str, str]) -> Tuple[str, str, str]:
    metadata_parts = [
        metadata.get("animals", ""),
        metadata.get("consequences", ""),
        metadata.get("climateaction", ""),
        metadata.get("setting", ""),
    ]
    metadata_text = " ".join(clean_support_parts(metadata_parts))
    blip_text = normalize_text(metadata.get("blip2_caption", ""))
    support_text = " ".join(clean_support_parts([metadata_text, blip_text]))
    return metadata_text, blip_text, support_text


# ============================================================================
# ENTITY EXTRACTION
# ============================================================================

ANIMAL_TERMS = {
    "bear", "polar bear", "fish", "whale", "dolphin", "seal", "penguin",
    "bird", "eagle", "owl", "deer", "elk", "cow", "sheep", "goat",
    "chicken", "bee", "butterfly", "turtle", "coral", "shark", "frog",
    "elephant", "lion", "tiger", "insect", "spider", "ant",
}

CLIMATE_EVENT_TERMS = {
    "climate change", "warming", "heatwave", "drought", "flood", "wildfire",
    "hurricane", "storm", "typhoon", "cyclone", "melting", "ice melt",
    "sea level", "acidification", "bleaching", "extinction", "pollution",
    "deforestation", "smog", "emissions", "greenhouse", "carbon",
}

LOCATION_TERMS = {
    "arctic", "antarctic", "pacific", "atlantic", "indian ocean",
    "north america", "south america", "europe", "asia", "africa",
    "australia", "great barrier reef", "reef", "coast", "river", "lake",
    "ocean", "sea", "glacier", "iceberg",
}

NUMERIC_PATTERNS = [
    r"\b(19|20)\d{2}\b",  # Years
    r"\b\d+(?:\.\d+)?%\b",  # Percentages
    r"\b-?\d+(?:\.\d+)?\s*(?:°c|°f|degrees?)\b",  # Temperatures
    r"\b\d+(?:\.\d+)?\s*(?:to|-)\s*\d+(?:\.\d+)?\b",  # Ranges
    r"\b\d+(?:\.\d+)?\s*ppm\b",  # PPM values
]


def _contains_term(text: str, term: str) -> bool:
    if " " in term or "-" in term:
        return term in text
    return re.search(r"\b" + re.escape(term) + r"\b", text) is not None


def extract_entities(text: str) -> Dict[str, set]:
    text_norm = normalize_text(text)
    entities = {
        "animals": set(),
        "climate_events": set(),
        "locations": set(),
        "numeric_claims": set(),
    }

    for term in ANIMAL_TERMS:
        if _contains_term(text_norm, term):
            entities["animals"].add(term)

    for term in CLIMATE_EVENT_TERMS:
        if _contains_term(text_norm, term):
            entities["climate_events"].add(term)

    for term in LOCATION_TERMS:
        if _contains_term(text_norm, term):
            entities["locations"].add(term)

    for pattern in NUMERIC_PATTERNS:
        for match in re.findall(pattern, text_norm):
            entities["numeric_claims"].add(match)

    return entities


def _mismatch_penalty(text_set: set, support_set: set, weight: float) -> float:
    if not text_set:
        return 0.0
    if not support_set:
        return weight
    unmatched = text_set - support_set
    if not unmatched:
        return 0.0
    return weight * (len(unmatched) / max(len(text_set), 1))


def compute_entity_penalty(
    text_entities: Dict[str, set],
    support_entities: Dict[str, set],
    support_text: str,
) -> float:
    penalty = 0.0
    penalty += _mismatch_penalty(text_entities["animals"], support_entities["animals"], 0.30)
    penalty += _mismatch_penalty(
        text_entities["climate_events"], support_entities["climate_events"], 0.30
    )
    penalty += _mismatch_penalty(
        text_entities["locations"], support_entities["locations"], 0.20
    )

    if text_entities["numeric_claims"]:
        has_numeric_support = bool(re.search(r"\d", support_text))
        if not has_numeric_support:
            penalty += 0.20 * min(len(text_entities["numeric_claims"]) / 2, 1.0)

    return min(penalty, 1.0)


# ============================================================================
# MODEL LOADING
# ============================================================================


def load_sbert_model() -> SentenceTransformer:
    logger.info(f"Loading SBERT model: {SBERT_MODEL_NAME}")
    return SentenceTransformer(SBERT_MODEL_NAME)


def load_clip_resources() -> Optional[Dict[str, Any]]:
    if not (CLIP_EMBEDDINGS_PATH.exists() and CLIP_INDEX_PATH.exists()):
        logger.warning("CLIP embeddings or index missing; skipping CLIP signal.")
        return None

    if not HAS_CLIP:
        logger.warning(f"CLIP dependencies missing; skipping CLIP signal: {CLIP_IMPORT_ERROR}")
        return None

    model_name = DEFAULT_CLIP_MODEL
    if CLIP_CONFIG_PATH.exists():
        try:
            config = json.loads(CLIP_CONFIG_PATH.read_text(encoding="utf-8"))
            model_name = config.get("model", model_name)
        except Exception as exc:
            logger.warning(f"Failed to read CLIP config: {exc}")

    try:
        image_embeddings = np.load(CLIP_EMBEDDINGS_PATH)
    except Exception as exc:
        logger.warning(f"Failed to load CLIP embeddings: {exc}")
        return None

    try:
        index_df = pd.read_csv(CLIP_INDEX_PATH, dtype=str, keep_default_na=False)
    except Exception as exc:
        logger.warning(f"Failed to load CLIP index: {exc}")
        return None

    if "id" not in index_df.columns:
        logger.warning("CLIP index missing 'id' column; skipping CLIP signal.")
        return None

    id_to_index = {str(row_id).strip(): idx for idx, row_id in enumerate(index_df["id"].tolist())}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        model.to(device)
        model.eval()
    except Exception as exc:
        logger.warning(f"Failed to load CLIP model '{model_name}': {exc}")
        return None

    # Normalize embeddings once
    norms = np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    image_embeddings = image_embeddings / norms

    return {
        "model": model,
        "processor": processor,
        "device": device,
        "image_embeddings": image_embeddings.astype(np.float32),
        "id_to_index": id_to_index,
        "model_name": model_name,
    }


# ============================================================================
# EMBEDDING HELPERS
# ============================================================================


def _cosine_similarity(vec_a: Optional[np.ndarray], vec_b: Optional[np.ndarray]) -> float:
    if vec_a is None or vec_b is None:
        return 0.0
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0:
        return 0.0
    score = float(np.dot(vec_a, vec_b) / denom)
    return max(0.0, min(1.0, score))


def encode_sbert_texts(
    texts: List[str],
    model: SentenceTransformer,
    cache: Dict[str, np.ndarray],
) -> None:
    unique_texts = [text for text in texts if text and text not in cache]
    if not unique_texts:
        return
    embeddings = model.encode(unique_texts, convert_to_numpy=True)
    for text, embedding in zip(unique_texts, embeddings):
        cache[text] = embedding


def encode_clip_texts(
    texts: List[str],
    clip_resources: Optional[Dict[str, Any]],
    cache: Dict[str, np.ndarray],
    batch_size: int = 32,
) -> None:
    if clip_resources is None:
        return

    unique_texts = [text for text in texts if text and text not in cache]
    if not unique_texts:
        return

    processor = clip_resources["processor"]
    model = clip_resources["model"]
    device = clip_resources["device"]

    for i in range(0, len(unique_texts), batch_size):
        batch = unique_texts[i:i + batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
        if hasattr(text_features, "pooler_output"):
            text_features = text_features.pooler_output
        elif isinstance(text_features, (tuple, list)) and text_features:
            text_features = text_features[0]

        if not torch.is_tensor(text_features):
            try:
                text_features = torch.tensor(text_features)
            except Exception as exc:
                logger.warning(f"CLIP text features not tensor; skipping batch: {exc}")
                continue

        if text_features.ndim == 1:
            text_features = text_features.unsqueeze(0)

        text_features = F.normalize(text_features, p=2, dim=-1)
        batch_embeddings = text_features.detach().cpu().numpy().astype(np.float32)
        for text, embedding in zip(batch, batch_embeddings):
            cache[text] = embedding


def compute_clip_similarity(
    text: str,
    image_id: str,
    clip_resources: Optional[Dict[str, Any]],
    clip_cache: Dict[str, np.ndarray],
    missing_clip_ids: set,
) -> Optional[float]:
    if clip_resources is None:
        return None

    image_index = clip_resources["id_to_index"].get(image_id)
    if image_index is None:
        if image_id not in missing_clip_ids:
            missing_clip_ids.add(image_id)
            logger.warning(f"Missing CLIP embedding for image_id={image_id}")
        return None

    text_embedding = clip_cache.get(text)
    if text_embedding is None:
        encode_clip_texts([text], clip_resources, clip_cache)
        text_embedding = clip_cache.get(text)

    if text_embedding is None:
        return None

    image_embedding = clip_resources["image_embeddings"][image_index]
    denom = (np.linalg.norm(text_embedding) * np.linalg.norm(image_embedding))
    if denom == 0:
        return None

    score = float(np.dot(text_embedding, image_embedding) / denom)
    score = (score + 1.0) / 2.0
    return max(0.0, min(1.0, score))


# ============================================================================
# DATA LOADING
# ============================================================================


def _parse_json_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return []
    return []


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def load_human_data() -> Dict[str, Dict[str, Any]]:
    logger.info(f"Loading human annotations from {ANNOTATED_CSV}")

    try:
        df = pd.read_csv(ANNOTATED_CSV)
    except FileNotFoundError:
        logger.error(f"File not found: {ANNOTATED_CSV}")
        return {}

    human_data: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        image_id = str(row.get("id", "")).strip()
        if not image_id:
            continue

        premises = _parse_json_list(row.get("premises"))
        conclusions = _parse_json_list(row.get("conclusions"))

        metadata = {
            "animals": _safe_text(row.get("animals", "")),
            "consequences": _safe_text(row.get("consequences", "")),
            "climateaction": _safe_text(row.get("climateaction", "")),
            "setting": _safe_text(row.get("setting", "")),
            "blip2_caption": _safe_text(row.get("blip2_caption", "")),
        }

        human_data[image_id] = {
            "metadata": metadata,
            "premises": premises,
            "conclusions": conclusions,
        }

    logger.info(f"Loaded human data for {len(human_data)} images")
    return human_data


def _extract_text_from_item(item: Any) -> List[str]:
    if item is None:
        return []
    if isinstance(item, list):
        texts: List[str] = []
        for entry in item:
            texts.extend(_extract_text_from_item(entry))
        return texts
    if isinstance(item, dict):
        for key in ["text", "observation", "inference", "premise", "conclusion"]:
            if key in item and isinstance(item[key], str):
                return [item[key]]
        for value in item.values():
            if isinstance(value, str):
                return [value]
        return []
    if isinstance(item, str):
        return [item]
    return []


LABEL_REGEX = re.compile(r"(premise\s*\d*\s*:|conclusion\s*\d*\s*:)", re.IGNORECASE)


def _clean_sentence_text(text: str, sentence_type: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r"\s+", " ", text.replace("\n", " ").replace("\r", " ")).strip()
    lowered = cleaned.lower()

    if sentence_type == "premise":
        idx = lowered.find("conclusion:")
        if idx != -1:
            cleaned = cleaned[:idx]
    elif sentence_type == "conclusion":
        idx = lowered.rfind("conclusion:")
        if idx != -1:
            cleaned = cleaned[idx + len("conclusion:"):]

    cleaned = re.sub(
        r"^\s*(premise|conclusion)\s*\d*\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^\s*[\-\*\d]+\.\s*", "", cleaned)
    return cleaned.strip()


def _split_labeled_sections(text: str, sentence_type: str) -> List[str]:
    matches = list(LABEL_REGEX.finditer(text))
    if not matches:
        cleaned = _clean_sentence_text(text, sentence_type)
        return [cleaned] if cleaned else []

    sections: List[str] = []
    for idx, match in enumerate(matches):
        label = match.group(0).lower()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        segment = text[start:end]
        if label.startswith(sentence_type):
            cleaned = _clean_sentence_text(segment, sentence_type)
            if cleaned:
                sections.append(cleaned)

    if sections:
        return sections

    cleaned = _clean_sentence_text(text, sentence_type)
    return [cleaned] if cleaned else []


def normalize_sentences(value: Any, sentence_type: str) -> List[str]:
    texts = _extract_text_from_item(value)
    normalized: List[str] = []
    for text in texts:
        normalized.extend(_split_labeled_sections(text, sentence_type))
    return [text for text in normalized if text]


def _parse_raw_output(raw_output: Any) -> Dict[str, Any]:
    if not isinstance(raw_output, str) or not raw_output.strip():
        return {}

    match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if not match:
        return {}

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        logger.warning(f"Malformed JSON in raw_output: {exc}")
        return {}

    return parsed if isinstance(parsed, dict) else {}


def load_model_outputs() -> Dict[str, Dict[str, Dict[str, Any]]]:
    logger.info(f"Loading model outputs from {MODELS_OUTPUT_DIR}")

    model_data: Dict[str, Dict[str, Dict[str, Any]]] = {}
    json_files = list(MODELS_OUTPUT_DIR.glob("*.json"))

    if not json_files:
        logger.warning(f"No JSON files found in {MODELS_OUTPUT_DIR}")
        return {}

    for json_file in json_files:
        model_name = json_file.stem.replace("_outputs", "")
        logger.info(f"Loading {model_name} from {json_file.name}")

        try:
            with open(json_file, "r") as handle:
                records = json.load(handle)
        except (json.JSONDecodeError, FileNotFoundError) as exc:
            logger.error(f"Failed to load {json_file}: {exc}")
            continue

        if isinstance(records, dict):
            records = [records]
        if not isinstance(records, list):
            logger.warning(f"Unexpected JSON structure in {json_file.name}")
            continue

        model_images: Dict[str, Dict[str, Any]] = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            image_id = str(record.get("image_id") or record.get("id") or "").strip()
            if not image_id:
                continue

            parsed_output = record.get("parsed_output") or {}
            if not parsed_output:
                parsed_output = _parse_raw_output(record.get("raw_output"))

            premises_value = parsed_output.get("premises", record.get("premises", []))
            conclusions_value = parsed_output.get("conclusions", record.get("conclusions", []))

            premises = normalize_sentences(premises_value, "premise")
            conclusions = normalize_sentences(conclusions_value, "conclusion")

            model_images[image_id] = {
                "premises": premises,
                "conclusions": conclusions,
                "raw_output": record.get("raw_output", ""),
            }

        if not model_images:
            logger.warning(f"No valid image records found for {model_name}")
            continue

        model_data[model_name] = model_images
        logger.info(f"Loaded {len(model_images)} image records for {model_name}")

    return model_data


# ============================================================================
# SCORING
# ============================================================================


def score_to_label(score: float) -> str:
    if score <= GROUNDED_MAX:
        return "grounded"
    if score <= WEAKLY_GROUNDED_MAX:
        return "weakly grounded"
    return "hallucinated"


def build_support_cache(
    human_data: Dict[str, Dict[str, Any]],
    sbert_model: SentenceTransformer,
) -> Dict[str, Dict[str, Any]]:
    image_ids = sorted(human_data.keys())
    metadata_texts: List[str] = []
    blip_texts: List[str] = []
    contexts: Dict[str, Dict[str, Any]] = {}

    for image_id in image_ids:
        metadata = human_data[image_id]["metadata"]
        metadata_text, blip_text, support_text = build_support_context(metadata)
        metadata_texts.append(metadata_text)
        blip_texts.append(blip_text)
        contexts[image_id] = {
            "metadata_text": metadata_text,
            "blip_text": blip_text,
            "support_text": support_text,
            "metadata": metadata,
        }

    if metadata_texts:
        metadata_embeddings = sbert_model.encode(
            [text if text else " " for text in metadata_texts], convert_to_numpy=True
        )
    else:
        metadata_embeddings = []

    if blip_texts:
        blip_embeddings = sbert_model.encode(
            [text if text else " " for text in blip_texts], convert_to_numpy=True
        )
    else:
        blip_embeddings = []

    for idx, image_id in enumerate(image_ids):
        metadata_text = contexts[image_id]["metadata_text"]
        blip_text = contexts[image_id]["blip_text"]
        support_text = contexts[image_id]["support_text"]

        contexts[image_id]["metadata_embedding"] = (
            metadata_embeddings[idx] if metadata_text else None
        )
        contexts[image_id]["blip_embedding"] = (
            blip_embeddings[idx] if blip_text else None
        )
        contexts[image_id]["support_entities"] = extract_entities(support_text)

    return contexts


def score_sentence(
    sentence: str,
    sentence_type: str,
    context: Dict[str, Any],
    sbert_cache: Dict[str, np.ndarray],
    sbert_model: SentenceTransformer,
    clip_resources: Optional[Dict[str, Any]],
    clip_cache: Dict[str, np.ndarray],
    missing_clip_ids: set,
    image_id: str,
) -> Dict[str, Any]:
    cleaned = _clean_sentence_text(sentence, sentence_type)
    if not cleaned:
        return {
            "text": sentence,
            "cleaned_text": "",
            "metadata_similarity": 0.0,
            "blip_similarity": 0.0,
            "clip_similarity": None,
            "entity_penalty": 1.0,
            "hallucination_score": 1.0,
            "label": "hallucinated",
        }

    encode_sbert_texts([cleaned], sbert_model, sbert_cache)
    sentence_embedding = sbert_cache.get(cleaned)

    metadata_similarity = _cosine_similarity(sentence_embedding, context["metadata_embedding"])
    blip_similarity = _cosine_similarity(sentence_embedding, context["blip_embedding"])
    clip_similarity = compute_clip_similarity(
        cleaned, image_id, clip_resources, clip_cache, missing_clip_ids
    )

    signal_values = [blip_similarity]
    if clip_similarity is not None:
        signal_values.append(clip_similarity)
    signal_b = sum(signal_values) / max(len(signal_values), 1)

    text_entities = extract_entities(cleaned)
    entity_penalty = compute_entity_penalty(
        text_entities, context["support_entities"], context["support_text"]
    )

    hallucination_score = (
        WEIGHT_METADATA * (1.0 - metadata_similarity)
        + WEIGHT_SIGNAL_B * (1.0 - signal_b)
        + WEIGHT_ENTITY * entity_penalty
    )
    hallucination_score = max(0.0, min(1.0, hallucination_score))

    return {
        "text": sentence,
        "cleaned_text": cleaned,
        "metadata_similarity": float(metadata_similarity),
        "blip_similarity": float(blip_similarity),
        "clip_similarity": None if clip_similarity is None else float(clip_similarity),
        "entity_penalty": float(entity_penalty),
        "hallucination_score": float(hallucination_score),
        "label": score_to_label(hallucination_score),
    }


def _compute_sentence_stats(score_details: List[Dict[str, Any]]) -> Tuple[float, float]:
    if not score_details:
        return 1.0, 1.0
    scores = [entry["hallucination_score"] for entry in score_details]
    hallucinated = [entry for entry in score_details if entry["label"] == "hallucinated"]
    return float(np.mean(scores)), float(len(hallucinated) / len(score_details))


def evaluate_record(
    record: Dict[str, Any],
    context: Dict[str, Any],
    sbert_cache: Dict[str, np.ndarray],
    sbert_model: SentenceTransformer,
    clip_resources: Optional[Dict[str, Any]],
    clip_cache: Dict[str, np.ndarray],
    missing_clip_ids: set,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    image_id = record["image_id"]
    premises = record.get("premises", [])
    conclusions = record.get("conclusions", [])

    if not premises:
        logger.warning(f"Empty premises for image_id={image_id}")
    if not conclusions:
        logger.warning(f"Empty conclusions for image_id={image_id}")

    premise_details = [
        score_sentence(
            sentence,
            "premise",
            context,
            sbert_cache,
            sbert_model,
            clip_resources,
            clip_cache,
            missing_clip_ids,
            image_id,
        )
        for sentence in premises
    ]
    conclusion_details = [
        score_sentence(
            sentence,
            "conclusion",
            context,
            sbert_cache,
            sbert_model,
            clip_resources,
            clip_cache,
            missing_clip_ids,
            image_id,
        )
        for sentence in conclusions
    ]

    premise_score, premise_halluc_rate = _compute_sentence_stats(premise_details)
    conclusion_score, conclusion_halluc_rate = _compute_sentence_stats(conclusion_details)
    overall_score = 0.7 * premise_score + 0.3 * conclusion_score
    overall_label = score_to_label(overall_score)

    if overall_score > 0.9 or overall_score < 0.1:
        logger.info(f"Extreme overall score for image_id={image_id}: {overall_score:.4f}")

    summary = {
        "image_id": image_id,
        "premise_score": float(premise_score),
        "conclusion_score": float(conclusion_score),
        "overall_score": float(overall_score),
        "label": overall_label,
        "premise_hallucination_rate": float(premise_halluc_rate),
        "conclusion_hallucination_rate": float(conclusion_halluc_rate),
    }

    details = {
        "image_id": image_id,
        "premises": premises,
        "conclusions": conclusions,
        "premise_details": premise_details,
        "conclusion_details": conclusion_details,
        "premise_score": float(premise_score),
        "conclusion_score": float(conclusion_score),
        "overall_score": float(overall_score),
        "label": overall_label,
        "metadata": context["metadata"],
        "blip2_caption": context["metadata"].get("blip2_caption", ""),
    }

    return summary, details


def evaluate_model(
    model_name: str,
    model_outputs: Dict[str, Dict[str, Any]],
    human_data: Dict[str, Dict[str, Any]],
    contexts: Dict[str, Dict[str, Any]],
    sbert_model: SentenceTransformer,
    clip_resources: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    summary_results: List[Dict[str, Any]] = []
    detail_results: List[Dict[str, Any]] = []

    sbert_cache: Dict[str, np.ndarray] = {}
    clip_cache: Dict[str, np.ndarray] = {}
    missing_clip_ids: set = set()

    all_sentences: List[str] = []
    for image_id in human_data.keys():
        model_record = model_outputs.get(image_id, {})
        all_sentences.extend(
            [_clean_sentence_text(text, "premise") for text in model_record.get("premises", [])]
        )
        all_sentences.extend(
            [_clean_sentence_text(text, "conclusion") for text in model_record.get("conclusions", [])]
        )

    all_sentences = [text for text in all_sentences if text]
    encode_sbert_texts(all_sentences, sbert_model, sbert_cache)
    encode_clip_texts(all_sentences, clip_resources, clip_cache)

    for image_id in sorted(human_data.keys()):
        model_record = model_outputs.get(image_id, {"premises": [], "conclusions": []})
        record = {
            "image_id": image_id,
            "model_name": model_name,
            "premises": model_record.get("premises", []),
            "conclusions": model_record.get("conclusions", []),
        }
        summary, details = evaluate_record(
            record,
            contexts[image_id],
            sbert_cache,
            sbert_model,
            clip_resources,
            clip_cache,
            missing_clip_ids,
        )
        summary_results.append(summary)
        detail_results.append(details)

    df = pd.DataFrame(summary_results)
    hallucinated_rate = float((df["label"] == "hallucinated").mean())

    aggregate_stats = {
        "model": model_name,
        "n_images": len(summary_results),
        "premise_score_mean": float(df["premise_score"].mean()),
        "premise_score_std": float(df["premise_score"].std()),
        "conclusion_score_mean": float(df["conclusion_score"].mean()),
        "conclusion_score_std": float(df["conclusion_score"].std()),
        "overall_score_mean": float(df["overall_score"].mean()),
        "overall_score_std": float(df["overall_score"].std()),
        "hallucination_rate": hallucinated_rate,
    }

    return summary_results, detail_results, aggregate_stats


# ============================================================================
# OUTPUT GENERATION
# ============================================================================


def save_per_model_csv(results: List[Dict[str, Any]], model_name: str, output_dir: Path) -> Path:
    df = pd.DataFrame(results)
    df = df[["image_id", "premise_score", "conclusion_score", "overall_score", "label"]]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(4)

    output_path = output_dir / f"{model_name}_{PER_MODEL_SUFFIX}"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved per-model deep hallucination results: {output_path}")
    return output_path


def save_summary_csv(all_stats: Dict[str, Dict[str, Any]], output_dir: Path) -> Path:
    summary_rows = []
    for model_name, stats in all_stats.items():
        summary_rows.append({
            "Model": model_name,
            "N_Images": stats["n_images"],
            "Premise_Mean": round(stats["premise_score_mean"], 4),
            "Premise_Std": round(stats["premise_score_std"], 4),
            "Conclusion_Mean": round(stats["conclusion_score_mean"], 4),
            "Conclusion_Std": round(stats["conclusion_score_std"], 4),
            "Overall_Mean": round(stats["overall_score_mean"], 4),
            "Overall_Std": round(stats["overall_score_std"], 4),
            "Hallucination_Rate": round(stats["hallucination_rate"], 4),
        })

    df_summary = pd.DataFrame(summary_rows)
    output_path = output_dir / SUMMARY_FILE
    df_summary.to_csv(output_path, index=False)
    logger.info(f"Saved deep hallucination summary: {output_path}")
    return output_path


def save_details_json(all_details: Dict[str, List[Dict[str, Any]]], output_dir: Path) -> Path:
    details = {}
    for model_name, results in all_details.items():
        sorted_results = sorted(results, key=lambda x: x["overall_score"], reverse=True)
        details[model_name] = sorted_results[:10]

    output_path = output_dir / DETAILS_FILE
    with open(output_path, "w") as handle:
        json.dump(details, handle, indent=2)
    logger.info(f"Saved deep hallucination details: {output_path}")
    return output_path


# ============================================================================
# VALIDATION
# ============================================================================


def validate_results(results: List[Dict[str, Any]], model_name: str) -> bool:
    df = pd.DataFrame(results)
    if df.isnull().any().any():
        logger.error(f"NaN values found in {model_name} results")
        return False

    for col in ["premise_score", "conclusion_score", "overall_score"]:
        if (df[col] < 0).any() or (df[col] > 1).any():
            logger.error(f"Invalid scores in {model_name}: {col}")
            return False

    logger.info(f"Validation passed for {model_name}: {len(results)} images")
    return True


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main() -> bool:
    logger.info("=" * 70)
    logger.info("Deep Hallucination Detection Pipeline")
    logger.info("=" * 70)

    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("\n[Phase 1] Loading data...")
    human_data = load_human_data()
    model_data = load_model_outputs()

    if not human_data:
        logger.error("No human data loaded. Exiting.")
        return False
    if not model_data:
        logger.error("No model data loaded. Exiting.")
        return False

    logger.info("\n[Phase 2] Loading models...")
    sbert_model = load_sbert_model()
    clip_resources = load_clip_resources()
    if clip_resources:
        logger.info(f"CLIP enabled: {clip_resources['model_name']}")
    else:
        logger.info("CLIP disabled; proceeding with metadata + BLIP signals only")

    logger.info("\n[Phase 3] Precomputing support embeddings...")
    contexts = build_support_cache(human_data, sbert_model)

    logger.info("\n[Phase 4] Computing hallucination scores...")
    all_model_results: Dict[str, List[Dict[str, Any]]] = {}
    all_model_details: Dict[str, List[Dict[str, Any]]] = {}
    all_model_stats: Dict[str, Dict[str, Any]] = {}

    for model_name in sorted(model_data.keys()):
        logger.info(f"\nEvaluating {model_name}...")
        summary_results, detail_results, stats = evaluate_model(
            model_name,
            model_data[model_name],
            human_data,
            contexts,
            sbert_model,
            clip_resources,
        )

        if not validate_results(summary_results, model_name):
            logger.warning(f"Validation failed for {model_name}, skipping output")
            continue

        all_model_results[model_name] = summary_results
        all_model_details[model_name] = detail_results
        all_model_stats[model_name] = stats

        logger.info(
            f"{model_name} - Overall Mean: {stats['overall_score_mean']:.4f} "
            f"± {stats['overall_score_std']:.4f}"
        )

    logger.info("\n[Phase 5] Saving outputs...")
    saved_files = []

    for model_name, results in all_model_results.items():
        saved_files.append(save_per_model_csv(results, model_name, EVAL_OUTPUT_DIR))

    saved_files.append(save_summary_csv(all_model_stats, EVAL_OUTPUT_DIR))
    saved_files.append(save_details_json(all_model_details, EVAL_OUTPUT_DIR))

    logger.info("\n" + "=" * 70)
    logger.info("Deep Hallucination Detection Complete!")
    logger.info("=" * 70)
    logger.info(f"Models evaluated: {len(all_model_stats)}")
    if all_model_stats:
        first_model = next(iter(all_model_stats.values()))
        logger.info(f"Images per model: {first_model['n_images']}")

    logger.info("\nOutput files saved:")
    for file_path in saved_files:
        logger.info(f"  - {file_path}")

    return True


if __name__ == "__main__":
    success = main()
    raise SystemExit(0 if success else 1)