"""
NLI Entailment Evaluation Script (v4)

Changes from v3:
- Consistent mean aggregation (agg_e, agg_n, agg_c all use mean)
- Chunk scores averaged per premise (not max-chunk selection)
- Per-premise scores exported to CSV columns
- Removed sidecar JSON (CSV is sufficient)
"""

import json
import logging
import numpy as np
import pandas as pd
import torch
import random

from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
MODEL_DIR        = Path(__file__).parent.parent.parent / "models" / "output_model"
EVALUATION_DIR   = Path(__file__).parent
NLI_MODEL_NAME   = "microsoft/deberta-large-mnli"

BATCH_SIZE        = 16
MAX_TOKEN_LENGTH  = 512
MAX_PREMISE_CHARS = 800
RANDOM_SEED       = 42

# Aggregation: mean all premise scores, then apply contradiction override
# if max_contradiction > 0.8 AND > max_entailment
CONTRADICTION_THRESHOLD = 0.8

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ─────────────────────────────────────────────
# File loading
# ─────────────────────────────────────────────
def load_json_files() -> Dict[str, Path]:
    if not MODEL_DIR.exists():
        log.error("MODEL_DIR not found: %s", MODEL_DIR)
        return {}
    files = {}
    for p in MODEL_DIR.glob("*.json"):
        if "processing_log" not in p.name:
            files[p.stem.replace("_outputs", "")] = p
    log.info("Found %d model output file(s)", len(files))
    for name, path in files.items():
        log.info("  %s → %s", name, path.name)
    return files


def load_json_data(path: Path) -> List[Dict]:
    try:
        with open(path) as f:
            data = json.load(f)
        return [data] if isinstance(data, dict) else data
    except Exception as e:
        log.error("Error loading %s: %s", path, e)
        return []

# ─────────────────────────────────────────────
# NLI model
# ─────────────────────────────────────────────
def load_nli_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading %s on %s", NLI_MODEL_NAME, device.upper())
    if device == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
    model = model.to(device).eval()

    # Read label mapping from model config — never hardcode
    id2label = {k: v.lower() for k, v in model.config.id2label.items()}
    log.info("Label mapping from model config: %s", id2label)
    return tokenizer, model, device, id2label

# ─────────────────────────────────────────────
# Premise chunking
# ─────────────────────────────────────────────
def chunk_premise(premise: str, max_chars: int = MAX_PREMISE_CHARS) -> List[str]:
    """Split long premises at sentence boundaries to avoid silent truncation."""
    if len(premise) <= max_chars:
        return [premise]
    sentences = premise.replace("! ", ". ").replace("? ", ". ").split(". ")
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) + 2 > max_chars and current:
            chunks.append(current.strip())
            current = sent
        else:
            current = current + ". " + sent if current else sent
    if current:
        chunks.append(current.strip())
    return chunks or [premise[:max_chars]]

# ─────────────────────────────────────────────
# Batched inference
# ─────────────────────────────────────────────
def score_pairs_batched(
    tokenizer, model, device: str, id2label: Dict,
    pairs: List[Tuple[str, str]],
) -> List[Dict]:
    results = []
    for start in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[start: start + BATCH_SIZE]
        try:
            enc = tokenizer(
                [p for p, _ in batch], [c for _, c in batch],
                truncation=True, max_length=MAX_TOKEN_LENGTH,
                padding=True, return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits.cpu()
            probs   = torch.softmax(logits, dim=-1).numpy()
            pred_ids = torch.argmax(logits, dim=-1).numpy()
            for prob_row, pred_id in zip(probs, pred_ids):
                score = {id2label[i]: float(prob_row[i]) for i in range(len(prob_row))}
                score["predicted_label"] = id2label[int(pred_id)]
                results.append(score)
        except Exception as e:
            log.warning("Batch error: %s", e)
            for _ in batch:
                results.append({"entailment": 0.0, "neutral": 0.0,
                                 "contradiction": 0.0, "predicted_label": "error"})
    return results

# ─────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────
def aggregate(premise_scores: List[Dict]) -> Dict:
    """
    Aggregate per-premise NLI scores.

    Design decisions (for thesis documentation):
    - Mean pooling across all premises: consistent, symmetric, easy to defend
    - Contradiction override: only when max_contradiction > 0.8 AND > max_entailment,
      i.e., a premise must strongly and dominantly contradict to override
    - nli_confidence = agg_entailment - agg_contradiction ∈ [-1, 1]
    """
    if not premise_scores:
        return {
            "agg_entailment": 0.0, "agg_neutral": 0.0, "agg_contradiction": 0.0,
            "agg_predicted_label": "unknown",
            "nli_confidence": 0.0, "max_entailment": 0.0, "max_contradiction": 0.0,
            "support_ratio": 0.0, "contradiction_ratio": 0.0,
            "premise_count_used": 0,
        }

    ents   = [s["entailment"]    for s in premise_scores]
    neus   = [s["neutral"]       for s in premise_scores]
    cons   = [s["contradiction"] for s in premise_scores]
    labels = [s["predicted_label"] for s in premise_scores]

    # Consistent mean pooling across all three scores
    agg_e = float(np.mean(ents))
    agg_n = float(np.mean(neus))
    agg_c = float(np.mean(cons))

    max_ent = max(ents)
    max_con = max(cons)
    n = len(premise_scores)

    # Contradiction override: strong single contradiction can invalidate argument
    if max_con > CONTRADICTION_THRESHOLD and max_con > max_ent:
        agg_label = "contradiction"
    else:
        scores = {
            "entailment": agg_e,
            "neutral": agg_n,
            "contradiction": agg_c
        }
        agg_label = max(scores, key=scores.get)

    return {
        "agg_entailment":      agg_e,
        "agg_neutral":         agg_n,
        "agg_contradiction":   agg_c,
        "agg_predicted_label": agg_label,
        "nli_confidence":      agg_e - agg_c,
        "max_entailment":      max_ent,
        "max_contradiction":   max_con,
        "support_ratio":       sum(1 for l in labels if l == "entailment") / n,
        "contradiction_ratio": sum(1 for l in labels if l == "contradiction") / n,
        "premise_count_used":  n,
    }

# ─────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────
def evaluate_samples(data: List[Dict], tokenizer, model, device: str, id2label: Dict) -> List[Dict]:
    """
    Per conclusion:
      - score each (premise_chunk, conclusion) pair
      - average chunk scores per premise  ← consistent, no bias
      - aggregate premise scores via mean pooling
      - export per-premise scores as flat CSV columns
    """
    rows: List[Dict] = []
    all_pairs: List[Tuple[str, str]] = []
    # (row_idx, premise_idx, chunk_idx)
    pair_map: List[Tuple[int, int, int]] = []

    for sample in data:
        image_id    = str(sample.get("image_id", "unknown"))
        model_name  = sample.get("model", "unknown")
        timestamp   = sample.get("timestamp", "")
        parsed      = sample.get("parsed_output", {})
        premises    = parsed.get("premises", [])
        conclusions = parsed.get("conclusions", [])

        for c_idx, conclusion in enumerate(conclusions):
            if not conclusion or not isinstance(conclusion, str):
                continue
            row_idx = len(rows)
            rows.append({
                "image_id":         image_id,
                "model":            model_name,
                "timestamp":        timestamp,
                "conclusion_index": c_idx,
                "conclusion_text":  conclusion,
                "num_premises":     len(premises),
                "_chunk_scores":    {},  # {premise_idx: [chunk_scores]}
            })

            for p_idx, premise in enumerate(premises):
                if not premise or not isinstance(premise, str):
                    continue
                for chunk_idx, chunk in enumerate(chunk_premise(premise)):
                    all_pairs.append((chunk, conclusion))
                    pair_map.append((row_idx, p_idx, chunk_idx))

    if not all_pairs:
        return rows

    log.info("Scoring %d pairs (batch=%d)...", len(all_pairs), BATCH_SIZE)
    pair_results = score_pairs_batched(tokenizer, model, device, id2label, all_pairs)

    # Collect all chunk scores per (row, premise)
    for (row_idx, p_idx, _), score in zip(pair_map, pair_results):
        bucket = rows[row_idx]["_chunk_scores"]
        if p_idx not in bucket:
            bucket[p_idx] = []
        bucket[p_idx].append(score)

    # Average chunk scores per premise, then aggregate + flatten to CSV columns
    for row in rows:
        chunk_scores = row.pop("_chunk_scores")

        # Per-premise: average over chunks
        premise_scores = []
        per_premise_cols = {}
        for p_idx in sorted(chunk_scores.keys()):
            chunks = chunk_scores[p_idx]
            avg = {
                "entailment":    float(np.mean([c["entailment"]    for c in chunks])),
                "neutral":       float(np.mean([c["neutral"]       for c in chunks])),
                "contradiction": float(np.mean([c["contradiction"] for c in chunks])),
            }
            avg["predicted_label"] = max(
                ["entailment", "neutral", "contradiction"],
                key=lambda k: avg[k]
            )
            premise_scores.append(avg)
            per_premise_cols[f"premise_{p_idx}_entailment"]    = avg["entailment"]
            per_premise_cols[f"premise_{p_idx}_neutral"]       = avg["neutral"]
            per_premise_cols[f"premise_{p_idx}_contradiction"] = avg["contradiction"]
            per_premise_cols[f"premise_{p_idx}_label"]         = avg["predicted_label"]

        row.update(aggregate(premise_scores))
        row.update(per_premise_cols)

    return rows

# ─────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────
def save_csv(rows: List[Dict], model_name: str) -> Path:
    df = pd.DataFrame(rows)

    # Fixed columns first, then per-premise columns sorted
    fixed_cols = [
        "image_id", "model", "timestamp", "conclusion_index", "conclusion_text",
        "num_premises", "premise_count_used", "agg_entailment", "agg_neutral", "agg_contradiction",
        "agg_predicted_label", "nli_confidence", "max_entailment", "max_contradiction",
        "support_ratio", "contradiction_ratio",
    ]
    premise_cols = sorted([c for c in df.columns if c.startswith("premise_")])
    df = df[fixed_cols + premise_cols]

    path = EVALUATION_DIR / f"{model_name}_outputs_with_nli.csv"
    df.to_csv(path, index=False)
    log.info("Saved CSV: %s", path)
    return path

# ─────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────
def compute_statistics(rows: List[Dict]) -> Dict:
    df = pd.DataFrame(rows)
    df = df[df["agg_predicted_label"].notna() & ~df["agg_predicted_label"].isin(["unknown", "error"])]
    if df.empty:
        return {}

    n = len(df)
    per_image_ent = df.groupby("image_id").apply(
        lambda g: (g["agg_predicted_label"] == "entailment").mean()
    )

    return {
        "total_conclusions":    n,
        "num_images":           int(df["image_id"].nunique()),
        "mean_entailment":      float(df["agg_entailment"].mean()),
        "mean_neutral":         float(df["agg_neutral"].mean()),
        "mean_contradiction":   float(df["agg_contradiction"].mean()),
        "std_entailment":       float(df["agg_entailment"].std()),
        "std_neutral":          float(df["agg_neutral"].std()),
        "std_contradiction":    float(df["agg_contradiction"].std()),
        "pct_entailment":       float((df["agg_predicted_label"] == "entailment").sum() / n * 100),
        "pct_neutral":          float((df["agg_predicted_label"] == "neutral").sum() / n * 100),
        "pct_contradiction":    float((df["agg_predicted_label"] == "contradiction").sum() / n * 100),
        "mean_nli_confidence":  float(df["nli_confidence"].mean()),
        "std_nli_confidence":   float(df["nli_confidence"].std()),
        "mean_support_ratio":   float(df["support_ratio"].mean()),
        "mean_contradiction_ratio": float(df["contradiction_ratio"].mean()),
        "mean_premise_count_used": float(df["premise_count_used"].mean()),
        "per_image_entailment_mean": float(per_image_ent.mean()),
        "per_image_entailment_std":  float(per_image_ent.std()),
    }


def save_summary(all_stats: Dict[str, Dict]) -> Path:
    SEP = "=" * 100
    path = EVALUATION_DIR / "nli_summary_statistics.txt"

    with open(path, "w") as f:
        f.write(SEP + "\n")
        f.write("NLI ENTAILMENT EVALUATION SUMMARY\n")
        f.write(f"NLI Model:  {NLI_MODEL_NAME}\n")
        f.write(f"Aggregation: mean pooling + contradiction override "
                f"(threshold={CONTRADICTION_THRESHOLD})\n")
        f.write(f"Seed: {RANDOM_SEED}\n")
        f.write(SEP + "\n\n")

        for model_name, s in all_stats.items():
            if not s:
                continue
            f.write(f"Model: {model_name}  "
                    f"({s['total_conclusions']} conclusions | {s['num_images']} images)\n")
            f.write(f"  Scores (mean ± std):\n")
            f.write(f"    Entailment:    {s['mean_entailment']:.4f} ± {s['std_entailment']:.4f}\n")
            f.write(f"    Neutral:       {s['mean_neutral']:.4f} ± {s['std_neutral']:.4f}\n")
            f.write(f"    Contradiction: {s['mean_contradiction']:.4f} ± {s['std_contradiction']:.4f}\n")
            f.write(f"  Label distribution:\n")
            f.write(f"    Entailment:    {s['pct_entailment']:6.2f}%\n")
            f.write(f"    Neutral:       {s['pct_neutral']:6.2f}%\n")
            f.write(f"    Contradiction: {s['pct_contradiction']:6.2f}%\n")
            f.write(f"  Research metrics:\n")
            f.write(f"    NLI Confidence (mean ± std):  {s['mean_nli_confidence']:.4f} ± {s['std_nli_confidence']:.4f}\n")
            f.write(f"    Support ratio (mean):         {s['mean_support_ratio']:.4f}\n")
            f.write(f"    Contradiction ratio (mean):   {s['mean_contradiction_ratio']:.4f}\n")
            f.write(f"    Premises used per conclusion: {s['mean_premise_count_used']:.2f}\n")
            f.write(f"    Per-image entailment rate:    {s['per_image_entailment_mean']:.4f} ± {s['per_image_entailment_std']:.4f}\n")
            f.write("\n")

        total_n = sum(s.get("total_conclusions", 0) for s in all_stats.values() if s)
        if total_n > 0 and len(all_stats) > 1:
            def wavg(k):
                return sum(s[k] * s["total_conclusions"]
                           for s in all_stats.values() if s) / total_n
            f.write("OVERALL (all models combined)\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Total conclusions:  {total_n}\n")
            f.write(f"  Entailment:         {wavg('pct_entailment'):.2f}%\n")
            f.write(f"  Neutral:            {wavg('pct_neutral'):.2f}%\n")
            f.write(f"  Contradiction:      {wavg('pct_contradiction'):.2f}%\n")
            f.write(f"  NLI Confidence:     {wavg('mean_nli_confidence'):.4f}\n")
            f.write(f"  Support ratio:      {wavg('mean_support_ratio'):.4f}\n\n")

        f.write(SEP + "\n")
        f.write("Output files:\n")
        f.write("  {model}_outputs_with_nli.csv  – per-conclusion scores + per-premise columns\n")
        f.write("  nli_summary_statistics.txt    – this file\n")
        f.write(SEP + "\n")

    log.info("Saved summary: %s", path)
    return path

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    set_seed(RANDOM_SEED)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    log.info("NLI EVALUATION PIPELINE v4")
    log.info("Batch: %d | Max tokens: %d | Contradiction threshold: %.1f",
             BATCH_SIZE, MAX_TOKEN_LENGTH, CONTRADICTION_THRESHOLD)

    json_files = load_json_files()
    if not json_files:
        log.error("No JSON files found in %s", MODEL_DIR)
        return

    tokenizer, nli_model, device, id2label = load_nli_model()
    all_stats: Dict[str, Dict] = {}

    for model_name, json_path in json_files.items():
        log.info("─" * 50)
        log.info("Processing: %s", model_name)

        data = load_json_data(json_path)
        if not data:
            log.warning("No data for %s, skipping.", model_name)
            continue
        log.info("Loaded %d samples", len(data))

        rows = evaluate_samples(data, tokenizer, nli_model, device, id2label)
        log.info("Evaluated %d conclusion(s)", len(rows))

        save_csv(rows, model_name)

        stats = compute_statistics(rows)
        all_stats[model_name] = stats
        if stats:
            log.info(
                "Entailment: %.2f%%  Neutral: %.2f%%  Contradiction: %.2f%%  "
                "Confidence: %.4f  Support: %.4f",
                stats["pct_entailment"], stats["pct_neutral"], stats["pct_contradiction"],
                stats["mean_nli_confidence"], stats["mean_support_ratio"],
            )

    if all_stats:
        save_summary(all_stats)

    log.info("PIPELINE COMPLETE — outputs in %s", EVALUATION_DIR)


if __name__ == "__main__":
    main()