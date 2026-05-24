"""
NLI Entailment Evaluation Script

Evaluates model-generated arguments using Natural Language Inference (NLI) to assess
whether conclusions are entailed by premises.

Workflow:
1. Load JSON files from models/output_model/
2. Create expanded CSV format (one row per conclusion)
3. Score each conclusion against merged premises using microsoft/deberta-large-mnli
4. Generate CSV and JSON outputs with NLI scores in evaluation/
5. Compute and export summary statistics
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import numpy as np

# Configuration
MODEL_DIR = Path(__file__).parent.parent / "models" / "output_model"
EVALUATION_DIR = Path(__file__).parent
NLI_MODEL_NAME = "microsoft/deberta-large-mnli"

# NLI label mapping
NLI_LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}
NLI_LABEL_TO_ID = {v: k for k, v in NLI_LABELS.items()}


def load_json_files() -> Dict[str, Path]:
    """Find and load all JSON files in models/output_model/"""
    json_files = {}
    for json_file in MODEL_DIR.glob("*.json"):
        if "processing_log" not in json_file.name:
            model_name = json_file.stem.replace("_outputs", "")
            json_files[model_name] = json_file
    print(f"Found {len(json_files)} model output files:")
    for model_name, path in json_files.items():
        print(f"  - {model_name}: {path.name}")
    return json_files


def load_json_data(json_path: Path) -> List[Dict]:
    """Load JSON file with error handling"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        return data
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return []


def expand_to_csv_format(data: List[Dict]) -> List[Dict]:
    """
    Expand JSON to CSV format where each row represents one conclusion.
    
    Args:
        data: List of sample dicts from JSON
        
    Returns:
        List of expanded rows (one per conclusion per sample)
    """
    expanded_rows = []
    
    for sample in data:
        image_id = sample.get("image_id", "unknown")
        model = sample.get("model", "unknown")
        timestamp = sample.get("timestamp", "")
        
        parsed_output = sample.get("parsed_output", {})
        premises = parsed_output.get("premises", [])
        conclusions = parsed_output.get("conclusions", [])
        
        # Merge all premises into a single string
        merged_premises = " ".join(premises) if premises else ""
        
        # Skip if no conclusions
        if not conclusions:
            continue
        
        # Create one row per conclusion
        for conclusion_idx, conclusion in enumerate(conclusions):
            row = {
                "image_id": image_id,
                "model": model,
                "timestamp": timestamp,
                "conclusion_index": conclusion_idx,
                "premises_merged": merged_premises,
                "conclusion_text": conclusion,
                # NLI scores (to be filled)
                "entailment_score": None,
                "neutral_score": None,
                "contradiction_score": None,
                "predicted_label": None
            }
            expanded_rows.append(row)
    
    return expanded_rows


def load_nli_model(device: str = "cpu"):
    """Load NLI model and tokenizer"""
    print(f"\nLoading NLI model: {NLI_MODEL_NAME}")
    if device == "cuda" and torch.cuda.is_available():
        print("Using CUDA device")
    else:
        device = "cpu"
        print("Using CPU device")
    
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
    model = model.to(device)
    model.eval()
    
    return tokenizer, model, device


def score_conclusion_with_nli(
    tokenizer, 
    model, 
    device: str,
    premises: str, 
    conclusion: str,
    max_length: int = 512
) -> Tuple[float, float, float, str]:
    """
    Score a conclusion using NLI model.
    
    Args:
        tokenizer: Tokenizer for NLI model
        model: NLI model
        device: Device to run on (cuda/cpu)
        premises: Merged premise text
        conclusion: Conclusion text
        max_length: Max token length for tokenization
        
    Returns:
        Tuple of (entailment_score, neutral_score, contradiction_score, predicted_label)
    """
    if not premises or not conclusion:
        return 0.0, 0.0, 0.0, "unknown"
    
    try:
        # Tokenize premise-conclusion pair
        inputs = tokenizer(
            premises,
            conclusion,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get logits and convert to probabilities
        logits = outputs.logits[0].cpu()
        probabilities = torch.softmax(logits, dim=-1).numpy()
        
        # Extract scores for each label
        entailment_score = float(probabilities[NLI_LABEL_TO_ID["entailment"]])
        neutral_score = float(probabilities[NLI_LABEL_TO_ID["neutral"]])
        contradiction_score = float(probabilities[NLI_LABEL_TO_ID["contradiction"]])
        
        # Get predicted label
        predicted_label_id = torch.argmax(logits).item()
        predicted_label = NLI_LABELS[predicted_label_id]
        
        return entailment_score, neutral_score, contradiction_score, predicted_label
    
    except Exception as e:
        print(f"Error scoring conclusion: {e}")
        return 0.0, 0.0, 0.0, "error"


def evaluate_expanded_data(
    expanded_data: List[Dict],
    tokenizer,
    model,
    device: str
) -> List[Dict]:
    """
    Score all conclusions with NLI model.
    
    Args:
        expanded_data: List of expanded rows
        tokenizer: NLI tokenizer
        model: NLI model
        device: Device to run on
        
    Returns:
        List of rows with NLI scores filled in
    """
    scored_data = []
    
    print(f"\nScoring {len(expanded_data)} conclusions with NLI model...")
    for row in tqdm(expanded_data, desc="NLI Scoring"):
        premises = row["premises_merged"]
        conclusion = row["conclusion_text"]
        
        entailment_score, neutral_score, contradiction_score, predicted_label = \
            score_conclusion_with_nli(tokenizer, model, device, premises, conclusion)
        
        row["entailment_score"] = entailment_score
        row["neutral_score"] = neutral_score
        row["contradiction_score"] = contradiction_score
        row["predicted_label"] = predicted_label
        
        scored_data.append(row)
    
    return scored_data


def save_csv(scored_data: List[Dict], model_name: str) -> Path:
    """
    Save scored data to CSV format.
    
    Args:
        scored_data: List of scored rows
        model_name: Name of the model (for filename)
        
    Returns:
        Path to saved CSV file
    """
    df = pd.DataFrame(scored_data)
    
    # Select and order columns for output
    output_columns = [
        "image_id",
        "model",
        "timestamp",
        "conclusion_index",
        "premises_merged",
        "conclusion_text",
        "entailment_score",
        "neutral_score",
        "contradiction_score",
        "predicted_label"
    ]
    
    df = df[output_columns]
    csv_path = EVALUATION_DIR / f"{model_name}_outputs_with_nli.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    return csv_path


def save_json(original_data: List[Dict], scored_data: List[Dict], model_name: str) -> Path:
    """
    Create JSON output with NLI scores added to original structure.
    
    Args:
        original_data: Original data from JSON file
        scored_data: Expanded data with NLI scores
        model_name: Name of the model (for filename)
        
    Returns:
        Path to saved JSON file
    """
    # Create a mapping from (image_id, conclusion_index) to NLI scores
    nli_scores_map = {}
    for row in scored_data:
        key = (str(row["image_id"]), row["conclusion_index"])
        nli_scores_map[key] = {
            "entailment_score": row["entailment_score"],
            "neutral_score": row["neutral_score"],
            "contradiction_score": row["contradiction_score"],
            "predicted_label": row["predicted_label"]
        }
    
    # Add NLI scores to original data
    enhanced_data = []
    for sample in original_data:
        enhanced_sample = sample.copy()
        parsed_output = enhanced_sample.get("parsed_output", {})
        conclusions = parsed_output.get("conclusions", [])
        
        # Add NLI scores to conclusions
        enhanced_conclusions = []
        for idx, conclusion in enumerate(conclusions):
            key = (str(sample.get("image_id", "")), idx)
            if key in nli_scores_map:
                enhanced_conclusion = {
                    "text": conclusion,
                    "nli_scores": nli_scores_map[key]
                }
            else:
                enhanced_conclusion = {
                    "text": conclusion,
                    "nli_scores": {
                        "entailment_score": None,
                        "neutral_score": None,
                        "contradiction_score": None,
                        "predicted_label": "unknown"
                    }
                }
            enhanced_conclusions.append(enhanced_conclusion)
        
        enhanced_sample["parsed_output"]["conclusions"] = enhanced_conclusions
        enhanced_data.append(enhanced_sample)
    
    json_path = EVALUATION_DIR / f"{model_name}_outputs_with_nli.json"
    with open(json_path, 'w') as f:
        json.dump(enhanced_data, f, indent=2)
    print(f"Saved JSON: {json_path}")
    return json_path


def compute_statistics(scored_data: List[Dict]) -> Dict:
    """
    Compute aggregate NLI statistics.
    
    Args:
        scored_data: List of scored rows
        
    Returns:
        Dictionary of statistics
    """
    df = pd.DataFrame(scored_data)
    
    total_conclusions = len(df)
    
    stats = {
        "total_conclusions": total_conclusions,
        "mean_entailment": float(df["entailment_score"].mean()),
        "mean_neutral": float(df["neutral_score"].mean()),
        "mean_contradiction": float(df["contradiction_score"].mean()),
        "std_entailment": float(df["entailment_score"].std()),
        "std_neutral": float(df["neutral_score"].std()),
        "std_contradiction": float(df["contradiction_score"].std()),
        "pct_entailment": float((df["predicted_label"] == "entailment").sum() / total_conclusions * 100),
        "pct_neutral": float((df["predicted_label"] == "neutral").sum() / total_conclusions * 100),
        "pct_contradiction": float((df["predicted_label"] == "contradiction").sum() / total_conclusions * 100),
    }
    
    return stats


def save_summary_statistics(all_stats: Dict[str, Dict]) -> Path:
    """
    Create and save a summary statistics report.
    
    Args:
        all_stats: Dictionary mapping model names to their statistics
        
    Returns:
        Path to saved statistics file
    """
    summary_path = EVALUATION_DIR / "nli_summary_statistics.txt"
    
    with open(summary_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("NLI ENTAILMENT EVALUATION SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        
        # Per-model statistics
        f.write("PER-MODEL STATISTICS\n")
        f.write("-" * 100 + "\n\n")
        
        for model_name, stats in all_stats.items():
            f.write(f"Model: {model_name}\n")
            f.write(f"  Total Conclusions Evaluated: {stats['total_conclusions']}\n")
            f.write(f"  \n")
            f.write(f"  Mean Scores (± Std Dev):\n")
            f.write(f"    Entailment:     {stats['mean_entailment']:.4f} ± {stats['std_entailment']:.4f}\n")
            f.write(f"    Neutral:        {stats['mean_neutral']:.4f} ± {stats['std_neutral']:.4f}\n")
            f.write(f"    Contradiction:  {stats['mean_contradiction']:.4f} ± {stats['std_contradiction']:.4f}\n")
            f.write(f"  \n")
            f.write(f"  Predicted Label Distribution:\n")
            f.write(f"    Entailment:     {stats['pct_entailment']:6.2f}%\n")
            f.write(f"    Neutral:        {stats['pct_neutral']:6.2f}%\n")
            f.write(f"    Contradiction:  {stats['pct_contradiction']:6.2f}%\n")
            f.write(f"\n")
        
        # Overall statistics
        f.write("\n")
        f.write("OVERALL STATISTICS (All Models Combined)\n")
        f.write("-" * 100 + "\n\n")
        
        combined_data = []
        for stats in all_stats.values():
            # Approximate: create weighted average (this is simplified)
            combined_data.append(stats)
        
        total_conclusions = sum(s["total_conclusions"] for s in all_stats.values())
        mean_entailment = sum(s["mean_entailment"] * s["total_conclusions"] for s in all_stats.values()) / total_conclusions if total_conclusions > 0 else 0
        mean_neutral = sum(s["mean_neutral"] * s["total_conclusions"] for s in all_stats.values()) / total_conclusions if total_conclusions > 0 else 0
        mean_contradiction = sum(s["mean_contradiction"] * s["total_conclusions"] for s in all_stats.values()) / total_conclusions if total_conclusions > 0 else 0
        
        pct_entailment = sum(s["pct_entailment"] * s["total_conclusions"] for s in all_stats.values()) / total_conclusions if total_conclusions > 0 else 0
        pct_neutral = sum(s["pct_neutral"] * s["total_conclusions"] for s in all_stats.values()) / total_conclusions if total_conclusions > 0 else 0
        pct_contradiction = sum(s["pct_contradiction"] * s["total_conclusions"] for s in all_stats.values()) / total_conclusions if total_conclusions > 0 else 0
        
        f.write(f"Total Conclusions Evaluated (All Models): {total_conclusions}\n")
        f.write(f"  \n")
        f.write(f"Mean Scores (All Models):\n")
        f.write(f"  Entailment:     {mean_entailment:.4f}\n")
        f.write(f"  Neutral:        {mean_neutral:.4f}\n")
        f.write(f"  Contradiction:  {mean_contradiction:.4f}\n")
        f.write(f"  \n")
        f.write(f"Predicted Label Distribution (All Models):\n")
        f.write(f"  Entailment:     {pct_entailment:6.2f}%\n")
        f.write(f"  Neutral:        {pct_neutral:6.2f}%\n")
        f.write(f"  Contradiction:  {pct_contradiction:6.2f}%\n")
        f.write(f"\n")
        
        f.write("=" * 100 + "\n")
        f.write("Output Files:\n")
        f.write("  CSV Files:  {model_name}_outputs_with_nli.csv\n")
        f.write("  JSON Files: {model_name}_outputs_with_nli.json\n")
        f.write("=" * 100 + "\n")
    
    print(f"Saved summary statistics: {summary_path}")
    return summary_path


def main():
    """Main execution function"""
    print("\n" + "=" * 100)
    print("NLI ENTAILMENT EVALUATION PIPELINE")
    print("=" * 100)
    
    # Step 1: Load JSON files
    json_files = load_json_files()
    if not json_files:
        print("No JSON files found. Exiting.")
        return
    
    # Step 2: Load NLI model
    tokenizer, model, device = load_nli_model()
    
    # Step 3: Process each model
    all_stats = {}
    
    for model_name, json_path in json_files.items():
        print(f"\n{'=' * 100}")
        print(f"Processing: {model_name}")
        print(f"{'=' * 100}")
        
        # Load original JSON data
        original_data = load_json_data(json_path)
        if not original_data:
            print(f"Skipping {model_name}: no data loaded")
            continue
        
        print(f"Loaded {len(original_data)} samples from {json_path.name}")
        
        # Expand to CSV format
        expanded_data = expand_to_csv_format(original_data)
        print(f"Expanded to {len(expanded_data)} rows (one per conclusion)")
        
        # Score with NLI model
        scored_data = evaluate_expanded_data(expanded_data, tokenizer, model, device)
        
        # Save CSV
        save_csv(scored_data, model_name)
        
        # Save enhanced JSON
        save_json(original_data, scored_data, model_name)
        
        # Compute statistics
        stats = compute_statistics(scored_data)
        all_stats[model_name] = stats
        
        print(f"\nStatistics for {model_name}:")
        print(f"  Mean Entailment Score: {stats['mean_entailment']:.4f}")
        print(f"  Entailment Rate: {stats['pct_entailment']:.2f}%")
        print(f"  Contradiction Rate: {stats['pct_contradiction']:.2f}%")
    
    # Step 4: Save summary statistics
    if all_stats:
        save_summary_statistics(all_stats)
    
    print("\n" + "=" * 100)
    print("PIPELINE COMPLETE")
    print("=" * 100)
    print(f"\nOutput files saved to: {EVALUATION_DIR}")


if __name__ == "__main__":
    main()
