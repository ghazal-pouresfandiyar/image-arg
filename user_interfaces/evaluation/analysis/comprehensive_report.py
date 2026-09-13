import json
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "evaluations_export_final.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(DATA_PATH, "r") as file:
    data = json.load(file)

records = []
for image_id, models in data["evaluations"].items():
    for model_name, metrics in models.items():
        if metrics:
            row = {"image_id": image_id, "model": model_name}
            row.update(metrics)
            records.append(row)

df = pd.DataFrame(records)

report_path = os.path.join(OUTPUT_DIR, "full_analysis_report.txt")
with open(report_path, "w") as f:
    f.write("=" * 70 + "\n")
    f.write("  COMPREHENSIVE EVALUATION REPORT — Image-Argument Analysis\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Generated from: evaluations_export_final.json\n")
    f.write(f"Total images evaluated: {df['image_id'].nunique()}\n")
    f.write(f"Total evaluations: {len(df)}\n")
    f.write(f"Models: {', '.join(df['model'].unique())}\n\n")

    f.write("-" * 70 + "\n")
    f.write("SECTION 1: DATASET OVERVIEW\n")
    f.write("-" * 70 + "\n")
    f.write(df.describe(include="all").to_string() + "\n\n")

    f.write("-" * 70 + "\n")
    f.write("SECTION 2: CATEGORICAL METRICS — Percentage Distribution per Model\n")
    f.write("-" * 70 + "\n")
    categorical_metrics = [
        "conclusion_novelty", "conclusion_relevance_score", "conclusion_validity",
        "format_adherence", "human_alignment_1", "human_alignment_2",
        "premise_impact", "premise_relevance_score", "quantity_compliance", "visual_grounding",
    ]
    for metric in categorical_metrics:
        if metric in df.columns:
            f.write(f"\n{metric.upper()}\n")
            f.write(f"{'─' * 40}\n")
            stats = pd.crosstab(df["model"], df[metric], normalize="index") * 100
            f.write(stats.round(2).to_string() + "\n\n")

    f.write("-" * 70 + "\n")
    f.write("SECTION 3: NUMERICAL METRICS — Summary Statistics per Model\n")
    f.write("-" * 70 + "\n")
    for metric in ["premise_coverage_0", "premise_coverage_1"]:
        if metric in df.columns:
            f.write(f"\n{metric.upper()}\n")
            f.write(f"{'─' * 40}\n")
            stats = df.groupby("model")[metric].describe()
            f.write(stats.round(2).to_string() + "\n\n")

    f.write("-" * 70 + "\n")
    f.write("SECTION 4: MODEL COMPARISON SUMMARY\n")
    f.write("-" * 70 + "\n")
    for metric in categorical_metrics:
        if metric in df.columns:
            f.write(f"\n{metric} — Top performing model:\n")
            top = df.groupby("model")[metric].value_counts().groupby("model").idxmax()
            f.write(f"  {top.to_string()}\n")

    f.write("\n" + "=" * 70 + "\n")
    f.write("  END OF REPORT\n")
    f.write("=" * 70 + "\n")

print(f"Comprehensive report saved to: {report_path}")
print(f"   - {df['image_id'].nunique()} images evaluated")
print(f"   - {len(df)} total model evaluations")
print(f"   - Models: {', '.join(df['model'].unique())}")
