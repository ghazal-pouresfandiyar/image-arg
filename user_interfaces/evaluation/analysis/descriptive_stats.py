import json
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "evaluations_export _final.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "evaluations_flattened.csv")

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

categorical_metrics = [
    "conclusion_novelty", "conclusion_relevance_score", "conclusion_validity",
    "format_adherence", "human_alignment_1", "human_alignment_2",
    "premise_impact", "premise_relevance_score", "quantity_compliance", "visual_grounding",
]
numerical_metrics = ["premise_coverage_0", "premise_coverage_1"]

print("=" * 60)
print("  DESCRIPTIVE STATISTICS — Image-Argument Evaluation")
print("=" * 60)
print(f"\nTotal evaluations: {len(df)}")
print(f"Models: {', '.join(df['model'].unique())}")
print(f"Images: {df['image_id'].nunique()}")

print("\n--- Dataset Overview ---")
print(df.describe(include="all").to_string())

print("\n\n=== CATEGORICAL METRICS (Distribution per Model) ===")
for metric in categorical_metrics:
    if metric in df.columns:
        print(f"\n{'─' * 40}")
        print(f"  {metric.upper()}")
        print(f"{'─' * 40}")
        stats = pd.crosstab(df["model"], df[metric], normalize="index") * 100
        print(stats.round(2).to_string())
        counts = pd.crosstab(df["model"], df[metric])
        print("\nRaw counts:")
        print(counts.to_string())

print("\n\n=== NUMERICAL METRICS (Summary Statistics per Model) ===")
for metric in numerical_metrics:
    if metric in df.columns:
        print(f"\n{'─' * 40}")
        print(f"  {metric.upper()}")
        print(f"{'─' * 40}")
        stats = df.groupby("model")[metric].describe()
        print(stats.round(2).to_string())

print("\n\n=== OVERALL METRIC SUMMARY ===")
for metric in categorical_metrics + numerical_metrics:
    if metric in df.columns:
        counts = df[metric].value_counts()
        print(f"\n{metric}:")
        print(f"  Total unique values: {df[metric].nunique()}")
        print(f"  Top value: {counts.index[0]} ({counts.iloc[0]})")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nFlattened data exported to: {OUTPUT_CSV}")
