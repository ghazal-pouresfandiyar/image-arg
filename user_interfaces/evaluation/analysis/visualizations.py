import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "evaluations_export _final.json")
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

categorical_metrics = [
    "conclusion_novelty", "conclusion_relevance_score", "conclusion_validity",
    "format_adherence", "human_alignment_1", "human_alignment_2",
    "premise_impact", "premise_relevance_score", "quantity_compliance", "visual_grounding",
]

plt.rcParams.update({"figure.figsize": (12, 6), "font.size": 10})

for metric in categorical_metrics:
    if metric not in df.columns:
        continue
    counts = pd.crosstab(df["model"], df[metric])
    ax = counts.plot(kind="bar", stacked=True, figsize=(12, 6))
    plt.title(f"{metric.replace('_', ' ').title()} — Distribution by Model")
    plt.xlabel("Model")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.legend(title=metric.replace("_", " ").title(), bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, f"{metric}_chart.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filepath}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
numerical_metrics = ["premise_coverage_0", "premise_coverage_1"]
for i, metric in enumerate(numerical_metrics):
    if metric in df.columns:
        df.boxplot(column=metric, by="model", ax=axes[i])
        axes[i].set_title(f"{metric} Distribution by Model")
        axes[i].set_xlabel("Model")
        axes[i].set_ylabel(metric)
plt.suptitle("")
plt.tight_layout()
filepath = os.path.join(OUTPUT_DIR, "numerical_metrics_charts.png")
plt.savefig(filepath, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {filepath}")

model_performance = df.groupby("model").size().reset_index(name="count")
ax = model_performance.plot(kind="pie", y="count", labels=model_performance["model"], autopct="%1.1f%%", figsize=(8, 8))
plt.title("Evaluation Count per Model")
plt.ylabel("")
plt.tight_layout()
filepath = os.path.join(OUTPUT_DIR, "model_distribution_chart.png")
plt.savefig(filepath, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {filepath}")

print(f"\nAll visualizations saved to {OUTPUT_DIR}")
