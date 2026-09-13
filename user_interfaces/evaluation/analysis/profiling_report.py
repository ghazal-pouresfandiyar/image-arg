import json
import pandas as pd
import os
from ydata_profiling import ProfileReport

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

print("Generating profiling report... (this may take a minute)")
profile = ProfileReport(
    df,
    title="Image-Argument Evaluation — Automated Profiling Report",
    explorative=True,
    minimal=False,
)

output_path = os.path.join(OUTPUT_DIR, "evaluation_profiling_report.html")
profile.to_file(output_path)
print(f"\nProfiling report saved to: {output_path}")
print("Open in a browser to explore the full interactive dashboard.")
