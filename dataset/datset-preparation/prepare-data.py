import json
import csv
import os

# Your local folder path
BASE = r"F:/Passau/CoRNLP/Multimodal Argument Mining/image-arg\dataset"

INPUT_JSON = f"{BASE}/comic-dacts-main/data.json"
OUTPUT_CSV = f"{BASE}/filtered_data.csv"

# Load JSON
with open(INPUT_JSON, "r", encoding="utf8") as f:
    data = json.load(f)

# filter only error_code == 0
# # - `0` → contains speech balloon (real dialogue)
# - `1` → no text
# - `2` → narration only (no speech)
data = [item for item in data if item.get("error_code") == 0]

# Remove error_code column
csv_columns = [col for col in data[0].keys() if col !=
               "error_code"] if data else []

# Write CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()

    for item in data:
        # remove error_code from each row
        item.pop("error_code", None)
        writer.writerow(item)

print("CSV created:", OUTPUT_CSV)
