import json
import csv
import os

# Your local folder path
BASE = r"F:/Passau/CoRNLP/Multimodal Argument Mining/image-arg\dataset"

INPUT_JSON = f"{BASE}/comic-dacts-main/data.json"
OUTPUT_CSV = f"{BASE}/filtered_data.csv"
MISSING_CSV = f"{BASE}/missing_images.csv"
IMAGES_DIR = f"{BASE}/images"

# Load JSON
with open(INPUT_JSON, "r", encoding="utf8") as f:
    data = json.load(f)

# filter only error_code == 0
# # - `0` → contains speech balloon (real dialogue)
# - `1` → no text
# - `2` → narration only (no speech)
data = [item for item in data if item.get("error_code") == 0]

# Remove error_code column, put img_id first
csv_columns = [col for col in data[0].keys() if col not in
               ("error_code", "img_id")] if data else []
csv_columns = ["img_id"] + csv_columns

# Split into rows with/without matching image file
has_image = []
no_image = []
for item in data:
    item.pop("error_code", None)
    img_path = os.path.join(IMAGES_DIR, item["img_id"] + ".jpg")
    if os.path.isfile(img_path):
        has_image.append(item)
    else:
        no_image.append(item)

# Write main CSV (rows with images)
with open(OUTPUT_CSV, "w", newline="", encoding="utf8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()
    for item in has_image:
        writer.writerow(item)

# Write missing-image CSV
with open(MISSING_CSV, "w", newline="", encoding="utf8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()
    for item in no_image:
        writer.writerow(item)

print("CSV created:", OUTPUT_CSV)
print("Missing images CSV:", MISSING_CSV)
print(f"  {len(has_image)} rows with images, {len(no_image)} rows without")
