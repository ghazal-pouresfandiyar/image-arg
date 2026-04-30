#!/usr/bin/env python3
"""
0_image_captioning.py

Generate image captions using BLIP-2 and store them in annotated.csv
"""

from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
CSV_PATH = DATASET_DIR / "annotated.csv"
IMAGES_DIR = DATASET_DIR / "images_for_annotation"

# Load CSV
df = pd.read_csv(CSV_PATH)

# Add column if not exists
if "blip2_caption" not in df.columns:
	df["blip2_caption"] = ""

# Load BLIP-2
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained(
	"Salesforce/blip2-flan-t5-xl",
	torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

def get_image_path(image_id):
	return IMAGES_DIR / f"{image_id}.jpg"

def caption_image(image_path):
	image = Image.open(image_path).convert("RGB")

	inputs = processor(images=image, return_tensors="pt").to(device)

	with torch.no_grad():
		outputs = model.generate(**inputs, max_new_tokens=30)

	caption = processor.decode(outputs[0], skip_special_tokens=True)
	return caption

# Process all rows
for idx, row in df.iterrows():
	image_id = str(row["id"])
	image_path = get_image_path(image_id)

	if not image_path.exists():
		print(f"⚠️  Missing image: {image_id}")
		continue

	if pd.notna(row.get("blip2_caption")) and row["blip2_caption"]:
		continue  # already processed

	caption = caption_image(image_path)
	df.at[idx, "blip2_caption"] = caption
	print(f"✓ {image_id}: {caption}")

# Save CSV
df.to_csv(CSV_PATH, index=False)
print("✅ Captions saved to annotated.csv")