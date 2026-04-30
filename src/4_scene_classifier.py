#!/usr/bin/env python3
"""Extract scene/environment features using heuristic analysis.

Improved version:
- Outputs probabilistic scene scores
- Adds uncertainty handling
- Normalizes features
- Keeps lightweight (no heavy models)
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "dataset" / "annotated.csv"
IMAGES_DIR = ROOT_DIR / "dataset" / "images_for_annotation"
FEATURES_DIR = ROOT_DIR / "dataset" / "features"

IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".webp"]

CATEGORIES = ["urban", "rural", "industrial", "natural"]


def find_image_path(row_id):
    row_id = str(row_id).strip()
    if not row_id:
        return None

    for suffix in IMAGE_SUFFIXES:
        candidate = IMAGES_DIR / (row_id + suffix)
        if candidate.exists():
            return candidate
    return None


def classify_scene(image_path):
    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image)

    hsv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

    total_pixels = img_array.shape[0] * img_array.shape[1]

    # Color masks
    green_mask = (h >= 35) & (h <= 85)
    blue_mask = (h >= 100) & (h <= 130)
    red_mask = (h < 10) | (h >= 170)

    green_ratio = green_mask.sum() / total_pixels
    blue_ratio = blue_mask.sum() / total_pixels
    red_ratio = red_mask.sum() / total_pixels

    saturation_mean = float(s.mean())
    brightness_mean = float(v.mean())

    # Normalize brightness (0–1)
    brightness_norm = brightness_mean / 255.0
    saturation_norm = saturation_mean / 255.0

    # Initialize scores
    scores = {cat: 0.0 for cat in CATEGORIES}

    # Heuristic scoring (soft instead of hard decisions)
    scores["natural"] += green_ratio * 1.2
    scores["natural"] += blue_ratio * 1.0

    scores["industrial"] += (1 - saturation_norm) * 0.6
    scores["industrial"] += (1 - brightness_norm) * 0.4

    scores["urban"] += red_ratio * 0.8
    scores["urban"] += (1 - green_ratio) * 0.3

    scores["rural"] += green_ratio * 0.6
    scores["rural"] += (1 - red_ratio) * 0.2

    # Normalize scores to sum = 1
    total_score = sum(scores.values()) + 1e-8
    scores = {k: float(v / total_score) for k, v in scores.items()}

    # Determine primary category
    primary_category = max(scores, key=scores.get)
    confidence = scores[primary_category]

    # Uncertainty flag
    uncertain = confidence < 0.5

    result = {
        "scene_scores": {k: round(v, 4) for k, v in scores.items()},
        "primary_category": primary_category,
        "confidence": round(confidence, 4),
        "uncertain": uncertain,
        "method": "heuristic_color_based_v2",
        "color_analysis": {
            "green_ratio": round(green_ratio, 3),
            "blue_ratio": round(blue_ratio, 3),
            "red_ratio": round(red_ratio, 3),
            "brightness": round(brightness_norm, 3),
            "saturation": round(saturation_norm, 3),
        },
    }

    return result


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {DATASET_PATH}")
    if not IMAGES_DIR.exists():
        raise NotADirectoryError(f"Image folder not found: {IMAGES_DIR}")

    df = pd.read_csv(DATASET_PATH, dtype=str, keep_default_na=False)

    results = []
    skipped = 0
    category_counts = {cat: 0 for cat in CATEGORIES}

    print("Processing images...")

    for idx, (_, row) in enumerate(df.iterrows()):
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(df)}")

        row_id = str(row.get("id", "")).strip()
        image_path = find_image_path(row_id)

        if image_path is None:
            skipped += 1
            continue

        try:
            classification = classify_scene(image_path)

            results.append({
                "id": row_id,
                "setting": row.get("setting", ""),
                "classification": classification,
            })

            category_counts[classification["primary_category"]] += 1

        except Exception as e:
            skipped += 1
            continue

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FEATURES_DIR / "scene_features_v2.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nTotal: {len(df)}")
    print(f"✓ Processed: {len(results)}")
    print(f"⚠️  Skipped: {skipped}")

    print("\nCategory distribution:")
    for cat, count in category_counts.items():
        if count > 0:
            pct = 100 * count / len(results)
            print(f"  {cat}: {count} ({pct:.1f}%)")

    print(f"\n✓ Saved to: {output_path}")


if __name__ == "__main__":
    main()