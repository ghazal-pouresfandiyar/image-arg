#!/usr/bin/env python3
"""Remove CSV rows when id.jpg does not exist in the annotation folder."""

import csv
import os
from pathlib import Path

# Get root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT_DIR / "dataset" / "annotated.csv"
IMAGES_DIR = ROOT_DIR / "dataset" / "images_for_annotation"
ID_COLUMN = "id"
IMAGE_EXT = ".jpg"
OUTPUT_PATH = None  # Example: "./filtered/annotation/argumet-manual.filtered.csv"
DRY_RUN = False


def read_csv_with_fallback(path):
    # Try common encodings used by exported CSV files on Windows.
    encodings = ["utf-8-sig", "cp1252", "latin-1"]
    last_error = None

    for enc in encodings:
        try:
            with open(path, "r", newline="", encoding=enc) as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                if not fieldnames:
                    raise ValueError("CSV has no header: " + path)
                rows = list(reader)
                return fieldnames, rows
        except UnicodeDecodeError as e:
            last_error = e

    raise UnicodeDecodeError(
        last_error.encoding,
        last_error.object,
        last_error.start,
        last_error.end,
        "Could not decode CSV with utf-8-sig, cp1252, or latin-1",
    )


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError("CSV not found: " + str(CSV_PATH))

    if not IMAGES_DIR.is_dir():
        raise NotADirectoryError("Images directory not found: " + str(IMAGES_DIR))

    valid_ids = set()
    for name in os.listdir(str(IMAGES_DIR)):
        file_path = IMAGES_DIR / name
        if file_path.is_file() and name.lower().endswith(IMAGE_EXT.lower()):
            valid_ids.add(file_path.stem)

    if not valid_ids:
        raise ValueError("No image files found with extension " + IMAGE_EXT)

    fieldnames, rows = read_csv_with_fallback(str(CSV_PATH))
    if ID_COLUMN not in fieldnames:
        raise KeyError("ID column not found: " + ID_COLUMN)

    kept_rows = []
    for row in rows:
        row_id = (row.get(ID_COLUMN) or "").strip()
        if row_id in valid_ids:
            kept_rows.append(row)

    removed_count = len(rows) - len(kept_rows)
    print("Rows total:", len(rows))
    print("Rows kept:", len(kept_rows))
    print("Rows removed (missing image):", removed_count)

    if DRY_RUN:
        return

    target_path = Path(OUTPUT_PATH) if OUTPUT_PATH else CSV_PATH
    with open(str(target_path), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    if OUTPUT_PATH:
        print("Filtered CSV written to:", target_path)
    else:
        print("Updated in place:", target_path)


if __name__ == "__main__":
    main()
