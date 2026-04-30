#!/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# =========================
# CONFIG
# =========================

DATA_PATH = Path("dataset/annotated.csv")
RANDOM_SEED = 42

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(DATA_PATH)

print(f"Original dataset size: {len(df)}")

# =========================
# SPLIT (80 / 20)
# =========================

train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.2,
    random_state=RANDOM_SEED,
    shuffle=True
)

# Create split column
df["split"] = "train"
df.loc[test_idx, "split"] = "test"

# =========================
# SAVE (OVERWRITE ORIGINAL FILE)
# =========================

df.to_csv(DATA_PATH, index=False)

# =========================
# STATS
# =========================

print("\n✅ File overwritten successfully!")
print(f"Train samples: {sum(df['split'] == 'train')}")
print(f"Test samples:  {sum(df['split'] == 'test')}")
print(f"Saved to: {DATA_PATH}")