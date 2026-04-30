# 🧠 Multimodal Argument Generation Project — Context Transfer Document

## 📌 Overview

This project focuses on **multimodal argument generation**, where the goal is:

> **Generate a structured argument (premises + conclusion) from an image and related metadata.**

The system combines:

* image understanding
* structured labels
* textual reasoning
* external knowledge (facts)

---

# 📊 Dataset Description

Current dataset size: **58 samples** (will expand later)

Each row contains:

* `id`
* `hash_id`
* `url`
* `animals`
* `consequences`
* `climateaction`
* `type`
* `setting`
* `source_file`
* `premises` → list of sentences
* `conclusions` → list of sentences
* `human_generated_argument`

Additional resource:

* `facts.json` → collection of factual sentences (used for retrieval)

---

# 🎯 Task Definition

### Input:

* Image
* Metadata (labels)
* Retrieved facts (external knowledge)

### Output:

* Generated argument:

  * premises
  * conclusion

---

# 🧩 Feature Extraction Pipeline

The system is modular. Each component extracts a different type of information.

---

## 1️⃣ CLIP Image Embeddings

### Purpose:

Capture **global semantic meaning of the image**

### Output:

* `.npy` file
* shape: `(num_images, embedding_dim)`

### Nature:

* dense vector
* NOT interpretable

---

## 2️⃣ Climate Attribute Extraction

### Purpose:

Detect **explicit environmental signals**

### Examples:

* smoke
* fire
* pollution
* flood
* deforestation
* ice melting

### Method:

* image ↔ text similarity (CLIP prompts)
* threshold-based detection

### Output:

```json
{
  "smoke": 1,
  "pollution": 1,
  "flood": 0
}
```

### Nature:

* interpretable
* binary signals

---

## 3️⃣ Scene Classification

### Purpose:

Identify **environmental context**

### Categories:

* urban
* rural
* industrial
* natural

### Method:

* color-based heuristics (HSV analysis)

### Output:

```json
{
  "scene_scores": {...},
  "primary_category": "industrial"
}
```

### Nature:

* weak signal (heuristic)
* used as auxiliary feature

---

## 4️⃣ Caption Generation (planned/optional)

### Purpose:

Bridge **image → language**

### Role:

* helps connect visual input to text reasoning
* supports premise selection

---

## 5️⃣ Text Feature Extraction & Fact Retrieval (RAG)

### Purpose:

Add **external knowledge to reasoning**

---

### Step 1: Encode Premises

* each premise → embedding
* model: sentence transformer

---

### Step 2: Encode Facts

* all facts → embeddings
* stored for reuse

---

### Step 3: Retrieve Relevant Facts

For each image:

* compare premises with all facts
* select top-k relevant facts

---

### Output Files:

#### 1. Premise embeddings

```
premise_embeddings.npy
```

* shape: `(num_images, dim)`
* ⚠️ averaged per image

---

#### 2. Fact embeddings

```
fact_embeddings.npy
```

---

#### 3. Retrieval results (MOST IMPORTANT)

```
fact_retrievals.json
```

Example:

```json
{
  "id": "12",
  "retrieved_facts": [
    {"fact": "...", "score": 0.82}
  ]
}
```

---

# ⚠️ Key Design Issue: Multiple Premises

## Problem

Each image has variable number of premises:

```
Image A → 2 premises
Image B → 5 premises
Image C → 1 premise
```

But ML models expect:

```
fixed-size input
```

---

## Current Solution

### Averaging embeddings:

```
P1, P2, P3 → mean → single vector
```

### Why:

* simple
* fixed size
* easy to train

---

## Limitation

* loses structure
* mixes different ideas
* important premises may be diluted

---

## Important Insight

👉 Retrieval step DOES NOT lose information

* each premise is used individually
* best facts are selected across all premises

---

## Suggested Improvements

### Option 1 (current — acceptable)

* keep averaging
* use retrieved facts for reasoning

---

### Option 2 (better)

* store all premise embeddings
* use padding

---

### Option 3 (best, advanced)

* use attention mechanism
* model learns importance of each premise

---

# 🧠 Feature Types Summary

| Feature         | Type         | Role               |
| --------------- | ------------ | ------------------ |
| CLIP embedding  | dense vector | global meaning     |
| attributes      | binary       | reasoning triggers |
| scene           | categorical  | context            |
| caption         | text         | language bridge    |
| premises        | text         | reasoning input    |
| retrieved facts | text         | knowledge support  |

---

# 🧩 Final Pipeline

```
Image
 → CLIP embedding
 → Attribute detection
 → Scene classification
 → (Optional caption)

Text:
 → Premises
 → Fact retrieval

→ Combine all

→ Argument generation model
```

---

# 🎯 Key Concept

This system is:

> **Multimodal + Retrieval-Augmented Argument Generation**

It combines:

* vision
* language
* structured metadata
* external knowledge

---

# ⚙️ Engineering Design Decisions

## Modular pipeline (important)

Each feature is extracted in separate scripts:

* easier debugging
* reproducibility
* flexibility
* better for research/thesis

---

## Data storage strategy

* embeddings → `.npy`
* structured features → `.json`
* indices → `.csv`

---

# 🚀 Current Status

✔ dataset created (58 samples)
✔ image features extracted
✔ attribute detection implemented
✔ scene classification implemented
✔ text embeddings + retrieval implemented

---

# 🔜 Next Steps

1. Combine all features into unified input
2. Design model architecture
3. Train argument generation model
4. Evaluate generated arguments

---

# 🧠 Key Insight for Future Work

* embeddings = powerful but not interpretable
* attributes = interpretable but simple
* retrieval = adds knowledge

👉 Best performance comes from combining all three

---

# ✅ One-line Summary

This project builds a multimodal pipeline that generates arguments from images by combining visual features, structured signals, and retrieved knowledge using a modular and extensible architecture.

---
