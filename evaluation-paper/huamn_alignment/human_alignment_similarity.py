import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer, util

# ==========================================
# 1. CONFIGURATION (Update paths if needed)
# ==========================================
MODELS_DIR = Path("models/output_model")  # <-- Check exact folder name (e.g., models/output_models)
HUMAN1_CSV = Path("dataset/annotated.csv")
HUMAN2_CSV = Path("dataset/annotated_2.csv")
OUTPUT_DIR = Path("evaluation-paper/human_alignment/similarity_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# 2. DATA LOADING HELPERS
# ==========================================
def parse_csv_list(val) -> list[str]:
    """Safely parse JSON/Python list strings from CSV cells."""
    if pd.isna(val) or not str(val).strip():
        return []
    text = str(val).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = [text]
    if isinstance(parsed, list):
        return [str(x).strip() for x in parsed if str(x).strip()]
    return [str(parsed).strip()]


def load_human_csv(csv_path: Path) -> dict[str, list[str]]:
    """Returns {image_id_str: [conclusion_1, conclusion_2, ...]}."""
    df = pd.read_csv(csv_path)
    mapping = {}
    for _, row in df.iterrows():
        img_id = str(row["id"]).strip()
        # Handle float IDs if pandas read '10' as '10.0'
        if img_id.endswith(".0"):
            img_id = img_id[:-2]
        mapping[img_id] = parse_csv_list(row["conclusions"])
    return mapping


def load_model_json(json_path: Path) -> dict[str, list[str]]:
    """Returns {image_id_str: [conclusion_1, conclusion_2, ...]}."""
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    mapping = {}
    for item in records:
        img_id = str(item["image_id"]).strip()
        if img_id.endswith(".0"):
            img_id = img_id[:-2]
        parsed_out = item.get("parsed_output") or {}
        concs = parsed_out.get("conclusions") or []
        mapping[img_id] = [str(c).strip() for c in concs if str(c).strip()]
    return mapping


# ==========================================
# 3. SIMILARITY & AGGREGATION ENGINE
# ==========================================
def aggregate_matrix(mat: np.ndarray) -> dict[str, float | list[float]]:
    """
    Given a [|M|, |H|] similarity matrix:
    - Sorts all pairwise scores descending
    - Returns max, top-3 mean, and mean-max (precision-oriented)
    """
    sorted_scores = np.sort(mat.flatten())[::-1]
    return {
        "max": float(sorted_scores[0]),
        "top3": float(np.mean(sorted_scores[:3])),  # Uses all pairs if < 3
        "mean_max": float(np.mean(np.max(mat, axis=1))),
        "all_sorted": [round(float(x), 4) for x in sorted_scores],
    }


def compute_all_comparisons(
    comparisons: list[tuple[str, str, dict[str, list[str]], dict[str, list[str]]]],
    image_ids: list[str],
    st_model: SentenceTransformer,
    bert_scorer: BERTScorer,
):
    """
    Batches all pairs across all comparisons for fast GPU/CPU inference.
    """
    # Collect every (cand_sentence, ref_sentence) pair to score BERTScore in one batch
    all_cands = []
    all_refs = []
    slices = []  # Tracks where each (comparison, img_id) sits in all_cands

    for cand_name, ref_name, cand_map, ref_map in comparisons:
        for img_id in image_ids:
            M = cand_map.get(img_id, [])
            H = ref_map.get(img_id, [])
            if not M or not H:
                slices.append((cand_name, ref_name, cand_map, ref_map, img_id, 0, 0, None, None))
                continue

            start_idx = len(all_cands)
            for m in M:
                for h in H:
                    all_cands.append(m)
                    all_refs.append(h)
            end_idx = len(all_cands)
            slices.append((cand_name, ref_name, cand_map, ref_map, img_id, len(M), len(H), start_idx, end_idx))

    print(f"Scoring {len(all_cands)} total conclusion pairs with BERTScore ({DEVICE})...")
    _, _, bert_f1_all = bert_scorer.score(all_cands, all_refs, batch_size=64)
    bert_f1_all = bert_f1_all.cpu().numpy()

    # Encode unique sentences once for Sentence-Transformer Cosine Similarity
    print("Computing Sentence-Transformer embeddings...")
    unique_texts = list(set(all_cands + all_refs))
    unique_embs = st_model.encode(
        unique_texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False
    )
    text_to_emb = {txt: emb for txt, emb in zip(unique_texts, unique_embs)}

    per_image_rows = []
    raw_matrices = {}

    for cand_name, ref_name, cand_map, ref_map, img_id, n_m, n_h, start_idx, end_idx in slices:
        if start_idx is None:
            per_image_rows.append({
                "candidate": cand_name,
                "reference": ref_name,
                "image_id": img_id,
                "num_cand_concs": 0,
                "num_ref_concs": 0,
                "cos_max": np.nan,
                "cos_top3": np.nan,
                "cos_mean_max": np.nan,
                "bert_max": np.nan,
                "bert_top3": np.nan,
                "bert_mean_max": np.nan,
                "cos_sorted_pairs": "[]",
                "bert_sorted_pairs": "[]",
            })
            continue

        # 1. Reconstruct BERTScore [|M|, |H|] matrix
        bert_slice = bert_f1_all[start_idx:end_idx]
        bert_mat = bert_slice.reshape(n_m, n_h)

        # 2. Reconstruct Cosine Similarity [|M|, |H|] matrix
        M_texts = cand_map[img_id]
        H_texts = ref_map[img_id]
        emb_M = torch.stack([text_to_emb[t] for t in M_texts])
        emb_H = torch.stack([text_to_emb[t] for t in H_texts])
        cos_mat = util.cos_sim(emb_M, emb_H).cpu().numpy()

        cos_agg = aggregate_matrix(cos_mat)
        bert_agg = aggregate_matrix(bert_mat)

        per_image_rows.append({
            "candidate": cand_name,
            "reference": ref_name,
            "image_id": img_id,
            "num_cand_concs": n_m,
            "num_ref_concs": n_h,
            "cos_max": cos_agg["max"],
            "cos_top3": cos_agg["top3"],
            "cos_mean_max": cos_agg["mean_max"],
            "bert_max": bert_agg["max"],
            "bert_top3": bert_agg["top3"],
            "bert_mean_max": bert_agg["mean_max"],
            "cos_sorted_pairs": json.dumps(cos_agg["all_sorted"]),
            "bert_sorted_pairs": json.dumps(bert_agg["all_sorted"]),
        })

        key = f"{cand_name}_vs_{ref_name}"
        raw_matrices.setdefault(key, {})[img_id] = {
            "candidate_conclusions": M_texts,
            "reference_conclusions": H_texts,
            "cosine_matrix": np.round(cos_mat, 4).tolist(),
            "bert_f1_matrix": np.round(bert_mat, 4).tolist(),
        }

    return pd.DataFrame(per_image_rows), raw_matrices


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    # Load Human References
    h1_map = load_human_csv(HUMAN1_CSV)
    h2_map = load_human_csv(HUMAN2_CSV)

    # Use the intersection of image IDs present in human annotations (46 images)
    image_ids = sorted(set(h1_map.keys()) & set(h2_map.keys()), key=lambda x: int(x) if x.isdigit() else x)
    print(f"Loaded {len(image_ids)} aligned images from human CSVs.")

    # Combined Human Pool (Human 1 + Human 2 conclusions per image)
    h_combined_map = {
        img_id: h1_map.get(img_id, []) + h2_map.get(img_id, [])
        for img_id in image_ids
    }

    # Load all model JSON files automatically from MODELS_DIR
    model_files = sorted(MODELS_DIR.glob("*.json"))
    if not model_files:
        raise FileNotFoundError(f"No .json files found in {MODELS_DIR.resolve()}")

    models_data = {}
    for mf in model_files:
        model_name = mf.stem  # e.g., "llava", "minicpm", "nemotron"
        models_data[model_name] = load_model_json(mf)
        print(f"Loaded model '{model_name}' ({len(models_data[model_name])} items)")

    # Build comparison list:
    # 1. Human 1 vs Human 2 (Inter-annotator semantic upper bound)
    comparisons = [("human1", "human2", h1_map, h2_map)]
    # 2. Each model vs Human 1, Human 2, and Combined Human Pool
    for model_name, m_map in models_data.items():
        comparisons.append((model_name, "human1", m_map, h1_map))
        comparisons.append((model_name, "human2", m_map, h2_map))
        comparisons.append((model_name, "human_combined", m_map, h_combined_map))

    # Load Models
    st_model = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)
    bert_scorer = BERTScorer(
        model_type="roberta-large",
        lang="en",
        rescale_with_baseline=False,
        device=DEVICE,
    )

    # Run evaluation
    df_per_image, raw_matrices = compute_all_comparisons(
        comparisons, image_ids, st_model, bert_scorer
    )

    # Save per-image CSV
    per_image_path = OUTPUT_DIR / "per_image_similarity.csv"
    df_per_image.to_csv(per_image_path, index=False)

    # Save raw matrices JSON
    matrices_path = OUTPUT_DIR / "raw_pairwise_matrices.json"
    with open(matrices_path, "w", encoding="utf-8") as f:
        json.dump(raw_matrices, f, indent=2, ensure_ascii=False)

    # Create Summary Table (Mean ± SD across images)
    summary_rows = []
    for (cand, ref), group in df_per_image.groupby(["candidate", "reference"], sort=False):
        valid_n = group["cos_max"].notna().sum()
        summary_rows.append({
            "Candidate": cand,
            "Reference": ref,
            "Valid_N": valid_n,
            "Cos_Max_Mean": round(group["cos_max"].mean(), 4),
            "Cos_Max_SD": round(group["cos_max"].std(), 4),
            "Cos_Top3_Mean": round(group["cos_top3"].mean(), 4),
            "Cos_MeanMax_Mean": round(group["cos_mean_max"].mean(), 4),
            "BERT_Max_Mean": round(group["bert_max"].mean(), 4),
            "BERT_Max_SD": round(group["bert_max"].std(), 4),
            "BERT_Top3_Mean": round(group["bert_top3"].mean(), 4),
            "BERT_MeanMax_Mean": round(group["bert_mean_max"].mean(), 4),
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "summary_table.csv"
    df_summary.to_csv(summary_path, index=False)

    print("\n=== SUMMARY RESULTS ===")
    print(df_summary.to_string(index=False))
    print(f"\nSaved files to '{OUTPUT_DIR.resolve()}':")
    print(f" - {per_image_path.name}")
    print(f" - {summary_path.name}")
    print(f" - {matrices_path.name}")


if __name__ == "__main__":
    main()