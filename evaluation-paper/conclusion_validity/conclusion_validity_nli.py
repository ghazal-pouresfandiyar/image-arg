import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ==========================================
# 1. CONFIGURATION (Update paths if needed)
# ==========================================
MODELS_DIR = Path("models/output_model")
HUMAN1_CSV = Path("dataset/annotated.csv")
HUMAN2_CSV = Path("dataset/annotated_2.csv")
OUTPUT_DIR = Path("evaluation-paper/conclusion_validity/nli_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# DeBERTa-v3-large NLI checkpoint (other options: "cross-encoder/nli-deberta-v3-large", "roberta-large-mnli")
NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
BATCH_SIZE = 32
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


def load_human_csv(csv_path: Path) -> dict[str, dict[str, list[str]]]:
    """Returns {image_id_str: {'premises': [...], 'conclusions': [...]}}."""
    df = pd.read_csv(csv_path)
    if "premises" not in df.columns or "conclusions" not in df.columns:
        raise KeyError(
            f"Expected 'premises' and 'conclusions' columns in {csv_path}, found: {list(df.columns)}"
        )

    mapping = {}
    for _, row in df.iterrows():
        img_id = str(row["id"]).strip()
        if img_id.endswith(".0"):
            img_id = img_id[:-2]
        mapping[img_id] = {
            "premises": parse_csv_list(row["premises"]),
            "conclusions": parse_csv_list(row["conclusions"]),
        }
    return mapping


def load_model_json(json_path: Path) -> dict[str, dict[str, list[str]]]:
    """Returns {image_id_str: {'premises': [...], 'conclusions': [...]}}."""
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    mapping = {}
    for item in records:
        img_id = str(item["image_id"]).strip()
        if img_id.endswith(".0"):
            img_id = img_id[:-2]
        parsed_out = item.get("parsed_output") or {}
        prems = parsed_out.get("premises") or []
        concs = parsed_out.get("conclusions") or []
        mapping[img_id] = {
            "premises": [str(p).strip() for p in prems if str(p).strip()],
            "conclusions": [str(c).strip() for c in concs if str(c).strip()],
        }
    return mapping


def format_premise_block(premises: list[str]) -> str:
    """Combine multiple premises into a single clean paragraph for NLI."""
    cleaned = []
    for p in premises:
        p = p.strip()
        if p and p[-1] not in ".!?":
            p += "."
        if p:
            cleaned.append(p)
    return " ".join(cleaned)


# ==========================================
# 3. NLI SCORING ENGINE
# ==========================================
def run_nli_evaluation(
    sources: list[tuple[str, dict[str, dict[str, list[str]]]]],
    image_ids: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
):
    """Score self-consistency (Premises -> Conclusion) across all sources and images."""
    # Map label indices dynamically from model config
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    ent_idx = next(i for i, lbl in id2label.items() if "entail" in lbl)
    neu_idx = next(i for i, lbl in id2label.items() if "neutral" in lbl)
    con_idx = next(i for i, lbl in id2label.items() if "contradict" in lbl)

    # Collect all (premise_block, conclusion) pairs for batched inference
    all_premises = []
    all_conclusions = []
    pair_meta = []

    for source_name, data_map in sources:
        for img_id in image_ids:
            entry = data_map.get(img_id, {"premises": [], "conclusions": []})
            prems = entry["premises"]
            concs = entry["conclusions"]

            if not prems or not concs:
                continue

            premise_block = format_premise_block(prems)
            for c_idx, conc in enumerate(concs, start=1):
                all_premises.append(premise_block)
                all_conclusions.append(conc)
                pair_meta.append({
                    "source": source_name,
                    "image_id": img_id,
                    "conclusion_idx": c_idx,
                    "num_premises": len(prems),
                    "num_conclusions": len(concs),
                    "premise_block": premise_block,
                    "conclusion": conc,
                })

    print(f"Scoring {len(all_premises)} (premise_block, conclusion) pairs on {DEVICE}...")
    all_probs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(all_premises), BATCH_SIZE):
            batch_p = all_premises[start : start + BATCH_SIZE]
            batch_c = all_conclusions[start : start + BATCH_SIZE]
            inputs = tokenizer(
                batch_p,
                batch_c,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(DEVICE)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.vstack(all_probs)

    # Build per-conclusion DataFrame
    per_conc_rows = []
    for meta, probs in zip(pair_meta, all_probs):
        p_ent = float(probs[ent_idx])
        p_neu = float(probs[neu_idx])
        p_con = float(probs[con_idx])

        scores = {"entailment": p_ent, "neutral": p_neu, "contradiction": p_con}
        pred_label = max(scores, key=scores.get)

        per_conc_rows.append({
            **meta,
            "prob_entailment": round(p_ent, 4),
            "prob_neutral": round(p_neu, 4),
            "prob_contradiction": round(p_con, 4),
            "predicted_label": pred_label,
        })

    df_per_conc = pd.DataFrame(per_conc_rows)

    # Build per-image aggregated DataFrame (including NaN rows for invalid/unparsed outputs)
    per_image_rows = []
    raw_details = {}

    for source_name, data_map in sources:
        sub_source = df_per_conc[df_per_conc["source"] == source_name]
        for img_id in image_ids:
            entry = data_map.get(img_id, {"premises": [], "conclusions": []})
            sub_img = sub_source[sub_source["image_id"] == img_id]

            if sub_img.empty:
                per_image_rows.append({
                    "source": source_name,
                    "image_id": img_id,
                    "num_premises": len(entry["premises"]),
                    "num_conclusions": len(entry["conclusions"]),
                    "entailment_mean": np.nan,
                    "entailment_max": np.nan,
                    "neutral_mean": np.nan,
                    "contradiction_mean": np.nan,
                    "contradiction_max": np.nan,
                    "labels": "[]",
                })
                continue

            per_image_rows.append({
                "source": source_name,
                "image_id": img_id,
                "num_premises": int(sub_img["num_premises"].iloc[0]),
                "num_conclusions": int(sub_img["num_conclusions"].iloc[0]),
                "entailment_mean": round(float(sub_img["prob_entailment"].mean()), 4),
                "entailment_max": round(float(sub_img["prob_entailment"].max()), 4),
                "neutral_mean": round(float(sub_img["prob_neutral"].mean()), 4),
                "contradiction_mean": round(float(sub_img["prob_contradiction"].mean()), 4),
                "contradiction_max": round(float(sub_img["prob_contradiction"].max()), 4),
                "labels": json.dumps(sub_img["predicted_label"].tolist()),
            })

            raw_details.setdefault(source_name, {})[img_id] = {
                "premises": entry["premises"],
                "conclusions": [
                    {
                        "text": r["conclusion"],
                        "prob_entailment": r["prob_entailment"],
                        "prob_neutral": r["prob_neutral"],
                        "prob_contradiction": r["prob_contradiction"],
                        "predicted_label": r["predicted_label"],
                    }
                    for _, r in sub_img.iterrows()
                ],
            }

    return df_per_conc, pd.DataFrame(per_image_rows), raw_details


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    # Load Human Data (as empirical baseline)
    h1_map = load_human_csv(HUMAN1_CSV)
    h2_map = load_human_csv(HUMAN2_CSV)

    image_ids = sorted(
        set(h1_map.keys()) & set(h2_map.keys()),
        key=lambda x: int(x) if x.isdigit() else x,
    )
    print(f"Loaded {len(image_ids)} aligned images from human CSVs.")

    sources = [("human1", h1_map), ("human2", h2_map)]

    # Load all model JSON files automatically from MODELS_DIR
    model_files = sorted(MODELS_DIR.glob("*.json"))
    if not model_files:
        raise FileNotFoundError(f"No .json files found in {MODELS_DIR.resolve()}")

    for mf in model_files:
        model_name = mf.stem
        m_map = load_model_json(mf)
        sources.append((model_name, m_map))
        print(f"Loaded model '{model_name}' ({len(m_map)} items)")

    # Load NLI Model & Tokenizer
    print(f"Loading NLI model: {NLI_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME).to(DEVICE)

    # Run evaluation
    df_per_conc, df_per_image, raw_details = run_nli_evaluation(
        sources, image_ids, tokenizer, nli_model
    )

    # Save detailed outputs
    per_conc_path = OUTPUT_DIR / "per_conclusion_nli.csv"
    per_image_path = OUTPUT_DIR / "per_image_nli.csv"
    details_path = OUTPUT_DIR / "raw_nli_details.json"

    df_per_conc.to_csv(per_conc_path, index=False)
    df_per_image.to_csv(per_image_path, index=False)
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(raw_details, f, indent=2, ensure_ascii=False)

    # Build Primary Summary Table
    summary_rows = []
    for source_name, _ in sources:
        img_grp = df_per_image[df_per_image["source"] == source_name]
        conc_grp = df_per_conc[df_per_conc["source"] == source_name]

        valid_imgs = int(img_grp["entailment_mean"].notna().sum())
        total_concs = len(conc_grp)

        ent_rate = (conc_grp["predicted_label"] == "entailment").mean() * 100 if total_concs else np.nan
        neu_rate = (conc_grp["predicted_label"] == "neutral").mean() * 100 if total_concs else np.nan
        con_rate = (conc_grp["predicted_label"] == "contradiction").mean() * 100 if total_concs else np.nan

        summary_rows.append({
            "Source": source_name,
            "Valid_Images": valid_imgs,
            "Total_Conclusions": total_concs,
            "Entail_Prob_Mean": round(img_grp["entailment_mean"].mean(), 4),
            "Entail_Prob_SD": round(img_grp["entailment_mean"].std(), 4),
            "Neutral_Prob_Mean": round(img_grp["neutral_mean"].mean(), 4),
            "Contra_Prob_Mean": round(img_grp["contradiction_mean"].mean(), 4),
            "Entail_Label_%": round(ent_rate, 1),
            "Neutral_Label_%": round(neu_rate, 1),
            "Contra_Label_%": round(con_rate, 1),
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "nli_summary_table.csv"
    df_summary.to_csv(summary_path, index=False)

    print("\n=== PRIMARY NLI SELF-CONSISTENCY RESULTS ===")
    print(df_summary.to_string(index=False))
    print(f"\nSaved files to '{OUTPUT_DIR.resolve()}':")
    print(f" - {summary_path.name}")
    print(f" - {per_image_path.name}")
    print(f" - {per_conc_path.name}")
    print(f" - {details_path.name}")


if __name__ == "__main__":
    main()