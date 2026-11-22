import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
import open_clip

# ==========================================
# 1. CONFIGURATION (Update paths to match your repo)
# ==========================================
IMAGES_DIR = Path("dataset/images_for_annotation")  
MODELS_DIR = Path("models/output_model")
HUMAN1_CSV = Path("dataset/annotated.csv")
HUMAN2_CSV = Path("dataset/annotated_2.csv")
OUTPUT_DIR = Path("evaluation-paper/visual_grounding/clipscore_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# 2. DATA & IMAGE LOADING HELPERS
# ==========================================
def parse_csv_list(val) -> list[str]:
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
    return [str(x).strip() for x in parsed if str(x).strip()] if isinstance(parsed, list) else [str(parsed).strip()]


def load_human_csv(csv_path: Path) -> dict[str, dict[str, list[str]]]:
    df = pd.read_csv(csv_path)
    mapping = {}
    for _, row in df.iterrows():
        img_id = str(row["id"]).strip().removesuffix(".0")
        mapping[img_id] = {
            "premises": parse_csv_list(row["premises"]),
            "conclusions": parse_csv_list(row["conclusions"]),
        }
    return mapping


def load_model_json(json_path: Path) -> dict[str, dict[str, list[str]]]:
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    mapping = {}
    for item in records:
        img_id = str(item["image_id"]).strip().removesuffix(".0")
        parsed = item.get("parsed_output") or {}
        mapping[img_id] = {
            "premises": [str(p).strip() for p in (parsed.get("premises") or []) if str(p).strip()],
            "conclusions": [str(c).strip() for c in (parsed.get("conclusions") or []) if str(c).strip()],
        }
    return mapping


def find_image_path(images_dir: Path, img_id: str) -> Path | None:
    """Finds image file matching img_id across common extensions."""
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        candidate = images_dir / f"{img_id}{ext}"
        if candidate.exists():
            return candidate
    return None


# ==========================================
# 3. CLIPSCORE COMPUTATION
# ==========================================
@torch.no_grad()
def compute_clip_scores(
    sources: dict[str, dict[str, dict[str, list[str]]]],
    image_ids: list[str],
    model,
    preprocess,
    tokenizer,
):
    rows = []

    for img_id in image_ids:
        img_path = find_image_path(IMAGES_DIR, img_id)
        if img_path is None:
            print(f"Warning: Image not found for ID {img_id}")
            continue

        # Encode image once
        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
        img_feat = model.encode_image(image)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        for source_name, data_map in sources.items():
            entry = data_map.get(img_id, {"premises": [], "conclusions": []})

            def score_texts(text_list: list[str]) -> tuple[float, float]:
                if not text_list:
                    return np.nan, np.nan
                tokens = tokenizer(text_list).to(DEVICE)
                txt_feat = model.encode_text(tokens)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                # Standard CLIPScore: w * max(cos_sim, 0), with w = 2.5 (Hessel et al., 2021)
                cos_sims = (img_feat @ txt_feat.T).squeeze(0).cpu().numpy()
                clip_scores = 2.5 * np.maximum(cos_sims, 0.0)
                return float(np.mean(clip_scores)), float(np.min(clip_scores))

            prem_mean, prem_min = score_texts(entry["premises"])
            conc_mean, conc_min = score_texts(entry["conclusions"])

            rows.append({
                "source": source_name,
                "image_id": img_id,
                "premise_clip_mean": prem_mean,
                "premise_clip_min": prem_min,  # Lowest-grounded premise (catches single-premise hallucinations)
                "conclusion_clip_mean": conc_mean,
                "grounding_drop": prem_mean - conc_mean if not (np.isnan(prem_mean) or np.isnan(conc_mean)) else np.nan,
            })

    return pd.DataFrame(rows)


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    h1_map = load_human_csv(HUMAN1_CSV)
    h2_map = load_human_csv(HUMAN2_CSV)
    image_ids = sorted(set(h1_map.keys()) & set(h2_map.keys()), key=lambda x: int(x) if x.isdigit() else x)

    sources = {"human1": h1_map, "human2": h2_map}
    for mf in sorted(MODELS_DIR.glob("*.json")):
        sources[mf.stem] = load_model_json(mf)

    print(f"Loading CLIP ViT-L-14 on {DEVICE}...")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=DEVICE)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")

    df_per_image = compute_clip_scores(sources, image_ids, model, preprocess, tokenizer)
    df_per_image.to_csv(OUTPUT_DIR / "per_image_clipscore.csv", index=False)

    summary_rows = []
    for source_name, group in df_per_image.groupby("source", sort=False):
        summary_rows.append({
            "Source": source_name,
            "Premise_CLIP_Mean": round(group["premise_clip_mean"].mean(), 2),
            "Premise_CLIP_SD": round(group["premise_clip_mean"].std(), 2),
            "Premise_CLIP_Min_Avg": round(group["premise_clip_min"].mean(), 2),
            "Conclusion_CLIP_Mean": round(group["conclusion_clip_mean"].mean(), 2),
            "Grounding_Drop (Prem - Conc)": round(group["grounding_drop"].mean(), 2),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUTPUT_DIR / "clipscore_summary.csv", index=False)

    print("\n=== VISUAL GROUNDING (CLIPScore) SUMMARY ===")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()