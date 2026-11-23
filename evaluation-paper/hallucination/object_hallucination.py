import ast
import json
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
import spacy
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

IMAGES_DIR = Path("dataset/images_for_annotation")  # <-- Folder containing your 46 image files (e.g., 0.jpg, 10.jpg)
MODELS_DIR = Path("models/output_model")
HUMAN1_CSV = Path("dataset/annotated.csv")
HUMAN2_CSV = Path("dataset/annotated_2.csv")
OUTPUT_DIR = Path("evaluation-paper/hallucination/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
nlp = spacy.load("en_core_web_lg")

# Use Apple Silicon GPU (mps) if available, otherwise CUDA or CPU
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

BOX_THRESHOLD = 0.30

ABSTRACT_STOPWORDS = {
    "image", "picture", "photo", "scene", "background", "foreground", "view",
    "side", "left", "right", "top", "bottom", "center", "part", "area", "region",
    "environment", "nature", "climate", "change", "pollution", "damage", "effect",
    "impact", "problem", "issue", "situation", "condition", "sign", "evidence",
    "lack", "absence", "presence", "amount", "level", "number", "group", "lot",
    "way", "time", "day", "year", "future", "world", "earth", "planet", "life"
}

NEGATION_PATTERNS = re.compile(
    r"\b(no|not|without|lack of|lacking|absence of|missing|empty of|devoid of)\b",
    re.IGNORECASE,
)


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


def load_human_premises(csv_path: Path) -> dict[str, list[str]]:
    df = pd.read_csv(csv_path)
    return {
        str(row["id"]).strip().removesuffix(".0"): parse_csv_list(row["premises"])
        for _, row in df.iterrows()
    }


def load_model_premises(json_path: Path) -> dict[str, list[str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    mapping = {}
    for item in records:
        img_id = str(item["image_id"]).strip().removesuffix(".0")
        parsed = item.get("parsed_output") or {}
        mapping[img_id] = [str(p).strip() for p in (parsed.get("premises") or []) if str(p).strip()]
    return mapping


def find_image_path(images_dir: Path, img_id: str) -> Path | None:
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        p = images_dir / f"{img_id}{ext}"
        if p.exists():
            return p
    return None


def extract_physical_objects(premise: str, nlp) -> tuple[list[str], bool]:
    has_negation = bool(NEGATION_PATTERNS.search(premise))
    doc = nlp(premise)
    objects = []
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"}:
            lemma = token.lemma_.lower().strip()
            if len(lemma) <= 2 or lemma in ABSTRACT_STOPWORDS:
                continue
            compounds = [child.text.lower() for child in token.lefts if child.dep_ == "compound"]
            phrase = " ".join(compounds + [token.text.lower()])
            if phrase not in objects:
                objects.append(phrase)
    return objects, has_negation


@torch.no_grad()
def verify_objects_grounding_dino(
    image: Image.Image,
    objects: list[str],
    processor,
    model,
) -> dict[str, float]:
    """Queries Grounding DINO ONCE per image with all unique objects across all sources."""
    if not objects:
        return {}

    text_prompt = " . ".join(objects) + " ."
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.05,
        text_threshold=0.05,
        target_sizes=[image.size[::-1]],
    )[0]

    scores = results["scores"].cpu().tolist()
    labels = results.get("text_labels", results["labels"])

    obj_scores = {obj: 0.0 for obj in objects}
    for score, label in zip(scores, labels):
        clean_lbl = str(label).strip().lower()
        for obj in objects:
            if obj in clean_lbl or clean_lbl in obj:
                obj_scores[obj] = max(obj_scores[obj], float(score))

    return obj_scores


def main():
    start_time = time.time()
    model_id = "IDEA-Research/grounding-dino-base"
    print(f"Loading {model_id} on {DEVICE}...")
    processor = AutoProcessor.from_pretrained(model_id)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE).eval()

    h1_map = load_human_premises(HUMAN1_CSV)
    h2_map = load_human_premises(HUMAN2_CSV)
    image_ids = sorted(set(h1_map.keys()) & set(h2_map.keys()), key=lambda x: int(x) if x.isdigit() else x)

    sources = {"human1": h1_map, "human2": h2_map}
    for mf in sorted(MODELS_DIR.glob("*.json")):
        sources[mf.stem] = load_model_premises(mf)

    detail_rows = []
    total_imgs = len(image_ids)

    for idx, img_id in enumerate(image_ids, 1):
        img_path = find_image_path(IMAGES_DIR, img_id)
        if img_path is None:
            continue
        image = Image.open(img_path).convert("RGB")

        # 1. Extract objects for all sources on this image first
        parsed_per_source = {}
        all_unique_objs = set()
        for source_name, prem_map in sources.items():
            parsed_list = []
            for p_idx, premise in enumerate(prem_map.get(img_id, [])):
                objs, has_neg = extract_physical_objects(premise, nlp)
                parsed_list.append((p_idx, premise, objs, has_neg))
                all_unique_objs.update(objs)
            parsed_per_source[source_name] = parsed_list

        # 2. Run Grounding DINO ONCE for this image across all unique objects
        obj_scores_map = verify_objects_grounding_dino(image, sorted(all_unique_objs), processor, detector)

        # 3. Record premise-level results
        for source_name, parsed_list in parsed_per_source.items():
            for p_idx, premise, objects, has_negation in parsed_list:
                obj_scores = {o: obj_scores_map.get(o, 0.0) for o in objects}
                absent_objs = [o for o, sc in obj_scores.items() if sc < BOX_THRESHOLD]
                is_hallucinated_prem = (len(absent_objs) > 0) or has_negation

                detail_rows.append({
                    "source": source_name,
                    "image_id": img_id,
                    "premise_idx": p_idx,
                    "premise": premise,
                    "extracted_objects": json.dumps(objects),
                    "object_scores": json.dumps({k: round(v, 3) for k, v in obj_scores.items()}),
                    "absent_objects": json.dumps(absent_objs),
                    "has_negation_cue": has_negation,
                    "total_objects": len(objects),
                    "num_absent_objects": len(absent_objs),
                    "is_flagged_premise": is_hallucinated_prem,
                })

        print(f"[{idx}/{total_imgs}] Processed image {img_id} ({len(all_unique_objs)} objects) - {time.time() - start_time:.1f}s")

    df_details = pd.DataFrame(detail_rows)
    df_details.to_csv(OUTPUT_DIR / "premise_object_audit.csv", index=False)

    summary_rows = []
    for source_name, group in df_details.groupby("source", sort=False):
        total_prems = len(group)
        flagged_prems = group["is_flagged_premise"].sum()
        negation_prems = group["has_negation_cue"].sum()
        total_objs = group["total_objects"].sum()
        absent_objs = group["num_absent_objects"].sum()

        summary_rows.append({
            "Source": source_name,
            "Total_Premises": total_prems,
            "Flagged_Premises (%)": round(100 * flagged_prems / max(total_prems, 1), 1),
            "Negation_Cues (%)": round(100 * negation_prems / max(total_prems, 1), 1),
            "Total_Objects": total_objs,
            "Absent_Objects_CHAIRi (%)": round(100 * absent_objs / max(total_objs, 1), 1),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUTPUT_DIR / "object_hallucination_summary.csv", index=False)

    print("\n=== ABSENT-OBJECT HALLUCINATION SUMMARY ===")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()