"""Evaluation UI backend.

Run:  python user_interfaces/evaluation/eval_app.py
Then open http://127.0.0.1:5001/
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

# This file lives in <project>/user_interfaces/evaluation/
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parents[1]  # <project> root
DATASET_DIR = PROJECT_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images_for_annotation"
MODEL_OUTPUT_DIR = PROJECT_DIR / "models" / "output_model"
EVAL_FILE = DATASET_DIR / "evaluations.json"

HUMAN_CSV_FILES = [
    ("human1", DATASET_DIR / "annotated.csv"),
    ("human2", DATASET_DIR / "annotated_2.csv"),
]

MODEL_MAP = {
    "llava": ("llava_outputs.json", "LLaVA"),
    "minicpm": ("minicpm-v_outputs.json", "MiniCPM-V"),
    "nemotron": ("nemotron-3-nano-omni-30b-a3b-reasoning_outputs.json", "Nemotron-3"),
}

# Metrics loaded from metrics.html for reference; also exposed to the client.
METRICS = {
    "Output Format": [
        {
            "key": "format_adherence",
            "label": "Format Adherence",
            "what": "Did the model follow the required output format (e.g., sentence structure, labels, sections)?",
            "options": ["Perfect", "Minor errors", "Partial failure", "Complete failure"],
        },
        {
            "key": "quantity_compliance",
            "label": "Quantity Compliance",
            "what": "Did the model generate the correct number of premises and conclusions as specified in the rules?",
            "options": ["All counts correct", "2 or less than 2 counts off", "More than 2 counts off", "Ignores all limits"],
        },
    ],
    "Premises": [
        {
            "key": "premise_relevance_score",
            "label": "Premise Relevance Score (PRS)",
            "what": "Are the generated premises relevant to the observed image content and topic?",
            "options": ["All relevant", "Mostly relevant", "Partially relevant", "Mostly irrelevant"],
        },
        {
            "key": "premise_impact",
            "label": "Premise Impact",
            "what": "Does every premise support at least one conclusion? Are there any orphan premises that do not connect to any conclusion?",
            "options": ["All premises linked", "Most premises linked", "Some premises orphaned", "Most premises orphaned"],
        },
    ],
    "Conclusion": [
        {
            "key": "conclusion_relevance_score",
            "label": "Conclusion Relevance Score (CRS)",
            "what": "Is the generated conclusion relevant to the observed premises and image content?",
            "options": ["All relevant", "Mostly relevant", "Partially relevant", "Mostly irrelevant"],
        },
        {
            "key": "conclusion_validity",
            "label": "Conclusion Validity",
            "what": "Does the conclusion logically follow from the premises?",
            "options": ["Clearly follows", "Mostly follows", "Weakly follows", "Does not follow"],
        },
        {
            "key": "conclusion_novelty",
            "label": "Conclusion Novelty",
            "what": "Does the conclusion add new insight beyond restating the premises?",
            "options": ["Insightful", "Moderate", "Paraphrase", "Copy"],
        },
        {
            "key": "visual_grounding",
            "label": "Visual Grounding",
            "what": "Does the conclusion stay grounded in the visual evidence, or does it introduce external knowledge not supported by the image?",
            "options": ["Strictly visual", "Minor external", "Some external", "Heavy external"],
        },
        {
            "key": "premise_coverage",
            "label": "Premise Coverage",
            "what": "For each conclusion, how many premises support it? (numeric count, percentage computed)",
            "options": None,
        },
    ],
    "Human Alignment": [
        {
            "key": "human_alignment_1",
            "label": "Human Alignment (vs. Human Eval 1)",
            "what": "How closely does the model's reasoning and conclusion match how a human would reason about the same image?",
            "options": ["Strong match", "Conclusion only", "Different focus", "Unrelated"],
        },
        {
            "key": "human_alignment_2",
            "label": "Human Alignment (vs. Human Eval 2)",
            "what": "How closely does the model's reasoning and conclusion match how a human would reason about the same image?",
            "options": ["Strong match", "Conclusion only", "Different focus", "Unrelated"],
        },
    ],
}

app = Flask(__name__, static_folder=".", static_url_path="")


def parse_json_field(raw: str):
    if not raw or not raw.strip():
        return []
    raw = raw.strip()
    if raw in ("[]", ""):
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(raw.replace('""', '"'))
    except json.JSONDecodeError:
        return [raw]


def read_human_csv(path: Path) -> dict:
    """Return {image_id: {id, url, metadata, premises, conclusions, notes}}."""
    if not path.exists():
        return {}
    rows = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row.get("id", "").strip()
            if not image_id:
                continue
            rows[image_id] = {
                "id": image_id,
                "url": row.get("url", ""),
                "metadata": {
                    "animals": row.get("animals", ""),
                    "consequences": row.get("consequences", ""),
                    "climateaction": row.get("climateaction", ""),
                    "type": row.get("type", ""),
                    "setting": row.get("setting", ""),
                    "source_file": row.get("source_file", ""),
                    "hash_id": row.get("hash_id", ""),
                },
                "premises": parse_json_field(row.get("premises", "")),
                "conclusions": parse_json_field(row.get("conclusions", "")),
                "notes": row.get("notes", ""),
            }
    return rows


def read_model_outputs() -> dict:
    """Return {model_key: {image_id: {plan, premises, conclusions, raw}}}."""
    results = {}
    for key, (filename, _label) in MODEL_MAP.items():
        path = MODEL_OUTPUT_DIR / filename
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        per_image = {}
        for item in data:
            image_id = str(item.get("image_id", ""))
            parsed = item.get("parsed_output", {}) or {}
            per_image[image_id] = {
                "plan": parsed.get("plan", ""),
                "premises": parsed.get("premises", []) or [],
                "conclusions": parsed.get("conclusions", []) or [],
                "raw": item.get("raw_output", ""),
            }
        results[key] = per_image
    return results


@app.route("/")
def index():
    return send_from_directory(".", "evaluation_ui.html")


@app.route("/api/data")
def api_data():
    human1 = read_human_csv(HUMAN_CSV_FILES[0][1])
    human2 = read_human_csv(HUMAN_CSV_FILES[1][1])
    models = read_model_outputs()

    # Union of image ids across human files and model outputs, in id order.
    ids = set(human1) | set(human2)
    for per_image in models.values():
        ids |= set(per_image)
    ids = sorted(ids, key=lambda x: (len(x), x))

    images = []
    for image_id in ids:
        h1 = human1.get(image_id, {})
        h2 = human2.get(image_id, {})
        row = {
            "id": image_id,
            "has_image": (IMAGES_DIR / f"{image_id}.jpg").exists(),
            "url": h1.get("url") or h2.get("url") or "",
            "metadata": h1.get("metadata") or h2.get("metadata") or {},
            "human1": {
                "premises": h1.get("premises", []),
                "conclusions": h1.get("conclusions", []),
                "notes": h1.get("notes", ""),
            },
            "human2": {
                "premises": h2.get("premises", []),
                "conclusions": h2.get("conclusions", []),
                "notes": h2.get("notes", ""),
            },
            "models": {
                key: per_image.get(image_id, {"plan": "", "premises": [], "conclusions": [], "raw": ""})
                for key, per_image in models.items()
            },
        }
        images.append(row)

    return jsonify({"images": images, "models": {k: v[1] for k, v in MODEL_MAP.items()}, "metrics": METRICS})


@app.route("/api/images/<path:filename>")
def api_image(filename):
    return send_from_directory(IMAGES_DIR, filename)


@app.route("/api/metrics-doc")
def api_metrics_doc():
    return send_from_directory(DATASET_DIR, "metrics.html")


@app.route("/api/evaluations", methods=["GET"])
def api_get_evaluations():
    if EVAL_FILE.exists():
        try:
            return jsonify(json.loads(EVAL_FILE.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            return jsonify({})
    return jsonify({})


@app.route("/api/evaluations", methods=["POST"])
def api_save_evaluations():
    data = request.get_json(force=True)
    data["updated_at"] = __import__("datetime").datetime.now().isoformat()
    EVAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    EVAL_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return jsonify({"status": "ok"})


@app.route("/api/evaluations", methods=["DELETE"])
def api_reset_evaluations():
    if EVAL_FILE.exists():
        EVAL_FILE.unlink()
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print(f"Evaluation UI: http://127.0.0.1:5001/")
    app.run(host="127.0.0.1", port=5001, debug=True)