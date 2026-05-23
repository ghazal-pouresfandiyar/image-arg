import os
import json
from pathlib import Path
from PIL import Image
import base64
import requests
from datetime import datetime
import re

# =========================
# CONFIG
# =========================

IMAGE_DIR = Path("dataset/images_for_annotation")
OUTPUT_FILE = Path("models/output_model/minicpm-v_outputs.json")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "minicpm-v"

# =========================
# HELPER FUNCTIONS
# =========================

def extract_json_from_response(response_text):
    """
    Extract JSON from model response.
    Handles markdown code blocks (```json ... ```) and raw JSON.
    Returns (parsed_dict, raw_json_string) or (None, raw_text) if parsing fails.
    """
    # Try to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response_text)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_str = response_text.strip()
    
    try:
        parsed = json.loads(json_str)
        return parsed, json_str
    except json.JSONDecodeError as e:
        print(f"  Warning: Failed to parse JSON: {e}")
        return None, response_text


def normalize_items(items):
    """
    Normalize premises/conclusions to plain string arrays.
    Handles both formats:
    - ["string1", "string2"] → ["string1", "string2"]
    - [{"text": "string1"}, {"observation": "string1"}] → ["string1", "string2"]
    """
    if not isinstance(items, list):
        return []
    
    normalized = []
    for item in items:
        if isinstance(item, str):
            # Already a string
            normalized.append(item)
        elif isinstance(item, dict):
            # Extract text from common keys: text, observation, inference, description
            for key in ['text', 'observation', 'inference', 'description']:
                if key in item and item[key]:
                    normalized.append(item[key])
                    break
    
    return normalized


def build_nemotron_entry(image_id, raw_output_text, parsed_output=None):
    """
    Build a structured entry matching nemotron format.
    
    Args:
        image_id: Image identifier
        raw_output_text: Raw model response (string)
        parsed_output: Parsed JSON dict with 'premises' and 'conclusions' keys
    
    Returns:
        Dict with image_id, model, timestamp, parsed_output, raw_output
    """
    entry = {
        "image_id": image_id,
        "model": MODEL_NAME,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Ensure parsed_output has premises and conclusions
    if parsed_output and isinstance(parsed_output, dict):
        # Normalize to plain string arrays
        premises = normalize_items(parsed_output.get("premises", []))
        conclusions = normalize_items(parsed_output.get("conclusions", []))
        
        entry["parsed_output"] = {
            "premises": premises,
            "conclusions": conclusions
        }
    else:
        entry["parsed_output"] = {
            "premises": [],
            "conclusions": []
        }
    
    # Store raw output as string
    entry["raw_output"] = raw_output_text
    
    return entry


def load_prompts():
    """Load prompts from centralized JSON file"""
    prompt_file = Path(__file__).parent / "prompts.json"
    if prompt_file.exists():
        with open(prompt_file, 'r') as f:
            return json.load(f)
    return {}

# Load prompts at startup
PROMPTS = load_prompts()

def build_image_only_prompt():
    """Get prompt for this model from centralized prompts.json"""
    return PROMPTS.get(MODEL_NAME, "")


images = sorted(list(IMAGE_DIR.glob("*.jpg")))

print("Found:", len(images))

if OUTPUT_FILE.exists():
    results = json.load(open(OUTPUT_FILE))
else:
    results = []

done = {r["image_id"] for r in results}

for i, img_path in enumerate(images):

    image_id = img_path.stem

    if image_id in done:
        print("skip", image_id)
        continue

    print(f"[{i}] Processing {image_id}")

    # encode image
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "model": MODEL_NAME,
        "prompt": build_image_only_prompt(),
        "images": [img_b64],
        "stream": False
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload)
        raw_output = r.json()["response"]

        # Parse JSON from response
        parsed_json, _ = extract_json_from_response(raw_output)

        # Build entry in nemotron format
        entry = build_nemotron_entry(image_id, raw_output, parsed_json)

        results.append(entry)

        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2)

        print("saved:", image_id)

    except Exception as e:
        print("error:", image_id, e)