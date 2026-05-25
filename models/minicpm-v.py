# before running, make sure ollama is running with the model:
# ollama pull minicpm-v
# ollama run minicpm-v
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
    Extract JSON from model response with better fallback handling.
    Handles markdown code blocks and raw JSON objects/arrays anywhere in text.
    Returns (parsed_dict, raw_json_string) or (None, raw_text) if parsing fails.
    """
    # Try 1: Extract from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response_text)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # Try 2: Find JSON object/array in response (greedy search)
        json_match = re.search(r'\{[\s\S]*\}|\[[\s\S]*\]', response_text)
        if json_match:
            json_str = json_match.group(0)
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
    Handles: strings, objects with any key, nested structures, numbered keys.
    """
    if not isinstance(items, list):
        return []
    
    normalized = []
    for item in items:
        text = None
        
        if isinstance(item, str):
            # Already a string
            text = item
        elif isinstance(item, dict):
            # Try to find text in common keys first
            for key in ['text', 'observation', 'inference', 'description']:
                if key in item and item[key]:
                    text = str(item[key])
                    break
            
            # If no common key found, try any key (handles Observation_1, Observation_2, etc.)
            if not text:
                for key in sorted(item.keys()):
                    value = item[key]
                    if value:
                        text = str(value)
                        break
        
        if text:
            normalized.append(text)
    
    return normalized


def parse_text_format(text):
    """
    Fallback parser for plain text format when model doesn't return JSON.
    Handles multiple formats:
    1. "Premise N: ... Conclusion N: ..." (interleaved)
    2. "Premises: 1. ... Conclusions: 1. ..." (grouped sections)
    3. Plain numbered lists
    
    Returns dict with 'premises' and 'conclusions' keys.
    """
    premises = []
    conclusions = []
    
    # Try to extract "Premise N: ..." and "Conclusion N: ..." patterns
    # This handles interleaved format
    premise_pattern = r'Premise\s+\d+:\s*(.+?)(?=(?:Premise|Conclusion)\s+\d+:|$)'
    conclusion_pattern = r'Conclusion\s+\d+:\s*(.+?)(?=(?:Premise|Conclusion)\s+\d+:|$)'
    
    premise_matches = re.findall(premise_pattern, text, re.IGNORECASE | re.DOTALL)
    conclusion_matches = re.findall(conclusion_pattern, text, re.IGNORECASE | re.DOTALL)
    
    if premise_matches:
        premises = [m.strip() for m in premise_matches if m.strip()]
    
    if conclusion_matches:
        conclusions = [m.strip() for m in conclusion_matches if m.strip()]
    
    # If no matches with numbered format, try section-based format
    if not premises or not conclusions:
        text_lower = text.lower()
        
        # Find Premises section
        prem_idx = text_lower.find('premise')
        if prem_idx >= 0:
            conc_idx = text_lower.find('conclusion', prem_idx)
            if conc_idx < 0:
                conc_idx = len(text)
            
            premises_text = text[prem_idx:conc_idx]
            
            # Extract numbered items: "1. Text", "2. Text"
            prem_lines = re.findall(r'^\s*\d+\.\s*(.+?)(?=^\s*\d+\.|$)', premises_text, re.MULTILINE | re.DOTALL)
            if prem_lines:
                premises = [line.strip() for line in prem_lines if line.strip()]
        
        # Find Conclusions section
        conc_idx = text_lower.find('conclusion')
        if conc_idx >= 0:
            conclusions_text = text[conc_idx:]
            conclusions_text = re.sub(r'^\s*conclusions?:\s*', '', conclusions_text, flags=re.IGNORECASE).strip()
            
            # Try numbered conclusions first
            conc_lines = re.findall(r'^\s*\d+\.\s*(.+?)(?=^\s*\d+\.|$)', conclusions_text, re.MULTILINE | re.DOTALL)
            if conc_lines:
                conclusions = [line.strip() for line in conc_lines if line.strip()]
            else:
                # Fallback: take whole section as one conclusion
                if conclusions_text.strip():
                    conclusions = [conclusions_text.strip()]
    
    return {
        "premises": premises,
        "conclusions": conclusions
    }


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
        # Fallback: try to parse plain text format
        parsed_text = parse_text_format(raw_output_text)
        entry["parsed_output"] = {
            "premises": parsed_text.get("premises", []),
            "conclusions": parsed_text.get("conclusions", [])
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