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
OUTPUT_FILE = Path("models/output_model/llava_outputs.json")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llava"

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


def parse_text_format(text):
    """
    Parser for plain text format with Plan, Premises, and Conclusions.
    Handles:
    1. Plan section
    2. Premises section with numbered items
    3. Conclusions section with numbered items and (based on premise N) linking
    """
    plan = ""
    premises = []
    conclusions = []
    
    text_lower = text.lower()
    
    # Extract Plan section
    plan_idx = text_lower.find('plan:')
    if plan_idx >= 0:
        plan_start = plan_idx + len('plan:')
        # Find next section (premises or conclusions)
        prem_idx = text_lower.find('premise', plan_start)
        conc_idx = text_lower.find('conclusion', plan_start)
        
        if prem_idx >= 0:
            plan = text[plan_start:prem_idx].strip()
        elif conc_idx >= 0:
            plan = text[plan_start:conc_idx].strip()
        else:
            plan = text[plan_start:].strip()
    
    # Extract Premises section
    prem_idx = text_lower.find('premise')
    if prem_idx >= 0:
        # Find the end of premises section (start of conclusions or end of text)
        conc_idx = text_lower.find('conclusion', prem_idx)
        if conc_idx < 0:
            conc_idx = len(text)
        
        premises_text = text[prem_idx:conc_idx]
        
        # Extract numbered items: "1. Text", "2. Text"
        prem_lines = re.findall(r'^\s*\d+\.\s*(.+?)(?=^\s*\d+\.|$)', premises_text, re.MULTILINE | re.DOTALL)
        if prem_lines:
            premises = [line.strip() for line in prem_lines if line.strip()]
        else:
            premises = []
    
    # Extract Conclusions section
    conc_idx = text_lower.find('conclusion')
    if conc_idx >= 0:
        conclusions_text = text[conc_idx:]
        conclusions_text = re.sub(r'^\s*conclusions?:\s*', '', conclusions_text, flags=re.IGNORECASE).strip()
        
        # Try numbered conclusions with "based on" pattern
        # Pattern: "1. conclusion text (based on premise 1, 2)"
        concl_pattern = r'^\s*\d+\.\s*(.+?)(?=^\s*\d+\.|$)'
        concl_matches = re.findall(concl_pattern, conclusions_text, re.MULTILINE | re.DOTALL)
        
        conclusions = []
        for match in concl_matches:
            match = match.strip()
            if not match:
                continue
            
            # Extract "based on premise N, M" part
            based_on = []
            based_on_match = re.search(r'\(?\s*based\s+on\s+premise\s+([\d,\s]+)\s*\)?', match, re.IGNORECASE)
            if based_on_match:
                nums_str = based_on_match.group(1)
                based_on = [int(n.strip()) for n in nums_str.split(',') if n.strip().isdigit()]
            
            # Extract conclusion text (without the based_on part)
            concl_text = re.sub(r'\(?\s*based\s+on\s+premise\s+[\d,\s]+\s*\)?', '', match, flags=re.IGNORECASE).strip()
            
            if concl_text:
                conclusions.append({"text": concl_text, "based_on": based_on})
    
    return {
        "plan": plan,
        "premises": premises,
        "conclusions": conclusions
    }


def normalize_items(items):
    """
    Normalize premises/conclusions to plain string arrays.
    Handles both formats:
    - ["string1", "string2"] → ["string1", "string2"]
    - [{"text": "string1"}, {"observation": "string1"}] → ["string1", "string2"]
    - [{"text": "...", "based_on": [1, 2]}] → ["..."] (new format)
    """
    if not isinstance(items, list):
        return []
    
    normalized = []
    for item in items:
        if isinstance(item, str):
            # Already a string
            normalized.append(item)
        elif isinstance(item, dict):
            # Handle new format: {"text": "...", "based_on": [1, 2]}
            if "text" in item and item["text"]:
                normalized.append(item["text"])
            else:
                # Extract text from common keys: observation, inference, description
                for key in ['observation', 'inference', 'description']:
                    if key in item and item[key]:
                        normalized.append(item[key])
                        break
    
    return normalized


def normalize_conclusions_with_structure(items):
    """
    Normalize conclusions preserving structure (text + based_on).
    Returns list of dicts with 'text' and 'based_on' keys.
    """
    if not isinstance(items, list):
        return []
    
    normalized = []
    for item in items:
        entry = {"text": "", "based_on": []}
        
        if isinstance(item, str):
            entry["text"] = item
        elif isinstance(item, dict):
            # Handle new format: {"text": "...", "based_on": [1, 2]}
            if "text" in item and item["text"]:
                entry["text"] = str(item["text"])
            else:
                for key in ['observation', 'inference', 'description']:
                    if key in item and item[key]:
                        entry["text"] = str(item[key])
                        break
            
            if "based_on" in item:
                entry["based_on"] = item["based_on"]
        
        if entry["text"]:
            normalized.append(entry)
    
    return normalized


def build_nemotron_entry(image_id, raw_output_text, parsed_output=None):
    """
    Build a structured entry matching nemotron format.
    Parses text output and extracts plan, premises, and conclusions with based_on.
    
    Args:
        image_id: Image identifier
        raw_output_text: Raw model response (string)
        parsed_output: Not used (kept for backward compatibility)
    
    Returns:
        Dict with image_id, model, timestamp, parsed_output, raw_output
    """
    entry = {
        "image_id": image_id,
        "model": MODEL_NAME,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Parse text format
    parsed = parse_text_format(raw_output_text)
    
    # Normalize premises
    premises = normalize_items(parsed.get("premises", []))
    
    # Normalize conclusions - preserve based_on structure
    conclusions_raw = parsed.get("conclusions", [])
    conclusions = normalize_conclusions_with_structure(conclusions_raw)
    
    entry["parsed_output"] = {
        "plan": parsed.get("plan", ""),
        "premises": premises,
        "conclusions": conclusions
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

        # Parse text output (handles Plan, Premises, Conclusions with based_on)
        entry = build_nemotron_entry(image_id, raw_output)

        results.append(entry)

        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2)

        print("saved:", image_id)

    except Exception as e:
        print("error:", image_id, e)