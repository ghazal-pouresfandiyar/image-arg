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

def normalize_items(items):
    """
    Normalize premises/conclusions to plain string arrays.
    Handles: strings, objects with any key, nested structures, numbered keys.
    Also handles new format with 'text' and 'based_on' fields.
    """
    if not isinstance(items, list):
        return []
    
    normalized = []
    for item in items:
        text = None
        
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            # Handle new format: {"text": "...", "based_on": [1, 2]}
            if "text" in item and item["text"]:
                text = str(item["text"])
            else:
                for key in ['observation', 'inference', 'description']:
                    if key in item and item[key]:
                        text = str(item[key])
                        break
            
            if not text:
                for key in sorted(item.keys()):
                    value = item[key]
                    if value and key != "based_on":  # Skip based_on field
                        text = str(value)
                        break
        
        if text:
            normalized.append(text)
    
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
            # Handle new format
            if "text" in item and item["text"]:
                entry["text"] = str(item["text"])
            else:
                for key in ['observation', 'inference', 'description']:
                    if key in item and item[key]:
                        entry["text"] = str(item[key])
                        break
                        break
            
            if "based_on" in item:
                entry["based_on"] = item["based_on"]
        
        if entry["text"]:
            normalized.append(entry)
    
    return normalized


def parse_text_format(text):
    """
    Parser for plain text format.
    Handles multiple formats:
    1. "Premise N: ... Conclusion N: ..." (interleaved)
    2. "Premises: 1. ... Conclusions: 1. ..." (grouped sections)
    3. Plain numbered lists
    """
    premises = []
    conclusions = []
    
    # Try to extract "Premise N: ..." and "Conclusion N: ..." patterns
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


def build_nemotron_entry(image_id, raw_output_text):
    """
    Build a structured entry matching nemotron format.
    Tries JSON parsing first (new format), falls back to text parsing (old format).
    """
    entry = {
        "image_id": image_id,
        "model": MODEL_NAME,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Try JSON parsing first (new format with plan, premises, conclusions with based_on)
    parsed_json = None
    try:
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw_output_text)
        if json_match:
            parsed_json = json.loads(json_match.group(1).strip())
        else:
            # Try direct JSON parsing
            parsed_json = json.loads(raw_output_text.strip())
    except (json.JSONDecodeError, AttributeError):
        pass
    
    if parsed_json and isinstance(parsed_json, dict) and "premises" in parsed_json:
        # New JSON format
        premises = parsed_json.get("premises", [])
        conclusions = parsed_json.get("conclusions", [])
        plan = parsed_json.get("plan", "")
        
        entry["parsed_output"] = {
            "plan": plan,
            "premises": normalize_items(premises) if isinstance(premises, list) else [],
            "conclusions": normalize_conclusions_with_structure(conclusions) if isinstance(conclusions, list) else [],
        }
    else:
        # Fall back to text parsing (old format)
        parsed_text_dict = parse_text_format(raw_output_text)
        entry["parsed_output"] = {
            "premises": normalize_items(parsed_text_dict.get("premises", [])),
            "conclusions": normalize_items(parsed_text_dict.get("conclusions", []))
        }
    
    # Store raw output for debugging
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


# Ensure the output directory exists
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

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

        # Build entry directly from the text output
        entry = build_nemotron_entry(image_id, raw_output)

        results.append(entry)

        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2)

        print("saved:", image_id)

    except Exception as e:
        print("error:", image_id, e)