# Image-Only Climate Argument Generation via OpenRouter
# No metadata, no facts - just the image!

import os
import json
import base64
import pandas as pd
from pathlib import Path
from openai import OpenAI
import re
import time
from datetime import datetime

# =========================================================
# CONFIG
# =========================================================

def load_api_keys():
    """Load API keys from access file (supports multi-key format)"""
    ACCESS_FILE = Path(__file__).resolve().parent / "access"
    
    if not ACCESS_FILE.exists():
        raise ValueError(
            "❌ 'models/access' file not found. "
            "Please create it with format: open_router : YOUR_KEY\nhugging_face : YOUR_KEY"
        )
    
    content = ACCESS_FILE.read_text().strip()
    keys = {}
    
    for line in content.split('\n'):
        if ':' in line:
            key_name, key_value = line.split(':', 1)
            keys[key_name.strip()] = key_value.strip()
    
    return keys

# Load API keys
api_keys = load_api_keys()
OPENROUTER_API_KEY = api_keys.get("open_router")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "❌ 'open_router' key not found in 'models/access' file. "
        "Please add: open_router : YOUR_KEY"
    )

# Single model for image-only processing
MODEL_NAME = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
#MODEL_NAME = "qwen/qwen2.5-vl-72b-instruct:free"
#MODEL_NAME = "qwen/qwen2.5-vl-3b-instruct:free"
model_slug = MODEL_NAME.split("/")[-1].split(":")[0]

# Retry and rate-limit configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds between retries
REQUEST_DELAY = 2  # seconds between successful requests

ROOT_DIR = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT_DIR / "dataset" / "images_for_annotation"

OUTPUT_DIR = ROOT_DIR / "models" / "output_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Separate tracking files for image-only processing
OUTPUT_FILE = OUTPUT_DIR / f"{model_slug}-image-only.json"
TRACKING_FILE = OUTPUT_DIR / "tracking_image_only.json"
FAILED_FILE = OUTPUT_DIR / "failed_image_only.json"
LOG_FILE = OUTPUT_DIR / "processing_log_image_only.txt"

# =========================================================
# TRACKING & LOGGING
# =========================================================

def load_tracking():
    """Load processing tracking data"""
    if TRACKING_FILE.exists():
        with open(TRACKING_FILE, 'r') as f:
            return json.load(f)
    return {"last_processed": None, "total_processed": 0, "token_limit_hit": False}

def load_failed():
    """Load failed processing data"""
    if FAILED_FILE.exists():
        with open(FAILED_FILE, 'r') as f:
            return json.load(f)
    return {"failed": []}

def save_tracking(tracking):
    """Save processing tracking data"""
    with open(TRACKING_FILE, 'w') as f:
        json.dump(tracking, f, indent=2)

def save_failed(failed_data):
    """Save failed processing data"""
    with open(FAILED_FILE, 'w') as f:
        json.dump(failed_data, f, indent=2)

def log_message(msg, level="INFO"):
    """Log messages to both console and file"""
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{level}] {msg}"
    print(log_entry)
    with open(LOG_FILE, 'a') as f:
        f.write(log_entry + "\n")

def is_token_limit_error(error):
    """Check if error is related to token limits"""
    error_str = str(error).lower()
    return any(keyword in error_str for keyword in [
        "rate limit",
        "quota",
        "token",
        "limit",
        "429",
        "503",
    ])

# =========================================================
# HELPER FUNCTIONS FOR PROCESSING
# =========================================================

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


def parse_json_output(output_text):
    """
    Parse JSON from model output (fallback for old format).
    Handles cases where JSON is wrapped in markdown code blocks.
    """
    try:
        # Try direct parsing first
        return json.loads(output_text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code blocks
    patterns = [
        r'```(?:json)?\s*(\{.*?\})\s*```',  # markdown code blocks
        r'({.*})',  # raw JSON object
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
    
    # If parsing fails, return None
    return None


def normalize_items(items):
    """Normalize premises/conclusions to plain string arrays."""
    if not isinstance(items, list):
        return []
    
    normalized = []
    for item in items:
        if isinstance(item, str):
            normalized.append(item)
        elif isinstance(item, dict):
            if "text" in item and item["text"]:
                normalized.append(item["text"])
            else:
                for key in ['observation', 'inference', 'description']:
                    if key in item and item[key]:
                        normalized.append(item[key])
                        break
    
    return normalized


def normalize_conclusions_with_structure(items):
    """Normalize conclusions preserving structure (text + based_on)."""
    if not isinstance(items, list):
        return []
    
    normalized = []
    for item in items:
        entry = {"text": "", "based_on": []}
        
        if isinstance(item, str):
            entry["text"] = item
        elif isinstance(item, dict):
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


def normalize_conclusions_with_structure(items):
    """Normalize conclusions preserving structure (text + based_on)."""
    if not isinstance(items, list):
        return []
    
    normalized = []
    for item in items:
        entry = {"text": "", "based_on": []}
        
        if isinstance(item, str):
            entry["text"] = item
        elif isinstance(item, dict):
            if "text" in item and item["text"]:
                entry["text"] = str(item["text"])
            else:
                for key in ['observation', 'inference', 'description']:
                    if key in item and item[key]:
                        entry["text"] = str(item[key"])
                        break
            
            if "based_on" in item:
                entry["based_on"] = item["based_on"]
        
        if entry["text"]:
            normalized.append(entry)
    
    return normalized

def save_result_immediately(image_id, result_data):
    """Save result immediately after processing"""
    # Load existing results
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, "r") as f:
                existing_results = json.load(f)
        except Exception:
            existing_results = []
    else:
        existing_results = []
    
    # Append new result
    existing_results.append(result_data)
    
    # Save immediately
    with open(OUTPUT_FILE, "w") as f:
        json.dump(existing_results, f, indent=2)

def get_image_list():
    """
    Get list of images to process FROM ANNOTATED.CSV ONLY.
    Filters out already-processed images.
    """
    # Load annotated.csv to get the list of image IDs
    annotated_csv = ROOT_DIR / "dataset" / "annotated.csv"
    
    if not annotated_csv.exists():
        log_message(f"❌ annotated.csv not found at {annotated_csv}", "ERROR")
        return []
    
    try:
        df = pd.read_csv(annotated_csv)
        annotated_ids = sorted([str(img_id) for img_id in df['id'].tolist()])
    except Exception as e:
        log_message(f"❌ Error reading annotated.csv: {str(e)}", "ERROR")
        return []
    
    if not annotated_ids:
        log_message(f"❌ No image IDs found in annotated.csv", "WARN")
        return []
    
    log_message(f"Found {len(annotated_ids)} annotated images in annotated.csv")
    
    # Check which annotated images exist in the directory
    available_images = []
    for img_id in annotated_ids:
        img_path = IMAGE_DIR / f"{img_id}.jpg"
        if img_path.exists():
            available_images.append(img_id)
        else:
            log_message(f"⚠️  Image {img_id} not found in {IMAGE_DIR}", "WARN")
    
    log_message(f"Available in directory: {len(available_images)} images")
    
    # Load already-processed images
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, "r") as f:
                existing_results = json.load(f)
            processed_ids = {r["image_id"] for r in existing_results}
        except Exception:
            processed_ids = set()
    else:
        processed_ids = set()
    
    # Filter to only unprocessed from annotated list
    to_process = [img_id for img_id in available_images if img_id not in processed_ids]
    
    log_message(f"Already processed: {len(processed_ids)}")
    log_message(f"Remaining to process: {len(to_process)}")
    
    return to_process

# =========================================================
# CLIENT
# =========================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# =========================================================
# HELPERS
# =========================================================

def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def build_image_only_prompt():
    """Read prompt from prompts.txt file (same prompt used by all models)"""
    prompt_file = Path(__file__).parent / "prompts.txt"
    if prompt_file.exists():
        return prompt_file.read_text().strip()
    else:
        # Fallback prompt if file not found
        return """You are an AI system for climate-related visual reasoning.

You will be given an image.

TASK:
Analyze the image and generate a structured climate argument.

STEP 1: PLAN
Before generating premises and conclusions, reason about:
- What is the main climate message visible in this image?
- What is the best argument approach based ONLY on visual evidence?

STEP 2: GENERATE
Based on your plan, generate premises and conclusions following these rules:
- Each conclusion MUST be logically supported by at least one premise
- Conclusions MUST go beyond simple visual observation
- Arguments MUST be highly specific to what is visible in THIS image

OUTPUT FORMAT:
Plan:
[Your reasoning about the image and argument approach]

Premises:
1. [first premise - directly visible observation]
2. [second premise - directly visible observation]

Conclusions:
1. [conclusion] (based on premise 1)
2. [conclusion] (based on premise 1, 2)"""

# =========================================================
# GENERATION LOOP - IMAGE ONLY
# =========================================================

print("\n" + "="*70)
print("STARTING IMAGE-ONLY PROCESSING")
print("="*70)

# Load tracking and failed data
tracking = load_tracking()
failed_data = load_failed()

log_message(f"Starting image-only processing with {MODEL_NAME}")
log_message(f"Previously processed: {tracking['total_processed']}")

# Get list of images to process
images_to_process = get_image_list()

if not images_to_process:
    log_message("✅ All images already processed!", "INFO")
    log_message("="*70 + "\n")
    exit(0)

log_message(f"\n{'='*70}")
log_message(f"Processing {len(images_to_process)} images with IMAGE-ONLY prompt")
log_message('='*70)

token_limit_hit = False
successful_count = 0
failed_count = 0

for idx, image_id in enumerate(images_to_process):
    try:
        # Skip if previously failed
        if image_id in [f["image_id"] for f in failed_data.get("failed", [])]:
            log_message(f"⏭ Skipping image {image_id} (previously failed)")
            continue
        
        image_path = IMAGE_DIR / f"{image_id}.jpg"
        if not image_path.exists():
            log_message(f"❌ Missing image: {image_path}", "WARN")
            failed_data["failed"].append({
                "image_id": image_id,
                "error": "Image file not found"
            })
            save_failed(failed_data)
            failed_count += 1
            continue
        
        prompt = build_image_only_prompt()
        
        log_message(f"[{idx+1}/{len(images_to_process)}] Processing image {image_id}")
        
        # Retry logic
        success = False
        for attempt in range(MAX_RETRIES):
            try:
                # Encode image
                base64_image = encode_image(image_path)
                
                # Call API with image-only prompt
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    temperature=0.7,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You generate structured argumentative reasoning "
                                "from climate and environmental images."
                            ),
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                },
                            ],
                        },
                    ],
                )
                
                output_text = response.choices[0].message.content
                
                # Parse text format first (new format with Plan, Premises, Conclusions)
                parsed = parse_text_format(output_text)
                
                # Check if we got valid data from text parsing
                if parsed.get("premises") or parsed.get("conclusions"):
                    # Successfully parsed text format
                    premises = normalize_items(parsed.get("premises", []))
                    conclusions_raw = parsed.get("conclusions", [])
                    conclusions = normalize_conclusions_with_structure(conclusions_raw)
                    
                    parsed_output = {
                        "plan": parsed.get("plan", ""),
                        "premises": premises,
                        "conclusions": conclusions
                    }
                else:
                    # Fall back to JSON parsing (old format)
                    parsed_json = parse_json_output(output_text)
                    if parsed_json and isinstance(parsed_json, dict):
                        premises = normalize_items(parsed_json.get("premises", []))
                        conclusions_raw = parsed_json.get("conclusions", [])
                        conclusions = normalize_conclusions_with_structure(conclusions_raw)
                        
                        parsed_output = {
                            "plan": parsed_json.get("plan", ""),
                            "premises": premises,
                            "conclusions": conclusions
                        }
                    else:
                        parsed_output = {"plan": "", "premises": [], "conclusions": []}
                
                # Create result object
                result_data = {
                    "image_id": image_id,
                    "model": MODEL_NAME,
                    "timestamp": datetime.now().isoformat(),
                    "parsed_output": parsed_output,
                    "raw_output": output_text,
                }
                
                # Save immediately to file
                save_result_immediately(image_id, result_data)
                
                log_message(f"  ✓ Success (attempt {attempt + 1}/{MAX_RETRIES})")
                successful_count += 1
                success = True
                break
                
            except Exception as e:
                error_msg = str(e)[:100]
                
                if is_token_limit_error(e):
                    log_message(f"⚠️  TOKEN LIMIT REACHED", "ERROR")
                    log_message(f"   Error: {error_msg}", "ERROR")
                    log_message(f"   Will resume from image {image_id} on next run", "WARN")
                    
                    # Update tracking before exiting
                    tracking["last_processed"] = image_id
                    tracking["total_processed"] += successful_count
                    tracking["token_limit_hit"] = True
                    save_tracking(tracking)
                    
                    token_limit_hit = True
                    success = False
                    break  # Don't retry on token limit, move to next model
                elif "not support image" in error_msg.lower() or "vision" in error_msg.lower():
                    log_message(f"⚠️  MODEL DOES NOT SUPPORT IMAGES", "ERROR")
                    log_message(f"   Error: {error_msg}", "ERROR")
                    failed_data["failed"].append({
                        "image_id": image_id,
                        "error": "Model does not support images"
                    })
                    save_failed(failed_data)
                    success = False
                    break
                else:
                    if attempt < MAX_RETRIES - 1:
                        log_message(f"⚠️  Attempt {attempt + 1}/{MAX_RETRIES} failed: {error_msg}", "WARN")
                        time.sleep(RETRY_DELAY)
                    else:
                        log_message(f"❌ Failed after {MAX_RETRIES} attempts: {error_msg}", "ERROR")
                        
                        # Track failure
                        failed_data["failed"].append({
                            "image_id": image_id,
                            "error": error_msg
                        })
                        save_failed(failed_data)
                        failed_count += 1
                        success = False
        
        # Add delay between successful requests to avoid rate limiting
        if success:
            time.sleep(REQUEST_DELAY)
    
    except Exception as e:
        log_message(f"❌ Unexpected error on image {image_id}: {str(e)[:100]}", "ERROR")
        failed_count += 1
        continue
    
    # Break outer loop if token limit hit
    if token_limit_hit:
        break

# Update tracking
tracking["total_processed"] += successful_count
if not token_limit_hit:
    tracking["token_limit_hit"] = False
save_tracking(tracking)

# =========================================================
# FINAL SUMMARY
# =========================================================

log_message("\n" + "="*70)
log_message("🎉 PROCESSING COMPLETE")
log_message("="*70)

log_message(f"\n📊 RESULTS SUMMARY:")
log_message(f"Output file: {OUTPUT_FILE}")

if OUTPUT_FILE.exists():
    with open(OUTPUT_FILE, 'r') as f:
        data = json.load(f)
    log_message(f"  ✓ Total results stored: {len(data)}")

log_message(f"  ✓ Successfully processed in this session: {successful_count}")
log_message(f"  ✓ Failed in this session: {failed_count}")

log_message(f"\n📁 OUTPUT FILES:")
log_message(f"  ✓ Results: {OUTPUT_FILE}")
log_message(f"  ✓ Tracking: {TRACKING_FILE}")
log_message(f"  ✓ Failed: {FAILED_FILE}")
log_message(f"  ✓ Log: {LOG_FILE}")

if token_limit_hit:
    log_message(f"\n⚠️  TOKEN LIMIT REACHED")
    log_message(f"💡 TO RESUME:")
    log_message(f"  Run: python3 models/run_model_image.py")
    log_message(f"  The script will automatically resume from where it left off")
    log_message(f"  Last processed: {tracking.get('last_processed', 'N/A')}")
else:
    log_message(f"\n✅ All images processed successfully!")

log_message(f"\n📋 OUTPUT DATA STRUCTURE:")
log_message(f"  Each result includes:")
log_message(f"  - image_id: Identifier of the image")
log_message(f"  - model: Model used for processing")
log_message(f"  - timestamp: When it was processed")
log_message(f"  - parsed_output: JSON-parsed model response with premises and conclusions")
log_message(f"  - raw_output: Original text response from model")
log_message(f"  - NO metadata fields (image-only analysis)")

log_message("="*70 + "\n")
