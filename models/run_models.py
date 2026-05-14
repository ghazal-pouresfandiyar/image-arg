# Multi-Model Climate Argument Generation via OpenRouter

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

# Prefer a local `models/access` file for the API key. Fall back to the
# environment variable if the file isn't present.
ACCESS_FILE = Path(__file__).resolve().parent / "access"

if ACCESS_FILE.exists():
    OPENROUTER_API_KEY = ACCESS_FILE.read_text().strip()
else:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# List of models to process
MODELS = [
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    #"deepseek/deepseek-v4-flash:free", not image capable
    #"google/gemma-4-26b-a4b-it:free", 429 TOKEN LIMIT REACHED!
    #"meta-llama/llama-3.3-70b-instruct:free" #not image capable
]

# Retry and rate-limit configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds between retries
REQUEST_DELAY = 2  # seconds between successful requests
PROMPT_VERSION = "v1"

ROOT_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = ROOT_DIR / "dataset" / "annotated.csv"
FACTS_PATH = ROOT_DIR / "dataset" / "facts.json"

IMAGE_DIR = ROOT_DIR / "dataset" / "images_for_annotation"

OUTPUT_DIR = ROOT_DIR / "models" / "output_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRACKING_FILE = OUTPUT_DIR / "tracking.json"
FAILED_FILE = OUTPUT_DIR / "failed.json"
LOG_FILE = OUTPUT_DIR / "processing_log.txt"

# =========================================================
# TRACKING & LOGGING
# =========================================================

def load_tracking():
    """Load processing tracking data"""
    if TRACKING_FILE.exists():
        with open(TRACKING_FILE, 'r') as f:
            return json.load(f)
    return {model: [] for model in MODELS}

def load_failed():
    """Load failed processing data"""
    if FAILED_FILE.exists():
        with open(FAILED_FILE, 'r') as f:
            return json.load(f)
    return {"failed": {}}

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

def parse_json_output(output_text):
    """
    Parse JSON from model output.
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

def save_result_immediately(model_name, image_id, result_data, output_dir):
    """Save result immediately after processing"""
    model_slug = model_name.split("/")[-1].split(":")[0]
    output_path = output_dir / f"{model_slug}.json"
    
    # Load existing results
    if output_path.exists():
        try:
            with open(output_path, "r") as f:
                existing_results = json.load(f)
        except Exception:
            existing_results = []
    else:
        existing_results = []
    
    # Append new result
    existing_results.append(result_data)
    
    # Save immediately
    with open(output_path, "w") as f:
        json.dump(existing_results, f, indent=2)

# =========================================================
# CLIENT
# =========================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# =========================================================
# LOAD DATA
# =========================================================

print("Loading CSV...")
df = pd.read_csv(CSV_PATH)

print("Loading facts...")
with open(FACTS_PATH, "r") as f:
    facts_dict = json.load(f)

print(f"Loaded {len(df)} rows")

# =========================================================
# HELPERS
# =========================================================

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def build_prompt(row, facts):

    metadata = f"""
IMAGE METADATA:
- animals: {row.get('animals', '')}
- consequences: {row.get('consequences', '')}
- climateaction: {row.get('climateaction', '')}
- type: {row.get('type', '')}
- setting: {row.get('setting', '')}
- caption: {row.get('blip2_caption', '')}
"""

    return f"""
You are an AI system specialized in climate-related argument generation.

You are given:
1. An image
2. Structured metadata
3. External climate-related facts

TASK:
Generate:
- 2 to 4 argumentative premises
- 1 to 3 logical conclusions

RULES:
- Use ONLY the provided image + metadata + facts
- Do NOT invent information
- Be concise and logical
- Focus on climate reasoning

RETURN STRICT JSON ONLY:

{{
  "premises": [
    "...",
    "..."
  ],
  "conclusions": [
    "..."
  ]
}}

{metadata}

FACTS:
{facts}
"""


# =========================================================
# GENERATION LOOP - MULTI-MODEL
# =========================================================

print("\n" + "="*70)
print("STARTING MULTI-MODEL PROCESSING")
print("="*70)

# Load tracking and failed data
tracking = load_tracking()
failed_data = load_failed()
log_message(f"Loaded tracking data: {sum(len(v) for v in tracking.values())} processed items")

# Attempt to process each model
model_index = 0
while model_index < len(MODELS):
    MODEL_NAME = MODELS[model_index]
    model_slug = MODEL_NAME.split("/")[-1].split(":")[0]
    
    log_message(f"\n{'='*70}")
    log_message(f"Processing with: {MODEL_NAME}")
    log_message('='*70)
    
    OUTPUT_PATH = OUTPUT_DIR / f"{model_slug}.json"
    
    # Load existing results for this model to check what's already processed
    if OUTPUT_PATH.exists():
        try:
            with open(OUTPUT_PATH, "r") as f:
                existing_results = json.load(f)
            existing_image_ids = {r["image_id"] for r in existing_results}
        except Exception:
            existing_results = []
            existing_image_ids = set()
    else:
        existing_results = []
        existing_image_ids = set()
    
    token_limit_hit = False
    
    for idx, row in df.iterrows():
        try:
            image_id = str(row["id"])
            
            # skip if this image is already in the output JSON
            if image_id in existing_image_ids:
                log_message(f"⏭ Skipping image {image_id} (already in output)")
                continue
            
            # skip if this image previously failed for this model
            if MODEL_NAME in failed_data.get("failed", {}) and image_id in failed_data["failed"][MODEL_NAME]:
                log_message(f"⏭ Skipping image {image_id} (previously failed: {failed_data['failed'][MODEL_NAME][image_id]})")
                continue
            
            image_path = IMAGE_DIR / f"{image_id}.jpg"
            if not image_path.exists():
                log_message(f"❌ Missing image: {image_path}", "WARN")
                continue
            
            facts = facts_dict.get(image_id, [])
            prompt = build_prompt(row, facts)
            
            log_message(f"[{idx+1}/{len(df)}] Processing image {image_id} with {model_slug}")
            
            # Retry logic
            success = False
            for attempt in range(MAX_RETRIES):
                try:
                    # Encode image and send to all models
                    base64_image = encode_image(image_path)
                    
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        temperature=0.7,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You generate structured argumentative reasoning "
                                    "from climate-related multimodal evidence."
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
                    
                    # Parse JSON immediately
                    parsed_output = parse_json_output(output_text)
                    
                    # Create result object with full metadata
                    result_data = {
                        "image_id": image_id,
                        "model": MODEL_NAME,
                        "timestamp": datetime.now().isoformat(),
                        "prompt_version": PROMPT_VERSION,
                        "metadata": {
                            "animals": row.get("animals", ""),
                            "consequences": row.get("consequences", ""),
                            "climateaction": row.get("climateaction", ""),
                            "type": row.get("type", ""),
                            "setting": row.get("setting", ""),
                            "caption": row.get("blip2_caption", ""),
                        },
                        "facts": facts,
                        "parsed_output": parsed_output,
                        "raw_output": output_text,
                    }
                    
                    # Save immediately to file
                    save_result_immediately(MODEL_NAME, image_id, result_data, OUTPUT_DIR)
                    
                    # Update existing_image_ids set for current session
                    existing_image_ids.add(image_id)
                    
                    log_message(f"  ✓ Success (attempt {attempt + 1}/{MAX_RETRIES})")
                    success = True
                    break
                    
                except Exception as e:
                    error_msg = str(e)[:100]
                    
                    if is_token_limit_error(e):
                        log_message(f"⚠️  TOKEN LIMIT REACHED for {model_slug}", "ERROR")
                        log_message(f"   Error: {error_msg}", "ERROR")
                        token_limit_hit = True
                        success = False
                        break  # Don't retry on token limit, move to next model
                    elif "not support image" in error_msg.lower() or "vision" in error_msg.lower():
                        log_message(f"⚠️  MODEL DOES NOT SUPPORT IMAGES: {model_slug}", "ERROR")
                        log_message(f"   Remove this model from MODELS list on next run", "ERROR")
                        log_message(f"   Error: {error_msg}", "ERROR")
                        # Skip remaining images for this model
                        if "failed" not in failed_data:
                            failed_data["failed"] = {}
                        if MODEL_NAME not in failed_data["failed"]:
                            failed_data["failed"][MODEL_NAME] = {}
                        failed_data["failed"][MODEL_NAME][image_id] = "Model does not support images"
                        save_failed(failed_data)
                        token_limit_hit = True  # This will skip to next model
                        success = False
                        break
                    else:
                        if attempt < MAX_RETRIES - 1:
                            log_message(f"⚠️  Attempt {attempt + 1}/{MAX_RETRIES} failed: {error_msg}", "WARN")
                            time.sleep(RETRY_DELAY)
                        else:
                            log_message(f"❌ Failed after {MAX_RETRIES} attempts: {error_msg}", "ERROR")
                            
                            # Track failure
                            if "failed" not in failed_data:
                                failed_data["failed"] = {}
                            if MODEL_NAME not in failed_data["failed"]:
                                failed_data["failed"][MODEL_NAME] = {}
                            failed_data["failed"][MODEL_NAME][image_id] = error_msg
                            save_failed(failed_data)
                            success = False
            
            # Add delay between successful requests to avoid rate limiting
            if success:
                time.sleep(REQUEST_DELAY)
        
        except Exception as e:
            log_message(f"❌ Unexpected error on row {idx}: {str(e)[:100]}", "ERROR")
            continue
        
        # Break outer loop if token limit hit
        if token_limit_hit:
            break
    
    log_message(f"✅ Model {model_slug} session completed")
    
    # Move to next model if token limit was hit
    if token_limit_hit:
        log_message(f"⚠️  Model {model_slug} hit token limits. Moving to next model...", "WARN")
    
    model_index += 1

# =========================================================
# FINAL SUMMARY
# =========================================================

log_message("\n" + "="*70)
log_message("🎉 PROCESSING COMPLETE")
log_message("="*70)

log_message(f"\n📊 RESULTS SUMMARY:")
log_message(f"Output directory: {OUTPUT_DIR}")

for MODEL_NAME in MODELS:
    model_slug = MODEL_NAME.split("/")[-1].split(":")[0]
    output_file = OUTPUT_DIR / f"{model_slug}.json"
    if output_file.exists():
        with open(output_file, 'r') as f:
            data = json.load(f)
        processed_count = len(tracking.get(MODEL_NAME, []))
        failed_count = len(failed_data.get("failed", {}).get(MODEL_NAME, {}))
        log_message(f"  ✓ {model_slug}: {len(data)} total samples, {processed_count} processed, {failed_count} failed")

log_message(f"\n📁 OUTPUT FILES:")
log_message(f"  ✓ Results: {OUTPUT_DIR}/*.json")
log_message(f"  ✓ Tracking: {TRACKING_FILE}")
log_message(f"  ✓ Failed: {FAILED_FILE}")
log_message(f"  ✓ Log: {LOG_FILE}")

log_message(f"\n💡 TO RESUME:")
log_message(f"  Run: python3 models/run_models.py")
log_message(f"  The script will automatically resume from where it left off using {TRACKING_FILE}")

log_message(f"\n📋 DATA STRUCTURE:")
log_message(f"  Each result includes:")
log_message(f"  - parsed_output: JSON-parsed model response")
log_message(f"  - raw_output: Original text response")
log_message(f"  - timestamp: When it was processed")
log_message(f"  - prompt_version: Version of prompt used")

log_message("="*70 + "\n")