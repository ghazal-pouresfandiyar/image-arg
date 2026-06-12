# ollama run llava
# python3 models/run_models.py --model llava
import argparse
import json
import re
import time
import base64
from pathlib import Path

# =========================
# PATHS
# =========================

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(__file__).resolve().parent
IMAGE_DIR = ROOT_DIR / "dataset" / "images_for_annotation"
OUTPUT_DIR = MODELS_DIR / "output_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# PROMPTS
# =========================

PROMPT_FILE = MODELS_DIR / "prompts.txt"

# =========================
# PARSING HELPERS
# =========================

def clean_text(text):
    """Strip whitespace and normalize internal spacing."""
    return re.sub(r'\s+', ' ', text).strip()


def parse_text_format(text):
    """
    Robust parser for Plan/Premises/Conclusions text format.
    Returns {"plan": str, "premises": [str], "conclusions": [str]}
    """
    result = {"plan": "", "premises": [], "conclusions": []}

    # Extract Plan
    plan_match = re.search(r'Plan:\s*(.*?)(?=Premises:|$)', text, re.DOTALL | re.IGNORECASE)
    if plan_match:
        result["plan"] = plan_match.group(1).strip()

    # Extract Premises
    premises_match = re.search(r'Premises:\s*(.*?)(?=Conclusions:|$)', text, re.DOTALL | re.IGNORECASE)
    if premises_match:
        for line in premises_match.group(1).strip().split('\n'):
            clean_line = re.sub(r'^\d+\.\s*', '', line.strip())
            if clean_line and re.match(r'^\d+\.', line.strip()):
                result["premises"].append(clean_line)

    # Extract Conclusions
    conclusions_match = re.search(r'Conclusions:\s*(.*)', text, re.DOTALL | re.IGNORECASE)
    if conclusions_match:
        for line in conclusions_match.group(1).strip().split('\n'):
            line = line.strip()
            if not line or not re.match(r'^\d+\.', line):
                continue
            clean_line = re.sub(r'^\d+\.\s*', '', line)
            if clean_line:
                result["conclusions"].append(clean_line)

    return result


def parse_json_output(text):
    """Try to extract JSON from model response (handles markdown code blocks)."""
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def extract_premises_conclusions(raw_output):
    """
    Parse raw model output into plan, premises, and conclusions.
    Tries text parsing first, falls back to JSON.
    """
    # Try text parsing first (new format)
    result = parse_text_format(raw_output)
    if result["premises"] or result["conclusions"]:
        return result

    # Fall back to JSON parsing (old format)
    parsed_json = parse_json_output(raw_output)
    if parsed_json and isinstance(parsed_json, dict):
        premises = []
        for item in parsed_json.get("premises", []):
            if isinstance(item, str):
                premises.append(clean_text(item))
            elif isinstance(item, dict):
                for key in ['text', 'observation', 'inference', 'description']:
                    if key in item and item[key]:
                        premises.append(clean_text(str(item[key])))
                        break

        conclusions = []
        for item in parsed_json.get("conclusions", []):
            if isinstance(item, str):
                conclusions.append(clean_text(item))
            elif isinstance(item, dict):
                for key in ['text', 'observation', 'inference', 'description']:
                    if key in item and item[key]:
                        conclusions.append(clean_text(str(item[key])))
                        break

        return {
            "plan": parsed_json.get("plan", ""),
            "premises": premises,
            "conclusions": conclusions
        }

    return result


def build_entry(image_id, raw_output):
    parsed = extract_premises_conclusions(raw_output)

    return {
        "image_id": image_id,
        "parsed_output": {
            "plan": parsed.get("plan", ""),
            "premises": parsed.get("premises", []),
            "conclusions": parsed.get("conclusions", []),
        },
        "raw_output": raw_output,
    }

# =========================
# OLLAMA RUNNER
# =========================

def run_ollama(model_name, prompt, images):
    import requests

    OLLAMA_URL = "http://localhost:11434/api/generate"
    output_file = OUTPUT_DIR / f"{model_name}_outputs.json"

    existing = []
    if output_file.exists():
        with open(output_file, "r") as f:
            existing = json.load(f)
    done = {r["image_id"] for r in existing}


    for i, img_path in enumerate(images):
        image_id = img_path.stem
        if image_id in done:
            print("skip", image_id)
            continue

        print(f"[{i}] Processing {image_id}")

        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        payload = {
            "model": model_name,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
        }

        try:
            r = requests.post(OLLAMA_URL, json=payload)
            raw_output = r.json()["response"]
            entry = build_entry(image_id, raw_output)
            existing.append(entry)

            with open(output_file, "w") as f:
                json.dump(existing, f, indent=2)

            print("saved:", image_id)
        except Exception as e:
            print("error:", image_id, e)

# =========================
# OPENROUTER RUNNER
# =========================

def run_openrouter(model_name, model_id, prompt, images):
    from openai import OpenAI

    ACCESS_FILE = MODELS_DIR / "access"
    if not ACCESS_FILE.exists():
        raise FileNotFoundError("models/access file not found. Add: open_router : YOUR_KEY")

    content = ACCESS_FILE.read_text().strip()
    api_key = None
    for line in content.split("\n"):
        if "open_router" in line and ":" in line:
            api_key = line.split(":", 1)[1].strip()
            break

    if not api_key:
        raise ValueError("open_router key not found in models/access")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    model_slug = model_id.split("/")[-1].split(":")[0]
    output_file = OUTPUT_DIR / f"{model_slug}_outputs.json"
    MAX_RETRIES = 3
    RETRY_DELAY = 2
    REQUEST_DELAY = 2

    existing = []
    if output_file.exists():
        with open(output_file, "r") as f:
            existing = json.load(f)
    done = {r["image_id"] for r in existing}

    for i, img_path in enumerate(images):
        image_id = img_path.stem
        if image_id in done:
            print("skip", image_id)
            continue

        print(f"[{i}] Processing {image_id}")

        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        success = False
        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    temperature=0.7,
                    messages=[
                        {
                            "role": "system",
                            "content": "You generate structured argumentative reasoning from climate and environmental images. Return ONLY valid JSON.",
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                                },
                            ],
                        },
                    ],
                )

                raw_output = response.choices[0].message.content
                entry = build_entry(image_id, raw_output)
                existing.append(entry)

                with open(output_file, "w") as f:
                    json.dump(existing, f, indent=2)

                print("saved:", image_id)
                success = True
                break

            except Exception as e:
                error_msg = str(e)[:100]
                if attempt < MAX_RETRIES - 1:
                    print(f"  retry {attempt+1}/{MAX_RETRIES}: {error_msg}")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"  failed: {error_msg}")

        if success:
            time.sleep(REQUEST_DELAY)

# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser(description="Run climate image reasoning models")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["llava", "minicpm-v", "nemotron"],
        help="Model to run: llava, minicpm-v, nemotron",
    )
    args = parser.parse_args()

    model_name = args.model

    prompt = ""
    if PROMPT_FILE.exists():
        prompt = PROMPT_FILE.read_text().strip()

    if not prompt:
        print(f"No prompt found in {PROMPT_FILE}")
        return

    images = sorted(list(IMAGE_DIR.glob("*.jpg")))
    if not images:
        print("No images found in", IMAGE_DIR)
        return

    print(f"Model: {model_name}")
    print(f"Images found: {len(images)}")

    if model_name == "nemotron":
        model_id = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
        run_openrouter(model_name, model_id, prompt, images)
    else:
        run_ollama(model_name, prompt, images)


if __name__ == "__main__":
    main()
