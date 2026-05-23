#!/usr/bin/env python3
"""
Normalize all model output JSON files to nemotron format.
Transforms raw_output into parsed_output with normalized premises/conclusions arrays.
"""

import json
import re
from pathlib import Path

OUTPUT_DIR = Path("models/output_model")

def extract_json_from_text(text):
    """Extract JSON from text, handling markdown code blocks."""
    if not isinstance(text, str):
        return None
    
    # Try to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_str = text.strip()
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def normalize_items(items):
    """
    Normalize premises/conclusions to plain string arrays.
    Handles:
    - ["string1", "string2"] → ["string1", "string2"]
    - [{"text": "string1"}, {"observation": "string1"}] → ["string1", "string2"]
    """
    if not isinstance(items, list):
        return []
    
    normalized = []
    for item in items:
        if isinstance(item, str):
            normalized.append(item)
        elif isinstance(item, dict):
            # Try common keys
            for key in ['text', 'observation', 'inference', 'description', 'type', 'premise', 'conclusion']:
                if key in item and item[key]:
                    value = item[key]
                    if isinstance(value, str):
                        normalized.append(value)
                    break
    
    return normalized


def transform_entry(entry):
    """Transform a single entry to nemotron format."""
    # If already has parsed_output with proper structure, normalize it
    if 'parsed_output' in entry and isinstance(entry['parsed_output'], dict):
        parsed = entry['parsed_output']
        if 'premises' in parsed and 'conclusions' in parsed:
            # Normalize if needed
            entry['parsed_output']['premises'] = normalize_items(parsed.get('premises', []))
            entry['parsed_output']['conclusions'] = normalize_items(parsed.get('conclusions', []))
            return entry
    
    # If only has raw_output, parse it
    if 'raw_output' in entry:
        raw = entry['raw_output']
        parsed_json = extract_json_from_text(raw)
        
        if parsed_json and isinstance(parsed_json, dict):
            entry['parsed_output'] = {
                'premises': normalize_items(parsed_json.get('premises', [])),
                'conclusions': normalize_items(parsed_json.get('conclusions', []))
            }
        else:
            # Failed to parse, create empty structure
            entry['parsed_output'] = {
                'premises': [],
                'conclusions': []
            }
        return entry
    
    # Neither parsed_output nor raw_output
    entry['parsed_output'] = {
        'premises': [],
        'conclusions': []
    }
    return entry


def transform_file(file_path):
    """Transform a single JSON file."""
    print(f"\nProcessing {file_path.name}...")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"  ⚠️  Not a list, skipping")
            return
        
        # Transform each entry
        transformed_count = 0
        for entry in data:
            entry = transform_entry(entry)
            transformed_count += 1
        
        # Save back
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  ✅ Transformed {transformed_count} entries")
    
    except Exception as e:
        print(f"  ❌ Error: {e}")


def main():
    print("=" * 60)
    print("Normalizing all model output JSON files to nemotron format")
    print("=" * 60)
    
    if not OUTPUT_DIR.exists():
        print(f"❌ Directory not found: {OUTPUT_DIR}")
        return
    
    json_files = list(OUTPUT_DIR.glob("*.json"))
    
    # Skip certain files
    skip_files = {'tracking.json', 'failed.json'}
    json_files = [f for f in json_files if f.name not in skip_files]
    
    if not json_files:
        print("❌ No JSON files found")
        return
    
    print(f"\nFound {len(json_files)} JSON files to process\n")
    
    for file_path in sorted(json_files):
        transform_file(file_path)
    
    print("\n" + "=" * 60)
    print("✅ All files transformed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
