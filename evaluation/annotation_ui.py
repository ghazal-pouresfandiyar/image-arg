#!/usr/bin/env python3
"""
Blind Human Evaluation UI for Climate Argument Generation
=========================================================
A local web interface for human judges to blindly evaluate generated arguments.

Features:
  - Image display at the top
  - Blind testing: models are anonymized as "Model A/B/C" with randomized order
  - 5 evaluation criteria on 1-5 scale
  - Saves results to CSV with actual model names mapped back

Usage:
  python evaluation/annotation_ui.py
  Then open http://127.0.0.1:8001 in your browser
"""

from __future__ import annotations

import csv
import html
import json
import os
import random
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import parse_qs, urlparse

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
MODELS_OUTPUT_DIR = PROJECT_DIR / "models" / "output_model"
EVALUATION_DIR = Path(__file__).parent
IMAGES_DIR = DATASET_DIR / "images_for_annotation"
ANNOTATED_CSV = DATASET_DIR / "annotated.csv"
OUTPUT_CSV = EVALUATION_DIR / "human_evaluation_results.csv"

HOST = "127.0.0.1"
PORT = 8001

MODEL_LABELS = ["Model A", "Model B", "Model C"]
CRITERIA = [
    "Premise Accuracy",
    "Conclusion Relevance",
    "Argument Strength",
    "Climate Specificity",
    "Overall Quality",
]
CRITERIA_DESCRIPTIONS = {
    "Premise Accuracy": "Are the premises visually grounded in the image?",
    "Conclusion Relevance": "Does the conclusion logically follow from the premises?",
    "Argument Strength": "How persuasive is the overall argument?",
    "Climate Specificity": "Is the argument specific to climate (not generic)?",
    "Overall Quality": "Overall quality of the generated argument",
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_model_outputs() -> Dict[str, Dict[str, Dict]]:
    """
    Load model outputs organized by image_id.
    
    Returns:
        {image_id: {model_name: {"premises": [...], "conclusions": [...]}}}
    """
    image_data = {}
    model_names = []
    
    json_files = list(MODELS_OUTPUT_DIR.glob("*.json"))
    for json_file in json_files:
        model_name = json_file.stem.replace("_outputs", "")
        model_names.append(model_name)
        
        try:
            with open(json_file, 'r') as f:
                records = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            continue
        
        if isinstance(records, dict):
            records = [records]
        
        for record in records:
            image_id = str(record.get("image_id", ""))
            if not image_id:
                continue
            
            parsed = record.get("parsed_output", {})
            premises = parsed.get("premises", [])
            conclusions = parsed.get("conclusions", [])
            
            # Normalize conclusions
            norm_conclusions = []
            for c in conclusions:
                if isinstance(c, dict):
                    norm_conclusions.append(c.get("text", str(c)))
                elif isinstance(c, str):
                    norm_conclusions.append(c)
            
            if image_id not in image_data:
                image_data[image_id] = {}
            
            image_data[image_id][model_name] = {
                "premises": premises,
                "conclusions": norm_conclusions,
            }
    
    return image_data, sorted(model_names)


def get_image_list() -> List[str]:
    """Get list of image IDs that have both image files and model outputs."""
    image_data, _ = load_model_outputs()
    
    available_images = []
    for image_id in sorted(image_data.keys()):
        img_path = IMAGES_DIR / f"{image_id}.jpg"
        if img_path.exists():
            available_images.append(image_id)
    
    return available_images


def load_completed_evaluations() -> set:
    """Load already-evaluated image IDs."""
    if not OUTPUT_CSV.exists():
        return set()
    
    completed = set()
    try:
        with open(OUTPUT_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add(row.get("image_id", ""))
    except Exception:
        pass
    
    return completed


# ============================================================================
# HTTP HANDLER
# ============================================================================

class EvaluationHandler(BaseHTTPRequestHandler):
    """Handle HTTP requests for the evaluation UI."""
    
    image_data = {}
    model_names = []
    image_list = []
    current_index = 0
    shuffled_models = {}  # {image_id: [shuffled_model_names]}
    label_to_model = {}   # {image_id: {"Model A": "actual_model", ...}}
    
    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path == "/" or path == "/evaluate":
            self.show_evaluation_page()
        elif path == "/next":
            self.next_image()
        elif path == "/prev":
            self.prev_image()
        elif path.startswith("/image/"):
            self.serve_image(path)
        elif path == "/progress":
            self.show_progress()
        else:
            self.send_error(404)
    
    def do_POST(self):
        """Handle POST requests (saving evaluations)."""
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path == "/save":
            self.save_evaluation()
        else:
            self.send_error(404)
    
    def show_evaluation_page(self):
        """Display the evaluation page for the current image."""
        if not self.image_list:
            self.send_html("<h1>No images available for evaluation</h1>")
            return
        
        # Ensure current_index is valid
        if self.current_index >= len(self.image_list):
            self.current_index = 0
        
        image_id = self.image_list[self.current_index]
        image_models = self.image_data.get(image_id, {})
        
        # Get or create shuffled model order for this image
        if image_id not in self.shuffled_models:
            available_models = [m for m in self.model_names if m in image_models]
            if len(available_models) < 2:
                # Need at least 2 models for comparison
                self.send_html(f"<h1>Image {image_id} has fewer than 2 model outputs</h1><p><a href='/next'>Skip to next</a></p>")
                return
            
            shuffled = available_models.copy()
            random.shuffle(shuffled)
            self.shuffled_models[image_id] = shuffled[:3]  # Take up to 3
            
            # Create label mapping
            self.label_to_model[image_id] = {}
            for i, model in enumerate(self.shuffled_models[image_id]):
                self.label_to_model[image_id][MODEL_LABELS[i]] = model
        
        shuffled = self.shuffled_models[image_id]
        label_map = self.label_to_model[image_id]
        
        # Build HTML
        completed = load_completed_evaluations()
        is_completed = image_id in completed
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Blind Evaluation - Image {image_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .header {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .progress {{ color: #666; font-size: 14px; }}
        .nav {{ margin-top: 10px; }}
        .nav a {{ margin-right: 15px; padding: 8px 16px; background: #4361ee; color: white; text-decoration: none; border-radius: 4px; }}
        .nav a:hover {{ background: #3a56d4; }}
        .image-container {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
        .image-container img {{ max-width: 100%; max-height: 500px; border-radius: 4px; }}
        .model-output {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .model-label {{ font-size: 20px; font-weight: bold; color: #4361ee; margin-bottom: 15px; }}
        .section {{ margin-bottom: 15px; }}
        .section-title {{ font-weight: bold; color: #333; margin-bottom: 5px; }}
        .section-content {{ color: #555; line-height: 1.6; }}
        .criteria {{ margin-top: 15px; }}
        .criterion {{ margin-bottom: 15px; padding: 10px; background: #f8f9fa; border-radius: 4px; }}
        .criterion label {{ display: block; font-weight: bold; margin-bottom: 5px; }}
        .criterion .description {{ color: #666; font-size: 13px; margin-bottom: 8px; }}
        .criterion input[type="range"] {{ width: 100%; }}
        .criterion .value {{ text-align: center; font-size: 18px; font-weight: bold; color: #4361ee; }}
        .save-btn {{ background: #28a745; color: white; padding: 12px 24px; border: none; border-radius: 4px; font-size: 16px; cursor: pointer; }}
        .save-btn:hover {{ background: #218838; }}
        .saved {{ color: #28a745; font-weight: bold; }}
        .completed {{ background: #d4edda; border: 1px solid #c3e6cb; padding: 10px; border-radius: 4px; margin-bottom: 15px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Blind Evaluation</h1>
        <div class="progress">
            Image {self.current_index + 1} of {len(self.image_list)} | 
            Image ID: {image_id}
        </div>
        <div class="nav">
            <a href="/prev">← Previous</a>
            <a href="/next">Next →</a>
            <a href="/progress">View Progress</a>
        </div>
    </div>
    
    {"<div class='completed'>✅ This image has already been evaluated</div>" if is_completed else ""}
    
    <div class="image-container">
        <img src="/image/{image_id}.jpg" alt="Image {image_id}">
    </div>
    
    <form method="POST" action="/save">
        <input type="hidden" name="image_id" value="{image_id}">
"""
        
        # Add model outputs with randomization
        for i, model_name in enumerate(shuffled):
            label = MODEL_LABELS[i]
            model_output = image_models.get(model_name, {})
            premises = model_output.get("premises", [])
            conclusions = model_output.get("conclusions", [])
            
            html_content += f"""
    <div class="model-output">
        <div class="model-label">{label}</div>
        
        <div class="section">
            <div class="section-title">Premises:</div>
            <div class="section-content">
                <ol>
"""
            for prem in premises:
                html_content += f"                    <li>{html.escape(str(prem))}</li>\n"
            
            html_content += """                </ol>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">Conclusions:</div>
            <div class="section-content">
                <ul>
"""
            for concl in conclusions:
                html_content += f"                    <li>{html.escape(str(concl))}</li>\n"
            
            html_content += """                </ul>
            </div>
        </div>
        
        <div class="criteria">
"""
            for criterion in CRITERIA:
                desc = CRITERIA_DESCRIPTIONS[criterion]
                html_content += f"""
            <div class="criterion">
                <label>{criterion}</label>
                <div class="description">{desc}</div>
                <input type="range" name="{label}_{criterion}" min="1" max="5" value="3" 
                       oninput="this.nextElementSibling.textContent = this.value">
                <div class="value">3</div>
            </div>
"""
            
            html_content += """        </div>
    </div>
"""
        
        html_content += f"""
    <button type="submit" class="save-btn">Save and Next Image</button>
</form>

<script>
// Store label-to-model mapping in hidden field
document.querySelector('form').addEventListener('submit', function(e) {{
    var mapping = {json.dumps(label_map)};
    var hidden = document.createElement('input');
    hidden.type = 'hidden';
    hidden.name = 'label_mapping';
    hidden.value = JSON.stringify(mapping);
    this.appendChild(hidden);
}});
</script>

</body>
</html>"""
        
        self.send_html(html_content)
    
    def show_progress(self):
        """Show evaluation progress."""
        completed = load_completed_evaluations()
        total = len(self.image_list)
        done = len(completed & set(self.image_list))
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Evaluation Progress</title>
    <style>
        body {{ font-family: sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }}
        .progress-bar {{ background: #e0e0e0; border-radius: 10px; height: 30px; overflow: hidden; }}
        .progress-fill {{ background: #4361ee; height: 100%; width: {done/total*100 if total else 0}%; transition: width 0.3s; }}
        .stats {{ margin: 20px 0; }}
        .nav {{ margin-top: 20px; }}
        .nav a {{ padding: 10px 20px; background: #4361ee; color: white; text-decoration: none; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>Evaluation Progress</h1>
    <div class="stats">
        <p><strong>{done}</strong> of <strong>{total}</strong> images evaluated</p>
        <p>{done/total*100 if total else 0:.1f}% complete</p>
    </div>
    <div class="progress-bar">
        <div class="progress-fill"></div>
    </div>
    <div class="nav">
        <a href="/">Continue Evaluation</a>
    </div>
</body>
</html>"""
        self.send_html(html_content)
    
    def next_image(self):
        """Move to next image."""
        self.current_index = (self.current_index + 1) % len(self.image_list)
        self.send_redirect("/")
    
    def prev_image(self):
        """Move to previous image."""
        self.current_index = (self.current_index - 1) % len(self.image_list)
        self.send_redirect("/")
    
    def serve_image(self, path: str):
        """Serve image files."""
        filename = path.split("/")[-1]
        image_path = IMAGES_DIR / filename
        
        if not image_path.exists():
            self.send_error(404)
            return
        
        with open(image_path, "rb") as f:
            content = f.read()
        
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)
    
    def save_evaluation(self):
        """Save evaluation results to CSV."""
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length).decode("utf-8")
        params = parse_qs(post_data)
        
        image_id = params.get("image_id", [""])[0]
        label_mapping_str = params.get("label_mapping", ["{}"])[0]
        
        try:
            label_mapping = json.loads(label_mapping_str)
        except json.JSONDecodeError:
            label_mapping = {}
        
        if not image_id:
            self.send_html("<h1>Error: No image ID</h1><p><a href='/'>Go back</a></p>")
            return
        
        # Collect scores
        scores = {}
        for label in MODEL_LABELS:
            scores[label] = {}
            for criterion in CRITERIA:
                key = f"{label}_{criterion}"
                value = params.get(key, ["3"])[0]
                try:
                    scores[label][criterion] = int(value)
                except ValueError:
                    scores[label][criterion] = 3
        
        # Map labels back to actual model names
        actual_scores = {}
        for label, model_name in label_mapping.items():
            actual_scores[model_name] = scores.get(label, {})
        
        # Write to CSV
        file_exists = OUTPUT_CSV.exists()
        
        with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
            fieldnames = ["image_id"]
            for model in self.model_names:
                for criterion in CRITERIA:
                    fieldnames.append(f"{model}_{criterion}")
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            
            row = {"image_id": image_id}
            for model_name, model_scores in actual_scores.items():
                for criterion, score in model_scores.items():
                    row[f"{model_name}_{criterion}"] = score
            
            writer.writerow(row)
        
        self.send_html(f"""
<!DOCTYPE html>
<html>
<head>
    <title>Saved!</title>
    <style>
        body {{ font-family: sans-serif; text-align: center; margin-top: 100px; }}
        .success {{ color: #28a745; font-size: 24px; }}
        .nav {{ margin-top: 30px; }}
        .nav a {{ padding: 10px 20px; background: #4361ee; color: white; text-decoration: none; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="success">✅ Evaluation Saved!</div>
    <p>Image {image_id} has been evaluated.</p>
    <div class="nav">
        <a href="/next">Next Image →</a>
    </div>
</body>
</html>""")
    
    def send_html(self, content: str):
        """Send HTML response."""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(content.encode("utf-8"))
    
    def send_redirect(self, path: str):
        """Send redirect response."""
        self.send_response(302)
        self.send_header("Location", path)
        self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Start the evaluation UI server."""
    print("=" * 60)
    print("Blind Human Evaluation UI")
    print("=" * 60)
    
    # Load data
    print("\nLoading model outputs...")
    image_data, model_names = load_model_outputs()
    
    if not image_data:
        print("ERROR: No model outputs found!")
        return
    
    print(f"Found {len(model_names)} models: {model_names}")
    print(f"Found {len(image_data)} images with model outputs")
    
    # Get image list
    image_list = get_image_list()
    print(f"Images with both outputs and files: {len(image_list)}")
    
    if len(image_list) == 0:
        print("ERROR: No images available for evaluation!")
        return
    
    if len(model_names) < 2:
        print(f"WARNING: Only {len(model_names)} model(s) found. Need at least 2 for blind comparison.")
    
    # Check completed
    completed = load_completed_evaluations()
    remaining = len([img for img in image_list if img not in completed])
    print(f"\nAlready evaluated: {len(completed)}")
    print(f"Remaining: {remaining}")
    
    # Configure handler
    EvaluationHandler.image_data = image_data
    EvaluationHandler.model_names = model_names
    EvaluationHandler.image_list = image_list
    EvaluationHandler.current_index = 0
    
    # Start server
    server = ThreadingHTTPServer((HOST, PORT), EvaluationHandler)
    
    print(f"\n🖥️  Server running at: http://{HOST}:{PORT}")
    print(f"📊 Results will be saved to: {OUTPUT_CSV}")
    print(f"\nPress Ctrl+C to stop the server")
    
    # Open browser
    threading.Timer(1.0, lambda: webbrowser.open(f"http://{HOST}:{PORT}")).start()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
