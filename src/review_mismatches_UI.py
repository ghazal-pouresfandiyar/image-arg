#!/usr/bin/env python3
"""
Web UI for reviewing YOLO+CLIP validation mismatches.
Opens directly in browser, similar to annotate_UI.py
"""

import json
import base64
import webbrowser
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse
import cv2
import numpy as np

# Paths
DATASET_DIR = Path(__file__).parent.parent / "dataset"
FEATURES_DIR = DATASET_DIR / "features"
IMAGES_DIR = DATASET_DIR / "images_for_annotation"
YOLO_DETECTION_DIR = Path(__file__).parent.parent / "yolo_detection"

MISMATCHES_JSON = FEATURES_DIR / "yolo_clip_mismatches.json"
VALIDATION_JSON = FEATURES_DIR / "yolo_clip_validation.json"
DETECTED_OBJECTS_JSON = FEATURES_DIR / "detected_objects.json"

# Global state
mismatches_data = {}
validation_data = {}


class UIRequestHandler(SimpleHTTPRequestHandler):
    """Handle HTTP requests for the review UI."""
    
    def do_GET(self):
        """Handle GET requests."""
        path = urllib.parse.urlparse(self.path).path
        
        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(self.get_html().encode())
        
        elif path == "/api/mismatches":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(mismatches_data).encode())
        
        elif path.startswith("/api/image/"):
            image_id = path.split("/")[-1]
            self.send_image(image_id)
        
        elif path.startswith("/api/preview/"):
            image_id = path.split("/")[-1].split("_")[0]
            bbox_idx = int(path.split("/")[-1].split("_")[1])
            self.send_bbox_preview(image_id, bbox_idx)
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def send_image(self, image_id):
        """Send YOLO detection image as base64."""
        image_path = YOLO_DETECTION_DIR / f"{image_id}.jpg"
        if not image_path.exists():
            self.send_response(404)
            self.end_headers()
            return
        
        with open(image_path, "rb") as f:
            img_data = f.read()
        
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        
        b64 = base64.b64encode(img_data).decode()
        response = json.dumps({"image": f"data:image/jpeg;base64,{b64}"})
        self.wfile.write(response.encode())
    
    def send_bbox_preview(self, image_id, bbox_idx):
        """Send YOLO detection image (already has all bboxes drawn)."""
        image_path = YOLO_DETECTION_DIR / f"{image_id}.jpg"
        if not image_path.exists():
            self.send_response(404)
            self.end_headers()
            return
        
        # Just send the YOLO detection image as-is (already has bboxes)
        with open(image_path, "rb") as f:
            img_data = f.read()
        
        b64 = base64.b64encode(img_data).decode()
        
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        
        response = json.dumps({"image": f"data:image/jpeg;base64,{b64}"})
        self.wfile.write(response.encode())
    
    def get_html(self):
        """Generate HTML for the review UI."""
        return """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Review YOLO+CLIP Mismatches</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        header { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        h1 { font-size: 24px; color: #333; margin-bottom: 10px; }
        .stats { display: flex; gap: 30px; font-size: 14px; color: #666; }
        .stat-item { display: flex; flex-direction: column; }
        .stat-value { font-size: 20px; font-weight: bold; color: #e74c3c; }
        .controls { display: flex; gap: 15px; margin-top: 15px; align-items: center; }
        select, input { padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; }
        button { padding: 10px 20px; background: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }
        button:hover { background: #2980b9; }
        .main { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .panel { background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 20px; }
        .panel h2 { font-size: 18px; margin-bottom: 15px; color: #333; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        .image-container { background: #f9f9f9; border: 2px solid #e0e0e0; border-radius: 4px; display: flex; align-items: center; justify-content: center; overflow: auto; height: 500px; margin-bottom: 15px; }
        .image-container img { max-width: 100%; max-height: 100%; object-fit: contain; }
        .bbox-list { list-style: none; max-height: 500px; overflow-y: auto; }
        .bbox-item { padding: 12px; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 10px; cursor: pointer; transition: all 0.2s; }
        .bbox-item:hover { background: #f0f0f0; border-color: #3498db; }
        .bbox-item.active { background: #e3f2fd; border-color: #2196f3; font-weight: bold; }
        .bbox-label { font-weight: bold; color: #e74c3c; margin-bottom: 5px; }
        .scores { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 13px; margin-top: 8px; }
        .score { background: #f9f9f9; padding: 8px; border-radius: 3px; }
        .score-label { color: #666; font-size: 12px; }
        .score-value { font-weight: bold; color: #333; margin-top: 3px; }
        .details { background: #f9f9f9; padding: 15px; border-radius: 4px; font-size: 13px; line-height: 1.6; margin-top: 15px; }
        .detail-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #e0e0e0; }
        .detail-row:last-child { border-bottom: none; }
        .detail-label { color: #666; font-weight: bold; }
        .detail-value { color: #333; text-align: right; word-break: break-word; }
        .clip-suggestion-row { background: #fff3cd !important; padding: 12px !important; border-radius: 4px !important; margin: 15px 0 !important; border-left: 5px solid #ffc107 !important; border-bottom: none !important; }
        .clip-suggestion-row .detail-label { color: #856404 !important; font-weight: bold; }
        .clip-suggestion-row .detail-value { color: #856404 !important; font-weight: bold; font-size: 15px; }
        .empty-state { text-align: center; color: #999; padding: 40px 20px; }
        @media (max-width: 1200px) { .main { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔍 YOLO + CLIP Validation Review</h1>
            <div class="stats">
                <div class="stat-item"><span>Total Mismatches</span><span class="stat-value" id="total-mismatches">0</span></div>
                <div class="stat-item"><span>Images with Mismatches</span><span class="stat-value" id="image-count">0</span></div>
                <div class="stat-item"><span>Avg Confidence Gap</span><span class="stat-value" id="avg-gap">0%</span></div>
            </div>
            <div class="controls">
                <label for="image-select">Select Image:</label>
                <select id="image-select"><option value="">-- Loading --</option></select>
                <button id="export-btn">📥 Export Results</button>
            </div>
        </header>
        
        <div class="main">
            <div class="panel">
                <h2>Image Preview</h2>
                <div class="image-container"><img id="preview-image" src="" alt="Image preview"></div>
                <div class="details">
                    <div class="detail-row"><span class="detail-label">Image ID:</span><span class="detail-value" id="detail-image-id">-</span></div>
                    <div class="detail-row"><span class="detail-label">Total Detections:</span><span class="detail-value" id="detail-bbox-count">-</span></div>
                </div>
            </div>
            
            <div class="panel">
                <h2>CLIP Suggestions</h2>
                <ul class="bbox-list" id="bbox-list"><li class="empty-state">Select an image to see CLIP suggestions</li></ul>
            </div>
        </div>
    </div>
    
    <script>
        let allMismatches = {};
        let currentImage = null;
        let currentBboxIdx = null;
        
        async function init() {
            try {
                const response = await fetch('/api/mismatches');
                allMismatches = await response.json();
                console.log('Loaded mismatches:', allMismatches);
                console.log('Total images:', Object.keys(allMismatches).length);
                if (Object.keys(allMismatches).length === 0) { alert('No mismatches found'); return; }
                updateStats();
                populateImageSelect();
            } catch (err) { console.error('Error:', err); alert('Error loading data'); }
        }
        
        function updateStats() {
            const imageIds = Object.keys(allMismatches);
            const totalMismatches = imageIds.reduce((sum, id) => sum + allMismatches[id].length, 0);
            const gaps = imageIds.flatMap(id => allMismatches[id].map(b => b.gap));
            const avgGap = gaps.length > 0 ? (gaps.reduce((a, b) => a + b, 0) / gaps.length * 100).toFixed(1) : 0;
            document.getElementById('total-mismatches').textContent = totalMismatches;
            document.getElementById('image-count').textContent = imageIds.length;
            document.getElementById('avg-gap').textContent = avgGap + '%';
        }
        
        function populateImageSelect() {
            const select = document.getElementById('image-select');
            const imageIds = Object.keys(allMismatches).sort();
            select.innerHTML = '<option value="">-- Select Image --</option>';
            imageIds.forEach(id => {
                const option = document.createElement('option');
                option.value = id;
                option.textContent = `Image ${id} (${allMismatches[id].length} mismatches)`;
                select.appendChild(option);
            });
            select.addEventListener('change', (e) => { if (e.target.value) loadImage(e.target.value); });
        }
        
        async function loadImage(imageId) {
            currentImage = imageId;
            currentBboxIdx = null;
            try {
                const response = await fetch(`/api/image/${imageId}`);
                const data = await response.json();
                document.getElementById('preview-image').src = data.image;
            } catch (err) { console.error('Error:', err); }
            document.getElementById('detail-image-id').textContent = imageId;
            document.getElementById('detail-bbox-count').textContent = allMismatches[imageId].length;
            const bboxList = document.getElementById('bbox-list');
            bboxList.innerHTML = '';
            allMismatches[imageId].forEach((bbox, idx) => {
                const li = document.createElement('li');
                li.className = 'bbox-item';
                li.innerHTML = `<div class="bbox-label">${bbox.detected_label}</div>
                    <div class="detail-row" style="margin-top: 10px; padding: 8px 0; border-bottom: none;">
                        <span class="detail-label">YOLO:</span>
                        <span class="detail-value">${(bbox.yolo_confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div class="clip-suggestion-row" style="margin: 8px 0; padding: 8px;">
                        <span class="detail-label">🔍 CLIP:</span>
                        <span class="detail-value">${bbox.clip_suggestion}</span>
                    </div>
                    <div class="detail-row" style="padding: 8px 0; border-bottom: none;">
                        <span class="detail-label">Match:</span>
                        <span class="detail-value">${(bbox.clip_suggestion_similarity * 100).toFixed(1)}%</span>
                    </div>`;
                bboxList.appendChild(li);
            });
        }
        
        async function selectBbox(idx, bbox) {
            currentBboxIdx = idx;
            document.querySelectorAll('.bbox-item').forEach((item, i) => {
                item.classList.toggle('active', i === idx);
            });
            try {
                const response = await fetch(`/api/preview/${currentImage}_${idx}`);
                const data = await response.json();
                document.getElementById('preview-image').src = data.image;
            } catch (err) { console.error('Error:', err); }
            document.getElementById('selected-details').style.display = 'block';
            document.getElementById('selected-label').textContent = bbox.detected_label;
            document.getElementById('selected-yolo-conf').textContent = (bbox.yolo_confidence * 100).toFixed(2) + '%';
            document.getElementById('selected-clip-sim').textContent = (bbox.clip_similarity * 100).toFixed(2) + '%';
            
            // Debug
            console.log('Full bbox data:', bbox);
            console.log('clip_suggestion field:', bbox.clip_suggestion);
            console.log('clip_suggestion_similarity field:', bbox.clip_suggestion_similarity);
            
            document.getElementById('selected-clip-suggestion').textContent = bbox.clip_suggestion || 'N/A';
            document.getElementById('selected-clip-suggestion-sim').textContent = (bbox.clip_suggestion_similarity * 100).toFixed(2) + '%';
            document.getElementById('selected-gap').textContent = (bbox.gap * 100).toFixed(2) + '%';
            document.getElementById('selected-pos').textContent = `(${bbox.x1}, ${bbox.y1}) → (${bbox.x2}, ${bbox.y2})`;
        }
        
        document.getElementById('export-btn').addEventListener('click', () => {
            let csv = 'Image ID,Label,YOLO Confidence,CLIP Similarity,Gap\\n';
            Object.keys(allMismatches).forEach(imageId => {
                allMismatches[imageId].forEach(bbox => {
                    csv += `${imageId},"${bbox.detected_label}",${(bbox.yolo_confidence * 100).toFixed(2)},${(bbox.clip_similarity * 100).toFixed(2)},${(bbox.gap * 100).toFixed(2)}\\n`;
                });
            });
            const element = document.createElement('a');
            element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(csv));
            element.setAttribute('download', 'yolo_clip_mismatches.csv');
            element.style.display = 'none';
            document.body.appendChild(element);
            element.click();
            document.body.removeChild(element);
        });
        
        init();
    </script>
</body>
</html>"""
    
    def log_message(self, format, *args):
        """Suppress log messages."""
        pass


def start_server(port=5000):
    """Start HTTP server."""
    server = HTTPServer(("127.0.0.1", port), UIRequestHandler)
    print(f"🌐 Server running at http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n✓ Server stopped")
        server.shutdown()


def load_data():
    """Load mismatch data and merge with bbox coordinates."""
    global mismatches_data, validation_data
    
    if not MISMATCHES_JSON.exists():
        print(f"❌ Mismatches file not found: {MISMATCHES_JSON}")
        print("Run `python3 src/3_YOLO_objects.py` first to generate validation data")
        exit(1)
    
    # Load detected objects to get bbox coordinates
    detected_objects = {}
    if DETECTED_OBJECTS_JSON.exists():
        with open(DETECTED_OBJECTS_JSON) as f:
            objects_list = json.load(f)
            for obj in objects_list:
                detected_objects[obj["id"]] = obj["bboxes"]
    
    # Load mismatches
    with open(MISMATCHES_JSON) as f:
        raw_data = json.load(f)
    
    # Transform list format to dict grouped by image_id
    mismatches_data = {}
    if isinstance(raw_data, list):
        for item in raw_data:
            image_id = str(item.get("image_id"))
            bbox_idx = item.get("bbox_index", 0)
            
            if image_id not in mismatches_data:
                mismatches_data[image_id] = []
            
            # Get bbox coordinates from detected_objects
            x1, y1, x2, y2 = 0, 0, 0, 0
            if image_id in detected_objects and bbox_idx < len(detected_objects[image_id]):
                bbox = detected_objects[image_id][bbox_idx].get("bbox", [0, 0, 0, 0])
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            
            # Format bbox data for frontend
            bbox_info = {
                "bbox_index": bbox_idx,
                "detected_label": item.get("yolo_label", "unknown"),
                "yolo_confidence": item.get("yolo_confidence", 0),
                "clip_similarity": item.get("clip_similarity_to_yolo_label", 0),
                "clip_suggestion": item.get("clip_suggestion", "unknown"),
                "clip_suggestion_similarity": item.get("clip_similarity_to_suggestion", 0),
                "gap": abs(item.get("yolo_confidence", 0) - item.get("clip_similarity_to_yolo_label", 0)),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
            mismatches_data[image_id].append(bbox_info)
    else:
        mismatches_data = raw_data
    
    if VALIDATION_JSON.exists():
        with open(VALIDATION_JSON) as f:
            validation_data = json.load(f)
    
    # Debug output
    print(f"✓ Loaded {len(mismatches_data)} images with mismatches")
    if mismatches_data:
        first_image_id = list(mismatches_data.keys())[0]
        print(f"  Sample: Image {first_image_id} has {len(mismatches_data[first_image_id])} detections")
        if mismatches_data[first_image_id]:
            print(f"  First detection keys: {mismatches_data[first_image_id][0].keys()}")
            print(f"  First detection: {mismatches_data[first_image_id][0]}")
    else:
        print("❌ WARNING: No mismatches data loaded!")


if __name__ == "__main__":
    load_data()
    
    port = 5000
    webbrowser.open(f"http://127.0.0.1:{port}")
    
    # Run server in main thread
    start_server(port)
