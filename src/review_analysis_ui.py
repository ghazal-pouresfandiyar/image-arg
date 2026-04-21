#!/usr/bin/env python3
"""
Web UI for reviewing YOLO + CLIP detection analysis with rich insights.
Shows detection disagreement, uncertainty, and full-image context.
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

ANALYSIS_JSON = FEATURES_DIR / "yolo_clip_analysis.json"

# Global state
analysis_data = {}


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
        
        elif path == "/api/analysis":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(analysis_data).encode())
        
        elif path.startswith("/api/image/"):
            image_id = path.split("/")[-1]
            self.send_image(image_id)
        
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
    
    def get_html(self):
        """Generate HTML for the analysis review UI."""
        return """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO+CLIP Detection Analysis</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; background: #f5f5f5; }
        .container { max-width: 1600px; margin: 0 auto; padding: 20px; }
        header { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        h1 { font-size: 28px; color: #333; margin-bottom: 15px; }
        .controls { display: flex; gap: 15px; flex-wrap: wrap; align-items: center; }
        .control-group { display: flex; gap: 10px; align-items: center; }
        label { font-weight: bold; color: #666; }
        select { padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; }
        .filter-buttons { display: flex; gap: 10px; }
        button { padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; font-weight: 500; }
        button.filter-btn { background: #e0e0e0; color: #333; }
        button.filter-btn.active { color: white; font-weight: bold; }
        button.filter-btn.agree.active { background: #4caf50; }
        button.filter-btn.uncertain.active { background: #ff9800; }
        button.filter-btn.conflict.active { background: #f44336; }
        
        .layout { display: grid; grid-template-columns: 1.2fr 1fr; gap: 20px; }
        .panel { background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); overflow: hidden; }
        .panel-header { background: #f9f9f9; padding: 15px; border-bottom: 1px solid #e0e0e0; font-weight: bold; }
        .panel-content { padding: 20px; }
        
        .image-view { display: flex; flex-direction: column; }
        .image-container { flex: 1; display: flex; align-items: center; justify-content: center; background: #f0f0f0; border-radius: 4px; overflow: auto; height: 500px; margin-bottom: 15px; }
        .image-container img { max-width: 100%; max-height: 100%; object-fit: contain; }
        
        .detection-list { list-style: none; max-height: 600px; overflow-y: auto; }
        .detection-item { padding: 12px; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 10px; cursor: pointer; transition: all 0.2s; }
        .detection-item:hover { background: #f5f5f5; border-color: #999; }
        .detection-item.selected { background: #e3f2fd; border-color: #2196f3; }
        
        .flag-badge { display: inline-block; padding: 4px 8px; border-radius: 3px; font-size: 12px; font-weight: bold; margin-left: 8px; }
        .flag-agree { background: #c8e6c9; color: #1b5e20; }
        .flag-uncertain { background: #ffe0b2; color: #e65100; }
        .flag-conflict { background: #ffcdd2; color: #b71c1c; }
        
        .detection-main { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
        .detection-label { font-weight: bold; font-size: 15px; }
        .detection-scores { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 13px; margin-top: 8px; }
        .score-box { background: #f9f9f9; padding: 8px; border-radius: 3px; }
        .score-label { color: #666; font-size: 11px; text-transform: uppercase; }
        .score-value { font-weight: bold; color: #333; margin-top: 3px; font-size: 14px; }
        .margin-box { background: #fff9e6; padding: 8px; border-radius: 3px; border-left: 3px solid #ffc107; }
        
        .details-section { background: #f9f9f9; padding: 15px; border-radius: 4px; margin-bottom: 12px; }
        .detail-row { display: flex; justify-content: space-between; padding: 6px 0; font-size: 13px; }
        .detail-label { color: #666; font-weight: bold; }
        .detail-value { color: #333; text-align: right; word-break: break-word; }
        
        .global-context { background: #f0f8ff; border-left: 4px solid #2196f3; padding: 12px; border-radius: 4px; }
        .global-title { font-weight: bold; color: #1976d2; margin-bottom: 8px; font-size: 13px; }
        .global-labels { display: flex; flex-direction: column; gap: 6px; }
        .label-item { display: flex; justify-content: space-between; font-size: 12px; padding: 4px 0; }
        .label-name { color: #333; }
        .label-score { font-weight: bold; color: #1976d2; }
        
        .empty-state { text-align: center; color: #999; padding: 40px 20px; }
        @media (max-width: 1200px) { .layout { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔍 YOLO + CLIP Detection Analysis</h1>
            <div class="controls">
                <div class="control-group">
                    <label for="image-select">Image:</label>
                    <select id="image-select"><option value="">-- Loading --</option></select>
                </div>
                <div class="filter-buttons">
                    <button class="filter-btn agree active" data-flag="agree">🟢 Agree</button>
                    <button class="filter-btn uncertain active" data-flag="uncertain">🟡 Uncertain</button>
                    <button class="filter-btn conflict active" data-flag="conflict">🔴 Conflict</button>
                </div>
            </div>
        </header>
        
        <div class="layout">
            <div class="panel image-view">
                <div class="image-container"><img id="preview-image" src="" alt="Image"></div>
            </div>
            
            <div class="panel">
                <div class="panel-header">Detections</div>
                <div class="panel-content">
                    <ul class="detection-list" id="detection-list"><li class="empty-state">Select an image</li></ul>
                    <div id="detail-panel" style="display: none; margin-top: 20px; padding-top: 20px; border-top: 2px solid #e0e0e0;">
                        <div class="panel-header" style="margin: -20px -20px 15px -20px; padding: 15px;">Detection Details</div>
                        <div class="details-section">
                            <div class="detail-row"><span class="detail-label">YOLO Prediction:</span><span class="detail-value" id="detail-yolo">-</span></div>
                            <div class="detail-row"><span class="detail-label">YOLO Confidence:</span><span class="detail-value" id="detail-yolo-conf">-</span></div>
                        </div>
                        <div class="details-section">
                            <div class="detail-row"><span class="detail-label">CLIP Top Label:</span><span class="detail-value" id="detail-clip-label">-</span></div>
                            <div class="detail-row"><span class="detail-label">CLIP Score:</span><span class="detail-value" id="detail-clip-score">-</span></div>
                            <div class="detail-row"><span class="detail-label">Margin (Uncertainty):</span><span class="detail-value" id="detail-margin">-</span></div>
                            <div class="detail-row"><span class="detail-label">Analysis:</span><span class="detail-value" id="detail-flag" style="font-weight: bold;">-</span></div>
                        </div>
                        <div class="global-context">
                            <div class="global-title">📊 Full-Image CLIP Context (Top 5)</div>
                            <div class="global-labels" id="global-labels"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let allAnalysis = {};
        let activeFilters = { agree: true, uncertain: true, conflict: true };
        let currentImage = null;
        let currentBboxIdx = null;
        
        async function init() {
            try {
                const response = await fetch('/api/analysis');
                const data = await response.json();
                allAnalysis = {};
                data.forEach(img => { allAnalysis[img.image_id] = img; });
                console.log('Loaded analysis for', Object.keys(allAnalysis).length, 'images');
                populateImageSelect();
                setupFilters();
            } catch (err) { console.error('Error loading analysis:', err); }
        }
        
        function setupFilters() {
            document.querySelectorAll('.filter-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const flag = e.target.dataset.flag;
                    activeFilters[flag] = !activeFilters[flag];
                    e.target.classList.toggle('active');
                    if (currentImage) loadImage(currentImage);
                });
            });
        }
        
        function populateImageSelect() {
            const select = document.getElementById('image-select');
            const imageIds = Object.keys(allAnalysis).sort();
            select.innerHTML = '<option value="">-- Select Image --</option>';
            imageIds.forEach(id => {
                const option = document.createElement('option');
                option.value = id;
                const count = allAnalysis[id].detections.length;
                option.textContent = `Image ${id} (${count} detections)`;
                select.appendChild(option);
            });
            select.addEventListener('change', (e) => { if (e.target.value) loadImage(e.target.value); });
        }
        
        async function loadImage(imageId) {
            currentImage = imageId;
            currentBboxIdx = null;
            document.getElementById('detail-panel').style.display = 'none';
            
            try {
                const response = await fetch(`/api/image/${imageId}`);
                const data = await response.json();
                document.getElementById('preview-image').src = data.image;
            } catch (err) { console.error('Error loading image:', err); }
            
            const imgData = allAnalysis[imageId];
            const detectionList = document.getElementById('detection-list');
            detectionList.innerHTML = '';
            
            const filtered = imgData.detections.filter(d => activeFilters[d.analysis_flag]);
            
            if (filtered.length === 0) {
                detectionList.innerHTML = '<li class="empty-state">No detections match filter</li>';
                return;
            }
            
            filtered.forEach((detection, idx) => {
                const li = document.createElement('li');
                li.className = 'detection-item';
                
                const flagEmoji = { agree: '🟢', uncertain: '🟡', conflict: '🔴' }[detection.analysis_flag];
                const flagClass = `flag-${detection.analysis_flag}`;
                
                li.innerHTML = `
                    <div class="detection-main">
                        <span class="detection-label">${detection.yolo_label}</span>
                        <span class="flag-badge ${flagClass}">${flagEmoji} ${detection.analysis_flag.toUpperCase()}</span>
                    </div>
                    <div class="detection-scores">
                        <div class="score-box">
                            <div class="score-label">YOLO</div>
                            <div class="score-value">${(detection.yolo_confidence * 100).toFixed(1)}%</div>
                        </div>
                        <div class="score-box">
                            <div class="score-label">CLIP</div>
                            <div class="score-value">${(detection.bbox_clip_top_score * 100).toFixed(1)}%</div>
                        </div>
                    </div>
                    <div class="margin-box">
                        <div class="score-label">Uncertainty (Margin)</div>
                        <div class="score-value">${(detection.bbox_clip_margin * 100).toFixed(1)}%</div>
                    </div>
                `;
                
                li.addEventListener('click', () => selectDetection(idx, detection, li));
                detectionList.appendChild(li);
            });
        }
        
        function selectDetection(idx, detection, element) {
            currentBboxIdx = idx;
            document.querySelectorAll('.detection-item').forEach(item => item.classList.remove('selected'));
            element.classList.add('selected');
            
            document.getElementById('detail-panel').style.display = 'block';
            document.getElementById('detail-yolo').textContent = detection.yolo_label;
            document.getElementById('detail-yolo-conf').textContent = (detection.yolo_confidence * 100).toFixed(2) + '%';
            document.getElementById('detail-clip-label').textContent = detection.bbox_clip_top_label;
            document.getElementById('detail-clip-score').textContent = (detection.bbox_clip_top_score * 100).toFixed(2) + '%';
            document.getElementById('detail-margin').textContent = (detection.bbox_clip_margin * 100).toFixed(2) + '%' + (detection.bbox_clip_margin < 0.05 ? ' (very uncertain)' : detection.bbox_clip_margin < 0.15 ? ' (uncertain)' : ' (confident)');
            
            const flagEmoji = { agree: '🟢 AGREE', uncertain: '🟡 UNCERTAIN', conflict: '🔴 CONFLICT' }[detection.analysis_flag];
            document.getElementById('detail-flag').textContent = flagEmoji;
            document.getElementById('detail-flag').style.color = { agree: '#1b5e20', uncertain: '#e65100', conflict: '#b71c1c' }[detection.analysis_flag];
            
            const globalLabels = document.getElementById('global-labels');
            globalLabels.innerHTML = detection.full_image_clip_top_labels.map(label => `
                <div class="label-item">
                    <span class="label-name">${label.label}</span>
                    <span class="label-score">${(label.score * 100).toFixed(1)}%</span>
                </div>
            `).join('');
        }
        
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
    """Load analysis data."""
    global analysis_data
    
    if not ANALYSIS_JSON.exists():
        print(f"❌ Analysis file not found: {ANALYSIS_JSON}")
        print("Run `python3 src/3_YOLO_objects.py` first to generate analysis data")
        exit(1)
    
    with open(ANALYSIS_JSON) as f:
        analysis_data = json.load(f)
    
    # Keep as list for API
    if not isinstance(analysis_data, list):
        analysis_data = [analysis_data]
    
    total_detections = sum(len(img.get("detections", [])) for img in analysis_data)
    print(f"✓ Loaded {len(analysis_data)} images with {total_detections} detections")


if __name__ == "__main__":
    load_data()
    
    port = 5000
    webbrowser.open(f"http://127.0.0.1:{port}")
    
    # Run server in main thread
    start_server(port)
