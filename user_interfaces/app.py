from flask import Flask, send_from_directory, jsonify, request
import json
import os
import csv

app = Flask(__name__, static_folder='.', static_url_path='')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATASET_DIR = os.path.join(PROJECT_DIR, 'dataset')
IMAGES_DIR = os.path.join(DATASET_DIR, 'images_for_annotation')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'models', 'output_model')
EVAL_DIR = os.path.join(PROJECT_DIR, 'evaluation')
DATA_FILE = os.path.join(DATASET_DIR, 'reviewer_data.json')
PROMPTS_FILE = os.path.join(PROJECT_DIR, 'models', 'prompts.txt')
CSV_FILE = os.path.join(DATASET_DIR, 'annotated.csv')

@app.route('/')
def index():
    return send_from_directory('.', 'visual_argument_evaluation_viewer.html')

@app.route('/api/prompts')
def get_prompts():
    if os.path.exists(PROMPTS_FILE):
        with open(PROMPTS_FILE, 'r') as f:
            return jsonify({'content': f.read()})
    return jsonify({'content': ''})

@app.route('/api/dataset')
def get_dataset():
    results = []
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row.get('id', row.get('ID', ''))
                if image_id and os.path.exists(os.path.join(IMAGES_DIR, f'{image_id}.jpg')):
                    # Parse premises and conclusions from CSV
                    premises = []
                    conclusions = []
                    try:
                        if row.get('premises'):
                            premises = json.loads(row['premises'].replace('""', '"'))
                    except:
                        pass
                    try:
                        if row.get('conclusions'):
                            conclusions = json.loads(row['conclusions'].replace('""', '"'))
                    except:
                        pass
                    results.append({
                        'id': image_id,
                        'premises': premises,
                        'conclusions': conclusions
                    })
    return jsonify(results)

@app.route('/api/images/<filename>')
def get_image(filename):
    return send_from_directory(IMAGES_DIR, filename)

@app.route('/api/outputs/<filename>')
def get_output(filename):
    # Try output_model first, then evaluation subdirs
    if os.path.exists(os.path.join(OUTPUT_DIR, filename)):
        return send_from_directory(OUTPUT_DIR, filename)
    # Check evaluation subdirectories
    for subdir in ['hallucination', 'similarity', 'nli']:
        path = os.path.join(EVAL_DIR, subdir, filename)
        if os.path.exists(path):
            return send_from_directory(os.path.join(EVAL_DIR, subdir), filename)
    return jsonify({'error': 'File not found'}), 404

@app.route('/api/outputs')
def list_outputs():
    files = []
    if os.path.exists(OUTPUT_DIR):
        files.extend([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.json') or f.endswith('.csv')])
    # Also list evaluation CSVs
    if os.path.exists(EVAL_DIR):
        for subdir in os.listdir(EVAL_DIR):
            subdir_path = os.path.join(EVAL_DIR, subdir)
            if os.path.isdir(subdir_path):
                for f in os.listdir(subdir_path):
                    if f.endswith('.csv'):
                        files.append(f)
    return jsonify(files)

@app.route('/api/reviewer', methods=['GET'])
def get_reviewer():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({})

@app.route('/api/reviewer', methods=['POST'])
def save_reviewer():
    data = request.get_json()
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
