"""
Climate Argument Results Viewer
Interactive Streamlit app to review and compare model outputs
run command: streamlit run "user interfaces/output.py"
"""

import streamlit as st
import json
import pandas as pd
from pathlib import Path
from PIL import Image
import os

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Model Results Viewer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .read-only-badge { 
        background-color: #e8f5e9; 
        padding: 8px 12px; 
        border-radius: 4px; 
        font-weight: bold;
        color: #2e7d32;
        margin-bottom: 1rem;
    }
    .model-box {
        background-color: #e3f2fd;
        padding: 12px 16px;
        border-radius: 6px;
        border-left: 4px solid #1976d2;
        font-weight: bold;
        font-size: 16px;
        margin-bottom: 12px;
        color: #0d47a1;
    }
    .human-box {
        background-color: #fff8e1;
        padding: 12px 16px;
        border-radius: 6px;
        border-left: 4px solid #f57f17;
        font-weight: bold;
        font-size: 16px;
        margin-bottom: 12px;
        color: #e65100;
    }
    .premises-box {
        background-color: #f3e5f5;
        padding: 8px 12px;
        border-radius: 4px;
        border-left: 3px solid #7b1fa2;
        font-weight: bold;
        margin-top: 12px;
        margin-bottom: 8px;
        color: #4a148c;
        font-size: 14px;
    }
    .conclusions-box {
        background-color: #e8f5e9;
        padding: 8px 12px;
        border-radius: 4px;
        border-left: 3px solid #388e3c;
        font-weight: bold;
        margin-top: 12px;
        margin-bottom: 8px;
        color: #1b5e20;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# ROOT DIRECTORY & PATHS
# ============================================================================

# Get the root directory relative to this script
SCRIPT_DIR = Path(__file__).parent.parent  # Go up from user_interfaces/ to project root
ROOT_DIR = SCRIPT_DIR
IMAGE_DIR = ROOT_DIR / 'dataset' / 'images_for_annotation'
CSV_PATH = ROOT_DIR / 'dataset' / 'annotated.csv'

# ============================================================================
# DATA LOADING & CACHING
# ============================================================================

@st.cache_resource
def load_human_annotations():
    """Load human annotations from CSV and transform to model-compatible format"""
    df = pd.read_csv(CSV_PATH)
    
    # Transform to model output format
    human_data = []
    for idx, row in df.iterrows():
        human_data.append({
            "image_id": str(row['id']),
            "model": "Human Annotator",
            "image_url": row.get('image_url', row.get('url', '')),  # Try different column names
            "metadata": {
                "animals": row.get('animals', 'N/A'),
                "consequences": row.get('consequences', 'N/A'),
                "climateaction": row.get('climateaction', 'N/A'),
                "type": row.get('type', 'N/A'),
                "setting": row.get('setting', 'N/A'),
                "caption": row.get('blip2_caption', row.get('caption', 'N/A')),
            },
            "facts": [],
            "premises": row.get('premises', '[]') if isinstance(row.get('premises', '[]'), str) else row.get('premises', '[]'),
            "conclusions": row.get('conclusions', '[]') if isinstance(row.get('conclusions', '[]'), str) else row.get('conclusions', '[]'),
            "is_human": True
        })
    
    return human_data

@st.cache_resource
def discover_model_outputs():
    """Discover all model output JSON files in root directory and models/output_model directory"""
    model_files = {}
    
    # Search in root directory
    for file_path in ROOT_DIR.glob('*.json'):
        if file_path.name == 'annotated.csv':
            continue
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Handle both list and dict formats
            if isinstance(data, dict):
                data = list(data.values()) if data else []
            
            if isinstance(data, list) and len(data) > 0:
                model_files[file_path.stem] = data
        except Exception as e:
            st.warning(f"Could not load {file_path.name}: {str(e)}")
    
    # Search in models/output_model directory
    output_model_dir = ROOT_DIR / 'models' / 'output_model'
    if output_model_dir.exists():
        for file_path in output_model_dir.glob('*.json'):
            # Skip tracking and failed files
            if file_path.name in ['tracking.json', 'failed.json']:
                continue
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                # Handle both list and dict formats
                if isinstance(data, dict):
                    data = list(data.values()) if data else []
                
                if isinstance(data, list) and len(data) > 0:
                    # Use more descriptive name for model files
                    model_name = file_path.stem.replace('-', ' ').title()
                    model_files[model_name] = data
            except Exception as e:
                st.warning(f"Could not load {file_path.name}: {str(e)}")
    
    return model_files

@st.cache_resource
def load_all_data():
    """Load and organize all data"""
    human_data = load_human_annotations()
    model_outputs = discover_model_outputs()
    
    # Create unified structure
    all_models = {"Human": human_data}
    all_models.update(model_outputs)
    
    # Create a lookup for image URLs from human data (same for all models)
    image_url_lookup = {item.get('image_id'): item.get('image_url', '') for item in human_data}
    
    return all_models, image_url_lookup

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_image_path(image_id):
    """Get image path for given image ID"""
    # Try different formats
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        path = IMAGE_DIR / f"{image_id}{ext}"
        if path.exists():
            return path
    return None

def resize_image(image, max_width=400, max_height=500):
    """Resize image to fit within max dimensions while maintaining aspect ratio"""
    img_width, img_height = image.size
    
    # Calculate scaling factor based on both width and height constraints
    width_ratio = max_width / img_width
    height_ratio = max_height / img_height
    
    # Use the smaller ratio to ensure image fits in both dimensions
    scale_ratio = min(width_ratio, height_ratio, 1.0)  # Don't upscale
    
    new_width = int(img_width * scale_ratio)
    new_height = int(img_height * scale_ratio)
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def parse_output(output_data):
    """Parse output data (can be string or dict)"""
    # If already a dict, return as-is
    if isinstance(output_data, dict):
        return output_data
    
    # Try to parse string as JSON
    if isinstance(output_data, str):
        try:
            return json.loads(output_data)
        except:
            return {"text": output_data}
    
    return output_data

def display_metadata(metadata, compact=False):
    """Display metadata in a formatted box"""
    if compact:
        # Compact 2-column layout
        cols = st.columns(2)
        with cols[0]:
            st.write(f"**Animals:** {metadata.get('animals', 'N/A')}")
            st.write(f"**Consequences:** {metadata.get('consequences', 'N/A')}")
            st.write(f"**Type:** {metadata.get('type', 'N/A')}")
        with cols[1]:
            st.write(f"**Setting:** {metadata.get('setting', 'N/A')}")
            st.write(f"**Climate Action:** {metadata.get('climateaction', 'N/A')}")
            if metadata.get('caption'):
                st.write(f"**Caption:** {metadata.get('caption')}")
    else:
        # Full 3-column layout
        cols = st.columns(3)
        with cols[0]:
            st.write(f"**Animals:** {metadata.get('animals', 'N/A')}")
            st.write(f"**Type:** {metadata.get('type', 'N/A')}")
        with cols[1]:
            st.write(f"**Consequences:** {metadata.get('consequences', 'N/A')}")
            st.write(f"**Setting:** {metadata.get('setting', 'N/A')}")
        with cols[2]:
            st.write(f"**Climate Action:** {metadata.get('climateaction', 'N/A')}")
            if metadata.get('caption'):
                st.write(f"**Caption:** {metadata.get('caption')}")

def display_results(model_output, model_name, is_human=False, show_header=True):
    """Display model results"""
    if show_header:
        st.write(f"📊 **{model_name}**")
    
    # Display output
    # Check multiple possible output fields: parsed_output, output, conclusions
    output_data = model_output.get('parsed_output', model_output.get('output', model_output.get('conclusions', '')))
    
    if is_human:
        # Display premises and conclusions from human annotation
        premises = model_output.get('premises', '')
        conclusions = model_output.get('conclusions', '')
        
        if premises:
            st.markdown('<div class="premises-box">Premises:</div>', unsafe_allow_html=True)
            try:
                if isinstance(premises, str):
                    premises_list = json.loads(premises) if premises.startswith('[') else [premises]
                else:
                    premises_list = premises if isinstance(premises, list) else [premises]
                for i, premise in enumerate(premises_list, 1):
                    st.write(f"{i}. {premise}")
            except:
                st.write(premises)
        
        if conclusions:
            st.markdown('<div class="conclusions-box">Conclusions:</div>', unsafe_allow_html=True)
            try:
                if isinstance(conclusions, str):
                    conclusions_list = json.loads(conclusions) if conclusions.startswith('[') else [conclusions]
                else:
                    conclusions_list = conclusions if isinstance(conclusions, list) else [conclusions]
                for i, conclusion in enumerate(conclusions_list, 1):
                    st.write(f"{i}. {conclusion}")
            except:
                st.write(conclusions)
    else:
        # Display parsed model output
        parsed = parse_output(output_data)
        
        if isinstance(parsed, dict):
            if 'premises' in parsed:
                st.markdown('<div class="premises-box">Premises:</div>', unsafe_allow_html=True)
                for i, premise in enumerate(parsed['premises'], 1):
                    st.write(f"{i}. {premise}")
            
            # Handle both 'conclusion' (singular) and 'conclusions' (plural)
            if 'conclusions' in parsed:
                st.markdown('<div class="conclusions-box">Conclusions:</div>', unsafe_allow_html=True)
                if isinstance(parsed['conclusions'], list):
                    for i, conclusion in enumerate(parsed['conclusions'], 1):
                        st.write(f"{i}. {conclusion}")
                else:
                    st.write(parsed['conclusions'])
            elif 'conclusion' in parsed:
                st.markdown('<div class="conclusions-box">Conclusion:</div>', unsafe_allow_html=True)
                st.write(parsed['conclusion'])
        else:
            st.write(parsed.get('text', output_data))

# ============================================================================
# SIDEBAR - Controls
# ============================================================================

st.sidebar.markdown("# 📋 Controls")
st.sidebar.markdown('<div class="read-only-badge">🔒 Read-Only Mode</div>', unsafe_allow_html=True)

# Load data
all_models, image_url_lookup = load_all_data()
model_names = list(all_models.keys())

if not model_names:
    st.error("❌ No model outputs found! Please ensure model output JSON files exist in the root directory or models/output_model/ directory.")
    st.stop()





# ============================================================================
# MAIN CONTENT - COMPARISON VIEW
# ============================================================================

st.header("🔀 Compare Models")

# Get all unique image IDs across all models
all_image_ids = set()
for model_data in all_models.values():
    all_image_ids.update([item.get('image_id') for item in model_data])
all_image_ids = sorted(list(all_image_ids))

if not all_image_ids:
    st.error("No images found!")
    st.stop()

# Model selection for comparison
col_select1, col_select2 = st.columns(2)

with col_select1:
    model_1 = st.selectbox("📊 Model 1", model_names, key="comp_model1")

with col_select2:
    # Allow any model in second slot, including same model
    model_2 = st.selectbox("📊 Model 2", model_names, index=min(1, len(model_names)-1), key="comp_model2")

# Get data for both models
data_1 = {item.get('image_id'): item for item in all_models[model_1]}
data_2 = {item.get('image_id'): item for item in all_models[model_2]}

# Initialize session state for image navigation
if 'current_image_idx' not in st.session_state:
    st.session_state.current_image_idx = 0

# Image selection
st.markdown("### 🖼️ Select Image")
nav_col1, nav_col2, nav_col3 = st.columns([0.6, 0.2, 0.2])

with nav_col1:
    selected_id = st.selectbox(
        "Image ID",
        all_image_ids,
        index=st.session_state.current_image_idx,
        format_func=lambda x: f"Image {x}",
        label_visibility="collapsed"
    )
    st.session_state.current_image_idx = all_image_ids.index(selected_id)

# Get current index
current_idx = st.session_state.current_image_idx

with nav_col2:
    if st.button("⬅️ Previous", use_container_width=True):
        if current_idx > 0:
            st.session_state.current_image_idx = current_idx - 1
            st.rerun()

with nav_col3:
    if st.button("Next ➡️", use_container_width=True):
        if current_idx < len(all_image_ids) - 1:
            st.session_state.current_image_idx = current_idx + 1
            st.rerun()

# Get the selected image ID based on current index
selected_id = all_image_ids[st.session_state.current_image_idx]

# Get samples for selected image
sample_1 = data_1.get(selected_id, {
    'image_id': selected_id,
    'model': model_1,
    'metadata': {},
    'parsed_output': {'premises': [], 'conclusions': []}
})
sample_2 = data_2.get(selected_id, {
    'image_id': selected_id,
    'model': model_2,
    'metadata': {},
    'parsed_output': {'premises': [], 'conclusions': []}
})

# TOP ROW: Image (left) + Metadata (right)
st.markdown("---")
col_img, col_meta = st.columns([0.4, 0.6])

with col_img:
    st.subheader("Image")
    
    # Get image URL from lookup (same for all models)
    image_url = image_url_lookup.get(selected_id, '')
    image_path = get_image_path(selected_id)
    
    if image_url:
        st.markdown(f'[🔗 Open Image URL]({image_url})', unsafe_allow_html=True)
    elif image_path:
        st.caption(f"Local: {image_path}")
    
    # Display the image
    if image_path:
        img = Image.open(image_path)
        img_resized = resize_image(img, max_width=400, max_height=500)
        st.image(img_resized, caption=f"Image ID: {selected_id}")
    else:
        st.warning(f"⚠️ Image not found for ID: {selected_id}")

with col_meta:
    st.subheader("Metadata")
    metadata = sample_1.get('metadata', {})
    display_metadata(metadata, compact=True)

# BOTTOM ROW: Model 1 Results (left) + Model 2 Results (right)
st.markdown("---")
st.markdown("### 📝 Results Comparison")

comp_col1, comp_col2 = st.columns(2)

with comp_col1:
    if sample_1.get('image_id'):
        is_human_1 = sample_1.get('is_human', False)
        box_class = "human-box" if is_human_1 else "model-box"
        st.markdown(f'<div class="{box_class}">{model_1}</div>', unsafe_allow_html=True)
        display_results(sample_1, model_1, is_human_1, show_header=False)
    else:
        st.info(f"ℹ️ No data for {model_1} on Image {selected_id}")

with comp_col2:
    if sample_2.get('image_id'):
        is_human_2 = sample_2.get('is_human', False)
        box_class = "human-box" if is_human_2 else "model-box"
        st.markdown(f'<div class="{box_class}">{model_2}</div>', unsafe_allow_html=True)
        display_results(sample_2, model_2, is_human_2, show_header=False)
    else:
        st.info(f"ℹ️ No data for {model_2} on Image {selected_id}")


