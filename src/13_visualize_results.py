"""
Climate Argument Generation - Results Visualization UI
Interactive Streamlit app to examine model outputs
"""

import streamlit as st
import json
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Climate Argument Generation - Results",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_resource
def load_data():
    """Load all results and metadata"""
    DATA_DIR = Path('/Users/poures/Desktop/PC/image-arg/dataset/unified_features')
    INFERENCE_DIR = Path('/Users/poures/Desktop/PC/image-arg/inference_results')
    IMAGE_DIR = Path('/Users/poures/Desktop/PC/image-arg/dataset/images_for_annotation')
    
    # Load inference results
    with open(INFERENCE_DIR / 'generated_conclusions.json', 'r') as f:
        inference_results = json.load(f)
    
    # Load evaluation metrics
    with open(INFERENCE_DIR / 'evaluation_metrics.json', 'r') as f:
        eval_metrics = json.load(f)
    
    # Load targets (ground truth)
    with open(DATA_DIR / 'targets.json', 'r') as f:
        targets = json.load(f)
    
    # Load retrieved facts
    with open(DATA_DIR / 'retrieved_facts.json', 'r') as f:
        facts_data = json.load(f)
    
    # Load unified features
    data = np.load(DATA_DIR / 'unified_features.npz')
    features = data['unified_features']
    
    return {
        'inference_results': inference_results,
        'eval_metrics': eval_metrics,
        'targets': targets,
        'facts': facts_data,
        'features': features,
        'image_dir': IMAGE_DIR,
        'data_dir': DATA_DIR
    }

# Load data
data = load_data()

# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
# 🌍 Climate Argument Generation
## Results Visualization & Analysis

This interactive tool shows what the trained AI model learned about generating climate arguments from images.
""")

# ============================================================================
# SIDEBAR - Navigation & Metrics
# ============================================================================

with st.sidebar:
    st.markdown("## 📊 Model Overview")
    
    st.metric("Total Samples", len(data['inference_results']))
    st.metric("Model Confidence (Avg)", f"{data['eval_metrics']['model_confidence']['mean']:.4f}")
    st.metric("BLEU Score", f"{data['eval_metrics']['bleu']['mean']:.4f}")
    st.metric("ROUGE Score", f"{data['eval_metrics']['rouge']['mean']:.4f}")
    st.metric("Semantic Similarity", f"{data['eval_metrics']['semantic_similarity']['mean']:.4f}")
    
    st.markdown("---")
    st.markdown("## 🎯 Navigation")
    
    mode = st.radio("Select View", [
        "📸 Sample Explorer",
        "📊 Metrics Dashboard",
        "🔍 Feature Analysis",
        "📈 Comparative Analysis"
    ])

# ============================================================================
# MODE 1: SAMPLE EXPLORER
# ============================================================================

if mode == "📸 Sample Explorer":
    st.header("Sample-by-Sample Explorer")
    
    # Select sample
    sample_idx = st.selectbox(
        "Select Sample #",
        options=[r['index'] for r in data['inference_results']],
        format_func=lambda x: f"Sample {x}"
    )
    
    # Get sample data
    sample = next(s for s in data['inference_results'] if s['index'] == sample_idx)
    
    col1, col2 = st.columns([1, 1])
    
    # LEFT COLUMN: IMAGE
    with col1:
        st.subheader("📷 Input Image")
        
        image_path = data['image_dir'] / f"{sample_idx}.jpg"
        if image_path.exists():
            img = Image.open(image_path)
            st.image(img, use_column_width=True, caption=f"Image {sample_idx}")
        else:
            st.warning(f"Image not found: {image_path}")
    
    # RIGHT COLUMN: TEXT CONTENT
    with col2:
        st.subheader("📝 Arguments")
        
        st.markdown("**Input Premise:**")
        st.info(sample['premise'])
        
        st.markdown("**Ground Truth Conclusion:**")
        st.success(sample['ground_truth'])
        
        st.markdown("**Generated Conclusion:**")
        st.warning(sample['generated'])
        
        st.markdown("**Model Confidence:** " + f"{sample['confidence']:.4f}")
    
    # Features breakdown
    st.markdown("---")
    st.subheader("🔧 Extracted Features (932-dimensional vector)")
    
    features_vec = data['features'][sample_idx]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Feature Breakdown:**")
        feature_info = {
            "CLIP Embeddings": f"Dims 0-511 (512 total)",
            "YOLO Objects": f"Dims 512-529 (18 total)",
            "Scene Classification": f"Dims 530-533 (4 total)",
            "Climate Attributes": f"Dims 534-547 (14 total)",
            "Premise Embeddings": f"Dims 548-931 (384 total)"
        }
        for name, info in feature_info.items():
            st.text(f"• {name}: {info}")
    
    with col2:
        st.markdown("**Feature Statistics:**")
        
        feature_ranges = {
            "CLIP": (features_vec[0:512], "0-511"),
            "YOLO": (features_vec[512:530], "512-529"),
            "Scene": (features_vec[530:534], "530-533"),
            "Climate": (features_vec[534:548], "534-547"),
            "Premises": (features_vec[548:932], "548-931")
        }
        
        stats_data = []
        for name, (feat, range_str) in feature_ranges.items():
            stats_data.append({
                "Modality": name,
                "Min": f"{feat.min():.4f}",
                "Max": f"{feat.max():.4f}",
                "Mean": f"{feat.mean():.4f}",
                "Std": f"{feat.std():.4f}"
            })
        
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
    
    # Metrics for this sample
    st.markdown("---")
    st.subheader("📈 Model Confidence")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Confidence Score", f"{sample['confidence']:.4f}")
    with col2:
        st.metric("Avg Confidence (All Samples)", f"{data['metrics']['model_confidence']['mean']:.4f}")
    
    # Retrieved facts
    st.markdown("---")
    st.subheader("💡 Retrieved Climate Facts")
    
    facts = data['facts'][sample_idx]['retrieved_facts']
    
    for i, fact_item in enumerate(facts, 1):
        with st.expander(f"Fact {i} (Score: {fact_item['relevance_score']:.4f})"):
            st.write(fact_item['fact'])

# ============================================================================
# MODE 2: METRICS DASHBOARD
# ============================================================================

elif mode == "📊 Metrics Dashboard":
    st.header("📊 Evaluation Metrics Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Summary Statistics")
        
        metrics_df = pd.DataFrame({
            "Metric": ["BLEU", "ROUGE", "Semantic Sim", "Confidence"],
            "Mean": [
                f"{data['eval_metrics']['bleu']['mean']:.4f}",
                f"{data['eval_metrics']['rouge']['mean']:.4f}",
                f"{data['eval_metrics']['semantic_similarity']['mean']:.4f}",
                f"{data['eval_metrics']['model_confidence']['mean']:.4f}"
            ],
            "Std": [
                f"{data['eval_metrics']['bleu']['std']:.4f}",
                f"{data['eval_metrics']['rouge']['std']:.4f}",
                f"{data['eval_metrics']['semantic_similarity']['std']:.4f}",
                f"{data['eval_metrics']['model_confidence']['std']:.4f}"
            ]
        })
        
        st.dataframe(metrics_df, use_container_width=True)
    
    with col2:
        st.markdown("### Interpretation")
        st.markdown("""
        **BLEU Score (0.0000):**
        No n-gram matches between generated and ground truth.
        
        **ROUGE Score (0.0603):**
        Minimal longest common subsequence overlap.
        
        **Semantic Similarity (0.0321):**
        Only 3% word vocabulary overlap.
        
        **Confidence (0.3863):**
        Model equally uncertain for all inputs → mode collapse.
        """)
    
    # Per-sample breakdown
    st.markdown("---")
    st.subheader("Per-Sample Metrics")
    
    detailed_df = pd.DataFrame({
        "Sample": [r['index'] for r in data['inference_results']],
        "BLEU": [f"{r['bleu']:.4f}" for r in data['inference_results']],
        "ROUGE": [f"{r['rouge']:.4f}" for r in data['inference_results']],
        "Semantic Sim": [f"{r['semantic_similarity']:.4f}" for r in data['inference_results']],
        "Confidence": [f"{r['confidence']:.4f}" for r in data['inference_results']]
    })
    
    st.dataframe(detailed_df, use_container_width=True)

# ============================================================================
# MODE 3: FEATURE ANALYSIS
# ============================================================================

elif mode == "🔍 Feature Analysis":
    st.header("🔍 Feature Space Analysis")
    
    st.markdown("""
    **932-dimensional unified feature vector:**
    - CLIP embeddings (512) + YOLO objects (18) + Scene (4) + Climate (14) + Premises (384)
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Modality Distribution")
        
        modalities = {
            "CLIP\n(512)": 512,
            "YOLO\n(18)": 18,
            "Scene\n(4)": 4,
            "Climate\n(14)": 14,
            "Premises\n(384)": 384
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=list(modalities.keys()),
            values=list(modalities.values()),
            hole=0.3
        )])
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Feature Statistics")
        
        stats_summary = {
            "Modality": ["CLIP", "YOLO", "Scene", "Climate", "Premises"],
            "Dims": [512, 18, 4, 14, 384],
            "Mean": [
                f"{data['features'][:, 0:512].mean():.4f}",
                f"{data['features'][:, 512:530].mean():.4f}",
                f"{data['features'][:, 530:534].mean():.4f}",
                f"{data['features'][:, 534:548].mean():.4f}",
                f"{data['features'][:, 548:932].mean():.4f}"
            ]
        }
        
        st.dataframe(pd.DataFrame(stats_summary), use_container_width=True)

# ============================================================================
# MODE 4: COMPARATIVE ANALYSIS  
# ============================================================================

elif mode == "📈 Comparative Analysis":
    st.header("📈 Generated vs Ground Truth")
    
    # Comparison table
    comparison_data = []
    for result in data['inference_results']:
        comparison_data.append({
            "Sample": result['index'],
            "Ground Truth": result['ground_truth'],
            "Generated": result['generated'],
            "BLEU": f"{result['bleu']:.3f}",
            "ROUGE": f"{result['rouge']:.3f}"
        })
    
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
    
    st.markdown("---")
    st.subheader("Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⚠️ Mode Collapse")
        st.markdown("""
        **Generated Output:**
        All samples: "floods can cause many car accidents"
        
        **Root Cause:**
        - Small dataset (58 images)
        - Limited diversity (1-2 conclusions/image)
        - Model converged to safe pattern
        """)
    
    with col2:
        st.markdown("### ✅ What Worked")
        st.markdown("""
        **Technical Success:**
        - Training converged ✓
        - Coherent sentences generated ✓
        - End-to-end pipeline working ✓
        
        **Next Steps:**
        - Collect 500+ images
        - Get 5-10 conclusions per image
        - Add regularization
        - Use pre-trained language models
        """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
## About This Dashboard

**Model:** Encoder-decoder with fact-augmented attention  
**Parameters:** 5.16M | **Training:** 5 epochs | **Features:** 7 modalities (932-dim)

**Status:** ✅ Complete | ⚠️ Mode collapse (needs larger dataset) | 🎯 Ready for extension
""")
