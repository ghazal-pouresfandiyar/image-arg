# 🌍 Climate Argument Generation - Results Dashboard
## Interactive Visualization Interface

### 📍 Access the Dashboard

**URL:** http://localhost:8501

Open this link in your web browser to view the interactive results dashboard.

---

## 🎯 What You Can Do

### 📸 View 1: Sample Explorer
Examine individual inference results in detail:

- **Select any sample** (0, 10, 20, 30, 40, 50)
- **View the climate image** - See the actual input image
- **Read the arguments:**
  - Input premise (what the image shows)
  - Ground truth conclusion (correct answer)
  - Generated conclusion (what model predicted)
- **Inspect features:**
  - Full 932-dimensional feature breakdown
  - CLIP embeddings, YOLO objects, scene type, climate attributes, premise embeddings
  - Min/Max/Mean/Std for each modality
- **Check metrics:**
  - BLEU, ROUGE, Semantic Similarity, Confidence
- **Review climate facts:**
  - Top-5 retrieved facts with relevance scores
  - Knowledge used by the model

### 📊 View 2: Metrics Dashboard
Analyze overall model performance:

- **Summary statistics** for all metrics:
  - BLEU, ROUGE, Semantic Similarity, Model Confidence
  - Mean, Std, Min, Max ranges
- **Interpretation guide** - Understand what each metric means
- **Per-sample breakdown** - See metrics for each of the 6 samples
- **Distribution analysis** - View metric ranges

### 🔍 View 3: Feature Analysis
Explore the feature space structure:

- **Feature modality distribution** - Pie chart showing:
  - CLIP (512 dims) - 55%
  - Premises (384 dims) - 41%
  - YOLO (18 dims) - 2%
  - Climate (14 dims) - 1.5%
  - Scene (4 dims) - 0.5%
- **Feature statistics** across all 58 samples
- **Detailed modality insights:**
  - CLIP embeddings (semantic visual encoding)
  - YOLO objects (concrete object detection)
  - Premise embeddings (text understanding)

### 📈 View 4: Comparative Analysis
Compare generated vs ground truth conclusions:

- **Side-by-side comparison table** showing:
  - Sample ID
  - Ground truth (correct answer)
  - Generated (model prediction)
  - Metrics (BLEU, ROUGE)
- **Mode collapse analysis:**
  - Why the model generates same output for all inputs
  - Root cause: small dataset (58 samples)
- **What worked well:**
  - Training converged successfully
  - Architecture works end-to-end
  - No implementation errors
- **Recommendations:**
  - Collect 500+ images
  - Get 5-10 conclusions per image
  - Add regularization
  - Use pre-trained language models

---

## 💡 Key Insights from Dashboard

### ✅ Model Strengths
- Encoder-decoder architecture functional
- Feature extraction robust (7 modalities)
- Training converged (loss: 5.92 → 3.19)
- Generated coherent sentences
- Inference pipeline working correctly

### ⚠️ Model Limitations
- Mode collapse: all outputs identical
  ```
  Generated for all samples: "floods can cause many car accidents"
  ```
- Low metrics:
  - BLEU: 0.0000 (no n-gram matches)
  - ROUGE: 0.0603 (minimal overlap)
  - Semantic Sim: 0.0321 (only 3% words overlap)
- Confidence uniform (0.3863) - indicates uncertainty

### 📊 Why Mode Collapse?

| Factor | Impact |
|--------|--------|
| **Small Dataset** | 58 images insufficient for diversity |
| **Few Conclusions** | Only 1-2 conclusions per image |
| **Limited Model** | 5.16M params vs billions for LLMs |
| **Training Converged** | Model found "safe" pattern |

---

## 🔧 Using the Dashboard

### Interactive Features:
- ✅ **Selectbox** - Choose which sample to examine
- ✅ **Expandable sections** - Click to show/hide facts
- ✅ **Data tables** - Sortable metrics
- ✅ **Plotly charts** - Interactive visualizations
- ✅ **Images** - Click to expand
- ✅ **Real-time stats** - Live metric display

### Dashboard Navigation:
1. Sidebar shows model overview metrics
2. Radio buttons select which view to display
3. Each view has specific controls
4. All data cached for fast browsing

---

## 📖 Feature Breakdown

### 932-Dimensional Feature Vector

The model uses features from 5 sources:

```
0-511     (512 dims) - CLIP semantic embeddings
512-529   (18 dims)  - YOLO object counts
530-533   (4 dims)   - Scene classification (one-hot)
534-547   (14 dims)  - Climate attributes + brightness/color
548-931   (384 dims) - Premise embeddings
```

**Why 5 modalities?**
- CLIP captures semantic visual content
- YOLO grounds concepts in real objects
- Scene provides environmental context
- Climate attributes detect domain-specific signals
- Premises understand the input statement

**Total: 512 + 18 + 4 + 14 + 384 = 932 dimensions**

---

## 🚀 Next Steps to Improve Results

### This Week:
- [ ] Examine individual samples in Sample Explorer
- [ ] Analyze metrics in Metrics Dashboard
- [ ] Understand feature space in Feature Analysis

### Next Week:
- [ ] Collect more climate images (100+)
- [ ] Get multiple conclusions per image (3-5 each)
- [ ] Add dropout regularization to model

### Next Month:
- [ ] Switch to transformer architecture
- [ ] Use pre-trained language models (DistilBERT)
- [ ] Implement beam search for diversity
- [ ] Get expert evaluation

---

## 📝 Dashboard Files

**Streamlit App:**
- `src/visualize_results.py` (389 lines)

**Data Files Used:**
- `inference_results/generated_conclusions.json`
- `inference_results/evaluation_metrics.json`
- `dataset/unified_features/unified_features.npz`
- `dataset/unified_features/targets.json`
- `dataset/unified_features/retrieved_facts.json`
- `dataset/images_for_annotation/*.jpg`

**Model Used:**
- `models/argument_model.pt` (trained weights)
- `models/vocabulary.json` (token mappings)

---

## 🔗 Quick Links

- **Dashboard URL:** http://localhost:8501
- **Code:** `/Users/poures/Desktop/PC/image-arg/src/visualize_results.py`
- **Documentation:** `PROJECT_GUIDE.md` (complete system documentation)
- **Quick Ref:** `QUICK_REFERENCE.md` (visual quick start)

---

## 📞 Troubleshooting

**Dashboard won't load?**
```bash
# Restart the app
cd /Users/poures/Desktop/PC/image-arg
streamlit run src/visualize_results.py
```

**Images not showing?**
- Check that images exist in `dataset/images_for_annotation/`
- Verify file names match sample indices (0.jpg, 10.jpg, etc.)

**Data loading error?**
- Ensure unified features exist: `dataset/unified_features/unified_features.npz`
- Check inference results: `inference_results/generated_conclusions.json`

---

**Status:** ✅ Dashboard Live | 🎯 Ready to Explore | 📊 4 Interactive Views
