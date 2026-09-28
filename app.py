"""
app.py — DocIQ: Advanced Intelligent Document Processing (IDP) Platform
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Light Theme Edition with Live Camera Optimization & Comprehensive OCR Accuracy Metrics
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io
import os
import sys

# Make src importable
sys.path.insert(0, os.path.dirname(__file__))

import importlib

# Ensure fresh imports across Streamlit hot reloads
for mod in ["src.preprocessor", "src.ocr_engine", "src.visualizer", "src.extractor", "src.embedder", "src.exporter", "src.sample_docs"]:
    if mod in sys.modules:
        try:
            importlib.reload(sys.modules[mod])
        except Exception:
            pass

from src.preprocessor import full_preprocess

try:
    from src.ocr_engine import run_ocr, get_full_text, compute_accuracy_metrics
except ImportError:
    if "src.ocr_engine" in sys.modules:
        del sys.modules["src.ocr_engine"]
    import src.ocr_engine
    from src.ocr_engine import run_ocr, get_full_text, compute_accuracy_metrics
from src.visualizer import draw_bounding_boxes, search_and_highlight, create_confidence_heatmap, draw_reading_order
from src.extractor import extract_all, redact_pii
from src.embedder import build_embedding_data
from src.exporter import (
    export_json,
    export_csv,
    export_txt,
    export_searchable_pdf,
    export_annotated_pdf,
    export_bundle_zip,
)
from src.sample_docs import SAMPLE_DOCUMENTS


# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="DocIQ — Intelligent Document Processing",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS: Modern Clean Light Theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    box-sizing: border-box;
}

/* Force light background on all app containers */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background: #f8fafc !important;
    background-color: #f8fafc !important;
    color: #0f172a !important;
}

.stApp > header { background: #f8fafc !important; }

/* Force light text color on ALL elements */
*, p, span, label, h1, h2, h3, h4, h5, h6, div {
    color: #1e293b;
}
div.stMarkdown, [data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] p,
[data-testid="stMarkdownContainer"] p {
    color: #1e293b !important;
}

/* Sidebar Light Theme */
section[data-testid="stSidebar"], [data-testid="stSidebarContent"], [data-testid="stSidebarNav"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
    box-shadow: 2px 0 16px rgba(15, 23, 42, 0.03) !important;
}
section[data-testid="stSidebar"] *,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {
    color: #334155 !important;
}

/* Toggle / Checkbox fix */
[data-testid="stToggle"] label, [data-testid="stToggle"] p,
[data-testid="stCheckbox"] label, [data-testid="stCheckbox"] p {
    color: #334155 !important;
    font-weight: 500 !important;
}

/* Multiselect selected items text */
[data-baseweb="tag"] span { color: #4338ca !important; }

/* Slider value label */
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"],
[data-testid="stSlider"] p { color: #334155 !important; }

/* Tab labels – always visible */
.stTabs [role="tab"] span, .stTabs [role="tab"] p {
    color: #64748b !important;
}
.stTabs [aria-selected="true"] span,
.stTabs [aria-selected="true"] p {
    color: #ffffff !important;
}

/* Expander header */
[data-testid="stExpander"] summary p {
    color: #1e293b !important;
    font-weight: 600 !important;
}

/* Success / warning / error message text */
[data-testid="stAlert"] p, [data-testid="stAlert"] div {
    color: #1e293b !important;
}

/* Cards */
.doc-card {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 1.3rem 1.5rem !important;
    margin-bottom: 1rem !important;
    box-shadow: 0 4px 14px -2px rgba(15, 23, 42, 0.05), 0 2px 6px -1px rgba(15, 23, 42, 0.02) !important;
    transition: all 0.25s ease !important;
}
.doc-card:hover {
    border-color: #cbd5e1 !important;
    box-shadow: 0 8px 24px -4px rgba(15, 23, 42, 0.08) !important;
}

/* Hero Header */
.hero-header {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #4338ca 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.8px !important;
    line-height: 1.15 !important;
}

.hero-sub {
    color: #64748b !important;
    font-size: 1.05rem !important;
    font-weight: 400 !important;
    margin-top: 0.3rem !important;
    margin-bottom: 1.5rem !important;
}

/* Badges */
.badge {
    display: inline-block !important;
    padding: 0.3rem 0.75rem !important;
    border-radius: 999px !important;
    font-size: 0.74rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}
.badge-green { background: #dcfce7 !important; color: #15803d !important; border: 1px solid #bbf7d0 !important; }
.badge-yellow { background: #fef9c3 !important; color: #a16207 !important; border: 1px solid #fef08a !important; }
.badge-red { background: #fee2e2 !important; color: #b91c1c !important; border: 1px solid #fecaca !important; }
.badge-indigo { background: #e0e7ff !important; color: #4338ca !important; border: 1px solid #c7d2fe !important; }
.badge-cyan { background: #e0f2fe !important; color: #0369a1 !important; border: 1px solid #bae6fd !important; }

/* Metric Cards */
.metric-card {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    padding: 1.1rem 1rem !important;
    text-align: center !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
    transition: all 0.2s ease !important;
}
.metric-card:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08) !important;
    border-color: #cbd5e1 !important;
}
.metric-value { font-size: 1.85rem !important; font-weight: 800 !important; color: #4338ca !important; line-height: 1.1 !important; }
.metric-label { font-size: 0.75rem !important; color: #64748b !important; margin-top: 0.3rem !important; text-transform: uppercase !important; font-weight: 600 !important; letter-spacing: 0.05em !important; }

/* Accuracy Meter Box */
.accuracy-banner {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 16px !important;
    padding: 1.3rem 1.6rem !important;
    margin-bottom: 1.2rem !important;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05) !important;
}

/* Tabs */
.stTabs [role="tablist"] {
    gap: 6px !important;
    background: #ffffff !important;
    border-radius: 12px !important;
    padding: 5px !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03) !important;
}
.stTabs [role="tab"] {
    border-radius: 8px !important;
    color: #64748b !important;
    font-weight: 600 !important;
    padding: 8px 18px !important;
    border: none !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: #4f46e5 !important;
    color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(79, 70, 229, 0.3) !important;
}

/* Primary Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 3px 10px rgba(79, 70, 229, 0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(79, 70, 229, 0.35) !important;
    color: #ffffff !important;
}

/* Text Inputs & Textareas */
textarea, input[type="text"], input[type="number"], .stTextInput > div > div > input {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
}
textarea:focus, input[type="text"]:focus, .stTextInput > div > div > input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
}

/* Selectbox & Multiselect */
[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
}
[data-baseweb="popover"], [data-baseweb="menu"], ul[role="listbox"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1) !important;
}
li[role="option"] {
    background-color: #ffffff !important;
    color: #0f172a !important;
}
li[role="option"]:hover, li[aria-selected="true"] {
    background-color: #eef2ff !important;
    color: #4338ca !important;
}
/* Multiselect Tag Chips */
[data-baseweb="tag"] {
    background-color: #e0e7ff !important;
    border: 1px solid #c7d2fe !important;
    border-radius: 8px !important;
}
[data-baseweb="tag"] span {
    color: #4338ca !important;
    font-weight: 600 !important;
}
[data-baseweb="tag"] svg {
    fill: #4338ca !important;
}

/* Radio & Checkbox */
[data-testid="stRadio"] label, [data-testid="stCheckbox"] label {
    color: #334155 !important;
    font-weight: 600 !important;
}

/* Slider Track & Thumb */
[data-testid="stSlider"] div[role="slider"] {
    background-color: #4f46e5 !important;
    border: 2px solid #ffffff !important;
    box-shadow: 0 2px 6px rgba(79, 70, 229, 0.4) !important;
}

/* Dataframe Light Table */
[data-testid="stDataFrame"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
}
[data-testid="stDataFrame"] * {
    color: #0f172a !important;
}

/* File Uploader Light */
[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 2px dashed #cbd5e1 !important;
    border-radius: 16px !important;
    padding: 1.2rem !important;
}
[data-testid="stFileUploader"] section {
    background: #ffffff !important;
}
[data-testid="stFileUploader"] * {
    color: #334155 !important;
}

/* Camera Scanner Light */
[data-testid="stCameraInput"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    padding: 12px !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
}
[data-testid="stCameraInput"] * {
    color: #0f172a !important;
}

/* Code & Alert Elements */
code {
    background: #f1f5f9 !important;
    color: #4338ca !important;
    padding: 2px 6px !important;
    border-radius: 6px !important;
    border: 1px solid #e2e8f0 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 7px; height: 7px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Sidebar Controls
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.8rem 0 0.4rem;">
        <div style="font-size:2.4rem;">🧠</div>
        <div style="font-size:1.3rem; font-weight:800; color:#1e1b4b; letter-spacing:-0.5px;">DocIQ</div>
        <div style="font-size:0.75rem; color:#64748b; font-weight:500;">Intelligent Document Processing</div>
    </div>
    <hr style="margin: 0.8rem 0; border-color:#e2e8f0;">
    """, unsafe_allow_html=True)

    st.markdown("#### ⚙️ OCR Settings")
    languages = st.multiselect(
        "Languages",
        ["en", "hi", "fr", "de", "es", "zh_sim", "ar"],
        default=["en", "hi"],
        help="Select languages present in your document (en=English, hi=Hindi)"
    )

    confidence_threshold = st.slider(
        "Confidence Filter",
        min_value=0.10, max_value=0.95, value=0.25, step=0.05,
        help="Discard text detected below this confidence score"
    )

    st.markdown("#### 🖼️ Image Preprocessing")
    preprocess_mode = st.selectbox(
        "Processing Mode",
        [
            "raw",
            "camera_enhance",
            "auto",
            "binarize",
            "warp",
        ],
        format_func=lambda x: {
            "raw": "🖼️ Original (No Preprocessing)",
            "camera_enhance": "📸 Camera & Mobile Capture (Shadows & Blur Fix)",
            "auto": "⚡ Auto Enhancement (Deskew + CLAHE)",
            "binarize": "⚪ High Contrast Black & White",
            "warp": "📐 Perspective Flatten (4-Corner)",
        }.get(x, x),
        index=0,
        help="Use 'Camera & Mobile Capture' for live camera photos with shadows or tilt."
    )

    camera_opt = st.checkbox(
        "📸 Two-Pass Live Photo Optimization",
        value=True,
        help="Runs multi-pass recognition with high sensitivity for difficult lighting, uneven angles, or camera glare."
    )

    st.markdown("#### 🛡️ Privacy & Compliance")
    auto_redact = st.toggle("Auto-Redact PII (Aadhaar, PAN, Cards)", value=False)

    st.markdown("#### 🔢 Semantic Settings")
    embed_unit = st.selectbox("Text Unit", ["word", "sentence", "line"], index=0)
    embed_reduction = st.selectbox("Reduction Method", ["PCA", "t-SNE"], index=0)
    embed_clusters = st.slider("K-Means Clusters", 2, 8, 4)

    st.markdown("<hr style='margin:1rem 0; border-color:#e2e8f0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.75rem; color:#64748b; text-align:center; line-height:1.5;">
        <strong>GPU Accelerated</strong> · EasyOCR Engine<br>
        OpenCV Preprocessing · Sentence-Transformers
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Main Header
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero-header">DocIQ — Intelligent Document Processing</div>
<div class="hero-sub">
    Capture or upload any document → Auto-enhance lighting & blur → Extract text & entities → Measure accuracy → Export
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Multi-Input Modes: Upload | Preset Sample | Camera Scanner
# ──────────────────────────────────────────────
input_tab1, input_tab2, input_tab3 = st.tabs([
    "📁 Upload Image",
    "✨ Preset Sample Documents",
    "📸 Live Camera Scanner",
])

pil_img = None
doc_name = "document.png"
is_camera_input = False

with input_tab1:
    uploaded_file = st.file_uploader(
        "Drop a document image here (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload scanned receipts, invoices, IDs, medical notes, or book pages.",
        key="file_uploader_input"
    )
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        doc_name = uploaded_file.name

with input_tab2:
    st.markdown("""
    <div style="color:#64748b; font-size:0.88rem; margin-bottom: 0.8rem;">
        No document image on hand? Click a pre-configured sample to test the complete OCR, accuracy metrics, and export pipeline:
    </div>
    """, unsafe_allow_html=True)

    sc1, sc2 = st.columns(2)
    for idx, (s_name, s_fn) in enumerate(SAMPLE_DOCUMENTS.items()):
        target_col = sc1 if idx % 2 == 0 else sc2
        with target_col:
            st.markdown(f"""
            <div class="doc-card" style="padding:1.1rem;">
                <div style="font-weight:700; color:#1e1b4b; font-size:1.05rem; margin-bottom:4px;">📄 {s_name}</div>
                <div style="color:#64748b; font-size:0.8rem; margin-bottom:12px;">Includes line items, monetary values, dates, and contact data.</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"⚡ Load {s_name}", key=f"btn_sample_{idx}", use_container_width=True):
                st.session_state["active_sample"] = s_name

    active_s = st.session_state.get("active_sample")
    if active_s and active_s in SAMPLE_DOCUMENTS and pil_img is None:
        pil_img = SAMPLE_DOCUMENTS[active_s]()
        doc_name = f"{active_s.lower().replace(' ', '_')}.png"
        st.success(f"✓ Loaded sample document: **{active_s}**")

with input_tab3:
    st.markdown("""
    <div style="background:#eff6ff; border:1px solid #bfdbfe; border-radius:12px; padding:1rem 1.2rem; margin-bottom:1rem;">
        <div style="font-weight:700; color:#1d4ed8; font-size:0.95rem; margin-bottom:4px;">📸 Live Camera Scanner</div>
        <div style="color:#1e40af; font-size:0.84rem; line-height:1.5;">
            Click <strong>Enable Camera</strong> to turn on your webcam, take a photo of any document,
            and DocIQ will extract all text from it. The image is shown in its natural colors — OCR runs on the original.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Camera is OFF by default — only enable when user explicitly clicks the button
    if "camera_enabled" not in st.session_state:
        st.session_state["camera_enabled"] = False

    col_cam_btn, col_cam_info = st.columns([1, 3])
    with col_cam_btn:
        if not st.session_state["camera_enabled"]:
            if st.button("📷 Enable Camera", key="btn_enable_camera", use_container_width=True):
                st.session_state["camera_enabled"] = True
                st.rerun()
        else:
            if st.button("🔴 Disable Camera", key="btn_disable_camera", use_container_width=True):
                st.session_state["camera_enabled"] = False
                st.session_state.pop("camera_capture", None)
                st.rerun()
    with col_cam_info:
        if st.session_state["camera_enabled"]:
            st.markdown("""
            <div style="color:#15803d; font-size:0.84rem; padding:0.4rem 0; font-weight:600;">
                ✅ Camera is ON — point at your document and click the capture button below
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="color:#64748b; font-size:0.84rem; padding:0.4rem 0;">
                Camera is off. Click Enable Camera to start your webcam.
            </div>""", unsafe_allow_html=True)

    if st.session_state["camera_enabled"]:
        cam_file = st.camera_input(
            "📷 Point at your document and capture",
            key="camera_scanner_input",
            help="Position the document clearly, ensure good lighting, then click the capture button."
        )
        if cam_file is not None:
            pil_img = Image.open(cam_file).convert("RGB")
            doc_name = "camera_capture.png"
            is_camera_input = True
            # Show preview of the natural captured image
            st.markdown("""
            <div style="color:#15803d; font-weight:600; font-size:0.9rem; margin:0.5rem 0 0.3rem;">✅ Image captured! OCR will run on the natural original image below:</div>
            """, unsafe_allow_html=True)
            st.image(pil_img, caption="📸 Captured Document (Natural Colors — OCR input)", use_container_width=True)


# ──────────────────────────────────────────────
# Empty State: Landing Feature Grid
# ──────────────────────────────────────────────
if pil_img is None:
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(4)
    features = [
        ("🎯", "OCR Accuracy Gauge", "Precise character & word reliability scores with quality grade"),
        ("📸", "Live Photo Ready", "Shadow removal, blur sharpening, and EXIF orientation fix"),
        ("🔍", "Interactive Canvas", "Confidence color bounding boxes, reading order, and heatmaps"),
        ("🛡️", "PII Redaction", "Auto-detect Aadhaar, PAN, credit cards with 1-click masking"),
        ("📊", "Structured Parsing", "Auto-classify document and extract dates, amounts, invoices"),
        ("🧬", "Embedding Space", "Cluster words in 2D vector space with similarity matrix"),
        ("📄", "Searchable PDF", "Export with invisible text layer for native Ctrl+F search"),
        ("📦", "All-In-One ZIP", "Single-click bundle with PDFs, JSON, CSV, and TXT"),
    ]
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="doc-card" style="min-height:130px;">
                <div style="font-size:1.8rem; margin-bottom:8px;">{icon}</div>
                <div style="font-weight:700; color:#0f172a; font-size:0.95rem;">{title}</div>
                <div style="color:#64748b; font-size:0.78rem; margin-top:4px; line-height:1.45;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    st.stop()


# ──────────────────────────────────────────────
# Show Natural Image Preview (always original colors)
# ──────────────────────────────────────────────
if not is_camera_input:  # Camera tab already shows its own preview
    st.markdown("""
    <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:12px;
                padding:0.8rem 1.2rem; margin-bottom:0.8rem; display:flex; align-items:center; gap:10px;">
        <span style="font-size:1.2rem;">🖼️</span>
        <div>
            <div style="font-weight:700; color:#15803d; font-size:0.9rem;">Image loaded — Natural original preview</div>
            <div style="color:#166534; font-size:0.8rem;">OCR is applied to the original image. Enhancement only runs internally to improve text detection accuracy.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.image(pil_img, caption=f"📄 {doc_name} — Original (Natural Colors)", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Preprocessing Pipeline (for OCR accuracy only)
# ──────────────────────────────────────────────
with st.spinner("🎨 Enhancing image for OCR inference..."):
    # Use user-selected mode; camera_enhance only activates if user explicitly picks it
    effective_mode = preprocess_mode
    processed_img, preprocess_meta = full_preprocess(pil_img, mode=effective_mode)


# ──────────────────────────────────────────────
# OCR Neural Inference
# ──────────────────────────────────────────────
with st.spinner("🧠 Running neural OCR inference (GPU accelerated)..."):
    ocr_results = run_ocr(
        processed_img,
        languages=languages,
        confidence_threshold=confidence_threshold,
        camera_mode=(camera_opt or is_camera_input)
    )

if not ocr_results:
    st.warning("⚠️ No text detected. Try switching the processing mode to 'Camera & Mobile Capture' in the sidebar or lowering the confidence filter.")
    st.stop()

full_text = get_full_text(ocr_results)
accuracy_stats = compute_accuracy_metrics(ocr_results, full_text=full_text)

# PII Redaction
redacted_text, pii_count = redact_pii(full_text) if auto_redact else (full_text, 0)
display_text = redacted_text if auto_redact else full_text

# Entity Extraction
with st.spinner("🔍 Extracting structured entities..."):
    entities = extract_all(full_text)


# ──────────────────────────────────────────────
# Accuracy & Quality Banner (Light Theme)
# ──────────────────────────────────────────────
acc_val = accuracy_stats["overall_accuracy"]
acc_grade = accuracy_stats["quality_grade"]
acc_label = accuracy_stats["reliability_label"]
acc_color = accuracy_stats["reliability_color"]

st.markdown(f"""
<div class="accuracy-banner">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px;">
        <div style="display:flex; align-items:center; gap:18px;">
            <div style="width:68px; height:68px; border-radius:50%; background:{acc_color}18; border:3px solid {acc_color}; display:flex; align-items:center; justify-content:center;">
                <span style="font-size:1.6rem; font-weight:800; color:{acc_color};">{acc_grade}</span>
            </div>
            <div>
                <div style="font-size:0.78rem; text-transform:uppercase; font-weight:700; color:#64748b; letter-spacing:0.05em;">
                    OCR Accuracy & Quality Score
                </div>
                <div style="font-size:1.8rem; font-weight:800; color:#0f172a; line-height:1.2;">
                    {acc_val}% <span style="font-size:1rem; font-weight:600; color:{acc_color}; margin-left:6px;">• {acc_label}</span>
                </div>
            </div>
        </div>
        <div style="display:flex; gap:16px; align-items:center; flex-wrap:wrap;">
            <div style="text-align:right;">
                <div style="font-size:0.75rem; color:#64748b; font-weight:600;">High Confidence (≥80%)</div>
                <div style="font-size:1.15rem; font-weight:700; color:#15803d;">{accuracy_stats['high_conf_pct']}% ({accuracy_stats['high_conf_count']} words)</div>
            </div>
            <div style="text-align:right; border-left:1px solid #e2e8f0; padding-left:16px;">
                <div style="font-size:0.75rem; color:#64748b; font-weight:600;">Lexical Validity</div>
                <div style="font-size:1.15rem; font-weight:700; color:#4338ca;">{accuracy_stats['lexical_validity']}%</div>
            </div>
        </div>
    </div>
    <!-- Confidence Distribution Bar -->
    <div style="margin-top:14px; background:#e2e8f0; height:8px; border-radius:999px; overflow:hidden; display:flex;">
        <div style="width:{accuracy_stats['high_conf_pct']}%; background:#22c55e;" title="High confidence"></div>
        <div style="width:{accuracy_stats['medium_conf_pct']}%; background:#eab308;" title="Moderate confidence"></div>
        <div style="width:{accuracy_stats['low_conf_pct']}%; background:#ef4444;" title="Low confidence"></div>
    </div>
    <div style="display:flex; justify-content:space-between; margin-top:6px; font-size:0.72rem; color:#64748b;">
        <span>🟢 High: {accuracy_stats['high_conf_pct']}%</span>
        <span>🟡 Medium: {accuracy_stats['medium_conf_pct']}%</span>
        <span>🔴 Low: {accuracy_stats['low_conf_pct']}%</span>
        <span>Total: {accuracy_stats['total_words']} words ({accuracy_stats['total_characters']} characters)</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Metrics Grid
# ──────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
metrics = [
    (m1, "Recognized Words", accuracy_stats["total_words"]),
    (m2, "Avg Confidence", f"{accuracy_stats['avg_confidence']:.1%}"),
    (m3, "High Quality Words", accuracy_stats["high_conf_count"]),
    (m4, "PII Detected", pii_count if auto_redact else "—"),
    (m5, "Document Type", entities.get("document_type", "General")[:18]),
]
for col, label, value in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Main Application Tabs (Light Mode)
# ──────────────────────────────────────────────
tab_vis, tab_search, tab_text, tab_entities, tab_embed, tab_export = st.tabs([
    "🖼️ Visual Canvas",
    "🔎 Visual Keyword Search",
    "📝 Extracted Text & Confidence",
    "🗂️ Entities & PII",
    "🧬 Embedding Space",
    "📤 Multi-Format Export",
])


# ────────────────
# TAB 1: Visual Canvas
# ────────────────
with tab_vis:
    c1, c2 = st.columns([3, 1])
    with c2:
        st.markdown("##### 🎨 Canvas Layer")
        view_mode = st.radio(
            "Canvas View",
            ["Bounding Boxes", "Confidence Heatmap", "Reading Order Flow", "Side-by-Side Comparison"],
            index=0,
            label_visibility="collapsed"
        )
        st.markdown("---")
        show_labels = st.toggle("Show Word Labels", value=True)
        show_conf = st.toggle("Show Confidence %", value=True)

        st.markdown("""
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:12px; margin-top:16px;">
            <div style="font-size:0.75rem; font-weight:700; color:#334155; margin-bottom:6px;">COLOR LEGEND</div>
            <div style="font-size:0.78rem; color:#15803d; margin-bottom:4px;">🟢 Green: ≥ 80% Confidence</div>
            <div style="font-size:0.78rem; color:#a16207; margin-bottom:4px;">🟡 Yellow: 50% – 79%</div>
            <div style="font-size:0.78rem; color:#b91c1c;">🔴 Red: &lt; 50% Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    with c1:
        if view_mode == "Bounding Boxes":
            # Always draw overlays on the original image to preserve true colors
            annotated = draw_bounding_boxes(
                pil_img, ocr_results,
                show_labels=show_labels,
                show_confidence=show_conf,
            )
            st.image(annotated, use_container_width=True, caption="Bounding Boxes with Confidence Color Coding")
        elif view_mode == "Confidence Heatmap":
            heatmap_img = create_confidence_heatmap(pil_img, ocr_results)
            st.image(heatmap_img, use_container_width=True, caption="Confidence Heatmap (Blue=High Certainty, Red=Low Certainty)")
        elif view_mode == "Reading Order Flow":
            order_img = draw_reading_order(pil_img, ocr_results)
            st.image(order_img, use_container_width=True, caption="Reconstructed Reading Order with Directional Arrows")
        else:
            col_o, col_p = st.columns(2)
            with col_o:
                st.image(pil_img, caption="Original Input", use_container_width=True)
            with col_p:
                st.image(processed_img, caption=f"Enhanced Preprocessed ({effective_mode})", use_container_width=True)

    # Preprocessing tags
    st.markdown(f"""
    <div class="doc-card" style="margin-top:0.8rem; padding:0.8rem 1.2rem;">
        <span style="color:#64748b; font-size:0.82rem; font-weight:600;">Active Preprocessing Enhancements: </span>
        {''.join(f'<span class="badge badge-cyan" style="margin-left:6px;">{s.replace("_", " ").title()}</span>' for s in preprocess_meta.get('steps', []))}
    </div>
    """, unsafe_allow_html=True)


# ────────────────
# TAB 2: Visual Keyword Search
# ────────────────
with tab_search:
    search_col, opts_col = st.columns([3, 1])
    with opts_col:
        case_sensitive = st.toggle("Case Sensitive Search", value=False)
    with search_col:
        keyword = st.text_input(
            "Search document keyword",
            placeholder="Type any word or phrase (e.g. invoice, total, patient)...",
            label_visibility="collapsed"
        )

    if keyword:
        # Use original image for search highlights to preserve true colors
        highlighted_img, matched_idx, matched_texts = search_and_highlight(
            pil_img, ocr_results, keyword, case_sensitive=case_sensitive
        )

        if matched_texts:
            st.markdown(f"""
            <div class="doc-card" style="background:#f0fdf4; border-color:#bbf7d0;">
                <span style="color:#15803d; font-weight:700; font-size:1rem;">✓ Found {len(matched_texts)} matching occurrence(s)</span>
                <div style="margin-top:8px;">
                    {'  '.join(f'<span class="badge badge-yellow" style="margin-right:4px;">{t}</span>' for t in matched_texts[:20])}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="doc-card" style="background:#fef2f2; border-color:#fecaca;">
                <span style="color:#b91c1c; font-weight:700;">✗ No text matching "{keyword}" was found in the document.</span>
            </div>
            """, unsafe_allow_html=True)

        st.image(highlighted_img, use_container_width=True, caption=f"Spatial Search Overlay: '{keyword}' highlighted in gold")
    else:
        st.info("👆 Enter a word or phrase above to instantly highlight its exact physical position on the document.")
        st.image(processed_img, use_container_width=True, caption="Document Preview")


# ────────────────
# TAB 3: Extracted Text & Quality Table
# ────────────────
with tab_text:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"""<div class="doc-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-weight:700; color:#0f172a; font-size:1.05rem;">
                    Extracted Text Content
                    {'<span class="badge badge-red" style="margin-left:8px;">PII REDACTED</span>' if auto_redact and pii_count > 0 else ''}
                </span>
                <span style="font-size:0.8rem; color:#64748b; font-weight:600;">
                    {len(display_text.split())} words · {len(display_text)} characters
                </span>
            </div>
        </div>""", unsafe_allow_html=True)
        st.text_area(
            "Extracted Text Content",
            value=display_text,
            height=400,
            label_visibility="collapsed",
            help="Select all and copy with Ctrl+C, or edit freely."
        )

    with c2:
        st.markdown("#### 📊 Word Recognition Table")
        df = pd.DataFrame([{
            "Text": r["text"],
            "Confidence": f"{r['confidence']:.1%}",
            "Width": r["width"],
            "Height": r["height"],
        } for r in sorted(ocr_results, key=lambda x: x["confidence"], reverse=True)])
        st.dataframe(df, use_container_width=True, height=430)


# ────────────────
# TAB 4: Entities & PII
# ────────────────
with tab_entities:
    doc_type = entities.get("document_type", "General Document")
    st.markdown(f"""
    <div class="doc-card" style="background:linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%); border-color:#c7d2fe;">
        <span style="color:#4338ca; font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em;">CLASSIFIED DOCUMENT TYPE</span>
        <div style="font-size:1.6rem; font-weight:800; color:#1e1b4b; margin-top:4px;">📋 {doc_type}</div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # Dates
        dates = entities.get("dates", [])
        st.markdown(f"""<div class="doc-card">
            <div style="font-weight:700; color:#0f172a; margin-bottom:8px;">📅 Detected Dates ({len(dates)})</div>
            {''.join(f'<span class="badge badge-cyan" style="margin:2px 3px;">{d}</span>' for d in dates) if dates else '<span style="color:#94a3b8; font-size:0.84rem;">None detected</span>'}
        </div>""", unsafe_allow_html=True)

        # Amounts
        amounts = entities.get("amounts", [])
        st.markdown(f"""<div class="doc-card">
            <div style="font-weight:700; color:#0f172a; margin-bottom:8px;">💰 Monetary Amounts ({len(amounts)})</div>
            {''.join(f'<span class="badge badge-green" style="margin:2px 3px;">{a}</span>' for a in amounts) if amounts else '<span style="color:#94a3b8; font-size:0.84rem;">None detected</span>'}
        </div>""", unsafe_allow_html=True)

        # Contact Info
        emails = entities.get("emails", [])
        phones = entities.get("phones", [])
        urls = entities.get("urls", [])
        st.markdown(f"""<div class="doc-card">
            <div style="font-weight:700; color:#0f172a; margin-bottom:8px;">📬 Contact Information</div>
            {''.join(f'<span class="badge badge-indigo" style="margin:2px 3px;">✉ {e}</span>' for e in emails)}
            {''.join(f'<span class="badge badge-indigo" style="margin:2px 3px;">📞 {p}</span>' for p in phones)}
            {''.join(f'<span class="badge badge-cyan" style="margin:2px 3px;">🔗 {u[:28]}...</span>' for u in urls[:3])}
            {'<span style="color:#94a3b8; font-size:0.84rem;">None detected</span>' if not emails and not phones and not urls else ''}
        </div>""", unsafe_allow_html=True)

    with col_b:
        # Key Document Fields
        inv = entities.get("invoice_fields", {})
        FIELD_ICONS = {
            "invoice_number": "🧾",
            "po_number": "📋",
            "total": "💰",
            "subtotal": "💵",
            "tax": "🏛️",
            "discount": "🏷️",
            "gstin": "🏢",
            "due_date": "⏳",
            "vendor": "🏬",
        }
        if inv:
            inv_rows = []
            for k, v in inv.items():
                icon = FIELD_ICONS.get(k, "📌")
                title = k.replace('_', ' ').title()
                inv_rows.append(f"""
                <div style="display:flex; justify-content:space-between; align-items:center; padding:10px 14px; margin-bottom:8px; background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;">
                    <span style="color:#475569; font-size:0.86rem; display:flex; align-items:center; gap:8px;">
                        <span>{icon}</span>
                        <strong style="color:#1e293b; font-weight:600;">{title}</strong>
                    </span>
                    <span style="color:#0369a1; font-size:0.88rem; font-weight:700; font-family:monospace; background:#e0f2fe; padding:3px 10px; border-radius:6px; border:1px solid #bae6fd;">
                        {v}
                    </span>
                </div>""")
            inv_html = "".join(inv_rows)
            st.markdown(f"""<div class="doc-card">
                <div style="font-weight:700; color:#0f172a; margin-bottom:12px; display:flex; align-items:center; justify-content:space-between;">
                    <span>🧾 Extracted Key-Value Fields</span>
                    <span class="badge badge-indigo">{len(inv)} detected</span>
                </div>
                {inv_html}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="doc-card">
                <div style="font-weight:700; color:#0f172a; margin-bottom:6px;">🧾 Key-Value Fields</div>
                <div style="color:#94a3b8; font-size:0.84rem;">No invoice or purchase order key-value fields detected.</div>
            </div>""", unsafe_allow_html=True)

        # PII Detected
        pii = entities.get("pii", {})
        pii_found = {k: v for k, v in pii.items() if v}
        if pii_found:
            pii_html = "".join(f"""
            <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #fee2e2;">
                <span style="color:#b91c1c; font-size:0.86rem; font-weight:600;">⚠ {k.upper()}</span>
                <span style="color:#b91c1c; font-size:0.84rem; font-weight:700;">{len(v)} detected</span>
            </div>""" for k, v in pii_found.items())
            st.markdown(f"""<div class="doc-card" style="border-color:#fecaca; background:#fff5f5;">
                <div style="font-weight:700; color:#b91c1c; margin-bottom:8px;">🛡️ Sensitive PII Detected</div>
                {pii_html}
                <div style="margin-top:10px; font-size:0.78rem; color:#64748b;">
                    Enable "Auto-Redact PII" in the sidebar to mask these identifiers before export.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="doc-card" style="border-color:#bbf7d0; background:#f0fdf4;">
                <div style="font-weight:700; color:#15803d; margin-bottom:4px;">✅ No Sensitive PII Found</div>
                <div style="font-size:0.82rem; color:#64748b;">No Aadhaar, PAN, SSN, or credit card patterns detected.</div>
            </div>""", unsafe_allow_html=True)


# ────────────────
# TAB 5: Embedding Space (Light Theme)
# ────────────────
with tab_embed:
    st.markdown("""<div class="doc-card">
        <div style="font-weight:700; color:#0f172a; margin-bottom:4px;">🧬 Semantic Vector Space Visualization</div>
        <div style="color:#64748b; font-size:0.85rem;">
            Converts text units into 384-dimensional dense semantic vectors using <code>all-MiniLM-L6-v2</code>, 
            then projects to 2D space. Words with related meanings cluster together automatically.
        </div>
    </div>""", unsafe_allow_html=True)

    if st.button("⚡ Generate Embedding Space", type="primary"):
        with st.spinner("🔢 Calculating sentence embeddings and 2D projection..."):
            embed_data = build_embedding_data(
                ocr_results,
                text_unit=embed_unit,
                reduction=embed_reduction,
                n_clusters=embed_clusters,
            )

        if embed_data and embed_data.get("coords_2d") is not None and len(embed_data["coords_2d"]) > 0:
            coords = np.array(embed_data["coords_2d"])
            texts = embed_data["texts"]
            labels = embed_data["cluster_labels"]

            CLUSTER_COLORS = [
                "#4f46e5", "#0284c7", "#16a34a", "#ca8a04", "#ea580c",
                "#db2777", "#7c3aed", "#059669", "#d97706", "#2563eb"
            ]

            fig = go.Figure()
            unique_labels = sorted(set(labels))
            for cluster_id in unique_labels:
                idx = [i for i, l in enumerate(labels) if l == cluster_id]
                cluster_texts = [texts[i] for i in idx]
                x_vals = [coords[i, 0] for i in idx]
                y_vals = [coords[i, 1] for i in idx]
                color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]

                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="markers+text",
                    name=f"Cluster {cluster_id + 1}",
                    marker=dict(
                        color=color,
                        size=10,
                        opacity=0.9,
                        line=dict(width=1.5, color="#ffffff"),
                    ),
                    text=[t[:18] for t in cluster_texts],
                    textposition="top center",
                    textfont=dict(size=10, color="#1e293b"),
                    hovertemplate="<b>%{customdata}</b><br>x=%{x:.2f}, y=%{y:.2f}<extra></extra>",
                    customdata=cluster_texts,
                ))

            fig.update_layout(
                title=dict(
                    text=f"<b>Semantic Embedding Space</b> — {embed_unit.title()}s via {embed_reduction}",
                    font=dict(size=16, color="#0f172a"),
                ),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#f8fafc",
                font=dict(color="#334155", size=11),
                xaxis=dict(
                    title=f"{embed_reduction} Axis 1",
                    gridcolor="#e2e8f0",
                    zerolinecolor="#cbd5e1",
                    title_font_color="#64748b",
                ),
                yaxis=dict(
                    title=f"{embed_reduction} Axis 2",
                    gridcolor="#e2e8f0",
                    zerolinecolor="#cbd5e1",
                    title_font_color="#64748b",
                ),
                legend=dict(
                    bgcolor="#ffffff",
                    bordercolor="#e2e8f0",
                    borderwidth=1,
                    font=dict(color="#334155"),
                ),
                height=520,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Cosine similarity heatmap
            sim_matrix = embed_data.get("similarity_matrix")
            if sim_matrix is not None and len(texts) <= 35:
                st.markdown("#### 🌡️ Semantic Similarity Heatmap")
                short_labels = [t[:14] + ("…" if len(t) > 14 else "") for t in texts]
                fig_heat = px.imshow(
                    sim_matrix,
                    x=short_labels,
                    y=short_labels,
                    color_continuous_scale="Viridis",
                    zmin=0, zmax=1,
                    aspect="auto",
                    title="<b>Cosine Similarity Heatmap</b>",
                )
                fig_heat.update_layout(
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#f8fafc",
                    font=dict(color="#334155", size=9),
                    height=480,
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.warning("⚠️ Document text is too brief to compute clusters. Try an image with more paragraphs.")
    else:
        st.info("👆 Click the button above to project words into semantic vector space.")


# ────────────────
# TAB 6: Multi-Format Export
# ────────────────
with tab_export:
    st.markdown("""<div class="doc-card">
        <div style="font-weight:700; color:#0f172a; margin-bottom:4px;">📤 Multi-Format Export Center</div>
        <div style="color:#64748b; font-size:0.85rem;">Download your processed document in standard formats or grab the complete archive.</div>
    </div>""", unsafe_allow_html=True)

    base_doc_stem = os.path.splitext(doc_name)[0]

    # One-click bundle ZIP
    st.markdown("##### 📦 All-In-One Bundle")
    if st.button("🗜️ Generate Complete Export Package (.ZIP)", type="primary", use_container_width=True):
        with st.spinner("Bundling Searchable PDF, Audit Report, JSON, CSV, and Plain Text..."):
            zip_bytes = export_bundle_zip(processed_img, ocr_results, entities, display_text, base_name=base_doc_stem)
        st.download_button(
            "⬇️ Download All Formats (.ZIP Archive)",
            data=zip_bytes,
            file_name=f"{base_doc_stem}_complete_export.zip",
            mime="application/zip",
            use_container_width=True,
        )

    st.markdown("<hr style='margin:1.4rem 0; border-color:#e2e8f0;'>", unsafe_allow_html=True)
    st.markdown("##### 📄 Individual Formats")

    ec1, ec2 = st.columns(2)

    with ec1:
        # Searchable PDF
        if st.button("📄 Generate Searchable PDF", use_container_width=True):
            with st.spinner("Generating searchable PDF with invisible text layer..."):
                pdf_bytes = export_searchable_pdf(processed_img, ocr_results, doc_name)
            st.download_button(
                "⬇️ Download Searchable PDF",
                data=pdf_bytes,
                file_name=f"{base_doc_stem}_searchable.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        # Annotated Audit PDF
        if st.button("📊 Generate Annotated Audit Report (PDF)", use_container_width=True):
            with st.spinner("Generating annotated audit PDF report..."):
                annotated_img = draw_bounding_boxes(processed_img, ocr_results, show_labels=True, show_confidence=True)
                pdf_bytes = export_annotated_pdf(annotated_img, ocr_results, entities, title="DocIQ OCR Report")
            st.download_button(
                "⬇️ Download Annotated PDF Report",
                data=pdf_bytes,
                file_name=f"{base_doc_stem}_audit_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        # JSON
        json_bytes = export_json(ocr_results, entities, {"filename": doc_name, "accuracy": accuracy_stats})
        st.download_button(
            "⬇️ Download Structured JSON Data",
            data=json_bytes,
            file_name=f"{base_doc_stem}_data.json",
            mime="application/json",
            use_container_width=True,
        )

    with ec2:
        # CSV
        csv_bytes = export_csv(ocr_results, entities)
        st.download_button(
            "⬇️ Download Words Table (CSV / Excel)",
            data=csv_bytes,
            file_name=f"{base_doc_stem}_words.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Plain Text
        txt_bytes = export_txt(ocr_results, full_text=display_text)
        st.download_button(
            f"⬇️ Download Clean Plain Text {'(Redacted)' if auto_redact else ''}",
            data=txt_bytes,
            file_name=f"{base_doc_stem}_text.txt",
            mime="text/plain",
            use_container_width=True,
        )

        # Entity Summary JSON
        st.download_button(
            "⬇️ Download Entity Summary (JSON)",
            data=export_json([], entities, {"filename": doc_name, "type": "entities_only"}),
            file_name=f"{base_doc_stem}_entities.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="doc-card" style="background:#f8fafc;">
        <div style="font-weight:700; color:#334155; margin-bottom:8px;">📋 Export Formats Reference</div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px;">
            <div><strong style="color:#0284c7;">Searchable PDF</strong><br><span style="color:#64748b; font-size:0.8rem;">Contains hidden text layer. Native Ctrl+F search works in Adobe Acrobat, Chrome, and Preview.</span></div>
            <div><strong style="color:#0284c7;">Annotated PDF Report</strong><br><span style="color:#64748b; font-size:0.8rem;">Visual bounding box document map, accuracy scores, and word-by-word audit trail.</span></div>
            <div><strong style="color:#0284c7;">Structured JSON</strong><br><span style="color:#64748b; font-size:0.8rem;">Precise pixel coordinates, confidence ratings, and parsed entities for downstream pipelines.</span></div>
            <div><strong style="color:#0284c7;">CSV / Excel</strong><br><span style="color:#64748b; font-size:0.8rem;">Tabular dataset ready to import directly into Microsoft Excel, Google Sheets, or Pandas.</span></div>
        </div>
    </div>""", unsafe_allow_html=True)
