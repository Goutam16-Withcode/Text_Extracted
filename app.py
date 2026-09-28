"""
app.py — Advanced Intelligent Document Processing (IDP) Platform
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Features:
  1. OpenCV auto-deskew / contrast / denoising preprocessing
  2. GPU-accelerated EasyOCR with bounding box visualization
  3. Confidence heatmap & reading-order visualization
  4. Interactive keyword search with spatial highlighting
  5. PII detection & one-click redaction
  6. Structured entity extraction (dates, amounts, invoice fields)
  7. Embedding vector space visualization (PCA / t-SNE)
  8. Semantic similarity heatmap
  9. Multi-format export: Searchable PDF, Annotated PDF, JSON, CSV, TXT
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

from src.preprocessor import full_preprocess
from src.ocr_engine import run_ocr, get_full_text, get_statistics
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
    page_title="DocIQ — Advanced Intelligent Document Processing",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS: Dark Premium Theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

/* Dark Background */
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0f172a 50%, #0d1117 100%) !important; }
.stApp > header { background: transparent !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15,23,42,0.95) !important;
    border-right: 1px solid rgba(99,102,241,0.25) !important;
}

/* Cards */
.doc-card {
    background: linear-gradient(145deg, rgba(30,41,59,0.95), rgba(15,23,42,0.98));
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(20px);
    transition: border-color 0.3s ease, transform 0.2s ease;
}
.doc-card:hover {
    border-color: rgba(99,102,241,0.6);
    transform: translateY(-2px);
}

/* Hero header */
.hero-header {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 40%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -1px;
    line-height: 1.1;
}

.hero-sub {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 400;
    margin-top: 0.4rem;
    margin-bottom: 2rem;
}

/* Badge */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-green { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
.badge-yellow { background: rgba(234,179,8,0.15); color: #eab308; border: 1px solid rgba(234,179,8,0.3); }
.badge-red { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.badge-indigo { background: rgba(99,102,241,0.15); color: #818cf8; border: 1px solid rgba(99,102,241,0.3); }
.badge-cyan { background: rgba(6,182,212,0.15); color: #22d3ee; border: 1px solid rgba(6,182,212,0.3); }

/* Metric cards */
.metric-card {
    background: rgba(30,41,59,0.7);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    transition: all 0.25s ease;
}
.metric-card:hover { border-color: rgba(99,102,241,0.5); background: rgba(30,41,59,0.95); }
.metric-value { font-size: 2rem; font-weight: 700; color: #818cf8; line-height: 1; }
.metric-label { font-size: 0.78rem; color: #64748b; margin-top: 0.3rem; text-transform: uppercase; letter-spacing: 0.05em; }

/* Entity chip */
.entity-chip {
    display: inline-block;
    margin: 2px 3px;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 500;
}

/* Text areas */
textarea { background: rgba(15,23,42,0.9) !important; color: #e2e8f0 !important; border: 1px solid rgba(99,102,241,0.3) !important; border-radius: 10px !important; }

/* Tabs */
.stTabs [role="tablist"] { gap: 4px; background: rgba(15,23,42,0.8); border-radius: 12px; padding: 4px; border: 1px solid rgba(99,102,241,0.2); }
.stTabs [role="tab"] { border-radius: 8px !important; color: #94a3b8 !important; font-weight: 500 !important; padding: 6px 16px !important; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #6366f1, #8b5cf6) !important; color: white !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(99,102,241,0.5) !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(15,23,42,0.8) !important;
    border: 2px dashed rgba(99,102,241,0.4) !important;
    border-radius: 16px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 3px; }

/* Divider */
hr { border-color: rgba(99,102,241,0.2) !important; }

/* selectbox / slider labels */
label { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <div style="font-size:2.5rem;">🧠</div>
        <div style="font-size:1.2rem; font-weight:700; background:linear-gradient(135deg,#6366f1,#06b6d4);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;">DocIQ</div>
        <div style="font-size:0.7rem; color:#475569; margin-top:2px;">Intelligent Document Processing</div>
    </div>
    <hr style="margin: 0.8rem 0;">
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ OCR Settings")
    languages = st.multiselect(
        "Languages",
        ["en", "hi", "fr", "de", "es", "zh_sim", "ar"],
        default=["en", "hi"],
        help="Select languages present in your document"
    )

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1, max_value=0.99, value=0.3, step=0.05,
        help="Discard text detected below this confidence"
    )

    st.markdown("### 🖼️ Preprocessing")
    preprocess_mode = st.selectbox(
        "Mode",
        ["auto", "raw", "binarize", "warp"],
        index=0,
        help="auto=deskew+CLAHE+denoise | binarize=B&W | warp=perspective flatten"
    )

    st.markdown("### 🔢 Embedding Settings")
    embed_unit = st.selectbox("Text Unit", ["word", "sentence", "line"], index=0)
    embed_reduction = st.selectbox("Reduction Method", ["PCA", "t-SNE", "UMAP"], index=0)
    embed_clusters = st.slider("K-Means Clusters", 2, 10, 5)

    st.markdown("### 🛡️ Privacy")
    auto_redact = st.toggle("Auto-detect & Redact PII", value=False)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.7rem; color:#334155; text-align:center;">
        GPU Accelerated · EasyOCR 1.7.2<br>
        sentence-transformers · PyMuPDF
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Main Content
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero-header">DocIQ — Intelligent Document Processing</div>
<div class="hero-sub">
    Upload any document image → Auto-enhance → OCR → Visualize Embeddings → Extract Entities → Export
</div>
""", unsafe_allow_html=True)

# Input Mode Tabs
input_tab1, input_tab2, input_tab3 = st.tabs([
    "📁 Upload File",
    "✨ Preset Sample Documents",
    "📸 Camera Scanner",
])

pil_img = None
doc_name = "document.png"

with input_tab1:
    uploaded_file = st.file_uploader(
        "Drop your document image here (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supports scanned documents, invoices, receipts, ID cards, forms, and business records.",
        key="file_uploader_input"
    )
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        doc_name = uploaded_file.name

with input_tab2:
    st.markdown("""
    <div style="color:#94a3b8; font-size:0.88rem; margin-bottom: 0.8rem;">
        No file handy? Pick a pre-configured synthetic sample to test the entire OCR, entity extraction, and embedding pipeline immediately:
    </div>
    """, unsafe_allow_html=True)

    sc1, sc2 = st.columns([1, 1])
    for idx, (s_name, s_fn) in enumerate(SAMPLE_DOCUMENTS.items()):
        target_col = sc1 if idx % 2 == 0 else sc2
        with target_col:
            st.markdown(f"""
            <div class="doc-card" style="padding:1rem;">
                <div style="font-weight:700; color:#818cf8; font-size:1rem; margin-bottom:4px;">📄 {s_name}</div>
                <div style="color:#64748b; font-size:0.78rem; margin-bottom:10px;">Includes tabular line items, totals, dates, and contact info.</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"⚡ Load {s_name}", key=f"btn_sample_{idx}", use_container_width=True):
                st.session_state["active_sample"] = s_name

    active_s = st.session_state.get("active_sample")
    if active_s and active_s in SAMPLE_DOCUMENTS and pil_img is None:
        pil_img = SAMPLE_DOCUMENTS[active_s]()
        doc_name = f"{active_s.lower().replace(' ', '_')}.png"
        st.success(f"✓ Loaded sample: **{active_s}**")

with input_tab3:
    cam_file = st.camera_input("Take a photo of a document", key="camera_scanner_input")
    if cam_file is not None and pil_img is None:
        pil_img = Image.open(cam_file).convert("RGB")
        doc_name = "camera_capture.png"

if pil_img is None:
    # Landing feature grid
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(4)
    features = [
        ("🔍", "Visual OCR", "Bounding boxes with confidence heatmap & reading-order"),
        ("🧬", "Embedding Space", "See your words as vector clusters in 2D space"),
        ("🛡️", "PII Redaction", "Auto-detect Aadhaar, PAN, credit cards & redact"),
        ("📊", "Entity Extraction", "Dates, amounts, invoice fields, emails auto-parsed"),
        ("📄", "Searchable PDF", "Export with invisible text layer for native search"),
        ("🔎", "Visual Search", "Search a keyword and light up its bounding box"),
        ("🌡️", "Confidence Map", "Heatmap overlay showing OCR certainty spatially"),
        ("⚡", "GPU Accelerated", "Torch CUDA 12.8 inference, model cached per session"),
    ]
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="doc-card" style="min-height:120px;">
                <div style="font-size:1.8rem; margin-bottom:6px;">{icon}</div>
                <div style="font-weight:600; color:#e2e8f0; font-size:0.9rem;">{title}</div>
                <div style="color:#64748b; font-size:0.75rem; margin-top:4px; line-height:1.4;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    st.stop()

with st.spinner("🎨 Preprocessing image..."):
    processed_img, preprocess_meta = full_preprocess(pil_img, mode=preprocess_mode)

# ──────────────────────────────────────────────
# OCR Inference
# ──────────────────────────────────────────────
with st.spinner("🧠 Running OCR (GPU accelerated)..."):
    ocr_results = run_ocr(processed_img, languages=languages, confidence_threshold=confidence_threshold)

if not ocr_results:
    st.warning("⚠️ No text detected. Try lowering the confidence threshold or switching to 'binarize' mode.")
    st.stop()

full_text = get_full_text(ocr_results)
stats = get_statistics(ocr_results)

# PII Redaction
redacted_text, pii_count = redact_pii(full_text) if auto_redact else (full_text, 0)
display_text = redacted_text if auto_redact else full_text

# Entity Extraction
with st.spinner("🔍 Extracting entities..."):
    entities = extract_all(full_text)

# ──────────────────────────────────────────────
# Metrics Row
# ──────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
m1, m2, m3, m4, m5 = st.columns(5)
metrics = [
    (m1, "Words", stats["total_words"], "badge-indigo"),
    (m2, "Avg Confidence", f"{stats['avg_confidence']:.1%}", "badge-green"),
    (m3, "High Conf ≥80%", stats["high_conf_count"], "badge-green"),
    (m4, "PII Detected", pii_count if auto_redact else "—", "badge-red" if pii_count > 0 else "badge-indigo"),
    (m5, "Doc Type", entities.get("document_type", "Unknown")[:18], "badge-cyan"),
]
for col, label, value, badge_class in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.4rem;">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────
tab_vis, tab_search, tab_text, tab_entities, tab_embed, tab_export = st.tabs([
    "🖼️ Visual Canvas",
    "🔎 Visual Search",
    "📝 Extracted Text",
    "🗂️ Entities & PII",
    "🧬 Embedding Space",
    "📤 Export",
])

# ────────────────
# TAB 1: Visual Canvas
# ────────────────
with tab_vis:
    c1, c2 = st.columns([3, 1])
    with c2:
        view_mode = st.radio(
            "Canvas View",
            ["Bounding Boxes", "Confidence Heatmap", "Reading Order", "Original"],
            index=0
        )
        show_labels = st.toggle("Word Labels", value=True)
        show_conf = st.toggle("Confidence %", value=True)

    with c1:
        if view_mode == "Bounding Boxes":
            annotated = draw_bounding_boxes(
                processed_img, ocr_results,
                show_labels=show_labels,
                show_confidence=show_conf,
            )
            st.image(annotated, use_container_width=True, caption="Bounding Boxes (Green≥80%, Yellow 50-80%, Red<50%)")
        elif view_mode == "Confidence Heatmap":
            heatmap_img = create_confidence_heatmap(processed_img, ocr_results)
            st.image(heatmap_img, use_container_width=True, caption="Confidence Heatmap (Blue=High, Red=Low)")
        elif view_mode == "Reading Order":
            order_img = draw_reading_order(processed_img, ocr_results)
            st.image(order_img, use_container_width=True, caption="Reading Order (numbered arrows)")
        else:
            col_o, col_p = st.columns(2)
            with col_o:
                st.image(pil_img, caption="Original", use_container_width=True)
            with col_p:
                st.image(processed_img, caption=f"Preprocessed ({preprocess_mode})", use_container_width=True)

    # Preprocessing steps info
    st.markdown(f"""
    <div class="doc-card" style="margin-top:0.5rem;">
        <span style="color:#64748b; font-size:0.8rem;">Preprocessing applied: </span>
        {''.join(f'<span class="badge badge-cyan" style="margin-left:4px;">{s}</span>' for s in preprocess_meta.get('steps', []))}
    </div>
    """, unsafe_allow_html=True)


# ────────────────
# TAB 2: Visual Search
# ────────────────
with tab_search:
    search_col, opts_col = st.columns([3, 1])
    with opts_col:
        case_sensitive = st.toggle("Case Sensitive", value=False)
    with search_col:
        keyword = st.text_input(
            "🔎 Search keyword in document",
            placeholder="Type a word or phrase...",
            label_visibility="collapsed"
        )

    if keyword:
        highlighted_img, matched_idx, matched_texts = search_and_highlight(
            processed_img, ocr_results, keyword, case_sensitive=case_sensitive
        )

        if matched_texts:
            st.markdown(f"""
            <div class="doc-card" style="background:rgba(34,197,94,0.08); border-color:rgba(34,197,94,0.4);">
                <span style="color:#22c55e; font-weight:600;">✓ Found {len(matched_texts)} match(es)</span>
                <br>
                {'  '.join(f'<span class="entity-chip" style="background:rgba(234,179,8,0.15);color:#eab308;border:1px solid rgba(234,179,8,0.3);">{t}</span>' for t in matched_texts[:20])}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="doc-card" style="background:rgba(239,68,68,0.08); border-color:rgba(239,68,68,0.4);">
                <span style="color:#ef4444; font-weight:600;">✗ No matches for "{keyword}"</span>
            </div>
            """, unsafe_allow_html=True)

        st.image(highlighted_img, use_container_width=True,
                 caption=f"Search Results: '{keyword}' highlighted in gold")
    else:
        st.info("👆 Enter a keyword above to highlight its position on the document canvas.")
        st.image(processed_img, use_container_width=True, caption="Document Preview")


# ────────────────
# TAB 3: Extracted Text
# ────────────────
with tab_text:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"""<div class="doc-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-weight:600; color:#e2e8f0;">
                    Extracted Text
                    {'<span class="badge badge-red" style="margin-left:8px;">PII REDACTED</span>' if auto_redact and pii_count > 0 else ''}
                </span>
                <span style="font-size:0.75rem; color:#64748b;">
                    {len(display_text.split())} words · {len(display_text)} chars
                </span>
            </div>
        </div>""", unsafe_allow_html=True)
        st.text_area(
            "Extracted Text Content",
            value=display_text,
            height=380,
            label_visibility="collapsed",
            help="Select all and Ctrl+C to copy, or edit text directly."
        )

    with c2:
        st.markdown("#### 📊 Word Confidence Table")
        df = pd.DataFrame([{
            "Text": r["text"],
            "Confidence": f"{r['confidence']:.1%}",
            "Width": r["width"],
            "Height": r["height"],
        } for r in sorted(ocr_results, key=lambda x: x["confidence"], reverse=True)])
        st.dataframe(df, use_container_width=True, height=420)


# ────────────────
# TAB 4: Entities & PII
# ────────────────
with tab_entities:
    doc_type = entities.get("document_type", "Unknown")
    st.markdown(f"""
    <div class="doc-card" style="background:linear-gradient(135deg,rgba(99,102,241,0.12),rgba(139,92,246,0.08));">
        <span style="color:#94a3b8; font-size:0.8rem;">CLASSIFIED AS</span>
        <div style="font-size:1.5rem; font-weight:700; color:#818cf8; margin-top:4px;">📋 {doc_type}</div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # Dates
        dates = entities.get("dates", [])
        st.markdown(f"""<div class="doc-card">
            <div style="font-weight:600;color:#e2e8f0;margin-bottom:8px;">📅 Dates ({len(dates)})</div>
            {''.join(f'<span class="entity-chip" style="background:rgba(6,182,212,0.12);color:#22d3ee;border:1px solid rgba(6,182,212,0.3);">{d}</span>' for d in dates) if dates else '<span style="color:#475569;font-size:0.82rem;">None detected</span>'}
        </div>""", unsafe_allow_html=True)

        # Amounts
        amounts = entities.get("amounts", [])
        st.markdown(f"""<div class="doc-card">
            <div style="font-weight:600;color:#e2e8f0;margin-bottom:8px;">💰 Amounts ({len(amounts)})</div>
            {''.join(f'<span class="entity-chip" style="background:rgba(34,197,94,0.12);color:#22c55e;border:1px solid rgba(34,197,94,0.3);">{a}</span>' for a in amounts) if amounts else '<span style="color:#475569;font-size:0.82rem;">None detected</span>'}
        </div>""", unsafe_allow_html=True)

        # Contact
        emails = entities.get("emails", [])
        phones = entities.get("phones", [])
        urls = entities.get("urls", [])
        st.markdown(f"""<div class="doc-card">
            <div style="font-weight:600;color:#e2e8f0;margin-bottom:8px;">📬 Contact Info</div>
            {''.join(f'<span class="entity-chip" style="background:rgba(99,102,241,0.12);color:#818cf8;border:1px solid rgba(99,102,241,0.3);">✉ {e}</span>' for e in emails)}
            {''.join(f'<span class="entity-chip" style="background:rgba(139,92,246,0.12);color:#a78bfa;border:1px solid rgba(139,92,246,0.3);">📞 {p}</span>' for p in phones)}
            {''.join(f'<span class="entity-chip" style="background:rgba(6,182,212,0.12);color:#22d3ee;border:1px solid rgba(6,182,212,0.3);">🔗 {u[:30]}...</span>' for u in urls[:3])}
            {'<span style="color:#475569;font-size:0.82rem;">None detected</span>' if not emails and not phones and not urls else ''}
        </div>""", unsafe_allow_html=True)

    with col_b:
        # Invoice fields
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
                <div style="display:flex; justify-content:space-between; align-items:center; padding:9px 12px; margin-bottom:6px; background:rgba(15,23,42,0.6); border:1px solid rgba(99,102,241,0.18); border-radius:10px;">
                    <span style="color:#94a3b8; font-size:0.84rem; display:flex; align-items:center; gap:6px;">
                        <span>{icon}</span>
                        <strong style="color:#cbd5e1; font-weight:500;">{title}</strong>
                    </span>
                    <span style="color:#38bdf8; font-size:0.86rem; font-weight:600; font-family:monospace; background:rgba(56,189,248,0.1); padding:2px 8px; border-radius:6px; border:1px solid rgba(56,189,248,0.25);">
                        {v}
                    </span>
                </div>""")
            inv_html = "".join(inv_rows)
            st.markdown(f"""<div class="doc-card">
                <div style="font-weight:600; color:#e2e8f0; margin-bottom:12px; display:flex; align-items:center; justify-content:space-between;">
                    <span>🧾 Key Document Fields</span>
                    <span class="badge badge-cyan">{len(inv)} detected</span>
                </div>
                {inv_html}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="doc-card">
                <div style="font-weight:600; color:#e2e8f0; margin-bottom:6px;">🧾 Key Document Fields</div>
                <div style="color:#64748b; font-size:0.82rem;">No structured invoice or PO fields detected in this document.</div>
            </div>""", unsafe_allow_html=True)

        # PII
        pii = entities.get("pii", {})
        pii_found = {k: v for k, v in pii.items() if v}
        if pii_found:
            pii_html = "".join(f"""
            <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid rgba(239,68,68,0.1);">
                <span style="color:#ef4444;font-size:0.82rem;">⚠ {k.upper()}</span>
                <span style="color:#fca5a5;font-size:0.82rem;">{len(v)} found</span>
            </div>""" for k, v in pii_found.items())
            st.markdown(f"""<div class="doc-card" style="border-color:rgba(239,68,68,0.3); background:rgba(239,68,68,0.05);">
                <div style="font-weight:600;color:#ef4444;margin-bottom:8px;">🛡️ PII Detected</div>
                {pii_html}
                <div style="margin-top:10px;font-size:0.75rem;color:#64748b;">Enable "Auto-redact PII" in sidebar to redact from exported text.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="doc-card" style="border-color:rgba(34,197,94,0.3); background:rgba(34,197,94,0.05);">
                <div style="font-weight:600;color:#22c55e;margin-bottom:4px;">✅ No PII Detected</div>
                <div style="font-size:0.8rem;color:#64748b;">No Aadhaar, PAN, credit card, or SSN patterns found.</div>
            </div>""", unsafe_allow_html=True)


# ────────────────
# TAB 5: Embedding Space
# ────────────────
with tab_embed:
    st.markdown("""<div class="doc-card">
        <div style="font-weight:600;color:#e2e8f0;margin-bottom:4px;">🧬 Semantic Embedding Vector Space</div>
        <div style="color:#64748b;font-size:0.82rem;">Each point represents a text unit projected to 2D using dimensionality reduction.
        Semantically similar words/phrases cluster together. Colors = K-Means clusters.</div>
    </div>""", unsafe_allow_html=True)

    if st.button("⚡ Generate Embedding Visualization", type="primary"):
        with st.spinner("🔢 Embedding text with sentence-transformers..."):
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
            dim = embed_data["embedding_dim"]

            # Metric row
            mc1, mc2, mc3 = st.columns(3)
            for col, val, lab in [
                (mc1, embed_data["n_texts"], f"{embed_unit.title()}s Embedded"),
                (mc2, dim, "Embedding Dimensions"),
                (mc3, embed_data["reduction_method"], "Reduction Method"),
            ]:
                with col:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-value">{val}</div>
                        <div class="metric-label">{lab}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Scatter plot
            CLUSTER_COLORS = [
                "#6366f1", "#22d3ee", "#22c55e", "#eab308", "#f97316",
                "#ec4899", "#a78bfa", "#34d399", "#fb923c", "#38bdf8"
            ]
            color_list = [CLUSTER_COLORS[l % len(CLUSTER_COLORS)] for l in labels]

            fig = go.Figure()

            # Plot each cluster as separate trace for legend
            unique_labels = sorted(set(labels))
            for cluster_id in unique_labels:
                idx = [i for i, l in enumerate(labels) if l == cluster_id]
                cluster_texts = [texts[i] for i in idx]
                x_vals = [coords[i, 0] for i in idx]
                y_vals = [coords[i, 1] for i in idx]
                color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]

                # Ellipse hull (convex hull approximation via scatter)
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="markers+text",
                    name=f"Cluster {cluster_id + 1}",
                    marker=dict(
                        color=color,
                        size=9,
                        opacity=0.85,
                        line=dict(width=1.5, color="rgba(255,255,255,0.3)"),
                    ),
                    text=[t[:20] for t in cluster_texts],
                    textposition="top center",
                    textfont=dict(size=9, color="rgba(255,255,255,0.7)"),
                    hovertemplate="<b>%{customdata}</b><br>x=%{x:.3f}, y=%{y:.3f}<extra></extra>",
                    customdata=cluster_texts,
                ))

            fig.update_layout(
                title=dict(
                    text=f"<b>Embedding Space</b> — {embed_unit.title()}s via {embed_reduction}",
                    font=dict(size=16, color="#e2e8f0"),
                ),
                paper_bgcolor="rgba(15,23,42,0)",
                plot_bgcolor="rgba(15,23,42,0.7)",
                font=dict(color="#94a3b8", size=11),
                xaxis=dict(
                    title=f"{embed_reduction} Dim 1",
                    gridcolor="rgba(99,102,241,0.1)",
                    zerolinecolor="rgba(99,102,241,0.2)",
                    title_font_color="#64748b",
                ),
                yaxis=dict(
                    title=f"{embed_reduction} Dim 2",
                    gridcolor="rgba(99,102,241,0.1)",
                    zerolinecolor="rgba(99,102,241,0.2)",
                    title_font_color="#64748b",
                ),
                legend=dict(
                    bgcolor="rgba(30,41,59,0.8)",
                    bordercolor="rgba(99,102,241,0.3)",
                    borderwidth=1,
                    font=dict(color="#94a3b8"),
                ),
                height=520,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Embedding raw values table
            with st.expander("🔢 Raw Embedding Vectors (first 8 dimensions)"):
                emb_array = embed_data["embeddings"]
                n_show = min(20, len(texts))
                table_data = {
                    "Text": texts[:n_show],
                    "Cluster": [f"C{l+1}" for l in labels[:n_show]],
                }
                n_dims = min(8, emb_array.shape[1])
                for d in range(n_dims):
                    table_data[f"dim_{d+1}"] = [f"{emb_array[i, d]:.4f}" for i in range(n_show)]
                st.dataframe(pd.DataFrame(table_data), use_container_width=True)

            # Cosine Similarity Heatmap
            sim_matrix = embed_data.get("similarity_matrix")
            if sim_matrix is not None and len(texts) <= 40:
                st.markdown("#### 🌡️ Semantic Similarity Heatmap")
                n = len(texts)
                short_labels = [t[:15] + ("…" if len(t) > 15 else "") for t in texts]
                fig_heat = px.imshow(
                    sim_matrix,
                    x=short_labels,
                    y=short_labels,
                    color_continuous_scale=[
                        [0.0, "rgba(15,23,42,1)"],
                        [0.3, "rgba(67,56,202,1)"],
                        [0.6, "rgba(99,102,241,1)"],
                        [0.8, "rgba(139,92,246,1)"],
                        [1.0, "rgba(6,182,212,1)"],
                    ],
                    zmin=0, zmax=1,
                    aspect="auto",
                    title="<b>Cosine Similarity Between Text Units</b>",
                )
                fig_heat.update_layout(
                    paper_bgcolor="rgba(15,23,42,0)",
                    plot_bgcolor="rgba(15,23,42,0.7)",
                    font=dict(color="#94a3b8", size=9),
                    height=500,
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                st.plotly_chart(fig_heat, use_container_width=True)

            # Top similar pairs
            similar_pairs = embed_data.get("similar_pairs", [])
            if similar_pairs:
                st.markdown("#### 🔗 Top Semantically Similar Pairs")
                pairs_df = pd.DataFrame([{
                    "Text A": p["text_i"],
                    "Text B": p["text_j"],
                    "Similarity": f"{p['similarity']:.1%}",
                } for p in similar_pairs[:10]])
                st.dataframe(pairs_df, use_container_width=True)

        else:
            st.warning("⚠️ Not enough text to generate embeddings. Try processing an image with more text.")
    else:
        st.markdown("""<div style="text-align:center;padding:3rem;color:#475569;">
            <div style="font-size:3rem; margin-bottom:1rem;">🧬</div>
            <div style="font-size:1rem;color:#64748b;">Click the button above to generate the embedding space visualization.<br>
            This uses <code>sentence-transformers/all-MiniLM-L6-v2</code> to convert each text unit into a 384-dim vector,
            then reduces to 2D for visualization.</div>
        </div>""", unsafe_allow_html=True)


# ────────────────
# TAB 6: Export
# ────────────────
with tab_export:
    st.markdown("""<div class="doc-card">
        <div style="font-weight:600;color:#e2e8f0;margin-bottom:4px;">📤 Export Formats & Downloads</div>
        <div style="color:#64748b;font-size:0.82rem;">Download your processed document in multiple formats or grab the all-in-one ZIP archive.</div>
    </div>""", unsafe_allow_html=True)

    base_doc_stem = os.path.splitext(doc_name)[0]

    # One-click all formats bundle
    st.markdown("""
    <div style="margin-bottom:12px;">
        <span style="font-size:0.88rem; font-weight:600; color:#818cf8;">📦 All-In-One Bundle</span>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🗜️ Prepare Complete Export Bundle (ZIP)", type="primary", use_container_width=True):
        with st.spinner("Bundling all formats (PDFs, JSON, CSV, TXT) into ZIP archive..."):
            zip_bytes = export_bundle_zip(processed_img, ocr_results, entities, display_text, base_name=base_doc_stem)
        st.download_button(
            "⬇️ Download All Formats (.ZIP Archive)",
            data=zip_bytes,
            file_name=f"{base_doc_stem}_complete_export.zip",
            mime="application/zip",
            use_container_width=True,
        )

    st.markdown("<hr style='margin:1.2rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom:12px;">
        <span style="font-size:0.88rem; font-weight:600; color:#cbd5e1;">📄 Individual Formats</span>
    </div>
    """, unsafe_allow_html=True)

    ec1, ec2 = st.columns(2)

    with ec1:
        # Searchable PDF
        if st.button("📄 Generate Searchable PDF", use_container_width=True):
            with st.spinner("Generating searchable PDF..."):
                pdf_bytes = export_searchable_pdf(processed_img, ocr_results, doc_name)
            st.download_button(
                "⬇️ Download Searchable PDF",
                data=pdf_bytes,
                file_name=f"{base_doc_stem}_searchable.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        # Annotated PDF
        if st.button("📊 Generate Annotated Report PDF", use_container_width=True):
            with st.spinner("Generating annotated PDF report..."):
                annotated_img = draw_bounding_boxes(processed_img, ocr_results, show_labels=True, show_confidence=True)
                pdf_bytes = export_annotated_pdf(annotated_img, ocr_results, entities, title="DocIQ OCR Report")
            st.download_button(
                "⬇️ Download Annotated PDF",
                data=pdf_bytes,
                file_name=f"{base_doc_stem}_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        # JSON
        json_bytes = export_json(ocr_results, entities, {"filename": doc_name})
        st.download_button(
            "⬇️ Download Structured JSON",
            data=json_bytes,
            file_name=f"{base_doc_stem}_data.json",
            mime="application/json",
            use_container_width=True,
        )

    with ec2:
        # CSV
        csv_bytes = export_csv(ocr_results, entities)
        st.download_button(
            "⬇️ Download CSV / Excel",
            data=csv_bytes,
            file_name=f"{base_doc_stem}_words.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Plain Text
        txt_bytes = export_txt(ocr_results, full_text=display_text)
        st.download_button(
            f"⬇️ Download Plain Text {'(Redacted)' if auto_redact else ''}",
            data=txt_bytes,
            file_name=f"{base_doc_stem}_text.txt",
            mime="text/plain",
            use_container_width=True,
        )

        # Export entity summary
        st.download_button(
            "⬇️ Download Entity Summary (JSON)",
            data=export_json([], entities, {"filename": doc_name, "type": "entities_only"}),
            file_name=f"{base_doc_stem}_entities.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="doc-card" style="background:rgba(99,102,241,0.06);">
        <div style="font-weight:600;color:#818cf8;margin-bottom:8px;">📋 Export Format Guide</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
            <div><span style="color:#22d3ee;font-weight:500;">Searchable PDF</span><br><span style="color:#475569;font-size:0.78rem;">Image + invisible text layer. Ctrl+F search works natively in any PDF viewer.</span></div>
            <div><span style="color:#22d3ee;font-weight:500;">Annotated PDF</span><br><span style="color:#475569;font-size:0.78rem;">Bounding box image + entity table + confidence table. Great for audit trails.</span></div>
            <div><span style="color:#22d3ee;font-weight:500;">Structured JSON</span><br><span style="color:#475569;font-size:0.78rem;">Full word coordinates, confidence, entities. Ready for downstream NLP pipelines.</span></div>
            <div><span style="color:#22d3ee;font-weight:500;">CSV / Excel</span><br><span style="color:#475569;font-size:0.78rem;">Tabular word data + entity section. Open directly in Excel or Google Sheets.</span></div>
        </div>
    </div>""", unsafe_allow_html=True)
