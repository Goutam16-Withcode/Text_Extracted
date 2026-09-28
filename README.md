# 🧠 DocIQ — Advanced Intelligent Document Processing (IDP) Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![EasyOCR](https://img.shields.io/badge/EasyOCR-PyTorch%20GPU-green.svg)](https://github.com/JaidedAI/EasyOCR)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**DocIQ** is an Intelligent Document Processing (IDP) platform built with Streamlit, EasyOCR, OpenCV, Sentence-Transformers, and PyMuPDF. It transforms raw document images, receipts, invoices, and scans into clean, structured, searchable, and semantically indexed data.

---

## 🚀 Key Features

### 1. 🖼️ Computer Vision Preprocessing Pipeline (`src/preprocessor.py`)
- **Auto-Deskew**: Automatically detects document tilt using Hough line transform and corrects orientation.
- **Adaptive Contrast (CLAHE)**: Enhances faded or low-contrast document scans.
- **Bilateral Denoising**: Smooths scanner grain while preserving crisp character edges.
- **Multiple Modes**:
  - `auto`: Deskew + CLAHE + Denoise
  - `binarize`: High-contrast black & white document thresholding (Otsu adaptive)
  - `warp`: Perspective transformation for flattened documents
  - `raw`: Process image as uploaded

### 2. ⚡ Accelerated OCR Inference (`src/ocr_engine.py`)
- Powered by **EasyOCR** with PyTorch CUDA GPU acceleration.
- Built-in multi-lingual support: **English (`en`)**, **Hindi (`hi`)**, French, German, Spanish, Chinese, and Arabic.
- Caching with `@st.cache_resource` for zero-redundancy model loading across sessions.
- Detailed bounding box coordinates `[x_min, y_min, x_max, y_max]`, confidence scores, and reading order.

### 3. 🎨 Visual Canvas & Spatial Inspection (`src/visualizer.py`)
- **Confidence Bounding Boxes**: Visual color cues (Green ≥80%, Yellow 50–80%, Red <50%).
- **Confidence Heatmap**: Continuous thermal density overlay showing spatial OCR certainty.
- **Reading Order Vector Flow**: Numbered direction arrows indicating extracted line-by-line reading flow.
- **Interactive Visual Keyword Search**: Search any keyword to immediately highlight its spatial bounding box in gold.

### 4. 🗂️ Structured Entity & Invoice Extraction (`src/extractor.py`)
- **Automated Document Classification**: Categorizes document as *Invoice / Receipt*, *Identity Document*, *Medical / Prescription*, *Legal / Contract*, or *Academic*.
- **Regex & Context Parsing**:
  - Dates (multiple international formats)
  - Monetary amounts & currencies (`$`, `₹`, `€`, `£`)
  - Contact information (emails, phones, URLs)
  - Invoice & purchase order numbers, subtotals, tax, discounts, and due dates
- **PII Detection & Redaction**: Detects sensitive credentials (Aadhaar, PAN, Credit Cards, SSN) with optional 1-click `████` redaction.

### 5. 🧬 Semantic Embedding Vector Space (`src/embedder.py`)
- Embeds extracted words or sentences using `sentence-transformers/all-MiniLM-L6-v2`.
- Projects 384-dimensional dense vectors down to 2D via **PCA**, **t-SNE**, or **UMAP**.
- Clusters related concepts using **K-Means** with interactive Plotly scatter plots.
- Computes pairwise **Cosine Similarity Heatmaps** to discover semantic connections in document content.

### 6. 📤 Multi-Format Export Engine (`src/exporter.py`)
- **Searchable PDF**: Embedded invisible text layer enabling native `Ctrl+F` search across standard PDF viewers.
- **Annotated Audit PDF**: Complete executive report with bounding box visual overview, confidence metrics, and entity tables.
- **Structured JSON**: Full word coordinates, confidence scores, and detected metadata for downstream NLP pipelines.
- **CSV / Excel**: Tabular word list with positions and confidence.
- **Plain Text**: Formatted clean text with optional PII redaction.
- **All-In-One ZIP Archive**: 1-click download of all formats bundled together.

### 7. ✨ Preset Samples & Camera Scanner
- **Preset Test Documents**: Built-in Commercial Invoice and Retail Receipt generators (`src/sample_docs.py`) for instant testing without needing local image files.
- **Live Camera Scanner**: Snap physical documents in real-time via camera/webcam.

---

## 📁 Project Architecture

```
Text_Extracted/
├── app.py                 # Streamlit UI, visual canvas, search, embeddings, and tabs
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
└── src/
    ├── __init__.py        # Package initialization
    ├── preprocessor.py    # OpenCV deskew, contrast CLAHE, binarization, perspective warp
    ├── ocr_engine.py      # EasyOCR GPU inference engine and result normalizer
    ├── visualizer.py      # Bounding box overlays, heatmaps, search highlighter
    ├── extractor.py       # Entity extraction (dates, amounts, contacts, invoice fields, PII)
    ├── embedder.py        # Sentence-transformers embedding & 2D dimensionality reduction
    ├── exporter.py        # Searchable PDF, Annotated PDF, JSON, CSV, TXT, and ZIP bundle
    └── sample_docs.py     # Synthetic document generator for 1-click demo testing
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+
- (Optional) NVIDIA GPU with CUDA for accelerated neural OCR inference

### 1. Clone the Repository
```bash
git clone https://github.com/Goutam16-Withcode/Text_Extracted.git
cd Text_Extracted
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser to begin.

---

## 🛡️ Privacy & Compliance
- **Local Execution**: All OCR inference, image preprocessing, and embedding calculations happen locally on your machine.
- **PII Redaction**: Sensitive IDs and financial numbers can be stripped or masked before exporting.

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
