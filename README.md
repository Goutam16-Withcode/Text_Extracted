# 🧠 DocIQ — Intelligent Document Processing

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![EasyOCR](https://img.shields.io/badge/EasyOCR-PyTorch-green.svg)](https://github.com/JaidedAI/EasyOCR)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![HF Space](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace-orange.svg)](https://huggingface.co/spaces/Goutam112/Text_Extracted)

**DocIQ** is an Intelligent Document Processing (IDP) platform. Upload any document image, use a preset sample, or capture with your webcam — DocIQ extracts text, measures accuracy, detects entities, and exports results in multiple formats.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **OCR Accuracy Score** | Confidence-weighted accuracy grade (A+ to D) for every scan |
| 🖼️ **Visual Canvas** | Bounding boxes, confidence heatmap, and reading order overlay on the **original image** |
| 🔎 **Keyword Search** | Highlight any word instantly on the document canvas |
| 🗂️ **Entity Extraction** | Auto-detect dates, amounts, invoice numbers, emails, phones |
| 🛡️ **PII Redaction** | Mask Aadhaar, PAN, credit cards with one toggle |
| 🧬 **Semantic Embeddings** | 2D word cluster map using sentence-transformers + PCA/t-SNE |
| 📤 **Multi-Format Export** | Searchable PDF, annotated PDF, JSON, CSV, plain text, ZIP bundle |
| 📸 **Camera Scanner** | On-demand webcam capture — camera only turns on when you click Enable |

---

## 🚀 Quick Start

### Run Locally
```bash
git clone https://github.com/Goutam16-Withcode/DocIQ---Intelligent-Document-Processing.git
cd DocIQ---Intelligent-Document-Processing
pip install -r requirements.txt
python -m streamlit run app.py
```
Open **http://localhost:8501** in your browser.

### Live Demo
👉 [huggingface.co/spaces/Goutam112/Text_Extracted](https://huggingface.co/spaces/Goutam112/Text_Extracted)

---

## 🔄 How It Works

```
Upload / Camera Capture
        ↓
Show original image (natural colors)
        ↓
Preprocessing (internal — for OCR accuracy only)
  • EXIF orientation fix
  • Auto deskew
  • CLAHE contrast enhancement
  • Shadow removal (camera mode)
  • Unsharp mask sharpening
        ↓
EasyOCR Neural Inference (GPU accelerated)
  • Primary pass + fallback high-sensitivity pass
        ↓
Structured Output
  • Bounding boxes on original image
  • Accuracy grade + confidence breakdown
  • Entity extraction & PII redaction
  • Export (PDF / JSON / CSV / ZIP)
```

---

## 📊 OCR Quality Grades

| Grade | Accuracy | Meaning |
|:-----:|:--------:|---------|
| **A+** | 90–100% | Exceptional — production ready |
| **A** | 80–89% | High accuracy — standard documents |
| **B** | 68–79% | Good — minor noise or tilt |
| **C** | 50–67% | Moderate — review recommended |
| **D** | < 50% | Low — retake with better lighting |

---

## 📁 Project Structure

```
DocIQ/
├── app.py              # Main Streamlit UI
├── requirements.txt    # Python dependencies
└── src/
    ├── preprocessor.py # Image enhancement pipeline
    ├── ocr_engine.py   # EasyOCR engine + accuracy metrics
    ├── visualizer.py   # Bounding boxes, heatmap, reading order
    ├── extractor.py    # Entity & invoice field extraction
    ├── embedder.py     # Semantic embeddings & clustering
    ├── exporter.py     # PDF, JSON, CSV, ZIP export
    └── sample_docs.py  # Built-in sample document generator
```

---

## 🛡️ Privacy

All OCR processing, image filtering, and vector embeddings run **entirely locally**. No images or text are sent to any external server.

---

## 📄 License

MIT License — open source and free to use.
