---
title: DocIQ - Intelligent Document Processing
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.40.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# 🧠 DocIQ — Intelligent Document Processing

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![EasyOCR](https://img.shields.io/badge/EasyOCR-PyTorch-green.svg)](https://github.com/JaidedAI/EasyOCR)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![HF Space](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace-orange.svg)](https://huggingface.co/spaces/Goutam112/Text_Extracted)

**DocIQ** is an Intelligent Document Processing (IDP) platform that transforms document images into structured, searchable, and exportable information.

Upload a document image, select a built-in sample, or capture an image using your webcam. DocIQ performs OCR, evaluates recognition confidence, visualizes detected text, extracts entities, detects sensitive information, and exports the results in multiple formats.

---

## ✨ Features

| Feature                    | Description                                                                         |
| -------------------------- | ----------------------------------------------------------------------------------- |
| 🎯 **OCR Accuracy Score**  | Confidence-weighted accuracy grade from A+ to D                                     |
| 🖼️ **Visual Canvas**      | Bounding boxes, confidence heatmap, and reading-order overlay on the original image |
| 🔎 **Keyword Search**      | Instantly search and highlight words on the document canvas                         |
| 🗂️ **Entity Extraction**  | Automatically detect dates, amounts, invoice numbers, emails, and phone numbers     |
| 🛡️ **PII Redaction**      | Detect and mask sensitive information such as Aadhaar, PAN, and credit-card numbers |
| 🧬 **Semantic Embeddings** | Generate 2D word-cluster visualizations using sentence-transformers with PCA/t-SNE  |
| 📤 **Multi-Format Export** | Export searchable PDFs, annotated PDFs, JSON, CSV, plain text, and ZIP bundles      |
| 📸 **Camera Scanner**      | Capture documents directly through the webcam when the scanner is enabled           |

---

## 🚀 Quick Start

### Run Locally

```bash
git clone https://github.com/Goutam16-Withcode/DocIQ---Intelligent-Document-Processing.git
cd DocIQ---Intelligent-Document-Processing
pip install -r requirements.txt
python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

### 🤗 Live Demo

[DocIQ — Hugging Face Space](https://huggingface.co/spaces/Goutam112/Text_Extracted?utm_source=chatgpt.com)

---

## 🔄 How It Works

```text
Upload / Camera Capture
        ↓
Display Original Image
        ↓
Image Preprocessing
        ├── EXIF orientation correction
        ├── Automatic deskew
        ├── CLAHE contrast enhancement
        ├── Shadow removal
        └── Unsharp-mask sharpening
        ↓
EasyOCR Neural Inference
        ├── Primary OCR pass
        └── High-sensitivity fallback pass
        ↓
Structured Document Output
        ├── Bounding boxes
        ├── Confidence analysis
        ├── OCR quality grade
        ├── Entity extraction
        ├── PII detection/redaction
        └── Multi-format export
```

---

## 📊 OCR Quality Grades

|  Grade | Accuracy | Meaning                                         |
| :----: | :------: | ----------------------------------------------- |
| **A+** |  90–100% | Exceptional recognition confidence              |
|  **A** |  80–89%  | High recognition confidence                     |
|  **B** |  68–79%  | Good recognition with possible minor errors     |
|  **C** |  50–67%  | Moderate recognition; manual review recommended |
|  **D** |   < 50%  | Low recognition; better image quality may help  |

> **Note:** The grade is an application-level confidence metric and should not be interpreted as a formal benchmark of OCR accuracy against ground-truth annotations.

---

## 🧠 Processing Pipeline

DocIQ separates the image shown to the user from the image transformations used internally for OCR.

### 1. Image Acquisition

Documents can be provided through:

* 📁 File upload
* 🧪 Built-in sample documents
* 📸 Webcam capture

### 2. Preprocessing

The OCR pipeline can apply several image-processing operations:

* EXIF orientation correction
* Deskewing
* CLAHE contrast enhancement
* Shadow removal for camera images
* Sharpening

These transformations are performed internally to improve OCR recognition while preserving the original document for visualization.

### 3. OCR

DocIQ uses **EasyOCR with PyTorch** to detect and recognize text.

The OCR output contains:

* Detected text
* Bounding boxes
* Confidence scores
* Reading order information

### 4. Information Extraction

The recognized text is further processed to identify structured information such as:

* Dates
* Currency amounts
* Invoice numbers
* Email addresses
* Phone numbers

### 5. PII Detection

Potentially sensitive identifiers can be detected and optionally redacted, including:

* Aadhaar numbers
* PAN numbers
* Credit-card numbers

### 6. Semantic Analysis

Recognized words can be converted into semantic embeddings and projected into a 2D visualization using dimensionality-reduction techniques such as PCA or t-SNE.

---

## 📤 Export Formats

DocIQ supports multiple output formats:

| Format             | Use Case                                     |
| ------------------ | -------------------------------------------- |
| **Searchable PDF** | Searchable version of the extracted document |
| **Annotated PDF**  | Document with OCR annotations                |
| **JSON**           | Structured OCR and entity data               |
| **CSV**            | Tabular extraction results                   |
| **TXT**            | Plain-text OCR output                        |
| **ZIP**            | Bundle of multiple generated outputs         |

---

## 📁 Project Structure

```text
DocIQ/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── src/
    ├── preprocessor.py # Image enhancement pipeline
    ├── ocr_engine.py   # EasyOCR engine and OCR metrics
    ├── visualizer.py   # Bounding boxes, heatmap, reading order
    ├── extractor.py    # Entity and invoice field extraction
    ├── embedder.py     # Semantic embeddings and clustering
    ├── exporter.py     # PDF, JSON, CSV, ZIP export
    └── sample_docs.py  # Built-in sample document generator
```

---

## ⚙️ Technology Stack

* **Python 3.10+**
* **Streamlit**
* **PyTorch**
* **EasyOCR**
* **OpenCV**
* **NumPy**
* **Pandas**
* **Sentence Transformers**
* **PCA / t-SNE**
* **ReportLab**
* **JSON / CSV processing**

---

## 🛡️ Privacy

DocIQ is designed with local processing in mind.

OCR processing, image preprocessing, and embedding generation are performed within the application environment. The application does not intentionally send uploaded document images or extracted text to external APIs as part of the core processing pipeline.

When deploying the application to a hosted environment such as Hugging Face Spaces, uploaded data is processed within that hosted runtime. Users should therefore avoid uploading sensitive documents unless they understand the hosting environment and its data-retention policies.

---

## 🤗 Hugging Face Space

**Live Demo:**

[Open DocIQ on Hugging Face](https://huggingface.co/spaces/Goutam112/Text_Extracted?utm_source=chatgpt.com)

---

## 📄 License

MIT License — open source and free to use.

---

## 🔗 Project Repository

[DocIQ GitHub Repository](https://github.com/Goutam16-Withcode/DocIQ---Intelligent-Document-Processing?utm_source=chatgpt.com)

---

Check out the configuration reference at [Hugging Face Spaces Configuration Reference](https://huggingface.co/docs/hub/spaces-config-reference?utm_source=chatgpt.com)
