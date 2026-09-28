"""
ocr_engine.py — Advanced EasyOCR inference engine with accuracy measurement.
Features:
  - GPU-accelerated inference (CUDA auto-detect)
  - @st.cache_resource singleton reader
  - High-sensitivity parameters tuned for live camera & scanned photos
  - Fallback multi-pass OCR for difficult lighting / low-contrast text
  - Comprehensive OCR accuracy and quality metrics calculation
"""

import streamlit as st
import easyocr
import numpy as np
import re
from PIL import Image
from typing import List, Dict, Any, Tuple


@st.cache_resource(show_spinner="🧠 Loading EasyOCR engine (one-time initialization)...")
def get_reader(languages: Tuple[str, ...]) -> easyocr.Reader:
    """
    Cached singleton OCR reader. Loaded once per session.
    Auto-detects CUDA GPU for acceleration.
    """
    try:
        import torch
        gpu = torch.cuda.is_available()
    except Exception:
        gpu = False
    return easyocr.Reader(list(languages), gpu=gpu)


def run_ocr(
    pil_img: Image.Image,
    languages: list = None,
    confidence_threshold: float = 0.25,
    camera_mode: bool = False,
) -> List[Dict[str, Any]]:
    """
    Execute neural OCR with parameters tuned for maximum detection recall and precision.
    Supports camera mode with multi-pass recognition for challenging live captures.
    """
    if languages is None:
        languages = ["en", "hi"]

    reader = get_reader(tuple(languages))
    img_array = np.array(pil_img.convert("RGB"))

    # Primary pass with optimized thresholds
    text_th = 0.35 if camera_mode else 0.40
    low_th = 0.20 if camera_mode else 0.25
    contrast_th = 0.05 if camera_mode else 0.10

    raw_results = reader.readtext(
        img_array,
        text_threshold=text_th,
        low_text=low_th,
        contrast_ths=contrast_th,
        adjust_contrast=0.7 if camera_mode else 0.5,
        link_threshold=0.35,
        decoder="greedy",
    )

    # If very few words detected on a live capture, attempt secondary high-sensitivity pass
    if len(raw_results) < 4:
        raw_results_fallback = reader.readtext(
            img_array,
            text_threshold=0.25,
            low_text=0.15,
            contrast_ths=0.03,
            adjust_contrast=0.9,
            link_threshold=0.25,
        )
        if len(raw_results_fallback) > len(raw_results):
            raw_results = raw_results_fallback

    structured = []
    for (bbox, text, prob) in raw_results:
        clean_text = text.strip()
        if prob < confidence_threshold:
            continue
        if not clean_text:
            continue

        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        cx = int(np.mean(xs))
        cy = int(np.mean(ys))
        w = int(max(xs) - min(xs))
        h = int(max(ys) - min(ys))

        structured.append({
            "bbox": bbox,
            "text": clean_text,
            "confidence": float(prob),
            "center": (cx, cy),
            "width": w,
            "height": h,
            "x_min": int(min(xs)),
            "y_min": int(min(ys)),
            "x_max": int(max(xs)),
            "y_max": int(max(ys)),
        })

    return structured


def get_full_text(results: List[Dict]) -> str:
    """Reconstruct full plain text from OCR results preserving natural reading order."""
    sorted_results = sorted(results, key=lambda r: (r["y_min"] // 20, r["x_min"]))
    return " ".join(r["text"] for r in sorted_results)


def compute_accuracy_metrics(results: List[Dict], full_text: str = "") -> Dict[str, Any]:
    """
    Measure comprehensive OCR accuracy, character recognition quality, and text coherence.
    Returns:
      - overall_accuracy: 0-100%
      - quality_grade: 'A+' | 'A' | 'B' | 'C' | 'D'
      - avg_confidence: float (0-1)
      - high_conf_pct: % words with conf >= 80%
      - medium_conf_pct: % words with conf 50-79%
      - low_conf_pct: % words with conf < 50%
      - lexical_validity: % of recognized tokens with standard alphanumeric structure
      - readability_status: Description of reliability
    """
    if not results:
        return {
            "overall_accuracy": 0.0,
            "quality_grade": "N/A",
            "avg_confidence": 0.0,
            "high_conf_count": 0,
            "medium_conf_count": 0,
            "low_conf_count": 0,
            "high_conf_pct": 0.0,
            "medium_conf_pct": 0.0,
            "low_conf_pct": 0.0,
            "total_words": 0,
            "total_characters": 0,
            "lexical_validity": 0.0,
            "reliability_label": "No Text Detected",
            "reliability_color": "#94a3b8",
        }

    confidences = [r["confidence"] for r in results]
    total_words = len(results)
    avg_conf = float(np.mean(confidences))

    high_c = sum(1 for c in confidences if c >= 0.80)
    med_c = sum(1 for c in confidences if 0.50 <= c < 0.80)
    low_c = sum(1 for c in confidences if c < 0.50)

    # Lexical validity: check if word contains intelligible alphanumeric patterns
    valid_tokens = 0
    total_chars = 0
    for r in results:
        t = r["text"]
        total_chars += len(t)
        # Token has letters or numbers (not pure punctuation noise)
        if re.search(r"[A-Za-z0-9\u0900-\u097F]", t):
            valid_tokens += 1

    lexical_validity = (valid_tokens / total_words) if total_words > 0 else 0.0

    # Composite OCR Accuracy Index (weighted: 70% confidence + 30% lexical validity)
    overall_accuracy = (avg_conf * 0.70 + lexical_validity * 0.30) * 100.0

    # Determine Quality Grade
    if overall_accuracy >= 90.0:
        grade = "A+"
        label = "Exceptional Accuracy"
        color = "#16a34a"  # Green
    elif overall_accuracy >= 80.0:
        grade = "A"
        label = "High Accuracy (Production Ready)"
        color = "#22c55e"
    elif overall_accuracy >= 68.0:
        grade = "B"
        label = "Good / Readable"
        color = "#eab308"  # Yellow
    elif overall_accuracy >= 50.0:
        grade = "C"
        label = "Moderate (Review Recommended)"
        color = "#f97316"  # Orange
    else:
        grade = "D"
        label = "Low (Retake photo with better lighting)"
        color = "#ef4444"  # Red

    return {
        "overall_accuracy": round(overall_accuracy, 1),
        "quality_grade": grade,
        "avg_confidence": round(avg_conf, 3),
        "high_conf_count": high_c,
        "medium_conf_count": med_c,
        "low_conf_count": low_c,
        "high_conf_pct": round((high_c / total_words) * 100.0, 1),
        "medium_conf_pct": round((med_c / total_words) * 100.0, 1),
        "low_conf_pct": round((low_c / total_words) * 100.0, 1),
        "total_words": total_words,
        "total_characters": total_chars,
        "lexical_validity": round(lexical_validity * 100.0, 1),
        "reliability_label": label,
        "reliability_color": color,
    }


def get_statistics(results: List[Dict]) -> Dict[str, Any]:
    """Compatibility wrapper that delegates to compute_accuracy_metrics."""
    return compute_accuracy_metrics(results)
