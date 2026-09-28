"""
ocr_engine.py — Cached EasyOCR inference engine.
Features:
  - GPU-accelerated inference (CUDA auto-detect)
  - @st.cache_resource singleton (loads weights once)
  - Returns rich structured results: bbox, text, confidence
  - Supports multi-language (English + Hindi)
"""

import streamlit as st
import easyocr
import numpy as np
from PIL import Image
from typing import List, Dict, Any


@st.cache_resource(show_spinner="🧠 Loading OCR engine (one-time)...")
def get_reader(languages: list = None) -> easyocr.Reader:
    """
    Cached singleton OCR reader. Loaded once per session.
    Auto-detects CUDA GPU for acceleration.
    """
    if languages is None:
        languages = ["en", "hi"]
    try:
        import torch
        gpu = torch.cuda.is_available()
    except ImportError:
        gpu = False
    return easyocr.Reader(languages, gpu=gpu)


def run_ocr(
    pil_img: Image.Image,
    languages: list = None,
    confidence_threshold: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Run OCR on a PIL image.

    Returns list of dicts:
    {
        "bbox": [[x1,y1],[x2,y1],[x2,y2],[x1,y2]],
        "text": "...",
        "confidence": 0.97,
        "center": (cx, cy),
        "width": w,
        "height": h
    }
    """
    if languages is None:
        languages = ["en", "hi"]

    reader = get_reader(tuple(languages))
    img_array = np.array(pil_img.convert("RGB"))

    raw_results = reader.readtext(img_array)

    structured = []
    for (bbox, text, prob) in raw_results:
        if prob < confidence_threshold:
            continue
        if not text.strip():
            continue

        # bbox is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        cx = int(np.mean(xs))
        cy = int(np.mean(ys))
        w = int(max(xs) - min(xs))
        h = int(max(ys) - min(ys))

        structured.append({
            "bbox": bbox,
            "text": text.strip(),
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
    """Reconstruct full plain text from OCR results, preserving reading order."""
    # Sort by y then x (reading order)
    sorted_results = sorted(results, key=lambda r: (r["y_min"], r["x_min"]))
    return " ".join(r["text"] for r in sorted_results)


def get_lines(results: List[Dict], line_gap_threshold: int = 15) -> List[List[Dict]]:
    """Group OCR results into lines based on vertical proximity."""
    if not results:
        return []

    sorted_results = sorted(results, key=lambda r: r["y_min"])
    lines = []
    current_line = [sorted_results[0]]

    for result in sorted_results[1:]:
        prev_y = current_line[-1]["center"][1]
        curr_y = result["center"][1]
        if abs(curr_y - prev_y) <= line_gap_threshold:
            current_line.append(result)
        else:
            lines.append(sorted(current_line, key=lambda r: r["x_min"]))
            current_line = [result]

    if current_line:
        lines.append(sorted(current_line, key=lambda r: r["x_min"]))

    return lines


def get_statistics(results: List[Dict]) -> Dict[str, Any]:
    """Compute statistics about the OCR results."""
    if not results:
        return {}

    confidences = [r["confidence"] for r in results]
    return {
        "total_words": len(results),
        "avg_confidence": float(np.mean(confidences)),
        "min_confidence": float(np.min(confidences)),
        "max_confidence": float(np.max(confidences)),
        "high_conf_count": sum(1 for c in confidences if c >= 0.8),
        "medium_conf_count": sum(1 for c in confidences if 0.5 <= c < 0.8),
        "low_conf_count": sum(1 for c in confidences if c < 0.5),
    }
