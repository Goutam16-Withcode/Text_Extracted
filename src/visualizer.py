"""
visualizer.py — Interactive visual bounding box overlay and keyword highlighting.
Features:
  - Confidence-coded color bounding boxes (Green/Yellow/Red)
  - Keyword search with box highlighting
  - Heatmap overlay for confidence distribution
  - Annotated word labels on image
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Optional


def _confidence_color(conf: float) -> tuple:
    """Return BGR color tuple based on confidence level."""
    if conf >= 0.8:
        return (34, 197, 94)   # Green
    elif conf >= 0.5:
        return (234, 179, 8)   # Yellow
    else:
        return (239, 68, 68)   # Red


def _confidence_color_rgba(conf: float) -> tuple:
    """Return RGBA color tuple for PIL drawing."""
    if conf >= 0.8:
        return (34, 197, 94, 200)
    elif conf >= 0.5:
        return (234, 179, 8, 200)
    else:
        return (239, 68, 68, 200)


def draw_bounding_boxes(
    pil_img: Image.Image,
    results: List[Dict[str, Any]],
    show_labels: bool = True,
    show_confidence: bool = True,
    highlight_indices: Optional[List[int]] = None,
    line_width: int = 2,
) -> Image.Image:
    """
    Draw bounding boxes with confidence color coding on a PIL image.

    Args:
        pil_img: Source image
        results: OCR result dicts from ocr_engine
        show_labels: Show text labels above each box
        show_confidence: Show confidence percentage on box
        highlight_indices: List of result indices to highlight in gold
        line_width: Border thickness

    Returns:
        Annotated PIL Image
    """
    img = pil_img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Try to load a small font
    try:
        font = ImageFont.truetype("arial.ttf", 12)
        small_font = ImageFont.truetype("arial.ttf", 10)
    except:
        font = ImageFont.load_default()
        small_font = font

    highlight_set = set(highlight_indices or [])

    for i, result in enumerate(results):
        bbox = result["bbox"]
        conf = result["confidence"]
        text = result["text"]

        pts = np.array(bbox, dtype=np.int32)
        x_min = min(p[0] for p in pts)
        y_min = min(p[1] for p in pts)
        x_max = max(p[0] for p in pts)
        y_max = max(p[1] for p in pts)

        if i in highlight_set:
            # Gold highlight for keyword matches
            color = (255, 215, 0, 230)
            bg_color = (255, 215, 0, 60)
        else:
            color = _confidence_color_rgba(conf)
            bg_color = (*_confidence_color_rgba(conf)[:3], 30)

        # Fill rectangle
        draw.rectangle([x_min, y_min, x_max, y_max], fill=bg_color, outline=color, width=line_width)

        # Label
        if show_labels:
            label = text[:20] + ("…" if len(text) > 20 else "")
            if show_confidence:
                label = f"{label} ({conf:.0%})"
            label_y = max(0, y_min - 14)
            draw.rectangle(
                [x_min, label_y, x_min + len(label) * 6, label_y + 12],
                fill=(30, 30, 30, 200)
            )
            draw.text((x_min + 1, label_y), label, fill=(255, 255, 255, 255), font=small_font)

    composed = Image.alpha_composite(img, overlay)
    return composed.convert("RGB")


def search_and_highlight(
    pil_img: Image.Image,
    results: List[Dict[str, Any]],
    keyword: str,
    case_sensitive: bool = False
) -> tuple[Image.Image, List[int], List[str]]:
    """
    Search for keyword in OCR results and return highlighted image.

    Returns:
        (annotated_image, matched_indices, matched_texts)
    """
    if not keyword.strip():
        return draw_bounding_boxes(pil_img, results), [], []

    matched_indices = []
    matched_texts = []
    kw = keyword if case_sensitive else keyword.lower()

    for i, result in enumerate(results):
        text = result["text"] if case_sensitive else result["text"].lower()
        if kw in text:
            matched_indices.append(i)
            matched_texts.append(result["text"])

    annotated = draw_bounding_boxes(
        pil_img,
        results,
        show_labels=True,
        show_confidence=True,
        highlight_indices=matched_indices,
    )
    return annotated, matched_indices, matched_texts


def create_confidence_heatmap(
    pil_img: Image.Image,
    results: List[Dict[str, Any]],
    alpha: float = 0.45
) -> Image.Image:
    """
    Overlay a confidence heatmap on the document image.
    High confidence = cool blue, Low confidence = hot red.
    """
    img_cv = np.array(pil_img.convert("RGB"))
    h, w = img_cv.shape[:2]

    heatmap = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    for result in results:
        bbox = result["bbox"]
        conf = result["confidence"]
        pts = np.array(bbox, dtype=np.int32)
        x_min = max(0, min(p[0] for p in pts))
        y_min = max(0, min(p[1] for p in pts))
        x_max = min(w - 1, max(p[0] for p in pts))
        y_max = min(h - 1, max(p[1] for p in pts))

        heatmap[y_min:y_max, x_min:x_max] += conf
        count_map[y_min:y_max, x_min:x_max] += 1.0

    count_map[count_map == 0] = 1.0
    heatmap = heatmap / count_map
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_heatmap_rgb = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(img_cv, 1 - alpha, colored_heatmap_rgb, alpha, 0)
    return Image.fromarray(blended)


def draw_reading_order(
    pil_img: Image.Image,
    results: List[Dict[str, Any]],
) -> Image.Image:
    """Draw reading order arrows connecting text regions sequentially."""
    img = np.array(pil_img.convert("RGB"))
    sorted_results = sorted(results, key=lambda r: (r["y_min"], r["x_min"]))

    for i, result in enumerate(sorted_results):
        cx, cy = result["center"]
        cv2.circle(img, (cx, cy), 8, (99, 102, 241), -1)
        cv2.putText(img, str(i + 1), (cx - 4, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        if i < len(sorted_results) - 1:
            nx, ny = sorted_results[i + 1]["center"]
            cv2.arrowedLine(img, (cx, cy), (nx, ny), (99, 102, 241), 1, tipLength=0.3)

    return Image.fromarray(img)
