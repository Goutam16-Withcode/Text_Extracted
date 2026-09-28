"""
exporter.py — Multi-format document export engine.
Features:
  - Searchable PDF with invisible text layer
  - Annotated PDF with colored bounding boxes
  - Structured JSON export with metadata
  - CSV / Excel tabular export
  - Plain text export
  - Redacted document export
"""

import io
import json
import csv
import datetime
import numpy as np
from typing import List, Dict, Any
from PIL import Image


def export_json(
    ocr_results: List[Dict],
    entities: Dict,
    metadata: Dict = None,
) -> bytes:
    """Export OCR results and entities to structured JSON bytes."""
    output = {
        "metadata": {
            "exported_at": datetime.datetime.utcnow().isoformat() + "Z",
            "total_words": len(ocr_results),
            "avg_confidence": float(np.mean([r["confidence"] for r in ocr_results])) if ocr_results else 0,
            **(metadata or {}),
        },
        "document_type": entities.get("document_type", "Unknown"),
        "entities": {
            "dates": entities.get("dates", []),
            "amounts": entities.get("amounts", []),
            "emails": entities.get("emails", []),
            "phones": entities.get("phones", []),
            "urls": entities.get("urls", []),
            "invoice_fields": entities.get("invoice_fields", {}),
            "pii_detected": {k: len(v) > 0 for k, v in entities.get("pii", {}).items()},
        },
        "words": [
            {
                "text": r["text"],
                "confidence": round(r["confidence"], 4),
                "x_min": r["x_min"],
                "y_min": r["y_min"],
                "x_max": r["x_max"],
                "y_max": r["y_max"],
                "center": list(r["center"]),
            }
            for r in ocr_results
        ],
    }
    return json.dumps(output, indent=2, ensure_ascii=False).encode("utf-8")


def export_csv(ocr_results: List[Dict], entities: Dict) -> bytes:
    """Export OCR results to CSV bytes."""
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "Index", "Text", "Confidence", "X_Min", "Y_Min", "X_Max", "Y_Max", "Width", "Height", "Center_X", "Center_Y"
    ])

    for i, r in enumerate(ocr_results):
        writer.writerow([
            i + 1,
            r["text"],
            f"{r['confidence']:.4f}",
            r["x_min"],
            r["y_min"],
            r["x_max"],
            r["y_max"],
            r["width"],
            r["height"],
            r["center"][0],
            r["center"][1],
        ])

    # Blank + entities section
    writer.writerow([])
    writer.writerow(["=== EXTRACTED ENTITIES ==="])
    writer.writerow(["Type", "Value"])
    for date in entities.get("dates", []):
        writer.writerow(["Date", date])
    for amt in entities.get("amounts", []):
        writer.writerow(["Amount", amt])
    for email in entities.get("emails", []):
        writer.writerow(["Email", email])
    for phone in entities.get("phones", []):
        writer.writerow(["Phone", phone])
    for field, value in entities.get("invoice_fields", {}).items():
        writer.writerow([f"Invoice: {field}", value])

    return output.getvalue().encode("utf-8")


def export_txt(ocr_results: List[Dict], redacted: bool = False, full_text: str = "") -> bytes:
    """Export clean plain text."""
    if full_text:
        return full_text.encode("utf-8")
    sorted_results = sorted(ocr_results, key=lambda r: (r["y_min"], r["x_min"]))
    text = " ".join(r["text"] for r in sorted_results)
    return text.encode("utf-8")


def export_searchable_pdf(
    pil_img: Image.Image,
    ocr_results: List[Dict],
    page_title: str = "OCR Document",
) -> bytes:
    """
    Generate a searchable PDF:
    - Original image as background
    - Invisible text layer placed at exact bbox coordinates
    Returns raw PDF bytes.
    """
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader

    buffer = io.BytesIO()

    # Get image dimensions
    img_w, img_h = pil_img.size
    page_w, page_h = img_w, img_h

    c = rl_canvas.Canvas(buffer, pagesize=(page_w, page_h))

    # Draw background image
    img_buffer = io.BytesIO()
    pil_img.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    c.drawImage(ImageReader(img_buffer), 0, 0, width=page_w, height=page_h)

    # Overlay invisible text at OCR positions
    c.setFillColorRGB(0, 0, 0, 0)  # Transparent text
    for result in ocr_results:
        text = result["text"]
        x = result["x_min"]
        # PDF coords: y=0 at bottom, image y=0 at top
        y = page_h - result["y_max"]
        font_size = max(4, result["height"])
        c.setFont("Helvetica", min(font_size, 72))
        c.drawString(x, y, text)

    c.save()
    buffer.seek(0)
    return buffer.read()


def export_annotated_pdf(
    annotated_pil: Image.Image,
    ocr_results: List[Dict],
    entities: Dict,
    title: str = "Annotated OCR Report",
) -> bytes:
    """
    Generate an annotated report PDF with:
    - Annotated image (bounding boxes)
    - Entity summary table
    - Full word-by-word confidence table
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.utils import ImageReader

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    story.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Words: {len(ocr_results)} | Type: {entities.get('document_type', 'Unknown')}",
        styles["Normal"]
    ))
    story.append(Spacer(1, 0.3*cm))

    # Annotated image
    img_buffer = io.BytesIO()
    # Resize to fit A4 width
    max_w = 500
    iw, ih = annotated_pil.size
    scale = min(max_w / iw, 1.0)
    resized = annotated_pil.resize((int(iw * scale), int(ih * scale)), Image.LANCZOS)
    resized.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    story.append(RLImage(img_buffer, width=int(iw * scale), height=int(ih * scale)))
    story.append(Spacer(1, 0.3*cm))

    # Entities section
    story.append(Paragraph("<b>Extracted Entities</b>", styles["Heading2"]))
    entity_data = [["Category", "Values"]]
    if entities.get("dates"):
        entity_data.append(["Dates", ", ".join(entities["dates"])])
    if entities.get("amounts"):
        entity_data.append(["Amounts", ", ".join(entities["amounts"])])
    if entities.get("emails"):
        entity_data.append(["Emails", ", ".join(entities["emails"])])
    if entities.get("phones"):
        entity_data.append(["Phones", ", ".join(entities["phones"])])
    if entities.get("invoice_fields"):
        for k, v in entities["invoice_fields"].items():
            entity_data.append([k.replace("_", " ").title(), str(v)])

    if len(entity_data) > 1:
        entity_table = Table(entity_data, colWidths=[120, 370])
        entity_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6366f1")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(entity_table)
        story.append(Spacer(1, 0.3*cm))

    # Word confidence table
    story.append(Paragraph("<b>Word-by-Word Confidence Table</b>", styles["Heading2"]))
    word_data = [["#", "Text", "Confidence", "Position"]]
    for i, r in enumerate(ocr_results[:100]):  # Limit to first 100
        conf_pct = f"{r['confidence']:.1%}"
        pos = f"({r['x_min']}, {r['y_min']})"
        word_data.append([str(i + 1), r["text"][:40], conf_pct, pos])

    word_table = Table(word_data, colWidths=[30, 260, 80, 120])
    word_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f1f5f9"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(word_table)

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


def export_bundle_zip(
    processed_img: Image.Image,
    ocr_results: List[Dict],
    entities: Dict,
    full_text: str,
    base_name: str = "document",
) -> bytes:
    """Generate a single ZIP archive containing all export formats."""
    import zipfile
    from src.visualizer import draw_bounding_boxes

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # JSON
        zip_file.writestr(f"{base_name}_data.json", export_json(ocr_results, entities, {"filename": base_name}))
        # CSV
        zip_file.writestr(f"{base_name}_words.csv", export_csv(ocr_results, entities))
        # Plain text
        zip_file.writestr(f"{base_name}_text.txt", export_txt(ocr_results, full_text=full_text))
        # Searchable PDF
        try:
            searchable_pdf = export_searchable_pdf(processed_img, ocr_results, base_name)
            zip_file.writestr(f"{base_name}_searchable.pdf", searchable_pdf)
        except Exception:
            pass
        # Annotated PDF
        try:
            annotated_img = draw_bounding_boxes(processed_img, ocr_results, show_labels=True, show_confidence=True)
            report_pdf = export_annotated_pdf(annotated_img, ocr_results, entities, title=f"DocIQ Report - {base_name}")
            zip_file.writestr(f"{base_name}_report.pdf", report_pdf)
        except Exception:
            pass

    zip_buffer.seek(0)
    return zip_buffer.read()
