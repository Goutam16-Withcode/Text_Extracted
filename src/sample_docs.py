"""
sample_docs.py — Generates realistic synthetic document images for instant 1-click testing in DocIQ.
"""

from PIL import Image, ImageDraw, ImageFont
import io


def _get_fonts():
    """Try to obtain clean TrueType fonts, fall back to default."""
    try:
        title_font = ImageFont.truetype("arial.ttf", 24)
        heading_font = ImageFont.truetype("arial.ttf", 18)
        body_font = ImageFont.truetype("arial.ttf", 15)
        mono_font = ImageFont.truetype("cour.ttf", 14)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        title_font = ImageFont.load_default()
        heading_font = title_font
        body_font = title_font
        mono_font = title_font
        small_font = title_font
    return title_font, heading_font, body_font, mono_font, small_font


def generate_invoice_sample() -> Image.Image:
    """Generate a clean business invoice document."""
    width, height = 800, 1000
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    title_font, heading_font, body_font, mono_font, small_font = _get_fonts()

    # Header Banner
    draw.rectangle([0, 0, width, 90], fill=(30, 41, 59))
    draw.text((40, 28), "ACME CLOUD SOLUTIONS INC.", fill=(255, 255, 255), font=title_font)
    draw.text((580, 32), "INVOICE", fill=(129, 140, 248), font=title_font)

    # Invoice Meta
    y = 120
    draw.text((40, y), "Billed To:", fill=(100, 116, 139), font=small_font)
    draw.text((40, y + 20), "TechCorp International Ltd.", fill=(15, 23, 42), font=heading_font)
    draw.text((40, y + 46), "100 Innovation Way, Suite 400", fill=(71, 85, 105), font=body_font)
    draw.text((40, y + 70), "Email: billing@techcorp.com", fill=(71, 85, 105), font=body_font)
    draw.text((40, y + 94), "Phone: +1 (555) 234-5678", fill=(71, 85, 105), font=body_font)

    draw.text((540, y), "Invoice Details:", fill=(100, 116, 139), font=small_font)
    draw.text((540, y + 20), "Invoice #: INV-2026-8842", fill=(15, 23, 42), font=mono_font)
    draw.text((540, y + 46), "Date: 15/09/2026", fill=(71, 85, 105), font=body_font)
    draw.text((540, y + 70), "Due Date: 30/09/2026", fill=(71, 85, 105), font=body_font)
    draw.text((540, y + 94), "Status: PENDING", fill=(217, 119, 6), font=mono_font)

    # Table Header
    y = 280
    draw.rectangle([40, y, width - 40, y + 40], fill=(241, 245, 249))
    draw.text((55, y + 10), "Description", fill=(30, 41, 59), font=heading_font)
    draw.text((420, y + 10), "Qty", fill=(30, 41, 59), font=heading_font)
    draw.text((520, y + 10), "Unit Price", fill=(30, 41, 59), font=heading_font)
    draw.text((660, y + 10), "Amount", fill=(30, 41, 59), font=heading_font)

    # Table Rows
    items = [
        ("Cloud Server Hosting - Cluster A", "2", "$450.00", "$900.00"),
        ("Database Backup & Storage (10TB)", "1", "$250.00", "$250.00"),
        ("API Gateway Bandwidth Overages", "1", "$150.00", "$150.00"),
        ("Security SSL & Compliance Pack", "1", "$199.00", "$199.00"),
    ]

    row_y = y + 55
    for desc, qty, price, total in items:
        draw.text((55, row_y), desc, fill=(51, 65, 85), font=body_font)
        draw.text((430, row_y), qty, fill=(51, 65, 85), font=body_font)
        draw.text((520, row_y), price, fill=(51, 65, 85), font=body_font)
        draw.text((660, row_y), total, fill=(15, 23, 42), font=body_font)
        draw.line([(40, row_y + 35), (width - 40, row_y + 35)], fill=(226, 232, 240), width=1)
        row_y += 48

    # Summary
    sum_y = row_y + 20
    draw.text((520, sum_y), "Subtotal:", fill=(71, 85, 105), font=body_font)
    draw.text((660, sum_y), "$1,499.00", fill=(15, 23, 42), font=body_font)

    draw.text((520, sum_y + 30), "Tax (10%):", fill=(71, 85, 105), font=body_font)
    draw.text((660, sum_y + 30), "$149.90", fill=(15, 23, 42), font=body_font)

    draw.rectangle([500, sum_y + 65, width - 40, sum_y + 115], fill=(238, 242, 255))
    draw.text((520, sum_y + 78), "Total Due:", fill=(67, 56, 202), font=heading_font)
    draw.text((650, sum_y + 78), "$1,648.90", fill=(67, 56, 202), font=heading_font)

    # Footer note
    draw.text((40, 850), "Payment Terms: Wire transfer payable to ACME Cloud Solutions within 15 days.", fill=(100, 116, 139), font=small_font)
    draw.text((40, 875), "Support inquiries: support@acmecloud.io | Website: https://acmecloud.io", fill=(100, 116, 139), font=small_font)
    draw.text((40, 920), "CONFIDENTIAL DOCUMENT - FOR RECIPIENT USE ONLY", fill=(148, 163, 184), font=small_font)

    return img


def generate_receipt_sample() -> Image.Image:
    """Generate a clean retail store receipt with PII and pricing."""
    width, height = 500, 750
    img = Image.new("RGB", (width, height), (252, 252, 252))
    draw = ImageDraw.Draw(img)
    title_font, heading_font, body_font, mono_font, small_font = _get_fonts()

    y = 30
    draw.text((160, y), "METRO MART", fill=(20, 20, 20), font=title_font)
    draw.text((140, y + 35), "Store #402 - 5th Avenue", fill=(80, 80, 80), font=body_font)
    draw.text((170, y + 60), "Tel: 555-019-2834", fill=(80, 80, 80), font=small_font)
    draw.line([(30, y + 85), (width - 30, y + 85)], fill=(180, 180, 180), width=1)

    y = 130
    draw.text((30, y), "Date: 22/09/2026", fill=(60, 60, 60), font=small_font)
    draw.text((300, y), "Time: 14:32:05", fill=(60, 60, 60), font=small_font)
    draw.text((30, y + 22), "Cashier: David M.", fill=(60, 60, 60), font=small_font)
    draw.text((300, y + 22), "Receipt: #89211", fill=(60, 60, 60), font=small_font)

    y = 180
    draw.line([(30, y), (width - 30, y)], fill=(180, 180, 180), width=1)
    y += 10
    draw.text((30, y), "ITEM", fill=(40, 40, 40), font=heading_font)
    draw.text((380, y), "PRICE", fill=(40, 40, 40), font=heading_font)
    y += 30

    items = [
        ("Organic Whole Milk 1L", "$3.49"),
        ("Artisan Sourdough Loaf", "$4.99"),
        ("Dark Roast Ground Coffee 250g", "$8.99"),
        ("Greek Yogurt Honey 500g", "$4.29"),
        ("Fresh Valencia Oranges 1kg", "$5.50"),
        ("Avocados Hass (Pack of 3)", "$3.99"),
    ]

    for item, price in items:
        draw.text((30, y), item, fill=(50, 50, 50), font=body_font)
        draw.text((390, y), price, fill=(20, 20, 20), font=mono_font)
        y += 28

    y += 15
    draw.line([(30, y), (width - 30, y)], fill=(180, 180, 180), width=1)
    y += 15
    draw.text((250, y), "SUBTOTAL:", fill=(80, 80, 80), font=body_font)
    draw.text((380, y), "$31.25", fill=(20, 20, 20), font=mono_font)
    y += 26
    draw.text((250, y), "TAX (8%):", fill=(80, 80, 80), font=body_font)
    draw.text((380, y), "$2.50", fill=(20, 20, 20), font=mono_font)
    y += 30
    draw.text((220, y), "TOTAL AMOUNT:", fill=(15, 23, 42), font=heading_font)
    draw.text((370, y), "$33.75", fill=(15, 23, 42), font=heading_font)

    y += 50
    draw.line([(30, y), (width - 30, y)], fill=(200, 200, 200), width=1)
    y += 15
    draw.text((30, y), "Card: Visa Ending in 4521", fill=(80, 80, 80), font=small_font)
    draw.text((30, y + 20), "Auth Code: 092813 - Approved", fill=(80, 80, 80), font=small_font)
    y += 50
    draw.text((150, y), "*** THANK YOU FOR SHOPPING ***", fill=(100, 100, 100), font=small_font)
    draw.text((175, y + 20), "Visit us at metromart.com", fill=(120, 120, 120), font=small_font)

    return img


SAMPLE_DOCUMENTS = {
    "Commercial Invoice": generate_invoice_sample,
    "Retail Store Receipt": generate_receipt_sample,
}
