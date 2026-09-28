"""
extractor.py — Structured entity & key-value extraction from OCR text.
Features:
  - Date extraction (multiple formats)
  - Monetary amount and currency detection
  - Email / phone / URL extraction
  - Invoice / receipt field detection
  - PII detection & redaction (Aadhaar, PAN, credit cards, SSN)
  - Document type classifier
"""

import re
from typing import Dict, List, Any, Optional


# ─────────────────────────────────────────────
# Regex Patterns
# ─────────────────────────────────────────────

DATE_PATTERNS = [
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
    r"\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b",
    r"\b(\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\b",
    r"\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4})\b",
]

MONEY_PATTERNS = [
    r"(?:₹|Rs\.?|INR|USD|\$|€|£|EUR|GBP)\s*[\d,]+(?:\.\d{1,2})?",
    r"[\d,]+(?:\.\d{1,2})?\s*(?:₹|Rs\.?|INR|USD|EUR|GBP)",
]

EMAIL_PATTERN = r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
PHONE_PATTERN = r"(?:\+91[\s\-]?)?[6-9]\d{4}[\s\-]?\d{5}|(?:\+?1[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}"
URL_PATTERN = r"https?://[^\s<>\"]+|www\.[^\s<>\"]{2,}"

# PII Patterns
AADHAAR_PATTERN = r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b"
PAN_PATTERN = r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"
CREDIT_CARD_PATTERN = r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"
SSN_PATTERN = r"\b\d{3}[\-]\d{2}[\-]\d{4}\b"
PASSPORT_PATTERN = r"\b[A-Z][1-9][0-9]{7}\b"

# Invoice / Receipt fields
INVOICE_FIELDS = {
    "invoice_number": r"(?i)\b(?:invoice|inv|bill|order)[\s#:\.-]*([A-Z0-9\-_/]{3,30})",
    "po_number": r"(?i)\b(?:p\.?o\.?\s*(?:#|num(?:ber)?)?|purchase\s+order(?:\s*#)?)[\s#:\.-]*([A-Z0-9\-_/]{3,30})",
    "gstin": r"(?i)\b(?:GSTIN|GST\s*No\.?)[\s:\.]*([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]Z[0-9A-Z])",
    "due_date": r"(?i)\b(?:due\s+date|payment\s+due|pay\s+by)[\s:]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
    "total": r"(?i)\b(?:grand\s+total|total\s+due|total\s+amount|amount\s+due|net\s+payable|total)[\s:₹$]*([0-9,]+(?:\.[0-9]{2})?)",
    "subtotal": r"(?i)\b(?:subtotal|sub-total|taxable\s+amount)[\s:₹$]*([0-9,]+(?:\.[0-9]{2})?)",
    "tax": r"(?i)\b(?:tax(?:\s*\(\d+%\))?|gst|vat|igst|cgst|sgst)[\s:₹$%]*([0-9,]+(?:\.[0-9]{2})?)",
    "discount": r"(?i)\b(?:discount|special\s+offer)[\s:₹$%-]*([0-9,]+(?:\.[0-9]{2})?)",
}

DOCUMENT_TYPE_KEYWORDS = {
    "Invoice / Receipt": ["invoice", "receipt", "bill", "amount due", "gst", "total", "subtotal", "payment"],
    "Identity Document": ["aadhaar", "pan", "passport", "driving licence", "voter", "date of birth", "dob", "nationality"],
    "Medical / Prescription": ["patient", "diagnosis", "prescription", "mg", "tablet", "dosage", "doctor", "hospital", "lab"],
    "Legal / Contract": ["whereas", "party", "agreement", "clause", "indemnify", "jurisdiction", "hereby", "execute"],
    "Academic": ["certificate", "degree", "university", "marks", "grade", "cgpa", "semester", "roll no"],
    "Business Card": ["ceo", "manager", "director", "email", "linkedin", "website", "@", "pvt ltd", "inc"],
}


def extract_dates(text: str) -> List[str]:
    dates = []
    for pattern in DATE_PATTERNS:
        dates.extend(re.findall(pattern, text, re.IGNORECASE))
    return list(set(dates))


def extract_amounts(text: str) -> List[str]:
    amounts = []
    for pattern in MONEY_PATTERNS:
        amounts.extend(re.findall(pattern, text))
    return list(set(amounts))


def extract_emails(text: str) -> List[str]:
    return list(set(re.findall(EMAIL_PATTERN, text)))


def extract_phones(text: str) -> List[str]:
    return list(set(re.findall(PHONE_PATTERN, text)))


def extract_urls(text: str) -> List[str]:
    return list(set(re.findall(URL_PATTERN, text)))


def detect_pii(text: str) -> Dict[str, List[str]]:
    """Detect all PII in text and return by category."""
    return {
        "aadhaar": re.findall(AADHAAR_PATTERN, text),
        "pan": re.findall(PAN_PATTERN, text),
        "credit_card": re.findall(CREDIT_CARD_PATTERN, text),
        "ssn": re.findall(SSN_PATTERN, text),
        "passport": re.findall(PASSPORT_PATTERN, text),
        "email": extract_emails(text),
        "phone": extract_phones(text),
    }


def redact_pii(text: str, redact_char: str = "█") -> tuple[str, int]:
    """
    Replace detected PII in text with redaction blocks.
    Returns (redacted_text, count_redacted).
    """
    redacted = text
    count = 0

    patterns = [
        AADHAAR_PATTERN,
        PAN_PATTERN,
        CREDIT_CARD_PATTERN,
        SSN_PATTERN,
        PASSPORT_PATTERN,
    ]
    for pattern in patterns:
        matches = re.findall(pattern, redacted)
        for match in matches:
            replacement = redact_char * len(match)
            redacted = redacted.replace(match, replacement)
            count += 1

    return redacted, count


def extract_invoice_fields(text: str) -> Dict[str, str]:
    """Extract structured key-value invoice fields."""
    fields = {}
    stopwords = {"se", "no", "is", "of", "to", "in", "on", "at", "by", "or", "and", "the", "for"}
    for field_name, pattern in INVOICE_FIELDS.items():
        match = re.search(pattern, text)
        if match:
            val = match.group(1).strip() if match.lastindex else match.group(0).strip()
            # Eliminate trivial noise and punctuation-only values
            clean_val = val.strip(" :#-.")
            if len(clean_val) >= 2 and clean_val.lower() not in stopwords:
                fields[field_name] = clean_val
    return fields


def classify_document(text: str) -> str:
    """Classify the document type based on keyword analysis."""
    text_lower = text.lower()
    scores = {}
    for doc_type, keywords in DOCUMENT_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[doc_type] = score

    if not scores:
        return "General Document"
    return max(scores, key=scores.get)


def extract_all(text: str) -> Dict[str, Any]:
    """Run all extractors and return a consolidated entities dict."""
    return {
        "document_type": classify_document(text),
        "dates": extract_dates(text),
        "amounts": extract_amounts(text),
        "emails": extract_emails(text),
        "phones": extract_phones(text),
        "urls": extract_urls(text),
        "pii": detect_pii(text),
        "invoice_fields": extract_invoice_fields(text),
    }
