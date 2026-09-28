"""
preprocessor.py — OpenCV Computer Vision preprocessing pipeline.
Features:
  - Auto-deskew via Hough line transform
  - CLAHE contrast enhancement
  - Bilateral denoising
  - Perspective warp (4-corner flattening)
  - Adaptive binarization
"""

import cv2
import numpy as np
from PIL import Image
import math


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL Image → OpenCV BGR ndarray."""
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR ndarray → PIL Image."""
    rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def auto_deskew(img: np.ndarray) -> np.ndarray:
    """Detect text tilt and rotate image to align text horizontally."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Find coordinates of white pixels
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        return img

    # minAreaRect gives angle
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Only correct if tilt is significant
    if abs(angle) < 0.5 or abs(angle) > 45:
        return img

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) per channel."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    merged = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def denoise(img: np.ndarray) -> np.ndarray:
    """Bilateral filter to remove noise while preserving text edges."""
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)


def binarize(img: np.ndarray) -> np.ndarray:
    """Adaptive Otsu binarization for high-contrast black/white output."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def detect_document_corners(img: np.ndarray):
    """
    Detect the 4 corners of a document for perspective warp.
    Returns corners array or None if not detected.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    screen_cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screen_cnt = approx
            break

    return screen_cnt


def order_points(pts):
    """Order points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def perspective_warp(img: np.ndarray, corners) -> np.ndarray:
    """Flatten a perspective-distorted document using 4-point transform."""
    pts = corners.reshape(4, 2).astype("float32")
    rect = order_points(pts)
    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))
    return warped


def full_preprocess(pil_img: Image.Image, mode: str = "auto") -> tuple[Image.Image, dict]:
    """
    Full preprocessing pipeline.
    mode: 'raw' | 'auto' | 'binarize' | 'warp'
    Returns (processed_pil_image, metadata_dict)
    """
    cv2_img = pil_to_cv2(pil_img)
    steps_applied = []

    if mode == "raw":
        return pil_img, {"steps": ["raw"]}

    # Step 1: Auto deskew
    deskewed = auto_deskew(cv2_img)
    if not np.array_equal(deskewed, cv2_img):
        steps_applied.append("deskew")
    cv2_img = deskewed

    # Step 2: Contrast enhancement
    cv2_img = enhance_contrast(cv2_img)
    steps_applied.append("CLAHE contrast")

    # Step 3: Denoise
    cv2_img = denoise(cv2_img)
    steps_applied.append("bilateral denoise")

    if mode == "binarize":
        cv2_img = binarize(cv2_img)
        steps_applied.append("Otsu binarize")
    elif mode == "warp":
        corners = detect_document_corners(cv2_img)
        if corners is not None:
            cv2_img = perspective_warp(cv2_img, corners)
            steps_applied.append("perspective warp")
        else:
            steps_applied.append("warp not detected")

    return cv2_to_pil(cv2_img), {"steps": steps_applied}
