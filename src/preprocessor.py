"""
preprocessor.py — Advanced Computer Vision preprocessing pipeline for documents and live camera photos.
Features:
  - EXIF rotation normalization (vital for mobile/camera photos)
  - Auto-deskew via Hough line transform
  - Background shadow & illumination normalization
  - Unsharp masking for camera blur correction
  - CLAHE adaptive contrast enhancement
  - Bilateral edge-preserving denoising
  - Adaptive Otsu binarization
  - Perspective warp for angled captures
"""

import cv2
import numpy as np
from PIL import Image, ImageOps
import math


def fix_exif_orientation(pil_img: Image.Image) -> Image.Image:
    """Normalize image orientation using EXIF data (crucial for phone & webcam photos)."""
    try:
        return ImageOps.exif_transpose(pil_img)
    except Exception:
        return pil_img


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL Image → OpenCV BGR ndarray."""
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR ndarray → PIL Image."""
    rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def remove_shadows_and_normalize(cv2_img: np.ndarray) -> np.ndarray:
    """
    Remove uneven background shadows and normalize illumination.
    Particularly effective for phone photos with cast shadows or glare.
    """
    # Split into color planes
    rgb_planes = cv2.split(cv2_img)
    result_planes = []

    for plane in rgb_planes:
        # Dilate to approximate background
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        # Difference between plane and estimated background
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        # Normalize to full dynamic range
        norm_img = cv2.normalize(
            diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1
        )
        result_planes.append(norm_img)

    return cv2.merge(result_planes)


def sharpen_image(cv2_img: np.ndarray, strength: float = 1.3) -> np.ndarray:
    """Apply unsharp mask filter to sharpen slightly blurry camera captures."""
    gaussian = cv2.GaussianBlur(cv2_img, (0, 0), 2.0)
    sharpened = cv2.addWeighted(cv2_img, 1.0 + strength, gaussian, -strength, 0)
    return sharpened


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

    # Only correct if tilt is moderate and significant
    if abs(angle) < 0.6 or abs(angle) > 40:
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


def enhance_contrast(img: np.ndarray, clip_limit: float = 2.5) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    merged = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def denoise(img: np.ndarray) -> np.ndarray:
    """Bilateral filter to remove camera noise while preserving text edges."""
    return cv2.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50)


def binarize(img: np.ndarray) -> np.ndarray:
    """Adaptive Gaussian thresholding for crisp black/white document text."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Background shadow removal prior to binarization
    dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg = cv2.medianBlur(dilated, 21)
    diff = 255 - cv2.absdiff(gray, bg)
    norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    
    binary = cv2.adaptiveThreshold(
        norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def detect_document_corners(img: np.ndarray):
    """Detect the 4 corners of a document for perspective warp."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > (img.shape[0] * img.shape[1] * 0.15):
            return approx

    return None


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

    if max_width < 100 or max_height < 100:
        return img

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
    Comprehensive document preprocessing pipeline.
    Modes:
      - 'auto': EXIF fix + deskew + shadow removal + CLAHE + sharpening + gentle denoise
      - 'camera_enhance': Specialized for live camera / phone photos (shadows, glare, blur compensation)
      - 'binarize': High contrast B&W text thresholding
      - 'warp': 4-corner perspective flattening
      - 'raw': Passthrough with EXIF rotation fixed
    """
    # Always normalize EXIF orientation first
    fixed_pil = fix_exif_orientation(pil_img)
    steps_applied = ["exif_orientation_fixed"]

    if mode == "raw":
        return fixed_pil, {"steps": ["raw_passthrough"]}

    cv2_img = pil_to_cv2(fixed_pil)

    # Resolution scaling check: if too small (e.g. low-res webcam), upscale with bicubic
    h, w = cv2_img.shape[:2]
    if min(h, w) < 700:
        scale = 800.0 / min(h, w)
        cv2_img = cv2.resize(cv2_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        steps_applied.append("rescale_upscale")

    # Step 1: Perspective Warp if explicitly selected
    if mode == "warp":
        corners = detect_document_corners(cv2_img)
        if corners is not None:
            cv2_img = perspective_warp(cv2_img, corners)
            steps_applied.append("perspective_warp")
        else:
            steps_applied.append("warp_not_detected")

    # Step 2: Auto Deskew
    deskewed = auto_deskew(cv2_img)
    if not np.array_equal(deskewed, cv2_img):
        steps_applied.append("auto_deskew")
    cv2_img = deskewed

    # Step 3: Camera photo shadow removal & illumination flattening
    if mode in ("camera_enhance", "auto"):
        cv2_img = remove_shadows_and_normalize(cv2_img)
        steps_applied.append("shadow_illumination_norm")

    # Step 4: CLAHE Contrast Enhancement
    cv2_img = enhance_contrast(cv2_img, clip_limit=3.0 if mode == "camera_enhance" else 2.2)
    steps_applied.append("clahe_contrast")

    # Step 5: Unsharp masking to sharpen characters
    cv2_img = sharpen_image(cv2_img, strength=1.2 if mode == "camera_enhance" else 0.8)
    steps_applied.append("unsharp_mask_sharpen")

    # Step 6: Denoise
    cv2_img = denoise(cv2_img)
    steps_applied.append("bilateral_denoise")

    # Step 7: Binarization if requested
    if mode == "binarize":
        cv2_img = binarize(cv2_img)
        steps_applied.append("adaptive_binarize")

    return cv2_to_pil(cv2_img), {"steps": steps_applied}
