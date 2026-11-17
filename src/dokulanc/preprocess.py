from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROFILES = {
    "szamla": "tablazat",
    "nyugta": "nyugta",
    "level": "alap",
    "oneletrajz": "alap",
    "urlap": "urlap",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SUPPORTED_EXTS = IMAGE_EXTS | {".pdf"}


class UnsupportedFileError(ValueError):
    pass


def load_image(path: str | Path, page: int = 0, dpi: int = 200) -> np.ndarray:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        import pymupdf as fitz
        with fitz.open(path) as pdf:
            if page >= pdf.page_count:
                raise UnsupportedFileError(f"A PDF-nek nincs {page + 1}. oldala: {path.name}")
            pix = pdf[page].get_pixmap(dpi=dpi)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
    if ext not in IMAGE_EXTS:
        raise UnsupportedFileError(f"Nem támogatott fájltípus: {path.suffix}")
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise UnsupportedFileError(f"A kép nem olvasható: {path.name}")
    return img


def decode_bytes(data: bytes, name: str) -> np.ndarray:
    if name.lower().endswith(".pdf"):
        import pymupdf as fitz
        with fitz.open(stream=data, filetype="pdf") as pdf:
            pix = pdf[0].get_pixmap(dpi=200)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise UnsupportedFileError(f"A kép nem olvasható: {name}")
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def find_document_quad(img: np.ndarray) -> np.ndarray | None:
    gray = to_gray(img)
    scale = 1000 / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray.copy()
    scale = min(scale, 1.0)
    area = small.shape[0] * small.shape[1]
    blur = cv2.GaussianBlur(small, (7, 7), 0)
    candidates = []
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    candidates.append(otsu)
    edges = cv2.dilate(cv2.Canny(blur, 50, 150), np.ones((5, 5), np.uint8))
    candidates.append(edges)
    for mask in candidates:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            a = cv2.contourArea(cnt)
            if a < 0.25 * area or a > 0.97 * area:
                continue
            peri = cv2.arcLength(cnt, True)
            for eps in (0.02, 0.03, 0.05):
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx) and not _touches_border(approx, small.shape):
                    return _order_corners(approx / scale)
    return None


def _touches_border(quad: np.ndarray, shape, margin: float = 0.015, max_corners: int = 0) -> bool:
    h, w = shape[:2]
    pts = quad.reshape(-1, 2)
    near = sum(1 for x, y in pts if x < margin * w or x > (1 - margin) * w or y < margin * h or y > (1 - margin) * h)
    return near > max_corners


def warp_document(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = quad
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def estimate_skew(img: np.ndarray, limit: float = 6.0, step: float = 0.25) -> float:
    gray = to_gray(img)
    scale = 900 / max(gray.shape)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    h, w = binary.shape
    center = (w / 2, h / 2)
    best, best_score = 0.0, -1.0
    for angle in np.arange(-limit, limit + 1e-6, step):
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        profile = rot.sum(axis=1, dtype=np.float64)
        score = float(np.sum(np.diff(profile) ** 2))
        if score > best_score:
            best, best_score = float(angle), score
    return best


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 1e-3:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def normalize_document(img: np.ndarray, detect: bool = True, deskew: bool = True) -> tuple[np.ndarray, dict]:
    info = {"perspektiva_javitva": False, "dolesszog": 0.0, "eredeti_meret": [int(img.shape[1]), int(img.shape[0])]}
    out = img
    if detect:
        quad = find_document_quad(out)
        if quad is not None:
            out = warp_document(out, quad)
            info["perspektiva_javitva"] = True
    if deskew:
        angle = estimate_skew(out)
        if abs(angle) >= 0.25:
            out = rotate(out, angle)
        info["dolesszog"] = round(angle, 2)
    long_side = max(out.shape[:2])
    if long_side > 2400:
        f = 2400 / long_side
        out = cv2.resize(out, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    info["meret"] = [int(out.shape[1]), int(out.shape[0])]
    return out, info


def remove_lines(gray: np.ndarray) -> np.ndarray:
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    h_len = max(40, gray.shape[1] // 15)
    v_len = max(40, gray.shape[0] // 25)
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1)))
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len)))
    mask = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((3, 3), np.uint8))
    out = gray.copy()
    out[mask > 0] = 255
    return out


def apply_profile(img: np.ndarray, profile: str) -> tuple[np.ndarray, float]:
    gray = to_gray(img)
    scale = 1.0
    if profile == "nyugta":
        target = 1100
        if gray.shape[1] < target:
            scale = target / gray.shape[1]
    elif profile in ("tablazat", "urlap", "alap") and max(gray.shape) < 1600:
        scale = 1600 / max(gray.shape)
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    if profile == "nyugta":
        blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
        gray = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
    elif profile in ("tablazat", "urlap"):
        gray = remove_lines(gray)
    return gray, scale


def classifier_image(img: np.ndarray, size: int = 256) -> Image.Image:
    gray = to_gray(img)
    h, w = gray.shape
    side = max(h, w)
    canvas = np.full((side, side), 255, np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = gray
    resized = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    return Image.fromarray(resized)
