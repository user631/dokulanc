from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np

from . import config
from .document import OcrLine
from .preprocess import apply_profile, to_gray


@dataclass
class Segment:
    text: str
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2


def split_columns(segs: list[Segment], min_gap_ratio: float = 0.015, min_share: float = 0.2,
                  max_alignment: float = 0.5) -> list[list[Segment]]:
    if len(segs) < 6:
        return [segs]
    left = min(s.x1 for s in segs)
    right = max(s.x2 for s in segs)
    width = right - left
    if width <= 0:
        return [segs]
    spans = sorted((s.x1, s.x2) for s in segs)
    best = None
    reach = spans[0][1]
    for x1, x2 in spans[1:]:
        if x1 > reach:
            gap_start, gap_end = reach, x1
            center = (gap_start + gap_end) / 2
            if gap_end - gap_start >= min_gap_ratio * width and left + 0.15 * width < center < left + 0.85 * width:
                if best is None or gap_end - gap_start > best[1] - best[0]:
                    best = (gap_start, gap_end)
        reach = max(reach, x2)
    if best is None:
        return [segs]
    cut = (best[0] + best[1]) / 2
    a = [s for s in segs if s.x2 <= cut]
    b = [s for s in segs if s.x1 >= cut]
    if min(len(a), len(b)) < min_share * len(segs):
        return [segs]
    if (_row_alignment(a, b) + _row_alignment(b, a)) / 2 >= max_alignment:
        return [segs]
    return [a, b]


def _row_alignment(ref: list[Segment], other: list[Segment]) -> float:
    hits = 0
    for s in other:
        if any(min(s.y2, t.y2) - max(s.y1, t.y1) > 0.5 * min(s.height, t.height) for t in ref):
            hits += 1
    return hits / max(len(other), 1)


def _rows(segs: list[Segment]) -> list[list[Segment]]:
    segs = sorted(segs, key=lambda s: (s.cy, s.x1))
    rows: list[list[Segment]] = []
    for s in segs:
        placed = False
        for row in reversed(rows[-6:]):
            top = min(r.y1 for r in row)
            bottom = max(r.y2 for r in row)
            overlap = min(bottom, s.y2) - max(top, s.y1)
            if overlap > 0.5 * min(s.height, bottom - top):
                row.append(s)
                placed = True
                break
        if not placed:
            rows.append([s])
    rows.sort(key=lambda r: min(s.y1 for s in r))
    return rows


def group_lines(segments: list[Segment], column_gap: float = 2.2, detect_columns: bool = True) -> list[OcrLine]:
    segs = [s for s in segments if s.text.strip()]
    if not segs:
        return []
    columns = split_columns(segs) if detect_columns else [segs]
    lines = []
    for column in columns:
        for row in _rows(column):
            row.sort(key=lambda s: s.x1)
            h = median(s.height for s in row) or 1.0
            parts = [row[0].text.strip()]
            for prev, cur in zip(row, row[1:]):
                gap = cur.x1 - prev.x2
                parts.append(" | " if gap > column_gap * h else " ")
                parts.append(cur.text.strip())
            text = "".join(parts)
            weights = [max(len(s.text), 1) for s in row]
            conf = sum(s.conf * w for s, w in zip(row, weights)) / sum(weights)
            box = (int(min(s.x1 for s in row)), int(min(s.y1 for s in row)),
                   int(max(s.x2 for s in row)), int(max(s.y2 for s in row)))
            lines.append(OcrLine(id=len(lines) + 1, text=text, bbox=box, confidence=float(conf)))
    return lines


class OcrEngine:
    name = "base"

    def segments(self, gray: np.ndarray) -> list[Segment]:
        raise NotImplementedError

    def read(self, img: np.ndarray, profile: str = "alap", use_profile: bool = True) -> list[OcrLine]:
        if use_profile:
            gray, scale = apply_profile(img, profile)
        else:
            gray, scale = to_gray(img), 1.0
        segs = self.segments(gray)
        if scale != 1.0:
            segs = [Segment(s.text, s.x1 / scale, s.y1 / scale, s.x2 / scale, s.y2 / scale, s.conf) for s in segs]
        return group_lines(segs)


class EasyOcrEngine(OcrEngine):
    name = "easyocr"

    def __init__(self, langs: list[str] | None = None, gpu: bool | None = None, canvas_size: int = 1800):
        import easyocr
        import torch
        self.torch = torch
        self.canvas_size = canvas_size
        self.reader = easyocr.Reader(langs or config.OCR_LANGS, gpu=torch.cuda.is_available() if gpu is None else gpu,
                                     verbose=False)

    def segments(self, gray: np.ndarray) -> list[Segment]:
        results = self.reader.readtext(gray, detail=1, paragraph=False, width_ths=0.7, mag_ratio=1.0,
                                       canvas_size=self.canvas_size)
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        out = []
        for pts, text, conf in results:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            out.append(Segment(text, min(xs), min(ys), max(xs), max(ys), float(conf)))
        return out


class TesseractEngine(OcrEngine):
    name = "tesseract"

    def __init__(self, lang: str = config.TESSERACT_LANG, cmd: str | None = None):
        import pytesseract
        exe = cmd or config.TESSERACT_CMD
        if Path(exe).exists():
            pytesseract.pytesseract.tesseract_cmd = exe
        elif shutil.which("tesseract") is None:
            raise RuntimeError("A Tesseract nincs telepítve (TESSERACT_CMD környezeti változóval adható meg az útvonala).")
        if config.TESSDATA_DIR.exists():
            os.environ["TESSDATA_PREFIX"] = str(config.TESSDATA_DIR)
        self.pt = pytesseract
        self.lang = lang if lang in self.pt.get_languages() else "eng"

    def segments(self, gray: np.ndarray) -> list[Segment]:
        data = self.pt.image_to_data(gray, lang=self.lang, config="--psm 3", output_type=self.pt.Output.DICT)
        out = []
        for i, text in enumerate(data["text"]):
            conf = float(data["conf"][i])
            if not text.strip() or conf < 0:
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            out.append(Segment(text, x, y, x + w, y + h, conf / 100.0))
        return out


_ENGINES: dict[str, OcrEngine] = {}


def get_engine(name: str = "easyocr") -> OcrEngine:
    if name not in _ENGINES:
        if name == "easyocr":
            _ENGINES[name] = EasyOcrEngine()
        elif name == "tesseract":
            _ENGINES[name] = TesseractEngine()
        else:
            raise ValueError(f"Ismeretlen OCR-motor: {name}")
    return _ENGINES[name]
