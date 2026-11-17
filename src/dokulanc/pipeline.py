from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from . import config
from .document import Document
from .extractor import Extractor, LlmError, OllamaClient
from .preprocess import PROFILES, SUPPORTED_EXTS, load_image, normalize_document


@contextmanager
def timed(doc: Document, step: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        doc.timings[step] = doc.timings.get(step, 0.0) + time.perf_counter() - start


class Pipeline:
    def __init__(self, classifier_path: str | Path = config.CLASSIFIER_PATH, ocr_engine: str = "easyocr",
                 llm_model: str = config.LLM_MODEL, use_llm: bool = True, llm_fallback: bool = True,
                 threshold: float = config.CLASSIFIER_THRESHOLD, use_profiles: bool = True):
        self.classifier_path = Path(classifier_path)
        self.ocr_name = ocr_engine
        self.use_llm = use_llm
        self.llm_fallback = llm_fallback
        self.threshold = threshold
        self.use_profiles = use_profiles
        self.llm_model = llm_model
        self._classifier = None
        self._ocr = None
        self._extractor = None
        self._validator = None

    @property
    def classifier(self):
        if self._classifier is None:
            from .classifier import DocumentClassifier
            self._classifier = DocumentClassifier(self.classifier_path)
        return self._classifier

    @property
    def ocr(self):
        if self._ocr is None:
            from .ocr import get_engine
            self._ocr = get_engine(self.ocr_name)
        return self._ocr

    @property
    def extractor(self) -> Extractor:
        if self._extractor is None:
            self._extractor = Extractor(OllamaClient(self.llm_model))
        return self._extractor

    @property
    def validator(self):
        if self._validator is None:
            from .validation import Validator
            self._validator = Validator()
        return self._validator

    def warmup(self) -> list[str]:
        problems = []
        for name in ("classifier", "ocr"):
            try:
                getattr(self, name)
            except Exception as e:
                problems.append(str(e))
        return problems

    def llm_status(self) -> tuple[bool, str]:
        if not self.use_llm:
            return False, "Az LLM-lépés ki van kapcsolva."
        return self.extractor.client.available()

    def process_path(self, path: str | Path) -> Document:
        doc = Document(source=str(path))
        try:
            with timed(doc, "betoltes"):
                img = load_image(path)
        except Exception as e:
            doc.errors.append(f"Betöltési hiba: {e}")
            return doc
        return self.process_image(img, str(path), doc)

    def process_image(self, img: np.ndarray, source: str = "kep", doc: Document | None = None,
                      forced_type: str | None = None) -> Document:
        doc = doc or Document(source=source)
        doc.original_image = img
        with timed(doc, "elofeldolgozas"):
            doc.image, doc.preprocessing = normalize_document(img)
        self._classify(doc, forced_type)
        try:
            self._ocr_step(doc)
        except Exception as e:
            doc.errors.append(f"OCR-hiba ({self.ocr_name}): {e}")
            return doc
        llm_ok, llm_msg = self.llm_status()
        if not llm_ok:
            doc.errors.append(llm_msg)
        if llm_ok and forced_type is None:
            self._fallback(doc)
        if llm_ok:
            self._extract(doc)
        with timed(doc, "ellenorzes"):
            if doc.fields:
                doc.parsed, doc.checks = self.validator.validate(doc)
        return doc

    def _classify(self, doc: Document, forced_type: str | None):
        if forced_type:
            doc.doc_type, doc.type_source = forced_type, "kézi"
            doc.type_probs = {forced_type: 1.0}
            return
        with timed(doc, "M1_tipus"):
            try:
                doc.type_probs = self.classifier.predict(doc.image)
                doc.doc_type = max(doc.type_probs, key=doc.type_probs.get)
                doc.type_source = "M1"
            except FileNotFoundError as e:
                doc.errors.append(str(e))
                doc.doc_type, doc.type_source = None, "nincs"

    def _ocr_step(self, doc: Document):
        doc.ocr_profile = PROFILES.get(doc.doc_type or "", "alap") if self.use_profiles else "nincs"
        with timed(doc, "M2_ocr"):
            doc.ocr_lines = self.ocr.read(doc.image, doc.ocr_profile, use_profile=self.use_profiles)
            doc.ocr_engine = self.ocr.name

    def _fallback(self, doc: Document):
        if not self.llm_fallback or not doc.ocr_lines:
            return
        if doc.doc_type is not None and doc.top_probability >= self.threshold:
            return
        with timed(doc, "M3_tipus_ellenorzes"):
            try:
                llm_type, reason = self.extractor.classify(doc.ocr_lines)
            except LlmError as e:
                doc.errors.append(str(e))
                return
        doc.llm_type = llm_type
        doc.preprocessing["llm_tipus_indoklas"] = reason
        if llm_type != doc.doc_type:
            old_profile = doc.ocr_profile
            doc.doc_type = llm_type
            doc.type_source = "M1 bizonytalan → LLM"
            if self.use_profiles and PROFILES.get(llm_type, "alap") != old_profile:
                self._ocr_step(doc)
        else:
            doc.type_source = "M1 + LLM megerősítés"

    def _extract(self, doc: Document):
        if not doc.doc_type or not doc.ocr_lines:
            doc.errors.append("Az adatkinyerés kimaradt: nincs dokumentumtípus vagy felismert szöveg.")
            return
        with timed(doc, "M3_kinyeres"):
            try:
                doc.fields = self.extractor.extract(doc.doc_type, doc.ocr_lines)
            except LlmError as e:
                doc.errors.append(str(e))

    def process_many(self, paths: Iterable[str | Path]) -> Iterator[Document]:
        for p in paths:
            yield self.process_path(p)


def iter_inputs(inputs: Iterable[str | Path]) -> Iterator[Path]:
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.suffix.lower() in SUPPORTED_EXTS:
                    yield child
        elif p.suffix.lower() in SUPPORTED_EXTS:
            yield p
