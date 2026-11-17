from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config
from .config import CLASS_LABELS
from .document import Check


def _pipeline(args):
    from .pipeline import Pipeline
    return Pipeline(ocr_engine=args.ocr, llm_model=args.model, use_llm=not args.no_llm,
                    llm_fallback=not args.no_fallback, use_profiles=not args.no_profiles)


def cmd_process(args):
    from .pipeline import iter_inputs
    from .report import write_csv, write_json
    paths = list(iter_inputs(args.inputs))
    if not paths:
        print("Nincs feldolgozható fájl.", file=sys.stderr)
        return 1
    pipe = _pipeline(args)
    for problem in pipe.warmup():
        print(f"! {problem}", file=sys.stderr)
    docs = []
    for i, p in enumerate(paths, 1):
        doc = pipe.process_path(p)
        docs.append(doc)
        label = CLASS_LABELS.get(doc.doc_type or "", "?")
        n_err = sum(1 for c in doc.checks if c.level == Check.ERROR)
        n_warn = sum(1 for c in doc.checks if c.level == Check.WARNING)
        print(f"[{i}/{len(paths)}] {p.name}: {label} ({doc.top_probability:.2f}, {doc.type_source}) "
              f"| {len(doc.ocr_lines)} sor | {doc.status()} ({n_err} hiba, {n_warn} figyelmeztetés) "
              f"| {sum(doc.timings.values()):.1f} s")
        for e in doc.errors:
            print(f"    ! {e}")
        if args.verbose:
            for c in doc.checks:
                if c.level != Check.OK:
                    print(f"    - [{c.level}] {c.field}: {c.message}")
    if args.json:
        write_json(docs, args.json)
        print(f"JSON: {args.json}")
    if args.csv:
        write_csv(docs, args.csv)
        print(f"CSV: {args.csv}")
    return 0


def cmd_classify(args):
    from .classifier import DocumentClassifier
    from .pipeline import iter_inputs
    from .preprocess import load_image, normalize_document
    clf = DocumentClassifier()
    for p in iter_inputs(args.inputs):
        img, _ = normalize_document(load_image(p))
        probs = clf.predict(img)
        top = max(probs, key=probs.get)
        detail = ", ".join(f"{k}={v:.2f}" for k, v in sorted(probs.items(), key=lambda kv: -kv[1]))
        print(f"{p.name}: {CLASS_LABELS[top]} ({detail})")
    return 0


def cmd_ocr(args):
    from .ocr import get_engine
    from .pipeline import iter_inputs
    from .preprocess import load_image, normalize_document
    engine = get_engine(args.ocr)
    for p in iter_inputs(args.inputs):
        img, _ = normalize_document(load_image(p))
        print(f"=== {p.name} ({engine.name}, profil: {args.profile}) ===")
        for line in engine.read(img, args.profile):
            print(f"[{line.id:3d}] ({line.confidence:.2f}) {line.text}")
    return 0


def cmd_status(args):
    import torch
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    print(f"M1 modell: {'megvan' if config.CLASSIFIER_PATH.exists() else 'HIÁNYZIK'} ({config.CLASSIFIER_PATH})")
    from .extractor import OllamaClient
    ok, msg = OllamaClient(args.model).available()
    print(f"LLM ({args.model}): {'elérhető' if ok else msg}")
    try:
        from .ocr import TesseractEngine
        t = TesseractEngine()
        print(f"Tesseract: elérhető (nyelv: {t.lang})")
    except Exception as e:
        print(f"Tesseract: {e}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dokulanc", description="Többmodelles dokumentumfeldolgozó lánc")
    ap.add_argument("--ocr", default="easyocr", choices=["easyocr", "tesseract"])
    ap.add_argument("--model", default=config.LLM_MODEL, help="Ollama modell neve")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("process", help="Teljes lánc futtatása fájlokon vagy mappán")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--json", type=Path)
    p.add_argument("--csv", type=Path)
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--no-fallback", action="store_true", help="Bizonytalan típusnál ne kérdezze az LLM-et")
    p.add_argument("--no-profiles", action="store_true", help="Típusfüggő OCR-előfeldolgozás kikapcsolása")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_process)
    c = sub.add_parser("classify", help="Csak típusfelismerés (M1)")
    c.add_argument("inputs", nargs="+")
    c.set_defaults(func=cmd_classify)
    o = sub.add_parser("ocr", help="Csak szövegfelismerés (M2)")
    o.add_argument("inputs", nargs="+")
    o.add_argument("--profile", default="alap", choices=["alap", "tablazat", "nyugta", "urlap"])
    o.set_defaults(func=cmd_ocr)
    s = sub.add_parser("status", help="Környezet és modellek állapota")
    s.set_defaults(func=cmd_status)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
