import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SYNTH_DIR = DATA_DIR / "synthetic"
RVL_DIR = DATA_DIR / "rvl_cdip"
REAL_DIR = DATA_DIR / "real"
DATASET_DIR = DATA_DIR / "dataset"
SAMPLES_DIR = DATA_DIR / "samples"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"

CLASSES = ["szamla", "nyugta", "level", "oneletrajz", "urlap"]
CLASS_LABELS = {
    "szamla": "Számla",
    "nyugta": "Nyugta",
    "level": "Levél",
    "oneletrajz": "Önéletrajz",
    "urlap": "Űrlap",
}

CLASSIFIER_PATH = MODELS_DIR / "classifier.pt"
CLASSIFIER_INPUT = 256
CLASSIFIER_THRESHOLD = float(os.environ.get("DOKULANC_THRESHOLD", "0.6"))

OCR_LANGS = ["hu", "en"]
TESSERACT_LANG = "hun"
TESSERACT_CMD = os.environ.get("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
TESSDATA_DIR = MODELS_DIR / "tessdata"
LOW_OCR_CONFIDENCE = 0.5
LOW_IMAGE_QUALITY = 0.55

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.environ.get("DOKULANC_LLM", "qwen2.5:7b")
LLM_TIMEOUT = 300
LLM_RETRIES = 2

FUZZY_FOUND_THRESHOLD = 85
AMOUNT_TOLERANCE = 1.0
