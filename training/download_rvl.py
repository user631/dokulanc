import argparse
import io

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

from dokulanc import config

REPO = "jordyvl/rvl_cdip_100_examples_per_class"
FILES = {
    "train": "data/train-00000-of-00001-81f1d229db782541.parquet",
    "val": "data/validation-00000-of-00001-00031909a6e73300.parquet",
    "test": "data/test-00000-of-00001-d5e0db6590d27073.parquet",
}
RVL_LABELS = ["letter", "form", "email", "handwritten", "advertisement", "scientific report",
              "scientific publication", "specification", "file folder", "news article", "budget",
              "invoice", "presentation", "questionnaire", "resume", "memo"]
MAPPING = {"letter": "level", "form": "urlap", "questionnaire": "urlap", "invoice": "szamla", "resume": "oneletrajz"}


def download(path: str, target):
    if target.exists():
        return target
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{path}"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return target


def main():
    ap = argparse.ArgumentParser(description="RVL-CDIP részhalmaz letöltése és a projekt osztályaira képezése")
    ap.parse_args()
    raw = config.RVL_DIR / "raw"
    counts = {}
    for split, path in FILES.items():
        pq = download(path, raw / f"{split}.parquet")
        df = pd.read_parquet(pq)
        label_col = "label" if "label" in df.columns else df.columns[-1]
        image_col = "image" if "image" in df.columns else df.columns[0]
        for i, row in tqdm(df.iterrows(), total=len(df), desc=split):
            label = row[label_col]
            name = RVL_LABELS[int(label)] if str(label).isdigit() else str(label)
            cls = MAPPING.get(name)
            if cls is None:
                continue
            img_data = row[image_col]
            data = img_data["bytes"] if isinstance(img_data, dict) else img_data
            img = Image.open(io.BytesIO(data)).convert("L")
            out = config.RVL_DIR / split / cls
            out.mkdir(parents=True, exist_ok=True)
            img.save(out / f"rvl_{name.replace(' ', '_')}_{i:05d}.png")
            counts[(split, cls)] = counts.get((split, cls), 0) + 1
    for k in sorted(counts):
        print(k, counts[k])


if __name__ == "__main__":
    main()
