import argparse
import csv
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

from dokulanc import config
from dokulanc.preprocess import IMAGE_EXTS, classifier_image, load_image, normalize_document

STORE_SIZE = 288


def collect() -> list[dict]:
    items = []
    for split in ("train", "val", "test"):
        for cls in config.CLASSES:
            for src_name, root in (("synthetic", config.SYNTH_DIR), ("rvl", config.RVL_DIR)):
                d = root / split / cls
                if not d.exists():
                    continue
                for p in sorted(d.iterdir()):
                    if p.suffix.lower() in IMAGE_EXTS:
                        variant = ""
                        meta = p.with_suffix(".json")
                        if meta.exists():
                            variant = json.loads(meta.read_text(encoding="utf-8")).get("variant", "")
                        items.append({"src": str(p), "split": split, "class": cls, "source": src_name, "variant": variant})
    for cls in config.CLASSES:
        d = config.REAL_DIR / cls
        if d.exists():
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in IMAGE_EXTS | {".pdf"}:
                    items.append({"src": str(p), "split": "real", "class": cls, "source": "real", "variant": "photo"})
    return items


def process(item: dict) -> dict:
    img = load_image(item["src"])
    norm, _ = normalize_document(img)
    out_dir = config.DATASET_DIR / item["split"] / item["class"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{item['source']}__{Path(item['src']).stem}.png"
    classifier_image(norm, STORE_SIZE).save(out)
    return {**item, "path": str(out.relative_to(config.ROOT))}


def main():
    ap = argparse.ArgumentParser(description="Tanítóadat összeállítása: előfeldolgozás és egységes méretre hozás")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if config.DATASET_DIR.exists():
        shutil.rmtree(config.DATASET_DIR)
    items = collect()
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for r in tqdm(ex.map(process, items, chunksize=8), total=len(items), desc="előkészítés"):
            rows.append(r)
    manifest = config.DATASET_DIR / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "split", "class", "source", "variant", "src"])
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for r in rows:
        key = (r["split"], r["source"])
        summary[key] = summary.get(key, 0) + 1
    for k in sorted(summary):
        print(k, summary[k])
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
