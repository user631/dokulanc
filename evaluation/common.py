import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dokulanc import config


def test_documents(n_per_type: int, split: str = "test", variants=None, corrupt=None, seed: int = 0):
    rng = random.Random(seed)
    out = []
    for cls in config.CLASSES:
        d = config.SYNTH_DIR / split / cls
        items = []
        for meta_path in sorted(d.glob("*.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if variants and meta["variant"] not in variants:
                continue
            if corrupt is not None and bool(meta["errors"]) != corrupt:
                continue
            items.append((meta_path.with_suffix(".jpg"), meta))
        rng.shuffle(items)
        out.extend(items[:n_per_type])
    return out


def save_table(df, name: str):
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path
