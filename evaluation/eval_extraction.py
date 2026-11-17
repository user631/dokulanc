import argparse
import json
import time

import common
import matplotlib
import pandas as pd
from tqdm import tqdm

from dokulanc import config
from dokulanc.extractor import Extractor, LlmError
from dokulanc.metrics import compare_fields, compare_items
from dokulanc.ocr import get_engine
from dokulanc.preprocess import PROFILES, load_image, normalize_document
from dokulanc.synth import ground_truth_lines

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description="M3 kiértékelése ideális szövegen és OCR-szövegen (hibaterjedés)")
    ap.add_argument("-n", type=int, default=10, help="Dokumentum típusonként")
    ap.add_argument("--ocr", default="easyocr")
    ap.add_argument("--plot-only", action="store_true", help="Csak az ábra újrarajzolása a mentett eredményekből")
    args = ap.parse_args()
    if args.plot_only:
        plot(pd.read_csv(config.RESULTS_DIR / "m3_tipusonkent.csv"), args.ocr)
        return
    extractor = Extractor()
    ok, msg = extractor.client.available()
    if not ok:
        raise SystemExit(msg)
    engine = get_engine(args.ocr)
    docs = common.test_documents(args.n, seed=2)
    field_rows, doc_rows = [], []
    for path, meta in tqdm(docs, desc="kinyerés"):
        t = meta["doc_type"]
        img, _ = normalize_document(load_image(path))
        inputs = {"idealis": ground_truth_lines(meta), "ocr": engine.read(img, PROFILES[t])}
        for cond, lines in inputs.items():
            start = time.perf_counter()
            try:
                pred = extractor.extract(t, lines)
                err = ""
            except LlmError as e:
                pred, err = {}, str(e)
            elapsed = time.perf_counter() - start
            matches = compare_fields(t, meta["fields"], pred)
            for name, ok_ in matches.items():
                field_rows.append({"fajl": path.name, "tipus": t, "valtozat": meta["variant"], "bemenet": cond, "mezo": name, "helyes": ok_,
                                   "elvart": json.dumps(meta["fields"].get(name), ensure_ascii=False),
                                   "kapott": json.dumps((pred.get(name) or {}).get("ertek") if isinstance(pred.get(name), dict) else pred.get(name), ensure_ascii=False)})
            doc_rows.append({"fajl": path.name, "tipus": t, "valtozat": meta["variant"], "bemenet": cond,
                             "mezo_pontossag": sum(matches.values()) / max(len(matches), 1),
                             "minden_mezo_helyes": all(matches.values()), "ido_s": elapsed, "hiba": err,
                             **compare_items(t, meta["fields"], pred)})
    fields = pd.DataFrame(field_rows)
    docs_df = pd.DataFrame(doc_rows)
    common.save_table(fields, "m3_mezonkent.csv")
    common.save_table(docs_df, "m3_dokumentumonkent.csv")
    by_type = docs_df.groupby(["tipus", "bemenet"]).agg(mezo_pontossag=("mezo_pontossag", "mean"),
                                                        hibatlan_dok=("minden_mezo_helyes", "mean"),
                                                        ido_s=("ido_s", "mean")).reset_index()
    common.save_table(by_type, "m3_tipusonkent.csv")
    by_field = fields.groupby(["tipus", "mezo", "bemenet"]).helyes.mean().unstack("bemenet").reset_index()
    common.save_table(by_field, "m3_mezo_pontossag.csv")
    by_variant = docs_df.groupby(["bemenet", "valtozat"]).mezo_pontossag.mean().unstack("bemenet")
    print(by_type.pivot(index="tipus", columns="bemenet", values="mezo_pontossag").round(3).to_string())
    print(by_variant.round(3).to_string())
    print(docs_df.groupby("bemenet")[["mezo_pontossag", "minden_mezo_helyes", "ido_s"]].mean().round(3).to_string())
    worst = by_field.assign(kulonbseg=by_field["idealis"] - by_field["ocr"]).sort_values("kulonbseg", ascending=False).head(10)
    print("Legnagyobb OCR okozta romlás mezőnként:")
    print(worst.round(3).to_string(index=False))
    plot(by_type, args.ocr)


def plot(by_type: pd.DataFrame, ocr_name: str):
    pv = by_type.pivot(index="tipus", columns="bemenet", values="mezo_pontossag")[["idealis", "ocr"]]
    pv.index = [config.CLASS_LABELS[i] for i in pv.index]
    pv.columns = ["ideális szöveg (ground truth)", f"OCR-szöveg ({ocr_name})"]
    ax = pv.plot.bar(figsize=(9, 4), rot=0, color=["#4C72B0", "#DD8452"])
    ax.set_ylim(0, 1.15)
    ax.legend(ncol=2, loc="upper center", fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("Mezőszintű pontosság")
    ax.set_title("M3 hibaterjedés: ugyanaz az LLM ideális és OCR-bemeneten")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.FIGURES_DIR / "m3_hibaterjedes.png", dpi=130)


if __name__ == "__main__":
    main()
