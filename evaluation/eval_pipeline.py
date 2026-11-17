import argparse
from collections import defaultdict

import common
import matplotlib
import pandas as pd
from tqdm import tqdm

from dokulanc import config
from dokulanc.document import Check
from dokulanc.metrics import compare_fields
from dokulanc.pipeline import Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description="A teljes lánc végponttól végpontig kiértékelése")
    ap.add_argument("-n", type=int, default=8, help="Ép dokumentum típusonként")
    ap.add_argument("--corrupt", type=int, default=6, help="Elrontott dokumentum típusonként (számla, nyugta, űrlap)")
    ap.add_argument("--ocr", default="easyocr")
    args = ap.parse_args()
    pipe = Pipeline(ocr_engine=args.ocr)
    ok, msg = pipe.llm_status()
    if not ok:
        raise SystemExit(msg)
    docs = common.test_documents(args.n, corrupt=False, seed=3)
    docs += [d for d in common.test_documents(args.corrupt, corrupt=True, seed=4) if d[1]["doc_type"] in ("szamla", "nyugta", "urlap")]
    rows, timing = [], defaultdict(list)
    for path, meta in tqdm(docs, desc="lánc"):
        doc = pipe.process_path(path)
        m1_type = max(doc.type_probs, key=doc.type_probs.get) if doc.type_probs else None
        matches = compare_fields(meta["doc_type"], meta["fields"], doc.fields) if doc.doc_type == meta["doc_type"] else {}
        error_fields = {c.field for c in doc.checks if c.level == Check.ERROR}
        expected = set(meta["errors"])
        rows.append({
            "fajl": path.name, "valodi_tipus": meta["doc_type"], "valtozat": meta["variant"],
            "m1_tipus": m1_type, "m1_valoszinuseg": doc.top_probability, "vegso_tipus": doc.doc_type,
            "tipus_forras": doc.type_source, "m1_helyes": m1_type == meta["doc_type"],
            "vegso_helyes": doc.doc_type == meta["doc_type"],
            "mezo_pontossag": sum(matches.values()) / max(len(matches), 1) if matches else 0.0,
            "elrontott": bool(expected), "elvart_hibak": ",".join(sorted(expected)),
            "jelzett_hibak": ",".join(sorted(error_fields)),
            "hiba_felismerve": bool(expected & error_fields) if expected else None,
            "teves_riasztas": bool(error_fields) if not expected else None,
            "figyelmeztetesek": sum(1 for c in doc.checks if c.level == Check.WARNING),
            "ido_s": sum(doc.timings.values()), "hibauzenetek": " | ".join(doc.errors),
        })
        for k, v in doc.timings.items():
            timing[k].append(v)
    df = pd.DataFrame(rows)
    common.save_table(df, "lanc_dokumentumonkent.csv")
    clean = df[~df.elrontott]
    corrupt = df[df.elrontott]
    summary = {
        "dokumentumok": len(df),
        "m1_tipus_pontossag": df.m1_helyes.mean(),
        "vegso_tipus_pontossag": df.vegso_helyes.mean(),
        "llm_visszacsatolas_aranya": df.tipus_forras.str.contains("LLM").mean(),
        "atlagos_mezo_pontossag_ep_dok": clean.mezo_pontossag.mean(),
        "elrontott_dok_hiba_felismerve": corrupt.hiba_felismerve.astype(float).mean() if len(corrupt) else None,
        "ep_dok_teves_hiba_riasztas": clean.teves_riasztas.astype(float).mean() if len(clean) else None,
        "atlagos_ido_s": df.ido_s.mean(),
    }
    s = pd.DataFrame([summary])
    common.save_table(s, "lanc_osszesites.csv")
    by_type = df[~df.elrontott].groupby("valodi_tipus").agg(m1=("m1_helyes", "mean"), vegso=("vegso_helyes", "mean"),
                                                         mezo_pontossag=("mezo_pontossag", "mean"), ido_s=("ido_s", "mean")).reset_index()
    common.save_table(by_type, "lanc_tipusonkent.csv")
    t = pd.DataFrame({"lepes": list(timing), "atlag_s": [sum(v) / len(v) for v in timing.values()],
                      "futasok": [len(v) for v in timing.values()]})
    common.save_table(t, "lanc_idok.csv")
    print(s.T.round(3).to_string(header=False))
    print(by_type.round(3).to_string(index=False))
    print(t.round(3).to_string(index=False))
    fig, ax = plt.subplots(figsize=(7, 3.2))
    labels = {"betoltes": "betöltés", "elofeldolgozas": "előfeldolgozás", "M1_tipus": "M1 típus", "M2_ocr": "M2 OCR",
              "M3_tipus_ellenorzes": "M3 típus-visszacsatolás", "M3_kinyeres": "M3 kinyerés", "ellenorzes": "ellenőrzés"}
    ax.barh([labels.get(k, k) for k in t.lepes], t.atlag_s, color="#4C72B0")
    ax.invert_yaxis()
    ax.set_xlabel("átlagos idő (s)")
    ax.set_title("A lánc lépéseinek futási ideje (RTX 3070)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.FIGURES_DIR / "lanc_idok.png", dpi=130)


if __name__ == "__main__":
    main()
