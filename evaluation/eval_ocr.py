import argparse
import time

import common
import matplotlib
import pandas as pd
from tqdm import tqdm

from dokulanc import config
from dokulanc.metrics import cer, wer, word_recall
from dokulanc.ocr import get_engine
from dokulanc.preprocess import PROFILES, load_image, normalize_document
from dokulanc.synth import ground_truth_lines

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description="M2 kiértékelése: OCR-motorok és előfeldolgozási profilok összevetése")
    ap.add_argument("-n", type=int, default=12, help="Dokumentum típusonként")
    ap.add_argument("--engines", nargs="+", default=["easyocr", "tesseract"])
    ap.add_argument("--plot-only", action="store_true", help="Csak az ábra újrarajzolása a mentett eredményekből")
    args = ap.parse_args()
    if args.plot_only:
        plot(pd.read_csv(config.RESULTS_DIR / "m2_tipusonkent.csv"))
        return
    docs = common.test_documents(args.n, seed=1)
    prepared = []
    for path, meta in tqdm(docs, desc="előfeldolgozás"):
        img, _ = normalize_document(load_image(path))
        ref = chr(10).join(l.text for l in ground_truth_lines(meta))
        prepared.append((path, meta, img, ref))
    rows = []
    for engine_name in args.engines:
        try:
            engine = get_engine(engine_name)
        except Exception as e:
            print(f"{engine_name} kihagyva: {e}")
            continue
        for use_profile in (True, False):
            for path, meta, img, ref in tqdm(prepared, desc=f"{engine_name} profil={'be' if use_profile else 'ki'}"):
                profile = PROFILES[meta["doc_type"]]
                start = time.perf_counter()
                lines = engine.read(img, profile, use_profile=use_profile)
                elapsed = time.perf_counter() - start
                hyp = "\n".join(l.text for l in lines)
                rows.append({"fajl": path.name, "tipus": meta["doc_type"], "valtozat": meta["variant"], "motor": engine_name,
                             "profil": profile if use_profile else "nincs", "profil_be": use_profile,
                             "cer": cer(ref, hyp), "wer": wer(ref, hyp), "szo_felidezes": word_recall(ref, hyp),
                             "ido_s": elapsed, "sorok": len(lines),
                             "atlag_konfidencia": sum(l.confidence for l in lines) / max(len(lines), 1)})
    df = pd.DataFrame(rows)
    common.save_table(df, "m2_dokumentumonkent.csv")
    summary = df.groupby(["motor", "profil_be"]).agg(cer=("cer", "mean"), wer=("wer", "mean"),
                                                     szo_felidezes=("szo_felidezes", "mean"), ido_s=("ido_s", "mean")).reset_index()
    by_type = df.groupby(["motor", "profil_be", "tipus"]).agg(cer=("cer", "mean"), szo_felidezes=("szo_felidezes", "mean")).reset_index()
    by_variant = df.groupby(["motor", "profil_be", "valtozat"]).agg(cer=("cer", "mean"), szo_felidezes=("szo_felidezes", "mean")).reset_index()
    common.save_table(summary, "m2_osszesites.csv")
    common.save_table(by_type, "m2_tipusonkent.csv")
    common.save_table(by_variant, "m2_valtozatonkent.csv")
    print(summary.to_string(index=False))
    print(by_type.pivot_table(index="tipus", columns=["motor", "profil_be"], values="szo_felidezes").round(3).to_string())
    print(by_variant.pivot_table(index="valtozat", columns=["motor", "profil_be"], values="cer").round(3).to_string())
    plot(by_type)


def plot(by_type: pd.DataFrame):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    label = by_type.motor + by_type.profil_be.map({True: " + profil", False: ""})
    pv = by_type.assign(beallitas=label).pivot(index="tipus", columns="beallitas", values="szo_felidezes")
    pv.index = [config.CLASS_LABELS[i] for i in pv.index]
    ax = pv.plot.bar(figsize=(9, 4), rot=0)
    ax.set_ylabel("Szó-felidézés")
    ax.set_ylim(0, 1.15)
    ax.legend(title=None, ncol=4, loc="upper center", fontsize=8)
    ax.set_xlabel("")
    ax.set_title("M2: OCR-motorok és típusfüggő profilok (szintetikus teszthalmaz)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "m2_ocr_osszevetes.png", dpi=130)


if __name__ == "__main__":
    main()
