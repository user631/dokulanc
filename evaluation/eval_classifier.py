import argparse

import common
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from dokulanc import config
from dokulanc.classifier import DocumentClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def predict(clf: DocumentClassifier, paths, batch=64):
    probs = []
    for i in range(0, len(paths), batch):
        imgs = [Image.open(config.ROOT / p).convert("L") for p in paths[i:i + batch]]
        probs.append(clf.predict_pil_batch(imgs))
    return np.concatenate(probs)


def plot_confusion(cm, title, path):
    labels = [config.CLASS_LABELS[c] for c in config.CLASSES]
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Becsült")
    ax.set_ylabel("Valódi")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="M1 kiértékelése a teszthalmazon")
    ap.parse_args()
    df = pd.read_csv(config.DATASET_DIR / "manifest.csv")
    test = df[df.split.isin(["test", "real"])].reset_index(drop=True)
    y = np.array([config.CLASSES.index(c) for c in test["class"]])
    rows = []
    for path in sorted(config.MODELS_DIR.glob("classifier_*.pt")):
        clf = DocumentClassifier(path)
        p = predict(clf, list(test.path))
        pred = p.argmax(1)
        mask = test.split == "test"
        rows.append({"arch": clf.meta["arch"], "teszt_accuracy": accuracy_score(y[mask], pred[mask]),
                     "teszt_makro_f1": f1_score(y[mask], pred[mask], average="macro"),
                     "parameterek_M": round(clf.meta["params"] / 1e6, 2), "tanitasi_ido_s": round(clf.meta["train_time_s"], 1)})
    arch_table = pd.DataFrame(rows)
    common.save_table(arch_table, "m1_architekturak_teszt.csv")
    print(arch_table.to_string(index=False))

    clf = DocumentClassifier(config.CLASSIFIER_PATH)
    probs = predict(clf, list(test.path))
    pred = probs.argmax(1)
    test["becsult"] = [config.CLASSES[i] for i in pred]
    test["valoszinuseg"] = probs.max(1)
    test["helyes"] = test["becsult"] == test["class"]
    groups = []
    for (split, source, variant), g in test.groupby(["split", "source", test["variant"].fillna("")]):
        groups.append({"halmaz": split, "forras": source, "valtozat": variant, "db": len(g),
                       "accuracy": g.helyes.mean(), "atlagos_valoszinuseg": g.valoszinuseg.mean()})
    by_source = pd.DataFrame(groups)
    common.save_table(by_source, "m1_forrasonkent.csv")
    print(by_source.to_string(index=False))
    mask = (test.split == "test").values
    report = classification_report(y[mask], pred[mask], target_names=config.CLASSES, output_dict=True)
    common.save_table(pd.DataFrame(report).T.reset_index().rename(columns={"index": "osztaly"}), "m1_osztalyonkent.csv")
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion(confusion_matrix(y[mask], pred[mask], labels=range(len(config.CLASSES))),
                   f"M1 konfúziós mátrix ({clf.meta['arch']})", config.FIGURES_DIR / "m1_konfuzios_matrix.png")
    thr = config.CLASSIFIER_THRESHOLD
    unsure = test[mask & (test.valoszinuseg < thr)]
    print(f"Küszöb alatti ({thr}) predikciók: {len(unsure)} / {mask.sum()}, ebből hibás: {(~unsure.helyes).sum()}; "
          f"küszöb feletti hibás: {(~test[mask & (test.valoszinuseg >= thr)].helyes).sum()}")
    if (test.split == "real").any():
        r = test[test.split == "real"]
        print(f"Valós fotók: {len(r)} db, accuracy: {r.helyes.mean():.3f}")
    common.save_table(test[~test.helyes][["path", "class", "becsult", "valoszinuseg", "source", "variant"]], "m1_hibak.csv")
    print(f"Teszt accuracy: {accuracy_score(y[mask], pred[mask]):.4f}, makró F1: {f1_score(y[mask], pred[mask], average='macro'):.4f}")


if __name__ == "__main__":
    main()
