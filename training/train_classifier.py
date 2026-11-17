import argparse
import json
import shutil
import time

import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dokulanc import config
from dokulanc.classifier import build_model, device, eval_transform, head_parameters, train_transform

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ManifestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.paths = [config.ROOT / p for p in df["path"]]
        self.labels = [config.CLASSES.index(c) for c in df["class"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("L")
        return self.transform(img), self.labels[i]


def evaluate(model, loader, dev):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(dev))
            ps.extend(out.argmax(1).cpu().tolist())
            ys.extend(y.tolist())
    return accuracy_score(ys, ps), f1_score(ys, ps, average="macro")


def train_one(arch: str, df: pd.DataFrame, args, dev) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_ds = ManifestDataset(df[df.split == "train"], train_transform(config.CLASSIFIER_INPUT))
    val_ds = ManifestDataset(df[df.split == "val"], eval_transform(config.CLASSIFIER_INPUT))
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, batch_size=64, num_workers=args.workers, persistent_workers=args.workers > 0)
    model = build_model(len(config.CLASSES), arch).to(dev)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    history = []
    best_f1, best_state = -1.0, None
    start = time.time()
    total_epochs = args.head_epochs + args.epochs
    for epoch in range(total_epochs):
        if epoch == 0:
            for p in model.parameters():
                p.requires_grad = False
            for p in head_parameters(model):
                p.requires_grad = True
            opt = torch.optim.AdamW(head_parameters(model), lr=1e-3, weight_decay=1e-4)
            sched = None
        if epoch == args.head_epochs:
            for p in model.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        model.train()
        running, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
            n += len(y)
        if sched:
            sched.step()
        acc, f1 = evaluate(model, val_dl, dev)
        history.append({"epoch": epoch + 1, "loss": running / n, "val_acc": acc, "val_f1": f1,
                        "fazis": "fej" if epoch < args.head_epochs else "teljes"})
        print(f"[{arch}] {epoch + 1:2d}/{total_epochs} loss={running / n:.4f} val_acc={acc:.4f} val_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    elapsed = time.time() - start
    params = sum(p.numel() for p in model.parameters())
    ckpt = {"state_dict": best_state, "classes": config.CLASSES, "arch": arch, "input_size": config.CLASSIFIER_INPUT,
            "val_f1": best_f1, "history": history, "train_time_s": elapsed, "params": params}
    out = config.MODELS_DIR / f"classifier_{arch}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)
    return {"arch": arch, "val_f1": best_f1, "train_time_s": round(elapsed, 1), "params_m": round(params / 1e6, 2),
            "path": str(out), "history": history}


def plot_history(results: list[dict]):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for r in results:
        h = pd.DataFrame(r["history"])
        axes[0].plot(h.epoch, h.loss, marker="o", label=r["arch"])
        axes[1].plot(h.epoch, h.val_f1, marker="o", label=r["arch"])
    axes[0].set_title("Tanítási veszteség")
    axes[1].set_title("Validációs makró F1")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "m1_tanitas.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="M1 dokumentumtípus-felismerő tanítása (transfer learning)")
    ap.add_argument("--archs", nargs="+", default=["mobilenet_v3_large", "efficientnet_b0", "resnet18"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--head-epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    df = pd.read_csv(config.DATASET_DIR / "manifest.csv")
    dev = device()
    print(f"Eszköz: {dev}, tanító: {(df.split == 'train').sum()}, validációs: {(df.split == 'val').sum()}")
    results = [train_one(a, df, args, dev) for a in args.archs]
    best = max(results, key=lambda r: r["val_f1"])
    shutil.copy(best["path"], config.CLASSIFIER_PATH)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([{k: v for k, v in r.items() if k != "history"} for r in results])
    table.to_csv(config.RESULTS_DIR / "m1_architekturak.csv", index=False)
    (config.RESULTS_DIR / "m1_tanitas.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    plot_history(results)
    print(table.to_string(index=False))
    print(f"Kiválasztott modell: {best['arch']} -> {config.CLASSIFIER_PATH}")


if __name__ == "__main__":
    main()
