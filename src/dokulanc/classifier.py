from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import models, transforms

from . import config
from .preprocess import classifier_image

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes: int, arch: str = "mobilenet_v3_large", pretrained: bool = True) -> nn.Module:
    if arch == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Ismeretlen architektúra: {arch}")
    return model


def head_parameters(model: nn.Module):
    if hasattr(model, "fc"):
        return model.fc.parameters()
    return model.classifier.parameters()


def eval_transform(size: int = config.CLASSIFIER_INPUT):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def train_transform(size: int = config.CLASSIFIER_INPUT):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomApply([transforms.RandomRotation(4, fill=255)], p=0.5),
        transforms.RandomApply([transforms.RandomPerspective(0.15, p=1.0, fill=255)], p=0.3),
        transforms.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class DocumentClassifier:
    def __init__(self, path: str | Path = config.CLASSIFIER_PATH, dev: torch.device | None = None):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Nincs betanított típusfelismerő modell: {self.path}. Futtasd: python training/train_classifier.py")
        self.device = dev or device()
        ckpt = torch.load(self.path, map_location=self.device, weights_only=False)
        self.classes: list[str] = ckpt["classes"]
        self.input_size: int = ckpt.get("input_size", config.CLASSIFIER_INPUT)
        self.model = build_model(len(self.classes), ckpt.get("arch", "mobilenet_v3_large"), pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.transform = eval_transform(self.input_size)
        self.meta = {k: v for k, v in ckpt.items() if k not in ("state_dict",)}

    @torch.no_grad()
    def predict(self, img: np.ndarray) -> dict[str, float]:
        pil = classifier_image(img, self.input_size)
        x = self.transform(pil).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0].cpu().numpy()
        return {c: float(p) for c, p in zip(self.classes, probs)}

    @torch.no_grad()
    def predict_pil_batch(self, images) -> np.ndarray:
        x = torch.stack([self.transform(im) for im in images]).to(self.device)
        return torch.softmax(self.model(x), dim=1).cpu().numpy()
