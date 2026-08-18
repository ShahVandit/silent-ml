"""P1 pipeline: small CNN on FashionMNIST (image classification).

Ported in spirit from the DeepCrime MNIST subject (an 8-layer conv model, Keras)
to a compact, CPU-fast PyTorch training script. This is the CLEAN template;
episode generators mutate a copy of this file to inject one silent bug.

The whole script is intentionally explicit and self-contained: the model, the
data pipeline, the config, and the training loop are all here so that any
injected fault is visible in the source the debugging agent reads and patches.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from silentml.pipelines.base import DataMeta, EvalMetrics, History

# --- Configuration -----------------------------------------------------------
CONFIG = {
    "lr": 1e-3,
    "batch_size": 64,
    "epochs": 3,
    "optimizer": "adam",
    "n_train": 6000,
    "n_val": 1000,
    "weight_decay": 0.0,
}

CLASS_NAMES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# FashionMNIST channel statistics (dataset-wide mean/std).
NORM_MEAN = 0.2860
NORM_STD = 0.3530

# Shared dataset cache, anchored to the installed package (not this file's path)
# so buggy/patched copies placed in episode dirs all reuse one download.
import silentml  # noqa: E402

_DATA_ROOT = Path(
    os.environ.get("SILENTML_DATA", Path(silentml.__file__).resolve().parent.parent / ".data")
)


# --- Model -------------------------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.act = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.act(self.conv1(x)))   # 28 -> 14
        x = self.pool(self.act(self.conv2(x)))   # 14 -> 7
        x = torch.flatten(x, 1)
        x = self.dropout(self.act(self.fc1(x)))
        x = self.fc2(x)
        return x


def build_model() -> nn.Module:
    return SmallCNN(num_classes=10)


# --- Data --------------------------------------------------------------------
def get_dataloaders(seed: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((NORM_MEAN,), (NORM_STD,)),
    ])
    train_full = datasets.FashionMNIST(
        root=str(_DATA_ROOT), train=True, download=True, transform=transform
    )
    val_full = datasets.FashionMNIST(
        root=str(_DATA_ROOT), train=False, download=True, transform=transform
    )

    g = torch.Generator().manual_seed(seed)
    train_idx = torch.randperm(len(train_full), generator=g)[: CONFIG["n_train"]]
    val_idx = torch.randperm(len(val_full), generator=g)[: CONFIG["n_val"]]
    train_set = Subset(train_full, train_idx.tolist())
    val_set = Subset(val_full, val_idx.tolist())

    loader_g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=0, generator=loader_g,
    )
    val_loader = DataLoader(
        val_set, batch_size=256, shuffle=False, num_workers=0,
    )
    meta = DataMeta(num_classes=10, class_names=CLASS_NAMES, input_shape=(1, 28, 28))
    return train_loader, val_loader, meta


# --- Training ----------------------------------------------------------------
def _make_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    if config["optimizer"] == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )
    return torch.optim.SGD(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )


def train(model, train_loader, val_loader, config, seed) -> History:
    device = torch.device("cpu")
    model.to(device)
    optimizer = _make_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()

    hist = History(train_loss=[], val_loss=[], train_acc=[], val_acc=[])
    for _epoch in range(config["epochs"]):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)
        hist.train_loss.append(running_loss / total)
        hist.train_acc.append(correct / total)

        val = evaluate(model, val_loader)
        hist.val_loss.append(val.loss)
        hist.val_acc.append(val.accuracy)
    return hist


@torch.no_grad()
def evaluate(model, val_loader) -> EvalMetrics:
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    per_class_correct: dict[int, int] = {}
    per_class_total: dict[int, int] = {}
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        total_loss += criterion(outputs, targets).item() * inputs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        for t, p in zip(targets.tolist(), preds.tolist()):
            per_class_total[t] = per_class_total.get(t, 0) + 1
            per_class_correct[t] = per_class_correct.get(t, 0) + int(p == t)
    per_class_acc = {
        c: per_class_correct.get(c, 0) / per_class_total[c]
        for c in sorted(per_class_total)
    }
    return EvalMetrics(
        accuracy=correct / total,
        per_class_accuracy=per_class_acc,
        loss=total_loss / total,
    )


if __name__ == "__main__":
    # Manual smoke run.
    from silentml.pipelines.base import run_pipeline

    result = run_pipeline(Path(__file__).parent, seed=0)
    print("val_acc:", result.eval.accuracy)
