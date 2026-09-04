"""End-to-end linear regression: y = 2x + 1 with noise."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loaders(batch_size: int = 16) -> tuple[DataLoader, DataLoader, float, float]:
    true_weight, true_bias = 2.0, 1.0
    x_train = torch.randn(200, 1) * 5
    y_train = true_weight * x_train + true_bias + 0.5 * torch.randn(200, 1)
    x_val = torch.randn(40, 1) * 5
    y_val = true_weight * x_val + true_bias + 0.5 * torch.randn(40, 1)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, true_weight, true_bias


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
    model.eval()
    total, n = 0.0, 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        total += loss_fn(model(features), labels).item()
        n += 1
    return total / n


def train(
    num_epochs: int = 80,
    lr: float = 1e-2,
) -> None:
    torch.manual_seed(0)
    device = get_device()
    train_loader, val_loader, true_w, true_b = make_loaders()

    model = nn.Linear(1, 1).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    print(f"Device: {device}")
    for epoch in range(1, num_epochs + 1):
        model.train()
        running, batches = 0.0, 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(features), labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
            batches += 1
        if epoch % 10 == 0 or epoch == 1:
            val_loss = evaluate(model, val_loader, loss_fn, device)
            print(f"Epoch {epoch:3d}/{num_epochs}  train {running / batches:.4f}  val {val_loss:.4f}")

    weight = model.weight.detach().cpu().item()
    bias = model.bias.detach().cpu().item()
    print(f"Learned  weight={weight:.3f}  bias={bias:.3f}")
    print(f"True     weight={true_w:.3f}  bias={true_b:.3f}")

    ARTIFACTS.mkdir(exist_ok=True)
    path = ARTIFACTS / "linear_regression.pt"
    torch.save({"state_dict": model.state_dict(), "weight": weight, "bias": bias}, path)
    print(f"Saved {path}")


if __name__ == "__main__":
    train()
