"""Log a short training run to TensorBoard.

    python 06-debug-visualization/04-tensorboard.py
    tensorboard --logdir runs
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

LOG_DIR = Path("runs") / "simple_experiment"


def main() -> None:
    torch.manual_seed(0)
    LOG_DIR.parent.mkdir(exist_ok=True)
    writer = SummaryWriter(str(LOG_DIR))

    model = nn.Linear(10, 2)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05)

    inputs = torch.randn(100, 10)
    targets = torch.randn(100, 2)

    print(f"Logging to {LOG_DIR}")
    num_epochs = 50

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            writer.add_scalar("Training/Loss", loss.item(), epoch)
            writer.add_histogram("Model/weights", model.weight, epoch)
            writer.add_histogram("Model/bias", model.bias, epoch)
            print(f"Epoch [{epoch + 1}/{num_epochs}]  loss: {loss.item():.4f}")

    writer.add_graph(model, inputs[0].unsqueeze(0))
    writer.close()
    print("Done. Run:  tensorboard --logdir runs")


if __name__ == "__main__":
    main()
