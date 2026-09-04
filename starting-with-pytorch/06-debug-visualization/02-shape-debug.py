"""A correct MLP, then the same net with a wrong last-layer size."""

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, last_in: int = 64):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(last_in, 10)

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        return self.layer3(x)


def run(last_in: int) -> None:
    dummy_input = torch.randn(4, 784)
    model = SimpleMLP(last_in=last_in)
    try:
        output = model(dummy_input)
        print(f"last_in={last_in}: OK, output {tuple(output.shape)}")
    except RuntimeError as e:
        print(f"last_in={last_in}: caught error:\n{e}\n")


if __name__ == "__main__":
    print("Correct architecture:")
    run(last_in=64)
    print("Broken architecture (layer3 expects 100 features, layer2 produced 64):")
    run(last_in=100)
