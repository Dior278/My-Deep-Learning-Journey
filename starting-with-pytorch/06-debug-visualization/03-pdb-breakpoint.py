"""Intentional shape bug with a pdb breakpoint just before it.

Run from a real terminal (pdb is awkward inside some notebook UIs):

    python 06-debug-visualization/03-pdb-breakpoint.py

At the prompt:

    p x.shape
    p self.layer3
    c
"""

import pdb

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(100, 10)  # bug: 100 should be 64

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        print("About to enter pdb...")
        pdb.set_trace()
        print("shape before layer3:", tuple(x.shape))
        return self.layer3(x)


if __name__ == "__main__":
    model = SimpleMLP()
    dummy_input = torch.randn(4, 784)
    model(dummy_input)
