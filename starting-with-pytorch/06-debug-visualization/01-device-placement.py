"""Show why the model and the input must live on the same device."""

import torch
import torch.nn as nn


def main() -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU (the mismatch below is simulated only when CUDA is present).")

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    model = SimpleNet().to(device)
    print("Model parameters are on:", next(model.parameters()).device)

    # Intentional mistake: tensor created on CPU even if the model is on GPU
    input_data = torch.randn(8, 10)
    print("Input data is on:", input_data.device)

    try:
        output = model(input_data)
        print("Forward pass succeeded (both were on the same device).")
        print("Output shape:", tuple(output.shape))
    except RuntimeError as e:
        print("\nCaught an error:")
        print(e)
        print("\nFix: input_data = input_data.to(device)  # or create it with device=device")
        output = model(input_data.to(device))
        print("After the fix, output shape:", tuple(output.shape))


if __name__ == "__main__":
    main()
