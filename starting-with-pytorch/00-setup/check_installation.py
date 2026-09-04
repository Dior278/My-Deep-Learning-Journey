"""Verify that PyTorch is installed and report the compute device."""

import torch


def main() -> None:
    print(f"PyTorch version: {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        device = torch.device("cuda")
    else:
        print("PyTorch is using the CPU.")
        device = torch.device("cpu")

    x = torch.rand(2, 3, device=device)
    print(f"Default device: {device}")
    print("Successfully created a tensor:")
    print(x)


if __name__ == "__main__":
    main()
