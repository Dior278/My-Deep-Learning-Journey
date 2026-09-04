# 02 — Autograd

PyTorch builds a computation graph on the fly; we call `.backward()` for gradients.

[01-autograd.ipynb](01-autograd.ipynb):

- `requires_grad` and gradient accumulation
- `torch.no_grad()` for inference
- `.detach()` vs cloning (not the same)
- `zero_()` before a new backward pass
