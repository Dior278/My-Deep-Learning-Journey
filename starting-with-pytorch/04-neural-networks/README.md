# 04 — Neural networks

`nn.Module`: layers in `__init__`, computation in `forward`.

| Notebook | Contents |
| --- | --- |
| [01-nn-modules.ipynb](01-nn-modules.ipynb) | `nn.Module`, `nn.Parameter`, `nn.Sequential`, losses, a tiny training step |
| [02-cnns.ipynb](02-cnns.ipynb) | `Conv2d`, pooling, flattening, a small CNN, TensorBoard logging |

We use `model.train()` during training and `model.eval()` during evaluation — dropout and batch-norm behave differently.
