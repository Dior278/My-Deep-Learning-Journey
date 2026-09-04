# 05 — Training

Our training step:

1. `optimizer.zero_grad()`
2. forward
3. loss
4. `loss.backward()`
5. `optimizer.step()`

| Notebook | Contents |
| --- | --- |
| [01-training-loop.ipynb](01-training-loop.ipynb) | Full loop on `y = 2x + 1`, validation, save weights, pause and resume via checkpoint |
| [02-metrics-logging.ipynb](02-metrics-logging.ipynb) | Reusable `train_one_epoch` / `evaluate`, accuracy, matplotlib curves |

Saved weights go to `artifacts/` (gitignored).
