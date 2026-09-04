# 07 — Mini-project

Synthetic data, linear model, training loop, validation, checkpoint — same pieces as the training notebooks.

```bash
python 07-mini-project/train_linear_regression.py
```

Learns \(y = 2x + 1\) with noise; writes `artifacts/linear_regression.pt`.

Possible next tweaks for us: 2-layer MLP, TensorBoard logging (`06-debug-visualization`), full checkpoint (`model` + `optimizer` + `epoch`) as in `05-training`.
