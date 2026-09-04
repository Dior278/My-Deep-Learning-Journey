# Starting PyTorch

Our notes on PyTorch: tensors, autograd, data loading, `nn.Module`, CNNs, training loops, debugging, TensorBoard.

## Setup

**Python 3.10+**. GPU optional — we run everything on CPU too.

From this folder (`starting-with-pytorch`):

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
python 00-setup/check_installation.py
```

Open notebooks in Jupyter, VS Code, or Cursor:

```bash
jupyter notebook
```

## Folders

| # | Folder | Entry point |
| --- | --- | --- |
| 0 | [00-setup](00-setup) | `check_installation.py` |
| 1 | [01-tensors](01-tensors) | creating tensors → operations → indexing / reshape → NumPy |
| 2 | [02-autograd](02-autograd) | `01-autograd.ipynb` |
| 3 | [03-data-loading](03-data-loading) | `01-datasets-dataloaders.ipynb` |
| 4 | [04-neural-networks](04-neural-networks) | `nn.Module`, then CNNs |
| 5 | [05-training](05-training) | training loop, then metrics |
| 6 | [06-debug-visualization](06-debug-visualization) | device, shapes, pdb, TensorBoard |
| 7 | [07-mini-project](07-mini-project) | `train_linear_regression.py` |
| 8 | [final_project](final_project) | fake vs real news (`code.ipynb`) |

Each folder has its own README.

## Layout

```text
starting-with-pytorch/
├── 00-setup/                  # install check
├── 01-tensors/                # the ndarray-like API
├── 02-autograd/               # gradients
├── 03-data-loading/           # Dataset & DataLoader
├── 04-neural-networks/        # nn.Module and Conv2d
├── 05-training/               # loops, checkpoints, metrics
├── 06-debug-visualization/    # pdb, devices, TensorBoard
├── 07-mini-project/           # one script that puts it together
├── final_project/             # fake vs real news (needs Kaggle CSVs)
└── requirements.txt
```

Folders 00–07 use **synthetic data** — no extra downloads. [final_project](final_project) needs `Fake.csv` and `True.csv` from Kaggle (see that folder's README).

Run scripts from **this folder**:

```bash
python 00-setup/check_installation.py
python 06-debug-visualization/04-tensorboard.py
python 07-mini-project/train_linear_regression.py
tensorboard --logdir runs
```

## Reminders

- `model.train()` vs `model.eval()` — dropout and batch-norm depend on it.
- `optimizer.zero_grad()` before `backward()`, or gradients accumulate.
- If `view()` fails, try `reshape` or `.contiguous()`.
