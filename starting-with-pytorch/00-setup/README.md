# 00 — Setup

We check that Python, PyTorch, and (optionally) CUDA work before opening notebooks.

## Run

```bash
python 00-setup/check_installation.py
```

Expect PyTorch version, CUDA availability, and a small random tensor.

## Environment

From `starting-with-pytorch`:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

Register a Jupyter kernel if needed:

```bash
python -m ipykernel install --user --name starting-pytorch --display-name "Python (Starting PyTorch)"
```
