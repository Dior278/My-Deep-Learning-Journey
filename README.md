# My Deep Learning Journey

Personal notes and projects as we learn deep learning. Each folder is one topic; we add a new one when we start it.

## Contents

| Folder | What it is |
| --- | --- |
| [starting-with-pytorch](starting-with-pytorch) | Tensors, autograd, `nn.Module`, training, then a fake vs real news classifier |


## Setup

Clone this repo, then go into the folder you want to work on:

```bash
git clone <repo-url>
cd My-Deep-Learning-Journey/starting-with-pytorch

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
python 00-setup/check_installation.py
```

Each project folder has its own README and `requirements.txt` when it needs one.
