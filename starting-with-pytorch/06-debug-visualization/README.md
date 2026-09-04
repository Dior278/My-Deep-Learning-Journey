# 06 — Debug and visualization

When something breaks, we usually look at **shapes** or **devices** first.

| File | Run | Contents |
| --- | --- | --- |
| [01-device-placement.py](01-device-placement.py) | `python 06-debug-visualization/01-device-placement.py` | Model and input on the same device |
| [02-shape-debug.py](02-shape-debug.py) | `python 06-debug-visualization/02-shape-debug.py` | Correct MLP vs layer-size mismatch |
| [03-pdb-breakpoint.py](03-pdb-breakpoint.py) | `python 06-debug-visualization/03-pdb-breakpoint.py` | `pdb.set_trace()` before an intentional bug |
| [04-tensorboard.py](04-tensorboard.py) | `python 06-debug-visualization/04-tensorboard.py` | Scalars, histograms, model graph |
| [05-pdb-in-notebook.ipynb](05-pdb-in-notebook.ipynb) | Jupyter | Same pdb idea in a notebook |

pdb at breakpoint: `p x.shape`, `n` (next), `c` (continue), `q` (quit).

TensorBoard:

```bash
tensorboard --logdir runs
```
