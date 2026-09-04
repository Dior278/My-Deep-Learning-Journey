# 03 — Data loading

`Dataset` = one sample; `DataLoader` = batching, shuffling, workers.

[01-datasets-dataloaders.ipynb](01-datasets-dataloaders.ipynb):

- Custom `Dataset` (`__len__` / `__getitem__`)
- `DataLoader` (`batch_size`, `shuffle`, `num_workers`, `pin_memory`)
- Transforms
- `WeightedRandomSampler` for imbalanced classes
- `collate_fn` for variable-length sequences

Examples use **synthetic tensors** — no CIFAR-10 or image folders. Optional torchvision snippets are commented out.
