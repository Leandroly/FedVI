# generate_data.py
from __future__ import annotations
from typing import List, Sequence, Tuple, Dict
import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

def get_mnist(root: str = "./data"):
    tfm = transforms.ToTensor()
    train = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    test  = datasets.MNIST(root=root, train=False, download=True, transform=tfm)
    return train, test

def _split_iid(n_items: int, n_clients: int, seed: int = 0) -> List[Sequence[int]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n_items)
    rng.shuffle(idx)
    splits = np.array_split(idx, n_clients)
    return [s.tolist() for s in splits]

def _split_dirichlet(labels: Sequence[int],
                     n_clients: int,
                     alpha: float = 0.5,
                     seed: int = 0,
                     min_size_per_client: int = 10) -> List[Sequence[int]]:

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    classes = np.unique(labels)

    idx_by_class: Dict[int, np.ndarray] = {c: np.where(labels == c)[0] for c in classes}
    for c in classes:
        rng.shuffle(idx_by_class[c])

    client_indices = [[] for _ in range(n_clients)]

    for c in classes:
        idx_c = idx_by_class[c]
        if len(idx_c) == 0:
            continue
        probs = rng.dirichlet(alpha * np.ones(n_clients))
        cuts = (np.cumsum(probs) * len(idx_c)).astype(int)[:-1]
        parts = np.split(idx_c, cuts)
        for k in range(n_clients):
            client_indices[k].extend(parts[k].tolist())

    all_idx = np.arange(len(labels))
    for k in range(n_clients):
        if len(client_indices[k]) < min_size_per_client:
            need = min_size_per_client - len(client_indices[k])
            extra = rng.choice(all_idx, size=need, replace=False).tolist()
            client_indices[k].extend(extra)

    for k in range(n_clients):
        rng.shuffle(client_indices[k])

    return client_indices

def mnist_subsets(n_clients: int,
                  scheme: str = "iid",
                  *,
                  alpha: float = 0.5,
                  seed: int = 0,
                  root: str = "./data") -> Tuple[List[Subset], datasets.MNIST]:

    train, test = get_mnist(root=root)

    if scheme.lower() in ("iid",):
        parts = _split_iid(len(train), n_clients, seed=seed)
    elif scheme.lower() in ("dirichlet", "noniid"):
        labels = np.array(train.targets)
        parts = _split_dirichlet(labels, n_clients, alpha=alpha, seed=seed)
    else:
        raise ValueError(f"Unknown partition scheme: {scheme}")

    subsets = [Subset(train, idxs) for idxs in parts]
    return subsets, test
