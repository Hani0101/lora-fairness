"""Metric functions selected by name from the task registry.

Implemented with numpy/scipy rather than `evaluate` so runs stay offline once
the dataset is cached.
"""

from __future__ import annotations

import numpy as np
import torch


def _accuracy(preds, labels):
    return float((preds == labels).mean())


def _f1(preds, labels, positive: int = 1):
    tp = float(((preds == positive) & (labels == positive)).sum())
    fp = float(((preds == positive) & (labels != positive)).sum())
    fn = float(((preds != positive) & (labels == positive)).sum())
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def _matthews(preds, labels):
    tp = float(((preds == 1) & (labels == 1)).sum())
    tn = float(((preds == 0) & (labels == 0)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denom) if denom else 0.0


def _pearson(preds, labels):
    return float(np.corrcoef(preds, labels)[0, 1])


def _spearman(preds, labels):
    rank = lambda x: np.argsort(np.argsort(x))
    return _pearson(rank(preds), rank(labels))


METRIC_FNS = {
    "accuracy": _accuracy,
    "f1": _f1,
    "matthews_corrcoef": _matthews,
    "pearson": _pearson,
    "spearman": _spearman,
}


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, task) -> dict[str, float]:
    labels_np = labels.numpy()
    if task.num_labels == 1:  # regression, e.g. STS-B
        preds = logits.squeeze(-1).numpy()
    else:
        preds = logits.argmax(dim=-1).numpy()
    return {name: METRIC_FNS[name](preds, labels_np) for name in task.metrics}
