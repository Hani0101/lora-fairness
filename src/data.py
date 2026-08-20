"""GLUE loading and tokenization, driven by a task registry.

Adding another GLUE task is a one-line entry, so MRPC is only the default and
not baked into the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding


@dataclass(frozen=True)
class GlueTask:
    name: str
    text_a: str
    text_b: str | None
    num_labels: int
    metrics: tuple[str, ...]
    validation_split: str = "validation"


GLUE_TASKS: dict[str, GlueTask] = {
    "mrpc": GlueTask("mrpc", "sentence1", "sentence2", 2, ("accuracy", "f1")),
    "sst2": GlueTask("sst2", "sentence", None, 2, ("accuracy",)),
    "rte": GlueTask("rte", "sentence1", "sentence2", 2, ("accuracy",)),
    "cola": GlueTask("cola", "sentence", None, 2, ("matthews_corrcoef",)),
    "qnli": GlueTask("qnli", "question", "sentence", 2, ("accuracy",)),
    "stsb": GlueTask("stsb", "sentence1", "sentence2", 1, ("pearson", "spearman")),
    "mnli": GlueTask(
        "mnli", "premise", "hypothesis", 3, ("accuracy",), "validation_matched"
    ),
}


def get_task(name: str) -> GlueTask:
    try:
        return GLUE_TASKS[name]
    except KeyError:
        raise ValueError(f"Unknown GLUE task '{name}'. Known: {sorted(GLUE_TASKS)}")


def build_dataloaders(
    task_name: str,
    tokenizer_name: str,
    batch_size: int = 16,
    max_length: int = 128,
    seed: int = 42,
    num_workers: int = 0,
    train_subset: int | None = None,
) -> tuple[DataLoader, DataLoader, GlueTask, AutoTokenizer]:
    """Return train and validation loaders for one GLUE task.

    train_subset truncates the training split, which is useful for smoke tests
    and for the low-data regime experiment in the paper (Appendix F.3).
    """
    task = get_task(task_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    raw = load_dataset("nyu-mll/glue", task.name)

    def tokenize(batch):
        args = (batch[task.text_a],) if task.text_b is None else (
            batch[task.text_a],
            batch[task.text_b],
        )
        return tokenizer(*args, truncation=True, max_length=max_length)

    keep = ["input_ids", "attention_mask", "label"]
    encoded = raw.map(tokenize, batched=True)
    encoded = encoded.remove_columns(
        [c for c in encoded["train"].column_names if c not in keep and c != "token_type_ids"]
    )
    encoded = encoded.rename_column("label", "labels")

    train_split = encoded["train"].shuffle(seed=seed)
    if train_subset is not None:
        train_split = train_split.select(range(min(train_subset, len(train_split))))

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(
        train_split,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
    )
    eval_loader = DataLoader(
        encoded[task.validation_split],
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
    )
    return train_loader, eval_loader, task, tokenizer
