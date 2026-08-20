"""Continued masked-language-model pretraining on a CDA-augmented corpus.

This is a separate line of experimentation from src/train.py: that script
fine-tunes a classifier on a GLUE task; this one continues MLM pretraining
on a gender-balanced corpus, so that both the trained checkpoint and the
untouched base model can be scored on CrowS-Pairs with the same architecture
(RobertaForMaskedLM) throughout. No classification head is involved anywhere
in this file, so there's no head-mismatch to work around.

Example:
    python -m src.fairness.train_mlm --method lora --model roberta-base --seed 0
    python -m src.fairness.train_mlm --method full --model roberta-base --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

from ..lora import LoRAConfig, count_parameters, inject_lora, mark_only_lora_trainable, lora_state_dict
from .cda import augment_corpus

METHODS = ("lora", "full")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_augmented_dataset(tokenizer, max_length: int, n_lines: int | None, seed: int):
    """WikiText-2 raw text, gender-augmented, tokenized for MLM."""
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    lines = [t.strip() for t in raw["text"] if len(t.strip()) > 20]
    if n_lines is not None:
        random.Random(seed).shuffle(lines)
        lines = lines[:n_lines]

    augmented = augment_corpus(lines)

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=max_length, padding=False
        )

    from datasets import Dataset

    ds = Dataset.from_dict({"text": augmented})
    return ds.map(tokenize, batched=True, remove_columns=["text"])


def build_model(args):
    model = AutoModelForMaskedLM.from_pretrained(args.model)

    if args.method == "lora":
        lora_config = LoRAConfig.for_model(
            model.config.model_type, r=args.rank, alpha=args.alpha
        )
        model, n_injected = inject_lora(model, lora_config)
        mark_only_lora_trainable(model, lora_config)
        print(f"Injected LoRA into {n_injected} projections: {lora_config.target_modules}")
    # "full": every parameter stays trainable.

    return model, count_parameters(model)


def train(args) -> dict:
    set_seed(args.seed)
    device = resolve_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = build_augmented_dataset(tokenizer, args.max_length, args.n_lines, args.seed)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator
    )

    model, param_stats = build_model(args)
    model.to(device)
    print(json.dumps(param_stats, indent=2))

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            seen += 1
            if args.log_every and step % args.log_every == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} loss {running_loss / seen:.4f}")

        print(f"epoch {epoch} mean loss {running_loss / max(seen, 1):.4f}")

    out_dir = Path(args.output_dir) / f"cda_{args.model.split('/')[-1]}_{args.method}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "lora":
        torch.save(lora_state_dict(model), out_dir / "lora_weights.pt")
    else:
        torch.save(model.state_dict(), out_dir / "model.pt")

    result = {
        "args": vars(args),
        "params": param_stats,
        "wall_clock_sec": round(time.time() - start, 1),
        "checkpoint_dir": str(out_dir),
    }
    (out_dir / "train_info.json").write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_dir}")
    return result


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=METHODS, default="lora")
    p.add_argument("--model", default="roberta-base")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=0.01, dest="weight_decay")
    p.add_argument("--warmup-ratio", type=float, default=0.06, dest="warmup_ratio")
    p.add_argument("--max-length", type=int, default=128, dest="max_length")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=8)
    p.add_argument(
        "--n-lines",
        type=int,
        default=5000,
        dest="n_lines",
        help="Subsample WikiText-2 to this many lines before augmenting (None = full corpus)",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", default="runs_fairness", dest="output_dir")
    p.add_argument("--log-every", type=int, default=50, dest="log_every")
    args = p.parse_args(argv)

    if args.lr is None:
        args.lr = {"lora": 4e-4, "full": 2e-5}[args.method]
    return args


if __name__ == "__main__":
    train(parse_args())
