"""Fine-tune a pretrained encoder on a GLUE task with LoRA, full FT, or head only.

The three methods differ only in which parameters are unfrozen, so they share
one training loop. That is what makes the comparison fair: same data order,
same schedule, same seed handling.

Example:
    python -m src.train --method lora --task mrpc --model roberta-base --seed 0
    python -m src.train --method full --task mrpc --model roberta-base --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoConfig, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from .data import build_dataloaders
from .lora import LoRAConfig, count_parameters, inject_lora, mark_only_lora_trainable
from .metrics import compute_metrics

METHODS = ("lora", "full", "head_only")


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


def build_model(args, num_labels: int) -> tuple[torch.nn.Module, dict]:
    config = AutoConfig.from_pretrained(args.model, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config)

    if args.method == "lora":
        lora_config = LoRAConfig.for_model(
            config.model_type,
            r=args.rank,
            alpha=args.alpha,
            dropout=args.lora_dropout,
        )
        model, n_injected = inject_lora(model, lora_config)
        mark_only_lora_trainable(model, lora_config)
        print(f"Injected LoRA into {n_injected} projections: {lora_config.target_modules}")
    elif args.method == "head_only":
        head = LoRAConfig().trainable_modules
        for name, param in model.named_parameters():
            param.requires_grad = any(h in name for h in head)
    # "full" leaves every parameter trainable.

    return model, count_parameters(model)


@torch.no_grad()
def evaluate(model, loader, task, device) -> dict[str, float]:
    model.eval()
    logits_all, labels_all = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels_all.append(batch.pop("labels").cpu())
        logits_all.append(model(**batch).logits.float().cpu())
    return compute_metrics(torch.cat(logits_all), torch.cat(labels_all), task)


def train(args) -> dict:
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_loader, eval_loader, task, _ = build_dataloaders(
        task_name=args.task,
        tokenizer_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=args.seed,
        train_subset=args.train_subset,
    )

    model, param_stats = build_model(args, task.num_labels)
    model.to(device)
    print(json.dumps(param_stats, indent=2))

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    history, best = [], {}
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * batch["labels"].size(0)
            seen += batch["labels"].size(0)
            if args.log_every and step % args.log_every == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss {running_loss / seen:.4f}")

        scores = evaluate(model, eval_loader, task, device)
        record = {"epoch": epoch, "train_loss": running_loss / seen, **scores}
        history.append(record)
        print(record)

        primary = task.metrics[0]
        if not best or scores[primary] > best[primary]:
            best = {"epoch": epoch, **scores}

    result = {
        "args": vars(args),
        "task": task.name,
        "params": param_stats,
        "history": history,
        "best": best,
        "wall_clock_sec": round(time.time() - start, 1),
    }

    out_dir = Path(args.output_dir) / f"{args.task}_{args.model.split('/')[-1]}_{args.method}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(result, indent=2))
    if args.save_model:
        torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"Wrote {out_dir / 'results.json'}")
    return result


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=METHODS, default="lora")
    p.add_argument("--task", default="mrpc")
    p.add_argument("--model", default="roberta-base")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    p.add_argument("--lr", type=float, default=None, help="Defaults per method if unset")
    p.add_argument("--weight-decay", type=float, default=0.01, dest="weight_decay")
    p.add_argument("--warmup-ratio", type=float, default=0.06, dest="warmup_ratio")
    p.add_argument("--max-grad-norm", type=float, default=1.0, dest="max_grad_norm")
    p.add_argument("--max-length", type=int, default=128, dest="max_length")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=8)
    p.add_argument("--lora-dropout", type=float, default=0.0, dest="lora_dropout")
    p.add_argument("--train-subset", type=int, default=None, dest="train_subset")
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", default="runs", dest="output_dir")
    p.add_argument("--log-every", type=int, default=50, dest="log_every")
    p.add_argument("--save-model", action="store_true", dest="save_model")
    args = p.parse_args(argv)

    if args.lr is None:
        # LoRA tolerates (and needs) a much larger LR than full fine-tuning.
        # Paper Table 9 uses 4e-4 for RoBERTa base on MRPC.
        args.lr = {"lora": 4e-4, "full": 2e-5, "head_only": 1e-3}[args.method]
    return args


if __name__ == "__main__":
    train(parse_args())
