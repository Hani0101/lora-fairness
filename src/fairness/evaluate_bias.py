"""Compare CrowS-Pairs bias scores across base / LoRA / full-fine-tuned checkpoints.

Loads a fresh pretrained RobertaForMaskedLM three times -- once untouched, once
with LoRA adapters injected and loaded from a saved checkpoint, once with a
full fine-tuned state dict loaded -- and scores each on CrowS-Pairs. Nothing
here trains anything; this is read-only evaluation.

Example:
    python -m src.fairness.evaluate_bias \\
        --model roberta-base \\
        --lora-checkpoint runs_fairness/cda_roberta-base_lora_seed0/lora_weights.pt \\
        --full-checkpoint runs_fairness/cda_roberta-base_full_seed0/model.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from ..lora import LoRAConfig, inject_lora
from .crows_pairs import evaluate_bias, load_crows_pairs


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_base(model_name: str) -> torch.nn.Module:
    return AutoModelForMaskedLM.from_pretrained(model_name)


def load_lora(model_name: str, checkpoint_path: str, rank: int, alpha: int) -> torch.nn.Module:
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    config = LoRAConfig.for_model(model.config.model_type, r=rank, alpha=alpha)
    model, _ = inject_lora(model, config)

    state = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [m for m in missing if "lora_" in m]
    if real_missing:
        raise RuntimeError(f"LoRA checkpoint did not cover expected keys: {real_missing}")
    return model


def load_full(model_name: str, checkpoint_path: str) -> torch.nn.Module:
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    return model


def run(args) -> dict:
    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pairs = load_crows_pairs()
    print(f"Loaded {len(pairs)} CrowS-Pairs examples")

    results = {}

    print("\n=== base (no fine-tuning) ===")
    base_model = load_base(args.model)
    results["base"] = evaluate_bias(pairs, base_model, tokenizer, device).__dict__
    del base_model

    if args.lora_checkpoint:
        print("\n=== lora ===")
        lora_model = load_lora(args.model, args.lora_checkpoint, args.rank, args.alpha)
        results["lora"] = evaluate_bias(pairs, lora_model, tokenizer, device).__dict__
        del lora_model

    if args.full_checkpoint:
        print("\n=== full fine-tuning ===")
        full_model = load_full(args.model, args.full_checkpoint)
        results["full"] = evaluate_bias(pairs, full_model, tokenizer, device).__dict__
        del full_model

    print("\n" + json.dumps(results, indent=2))

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "bias_comparison.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path / 'bias_comparison.json'}")
    return results


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="roberta-base")
    p.add_argument("--lora-checkpoint", default=None, dest="lora_checkpoint")
    p.add_argument("--full-checkpoint", default=None, dest="full_checkpoint")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", default="runs_fairness", dest="output_dir")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
