"""Attach LoRA to an arbitrary nn.Module by name-matching its Linear layers.

Nothing here is model specific. Which projections get adapted is passed in as
target suffixes, so the same code covers DistilBERT (q_lin, v_lin), RoBERTa
(query, value), GPT-2 (c_attn), or anything else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator

import torch.nn as nn

from .layers import LoRALinear

# Attention projection names per model family, matching the paper's Wq and Wv.
TARGET_PRESETS: dict[str, list[str]] = {
    "distilbert": ["q_lin", "v_lin"],
    "roberta": ["query", "value"],
    "bert": ["query", "value"],
    "deberta": ["query_proj", "value_proj"],
}


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: int = 8
    dropout: float = 0.0
    init_std: float | None = 0.02
    target_modules: list[str] = field(default_factory=lambda: ["query", "value"])
    # Modules kept trainable alongside the low-rank matrices. The classification
    # head is randomly initialized, so it has to be trained under every method.
    trainable_modules: list[str] = field(
        default_factory=lambda: ["classifier", "pre_classifier", "score"]
    )

    @classmethod
    def for_model(cls, model_type: str, **kwargs) -> "LoRAConfig":
        """Build a config with the right attention names for a model family."""
        key = next((k for k in TARGET_PRESETS if k in model_type.lower()), None)
        if key is None:
            raise ValueError(
                f"No target preset for model type '{model_type}'. "
                f"Known: {sorted(TARGET_PRESETS)}. Pass target_modules explicitly."
            )
        return cls(target_modules=list(TARGET_PRESETS[key]), **kwargs)


def _matches(name: str, patterns: Iterable[str]) -> bool:
    """True when the dotted module path ends in, or contains, one of the patterns."""
    return any(re.search(rf"(^|\.){re.escape(p)}(\.|$)", name) for p in patterns)


def _set_submodule(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parent_path, _, attr = dotted_name.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, attr, new_module)


def inject_lora(model: nn.Module, config: LoRAConfig) -> tuple[nn.Module, int]:
    """Replace every targeted nn.Linear with a LoRALinear. Returns (model, count)."""
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and _matches(name, config.target_modules)
    ]
    if not targets:
        raise ValueError(
            f"No Linear layers matched {config.target_modules}. "
            "Print [n for n, m in model.named_modules()] to check the names."
        )

    for name, module in targets:
        wrapped = LoRALinear(
            module,
            r=config.r,
            alpha=config.alpha,
            dropout=config.dropout,
            init_std=config.init_std,
        )
        _set_submodule(model, name, wrapped)

    return model, len(targets)


def mark_only_lora_trainable(model: nn.Module, config: LoRAConfig) -> None:
    """Freeze everything except lora_A/lora_B and the explicitly trainable modules."""
    for name, param in model.named_parameters():
        is_lora = name.endswith("lora_A") or name.endswith("lora_B")
        param.requires_grad = is_lora or _matches(name, config.trainable_modules)


def iter_lora_layers(model: nn.Module) -> Iterator[tuple[str, LoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def merge_lora(model: nn.Module) -> None:
    for _, layer in iter_lora_layers(model):
        layer.merge()


def unmerge_lora(model: nn.Module) -> None:
    for _, layer in iter_lora_layers(model):
        layer.unmerge()


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Trainable vs total parameter counts, the headline number of the paper."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = sum(
        p.numel()
        for n, p in model.named_parameters()
        if n.endswith("lora_A") or n.endswith("lora_B")
    )
    return {
        "total": total,
        "trainable": trainable,
        "lora_only": lora,
        "trainable_pct": round(100.0 * trainable / total, 4) if total else 0.0,
    }


def lora_state_dict(model: nn.Module) -> dict:
    """Just the adapter weights, which is the checkpoint you actually ship."""
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
