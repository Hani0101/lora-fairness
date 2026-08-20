"""Correctness checks for the LoRA implementation.

These catch the errors that are otherwise invisible: a LoRA run that silently
trains nothing, or one that changes the model's behavior before step 0.
Run with: python -m pytest tests -q
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lora import (  # noqa: E402
    LoRAConfig,
    LoRALinear,
    count_parameters,
    inject_lora,
    mark_only_lora_trainable,
    merge_lora,
    unmerge_lora,
)


class TinyAttention(nn.Module):
    """Stand-in for a transformer block, with the RoBERTa projection names."""

    def __init__(self, dim=32):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)

    def forward(self, x):
        return self.output(self.query(x) + self.key(x) + self.value(x))


class TinyModel(nn.Module):
    def __init__(self, dim=32, layers=2):
        super().__init__()
        self.layers = nn.ModuleList([TinyAttention(dim) for _ in range(layers)])
        self.classifier = nn.Linear(dim, 2)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.classifier(x)


@pytest.fixture
def x():
    torch.manual_seed(0)
    return torch.randn(4, 32)


def test_identity_at_init(x):
    base = nn.Linear(32, 32)
    lora = LoRALinear(base, r=4, alpha=8)
    # B starts at zero, so the adapted layer must equal the pretrained one.
    assert torch.allclose(lora(x), base(x), atol=1e-6)
    assert torch.count_nonzero(lora.delta_w()) == 0


def test_lora_changes_output_once_b_is_nonzero(x):
    lora = LoRALinear(nn.Linear(32, 32), r=4, alpha=8)
    baseline = lora(x).clone()
    with torch.no_grad():
        lora.lora_B.normal_(std=0.1)
    assert not torch.allclose(lora(x), baseline, atol=1e-6)


def test_merge_is_equivalent_to_unmerged_forward(x):
    lora = LoRALinear(nn.Linear(32, 32), r=4, alpha=16)
    with torch.no_grad():
        lora.lora_B.normal_(std=0.1)
    before = lora(x).clone()
    lora.merge()
    assert torch.allclose(lora(x), before, atol=1e-5)
    lora.unmerge()
    assert torch.allclose(lora(x), before, atol=1e-5)


def test_scaling_uses_alpha_over_r():
    lora = LoRALinear(nn.Linear(8, 8), r=4, alpha=16)
    assert lora.scaling == 4.0
    with torch.no_grad():
        lora.lora_A.fill_(1.0)
        lora.lora_B.fill_(1.0)
    # Each delta entry is sum over r of 1*1, scaled by alpha/r.
    assert torch.allclose(lora.delta_w(), torch.full((8, 8), 4.0 * 4.0))


def test_injection_targets_only_requested_projections():
    model = TinyModel()
    config = LoRAConfig(r=4, alpha=8, target_modules=["query", "value"])
    model, n = inject_lora(model, config)
    assert n == 4  # 2 layers x (query, value)
    assert isinstance(model.layers[0].query, LoRALinear)
    assert isinstance(model.layers[0].value, LoRALinear)
    assert isinstance(model.layers[0].key, nn.Linear)
    assert not isinstance(model.layers[0].key, LoRALinear)


def test_only_lora_and_head_are_trainable():
    model = TinyModel()
    config = LoRAConfig(r=4, alpha=8, target_modules=["query", "value"])
    model, _ = inject_lora(model, config)
    mark_only_lora_trainable(model, config)

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert all("lora_" in n or "classifier" in n for n in trainable)
    assert any("lora_A" in n for n in trainable)
    assert any("classifier" in n for n in trainable)

    stats = count_parameters(model)
    assert 0 < stats["trainable"] < stats["total"]
    assert stats["lora_only"] == 4 * 2 * 4 * 32  # 4 layers x (A + B) x r x dim


def test_frozen_base_receives_no_gradient(x):
    model = TinyModel()
    config = LoRAConfig(r=4, alpha=8, target_modules=["query", "value"])
    model, _ = inject_lora(model, config)
    mark_only_lora_trainable(model, config)

    model(x).sum().backward()
    assert model.layers[0].query.base_layer.weight.grad is None
    assert model.layers[0].query.lora_A.grad is not None
    assert model.layers[0].query.lora_B.grad is not None


def test_merge_all_and_unmerge_all_round_trip(x):
    model = TinyModel()
    config = LoRAConfig(r=4, alpha=8, target_modules=["query", "value"])
    model, _ = inject_lora(model, config)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                module.lora_B.normal_(std=0.05)

    before = model(x).clone()
    merge_lora(model)
    assert torch.allclose(model(x), before, atol=1e-4)
    unmerge_lora(model)
    assert torch.allclose(model(x), before, atol=1e-4)


def test_unmatched_targets_raise():
    with pytest.raises(ValueError, match="No Linear layers matched"):
        inject_lora(TinyModel(), LoRAConfig(target_modules=["not_a_layer"]))
