"""Low-rank adaptation layer, following Hu et al. (2021), Section 4.1.

h = W0 x + (alpha / r) * B A x

W0 is frozen. A is initialized from a random Gaussian, B is initialized to zero,
so BA = 0 at the start of training and the adapted model is exactly the
pretrained model on step 0.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable rank-r update.

    The wrapper owns the original layer rather than copying its weights, so the
    pretrained tensor is shared (no extra memory) and state_dict keys stay
    traceable back to the base checkpoint.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: int = 8,
        dropout: float = 0.0,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}")

        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.merged = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Paper notation: A is r x k (input side), B is d x r (output side).
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.reset_lora_parameters(init_std)

        # The frozen path never receives gradients.
        self.base_layer.weight.requires_grad_(False)
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad_(False)

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    def reset_lora_parameters(self, init_std: float = 0.02) -> None:
        """Gaussian A, zero B. Set init_std to None for Kaiming init on A.

        The paper says "random Gaussian" without fixing sigma. The reference
        implementation uses Kaiming uniform, which is what most libraries
        inherited. Both are exposed here so the choice can be ablated.
        """
        if init_std is None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            nn.init.normal_(self.lora_A, mean=0.0, std=init_std)
        nn.init.zeros_(self.lora_B)

    def delta_w(self) -> torch.Tensor:
        """The effective weight update, shaped like base_layer.weight."""
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            # Weights already folded in, so the low-rank path would double count.
            return self.base_layer(x)
        base_out = self.base_layer(x)
        lora_out = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return base_out + lora_out * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        """Fold BA into W0 for zero added inference latency (paper, Section 4.1)."""
        if self.merged:
            return
        self.base_layer.weight.data += self.delta_w().to(self.base_layer.weight.dtype)
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            return
        self.base_layer.weight.data -= self.delta_w().to(self.base_layer.weight.dtype)
        self.merged = False

    def extra_repr(self) -> str:
        return (
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.3f}, "
            f"merged={self.merged}"
        )
