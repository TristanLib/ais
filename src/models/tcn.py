"""Temporal convolutional baseline for AIS trajectory forecasting."""

from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """Remove right padding so convolutions remain causal."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size <= 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Residual dilated-convolution block used by the TCN baseline."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        return self.activation(self.net(x) + residual)


class TrajTCN(nn.Module):
    """Dilated temporal convolutional network for fixed-horizon forecasting."""

    def __init__(self, config: Dict) -> None:
        super().__init__()
        arch = config.get("model", {}).get("architecture", {})
        input_dim = int(arch.get("d_input", 5))
        channels = _as_int_list(arch.get("channels", [64, 96, 128]))
        kernel_size = int(arch.get("kernel_size", 3))
        dropout = float(arch.get("dropout", 0.1))
        self.forecast_steps = int(arch.get("forecast_steps", 15))
        self.output_dim = int(arch.get("output_dim", 2))

        layers: list[nn.Module] = []
        prev_channels = input_dim
        for layer_index, out_channels in enumerate(channels):
            layers.append(
                TemporalBlock(
                    prev_channels,
                    int(out_channels),
                    kernel_size=kernel_size,
                    dilation=2**layer_index,
                    dropout=dropout,
                )
            )
            prev_channels = int(out_channels)
        self.encoder = nn.Sequential(*layers)
        head_hidden = int(arch.get("head_hidden", prev_channels))
        self.head = nn.Sequential(
            nn.Linear(prev_channels, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, self.forecast_steps * self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is batch x time x features; Conv1d expects batch x features x time.
        encoded = self.encoder(x.transpose(1, 2))
        last_state = encoded[:, :, -1]
        output = self.head(last_state)
        return output.view(-1, self.forecast_steps, self.output_dim)

    def get_model_info(self) -> Dict:
        return {
            "model_name": "TrajTCN",
            "forecast_steps": self.forecast_steps,
            "output_dim": self.output_dim,
            "parameters": sum(p.numel() for p in self.parameters()),
        }


def _as_int_list(value: Iterable[int] | str | None) -> list[int]:
    if value is None:
        return [64, 96, 128]
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [int(item) for item in value]


def create_tcn_model(config: Dict, model_type: str = "basic") -> TrajTCN:
    if model_type != "basic":
        raise ValueError(f"Unsupported TCN model type: {model_type}")
    return TrajTCN(config)
