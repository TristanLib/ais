"""GRU model for ship trajectory prediction."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class TrajGRU(nn.Module):
    """GRU sequence encoder with a direct multi-step trajectory head."""

    def __init__(self, config: Dict):
        super().__init__()
        arch = config["model"]["architecture"]
        self.d_input = int(arch["d_input"])
        self.hidden_size = int(arch.get("hidden_size", 128))
        self.n_layers = int(arch.get("n_layers", 2))
        self.dropout = float(arch.get("dropout", 0.1))
        self.forecast_steps = int(arch["forecast_steps"])
        self.output_dim = int(arch["output_dim"])
        head_hidden = int(arch.get("head_hidden", self.hidden_size))

        self.gru = nn.GRU(
            input_size=self.d_input,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            dropout=self.dropout if self.n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, head_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(head_hidden, self.output_dim * self.forecast_steps),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)

    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.size(0)
        out, _ = self.gru(x)
        pred = self.head(out[:, -1, :])
        return pred.view(batch_size, self.forecast_steps, self.output_dim)

    def get_model_info(self) -> Dict:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_name": "TrajGRU",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "architecture": {
                "input_dim": self.d_input,
                "hidden_size": self.hidden_size,
                "num_layers": self.n_layers,
                "forecast_steps": self.forecast_steps,
                "output_dim": self.output_dim,
            },
        }


def create_gru_model(config: Dict, model_type: str = "basic") -> TrajGRU:
    if model_type != "basic":
        raise ValueError(f"Unknown GRU model type: {model_type}")
    return TrajGRU(config)
