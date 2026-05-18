"""Multi-stream classifier with pluggable encoders + fusion.

Family: `multi_stream_bigru`. Per-channel encoder + cross-channel fusion + FC.

`MultiStreamNet` is the generalized model. Two structural knobs:
  - encoder_type in {gru, lstm, bilstm, transformer, conv1d}  (default gru)
  - fusion      in {late_concat, mean_pool, max_pool, attention_pool, mid_fusion}
                                                              (default late_concat)

`MultiStreamBiGRU` is kept as a thin backward-compat subclass
(encoder_type=gru, fusion=late_concat) so pre-iter_0015 specs and tests are
unaffected. The `swap_encoder` and `swap_fusion` architectural mutation
operators flip these two knobs.

Spec hyperparams:
  - per_channel_hidden: unified capacity knob (GRU/LSTM hidden, Transformer
    d_model, conv base_channels). Default 32.
  - per_channel_layers: depth of each per-channel encoder. Default 1.
  - encoder_type, fusion: see above.
  - fusion_dropout: dropout before the final FC. Default 0.2.

Input shape: (B, T, C). Each channel becomes its own univariate encoder input
(except mid_fusion, which projects + concatenates channels before one shared
encoder).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn

from ai4pain.baselines import run_pytorch_model


ENCODER_TYPES = ("gru", "lstm", "bilstm", "transformer", "conv1d")
FUSION_MODES = ("late_concat", "mean_pool", "max_pool", "attention_pool",
                "mid_fusion")


def _pick_num_heads(d_model: int) -> int:
    """Largest head count in {4,2,1} that divides d_model."""
    for h in (4, 2, 1):
        if d_model % h == 0:
            return h
    return 1


def _positional_encoding(T: int, d_model: int,
                          device: torch.device | None = None) -> torch.Tensor:
    pe = torch.zeros(T, d_model, device=device)
    position = torch.arange(0, T, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float, device=device)
        * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class _RecurrentEncoder(nn.Module):
    """GRU / LSTM / BiLSTM encoder: (B, T, in_dim) -> (B, out_dim)."""

    def __init__(self, in_dim: int, hidden: int, layers: int,
                 kind: str, dropout: float):
        super().__init__()
        bidir = kind in ("gru", "bilstm")
        rnn_dropout = dropout if layers > 1 else 0.0
        cls = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn = cls(input_size=in_dim, hidden_size=hidden,
                       num_layers=layers, batch_first=True,
                       bidirectional=bidir, dropout=rnn_dropout)
        self.out_dim = hidden * (2 if bidir else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)            # (B, T, out_dim)
        return out.mean(dim=1)          # (B, out_dim)


class _TransformerEncoder(nn.Module):
    """Mini-Transformer encoder: (B, T, in_dim) -> (B, d_model)."""

    def __init__(self, in_dim: int, d_model: int, layers: int,
                 dropout: float):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.d_model = d_model
        heads = _pick_num_heads(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.out_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        pe = _positional_encoding(x.shape[1], self.d_model, device=x.device)
        x = x + pe.unsqueeze(0)
        x = self.encoder(x)
        return x.mean(dim=1)


class _ConvEncoder(nn.Module):
    """1D-conv encoder: (B, T, in_dim) -> (B, base_channels)."""

    def __init__(self, in_dim: int, base_channels: int, layers: int,
                 kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        blocks: list[nn.Module] = [
            nn.Conv1d(in_dim, base_channels, kernel_size, padding=pad),
            nn.BatchNorm1d(base_channels), nn.ReLU(inplace=True),
        ]
        for _ in range(layers):
            blocks += [
                nn.Conv1d(base_channels, base_channels, kernel_size, padding=pad),
                nn.BatchNorm1d(base_channels), nn.ReLU(inplace=True),
            ]
        self.conv = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = base_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_dim) -> (B, in_dim, T)
        x = x.transpose(1, 2)
        x = self.conv(x)
        return self.pool(x).squeeze(-1)


def _build_encoder(encoder_type: str, in_dim: int, hidden: int,
                   layers: int, dropout: float) -> nn.Module:
    if encoder_type in ("gru", "lstm", "bilstm"):
        return _RecurrentEncoder(in_dim, hidden, layers, encoder_type, dropout)
    if encoder_type == "transformer":
        return _TransformerEncoder(in_dim, hidden, layers, dropout)
    if encoder_type == "conv1d":
        return _ConvEncoder(in_dim, hidden, layers)
    raise ValueError(f"unknown encoder_type {encoder_type!r}; "
                     f"expected one of {ENCODER_TYPES}")


class MultiStreamNet(nn.Module):
    """Per-channel encoder + cross-channel fusion. (B, T, C) -> logits."""

    def __init__(self, in_channels: int = 4, per_channel_hidden: int = 32,
                 per_channel_layers: int = 1,
                 encoder_type: str = "gru",
                 fusion: str = "late_concat",
                 fusion_dropout: float = 0.2,
                 num_classes: int = 3):
        super().__init__()
        if encoder_type not in ENCODER_TYPES:
            raise ValueError(f"unknown encoder_type {encoder_type!r}; "
                             f"expected one of {ENCODER_TYPES}")
        if fusion not in FUSION_MODES:
            raise ValueError(f"unknown fusion {fusion!r}; "
                             f"expected one of {FUSION_MODES}")
        self.in_channels = in_channels
        self.encoder_type = encoder_type
        self.fusion = fusion
        self.per_channel_hidden = per_channel_hidden
        self.last_attn_weights: torch.Tensor | None = None
        self.dropout = nn.Dropout(fusion_dropout)

        if fusion == "mid_fusion":
            # Project each channel per-timestep, concat, then ONE shared encoder.
            proj_dim = max(2, per_channel_hidden // in_channels)
            self.channel_proj = nn.ModuleList(
                [nn.Linear(1, proj_dim) for _ in range(in_channels)])
            self.encoders = None
            self.shared_encoder = _build_encoder(
                encoder_type, in_channels * proj_dim, per_channel_hidden,
                per_channel_layers, fusion_dropout)
            fc_in = self.shared_encoder.out_dim
        else:
            self.channel_proj = None
            self.shared_encoder = None
            self.encoders = nn.ModuleList([
                _build_encoder(encoder_type, 1, per_channel_hidden,
                               per_channel_layers, fusion_dropout)
                for _ in range(in_channels)
            ])
            enc_out = self.encoders[0].out_dim
            if fusion == "late_concat":
                fc_in = in_channels * enc_out
            else:  # mean_pool, max_pool, attention_pool
                fc_in = enc_out
            if fusion == "attention_pool":
                self.attn = nn.Linear(enc_out, 1)

        self.fc = nn.Linear(fc_in, num_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Fused pre-FC embedding. (B, T, C) -> (B, fc_in). Exposed so
        MultiStreamAuxNet can reuse the body without the classification head."""
        if self.fusion == "mid_fusion":
            projected = [self.channel_proj[c](x[:, :, c:c + 1])
                         for c in range(self.in_channels)]
            fused_seq = torch.cat(projected, dim=-1)  # (B, T, C*proj)
            return self.shared_encoder(fused_seq)     # (B, out_dim)

        embs = [self.encoders[c](x[:, :, c:c + 1])
                for c in range(self.in_channels)]      # each (B, enc_out)
        if self.fusion == "late_concat":
            return torch.cat(embs, dim=1)
        if self.fusion == "mean_pool":
            return torch.stack(embs, dim=1).mean(dim=1)
        if self.fusion == "max_pool":
            return torch.stack(embs, dim=1).max(dim=1).values
        if self.fusion == "attention_pool":
            stacked = torch.stack(embs, dim=1)         # (B, C, enc_out)
            logits = self.attn(stacked).squeeze(-1)    # (B, C)
            weights = torch.softmax(logits, dim=1)     # (B, C)
            self.last_attn_weights = weights.detach()
            return (stacked * weights.unsqueeze(-1)).sum(dim=1)
        raise ValueError(self.fusion)  # pragma: no cover - guarded in __init__

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.embed(x)))


class MultiStreamBiGRU(MultiStreamNet):
    """Backward-compat: MultiStreamNet fixed to encoder_type=gru,
    fusion=late_concat (the pre-iter_0015 architecture)."""

    def __init__(self, in_channels: int = 4, per_channel_hidden: int = 32,
                 per_channel_layers: int = 1,
                 fusion: str = "late_concat",
                 fusion_dropout: float = 0.2,
                 num_classes: int = 3):
        super().__init__(
            in_channels=in_channels, per_channel_hidden=per_channel_hidden,
            per_channel_layers=per_channel_layers, encoder_type="gru",
            fusion=fusion, fusion_dropout=fusion_dropout,
            num_classes=num_classes)


def _multi_stream_factory(in_channels: int, T_max: int, model_cfg: dict,
                           num_classes: int) -> nn.Module:
    return MultiStreamNet(
        in_channels=in_channels,
        per_channel_hidden=int(model_cfg.get("per_channel_hidden", 32)),
        per_channel_layers=int(model_cfg.get("per_channel_layers", 1)),
        encoder_type=model_cfg.get("encoder_type", "gru"),
        fusion=model_cfg.get("fusion", "late_concat"),
        fusion_dropout=float(model_cfg.get("fusion_dropout", 0.2)),
        num_classes=num_classes,
    )


def train_multi_stream(spec: dict, data_root: Path, out_dir: Path) -> dict:
    return run_pytorch_model(_multi_stream_factory, spec, data_root, out_dir,
                              name_tag="multi_stream")


def run_from_dir(run_dir: Path, data_root: Path) -> dict:
    run_dir = Path(run_dir)
    spec = json.loads((run_dir / "spec.json").read_text())
    from ai4pain.multiseed import run_multiseed
    return run_multiseed(train_multi_stream, spec, Path(data_root), run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "raw")
    args = parser.parse_args()
    run_from_dir(args.run_dir, args.data_root)
