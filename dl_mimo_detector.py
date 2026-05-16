from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from sionna.phy import Block

ROOT = Path(__file__).resolve().parents[1]
COMSYS_DIR = ROOT / "comsys"
if str(COMSYS_DIR) not in sys.path:
    sys.path.insert(0, str(COMSYS_DIR))

from baseline_system import TraditionalMIMOOfdmSystem  # noqa: E402
from config import MIMOOfdmConfig  # noqa: E402


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        hidden_channels = channels * expansion
        self.conv1 = nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return F.gelu(x + residual)


class ResidualStage(nn.Module):
    def __init__(self, channels: int, num_blocks: int, expansion: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([ResidualConvBlock(channels, expansion=expansion) for _ in range(num_blocks)])
        self.stage_norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stage_input = x
        for block in self.blocks:
            x = block(x)
        return F.gelu(self.stage_norm(x + stage_input))


class GridNeuralMIMODetector(nn.Module):
    """A compact CNN that maps OFDM grids to coded-bit LLRs."""

    def __init__(
        self,
        cfg: MIMOOfdmConfig,
        hidden_channels: int = 96,
        num_stages: int = 4,
        blocks_per_stage: int = 2,
        expansion: int = 2,
        num_heads: int = 3,
        head_blocks: int = 2,
        gate_hidden_dim: int = 128,
        gate_temperature: float = 1.0,
        num_mcs: int = 1,
        mcs_embedding_dim: int = 8,
    ):
        super().__init__()
        self.cfg = cfg
        self.total_streams = cfg.total_streams
        self.num_heads = num_heads
        self.gate_temperature = gate_temperature
        self.num_mcs = num_mcs
        self.mcs_embedding_dim = mcs_embedding_dim

        num_rx_features = 2 * cfg.num_bs_ant
        num_channel_features = 2 * cfg.num_bs_ant * self.total_streams
        num_context_features = self.total_streams + (mcs_embedding_dim if num_mcs > 1 else 0)
        in_channels = num_rx_features + num_channel_features + num_context_features
        out_channels = self.total_streams * cfg.num_bits_per_symbol

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.backbone = nn.ModuleList(
            [ResidualStage(hidden_channels, num_blocks=blocks_per_stage, expansion=expansion) for _ in range(num_stages)]
        )
        self.global_skip_proj = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
        self.expert_heads = nn.ModuleList(
            [
                nn.Sequential(
                    ResidualStage(hidden_channels, num_blocks=head_blocks, expansion=expansion),
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.GELU(),
                    nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
                )
                for _ in range(num_heads)
            ]
        )
        self.gate_pool = nn.AdaptiveAvgPool2d(1)
        self.mcs_embedding = nn.Embedding(num_mcs, mcs_embedding_dim) if num_mcs > 1 else None
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_channels + 1 + (mcs_embedding_dim if num_mcs > 1 else 0), gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, num_heads),
        )
        nn.init.zeros_(self.gate_network[-1].weight)
        nn.init.zeros_(self.gate_network[-1].bias)

    def _prepare_features(
        self,
        y: torch.Tensor,
        h_hat: torch.Tensor,
        noise_variance: torch.Tensor,
        pilot_mask: torch.Tensor,
        mcs_index: torch.Tensor | int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch_size = y.shape[0]
        y = y.squeeze(1)  # [B, Nr, T, F]
        h_hat = h_hat.squeeze(1)  # [B, Nr, Nt, Ns, T, F]
        noise_variance = torch.as_tensor(noise_variance, dtype=torch.float32, device=y.device).reshape(-1)
        if noise_variance.numel() == 1:
            noise_variance = noise_variance.repeat(batch_size)
        noise_db = 10.0 * torch.log10(torch.clamp(noise_variance, min=1e-12))

        # Normalize the observation and CSI by the noise standard deviation so
        # the network sees a compressed dynamic range across a wide SNR sweep.
        noise_std = torch.sqrt(torch.clamp(noise_variance, min=1e-12))
        y = y / noise_std[:, None, None, None]
        h_hat = h_hat / noise_std[:, None, None, None, None, None]

        y_features = torch.cat([y.real, y.imag], dim=1)

        h_hat = h_hat.reshape(
            batch_size,
            self.cfg.num_bs_ant * self.total_streams,
            self.cfg.num_ofdm_symbols,
            self.cfg.fft_size,
        )
        h_features = torch.cat([h_hat.real, h_hat.imag], dim=1)

        pilot_feature = pilot_mask.unsqueeze(0).expand(batch_size, -1, -1, -1)
        feature_tensors = [y_features, h_features, pilot_feature]

        mcs_embedding = None
        if self.mcs_embedding is not None:
            if mcs_index is None:
                raise ValueError("mcs_index must be provided when num_mcs > 1")
            mcs_index = torch.as_tensor(mcs_index, dtype=torch.long, device=y.device).reshape(-1)
            if mcs_index.numel() == 1:
                mcs_index = mcs_index.repeat(batch_size)
            mcs_embedding = self.mcs_embedding(mcs_index)
            mcs_feature = mcs_embedding[:, :, None, None].expand(
                batch_size, self.mcs_embedding_dim, self.cfg.num_ofdm_symbols, self.cfg.fft_size
            )
            feature_tensors.append(mcs_feature)

        return torch.cat(feature_tensors, dim=1), noise_db, mcs_embedding

    def forward(
        self,
        y: torch.Tensor,
        h_hat: torch.Tensor,
        noise_variance: torch.Tensor,
        pilot_mask: torch.Tensor,
        mcs_index: torch.Tensor | int | None = None,
        return_gate_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features, noise_db, mcs_embedding = self._prepare_features(y, h_hat, noise_variance, pilot_mask, mcs_index)
        x = self.input_proj(features)
        global_skip = self.global_skip_proj(x)
        for stage in self.backbone:
            x = stage(x)
        x = F.gelu(x + global_skip)

        pooled = self.gate_pool(x).flatten(1)
        gate_inputs = [pooled, noise_db[:, None]]
        if mcs_embedding is not None:
            gate_inputs.append(mcs_embedding)
        gate_input = torch.cat(gate_inputs, dim=1)
        gate_logits = self.gate_network(gate_input)
        gate_weights = torch.softmax(gate_logits / self.gate_temperature, dim=1)

        expert_logits = torch.stack([head(x) for head in self.expert_heads], dim=1)
        logits = torch.sum(expert_logits * gate_weights[:, :, None, None, None], dim=1)
        if return_gate_weights:
            return logits, gate_weights
        return logits


class NeuralReceiverSystem(Block):
    """
    End-to-end system using a neural detector that directly predicts coded-bit LLRs.
    """

    def __init__(
        self,
        cfg: MIMOOfdmConfig,
        use_perfect_csi_for_ai: bool = False,
        hidden_channels: int = 96,
        num_stages: int = 4,
        blocks_per_stage: int = 2,
        expansion: int = 2,
        num_heads: int = 3,
        head_blocks: int = 2,
        gate_hidden_dim: int = 128,
        gate_temperature: float = 1.0,
        num_mcs: int = 1,
        mcs_embedding_dim: int = 8,
    ):
        super().__init__()
        self.cfg = cfg
        self.reference_link = TraditionalMIMOOfdmSystem(cfg)
        self.total_streams = self.reference_link.cfg.total_streams
        self.detector = GridNeuralMIMODetector(
            self.reference_link.cfg,
            hidden_channels=hidden_channels,
            num_stages=num_stages,
            blocks_per_stage=blocks_per_stage,
            expansion=expansion,
            num_heads=num_heads,
            head_blocks=head_blocks,
            gate_hidden_dim=gate_hidden_dim,
            gate_temperature=gate_temperature,
            num_mcs=num_mcs,
            mcs_embedding_dim=mcs_embedding_dim,
        ).to(self.reference_link.cfg.device)
        self.decoder = self.reference_link.decoder
        self.use_perfect_csi_for_ai = use_perfect_csi_for_ai

        pilot_mask = self.reference_link.resource_grid.pilot_pattern.mask.float()
        self.pilot_mask = pilot_mask.reshape(
            self.reference_link.cfg.total_streams,
            self.reference_link.cfg.num_ofdm_symbols,
            self.reference_link.cfg.fft_size,
        ).to(self.reference_link.cfg.device)
        self.data_mask = (~self.reference_link.resource_grid.pilot_pattern.mask.bool()).to(self.reference_link.cfg.device)

    def _extract_data_llrs(self, llr_grid: torch.Tensor) -> torch.Tensor:
        batch_size = llr_grid.shape[0]
        llr_grid = llr_grid.reshape(
            batch_size,
            1,
            self.total_streams,
            self.cfg.num_bits_per_symbol,
            self.cfg.num_ofdm_symbols,
            self.cfg.fft_size,
        )

        outputs = []
        for tx_idx in range(1):
            stream_outputs = []
            for stream_idx in range(self.total_streams):
                mask = self.data_mask[tx_idx, stream_idx].reshape(-1)
                values = llr_grid[:, tx_idx, stream_idx].reshape(batch_size, self.cfg.num_bits_per_symbol, -1)
                values = values[:, :, mask].permute(0, 2, 1).reshape(batch_size, -1)
                stream_outputs.append(values)
            outputs.append(torch.stack(stream_outputs, dim=1))
        return torch.stack(outputs, dim=1)

    def call(
        self,
        batch_size: int,
        ebno_db: torch.Tensor | float,
        mcs_index: torch.Tensor | int | None = None,
        return_intermediates: bool = False,
    ):
        inter = self.reference_link(batch_size, ebno_db, return_intermediates=True)
        h_input = inter["channel_freq"] if self.use_perfect_csi_for_ai else inter["channel_estimate"]
        detector_output = self.detector(
            inter["rx_grid"],
            h_input,
            inter["noise_variance"],
            self.pilot_mask,
            mcs_index=mcs_index,
            return_gate_weights=return_intermediates,
        )
        if return_intermediates:
            llr_grid, gate_weights = detector_output
        else:
            llr_grid = detector_output
        llr = self._extract_data_llrs(llr_grid)
        bits_hat = self.decoder(llr)

        if not return_intermediates:
            return inter["bits"], bits_hat

        result: Dict[str, torch.Tensor] = dict(inter)
        result["predicted_llr"] = llr
        result["predicted_llr_grid"] = llr_grid
        result["decoded_bits_ai"] = bits_hat
        result["gate_weights"] = gate_weights
        if mcs_index is not None:
            result["mcs_index"] = torch.as_tensor(mcs_index, device=bits_hat.device)
        return result


def coded_bce_loss(predicted_llr: torch.Tensor, codewords: torch.Tensor) -> torch.Tensor:
    # For the PyTorch backend in this environment, positive logits align with bit=1.
    return F.binary_cross_entropy_with_logits(predicted_llr, codewords.float())
