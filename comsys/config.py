from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class MIMOOfdmConfig:
    seed: int = 42
    device: str = "cuda"  # falls back to cpu at runtime if unavailable
    carrier_frequency_hz: float = 3.5e9
    subcarrier_spacing_hz: float = 30e3
    fft_size: int = 64
    num_ofdm_symbols: int = 14
    cyclic_prefix_length: int = 16
    pilot_ofdm_symbol_indices: Tuple[int, ...] = (2, 11)
    num_guard_carriers: Tuple[int, int] = (0, 0)
    dc_null: bool = False

    # Point-to-point MIMO is modeled as one logical transmitter carrying
    # `total_streams = num_tx * num_streams_per_tx` spatial streams.
    num_tx: int = 2
    num_streams_per_tx: int = 1
    num_bs_ant: int = 2

    num_bits_per_symbol: int = 2
    coderate: float = 0.5
    code_family: str = "ldpc"  # ldpc | polar

    channel_type: str = "cdl"  # cdl | tdl
    channel_profile: str = "A"
    delay_spread_s: float = 100e-9
    min_speed_mps: float = 0.0
    max_speed_mps: float = 0.0
    direction: str = "uplink"

    estimator_type: str = "ls"  # ls | lmmse | perfect
    interpolation_type: str = "lin"
    lmmse_order: str = "t-f-s"

    decoder_iterations: int = 12
    batch_size: int = 64
    eval_batch_size: int = 128

    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "outputs")

    @property
    def modulation_name(self) -> str:
        return {2: "QPSK", 4: "16QAM", 6: "64QAM"}[self.num_bits_per_symbol]

    @property
    def mimo_name(self) -> str:
        total_streams = self.num_tx * self.num_streams_per_tx
        return f"{total_streams}x{self.num_bs_ant}"

    @property
    def total_streams(self) -> int:
        return self.num_tx * self.num_streams_per_tx


DEFAULT_MCS_TABLE = [
    {"mcs_index": 0, "num_bits_per_symbol": 2, "coderate": 0.50, "code_family": "ldpc"},
    {"mcs_index": 1, "num_bits_per_symbol": 2, "coderate": 0.75, "code_family": "ldpc"},
    {"mcs_index": 2, "num_bits_per_symbol": 4, "coderate": 0.50, "code_family": "ldpc"},
    {"mcs_index": 3, "num_bits_per_symbol": 4, "coderate": 0.75, "code_family": "ldpc"},
    {"mcs_index": 4, "num_bits_per_symbol": 6, "coderate": 2.0 / 3.0, "code_family": "ldpc"},
    {"mcs_index": 5, "num_bits_per_symbol": 6, "coderate": 0.75, "code_family": "ldpc"},
]
