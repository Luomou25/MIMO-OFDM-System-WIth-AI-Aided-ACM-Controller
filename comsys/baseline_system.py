from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from sionna.phy import Block
from sionna.phy.channel import OFDMChannel
from sionna.phy.channel.tr38901 import Antenna, AntennaArray, CDL, TDL
from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder
from sionna.phy.fec.polar import Polar5GDecoder, Polar5GEncoder
from sionna.phy.mapping import BinarySource, Demapper, Mapper
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import (
    LMMSEEqualizer,
    LMMSEInterpolator,
    LSChannelEstimator,
    ResourceGrid,
    ResourceGridMapper,
)
from sionna.phy.utils import ebnodb2no

from config import MIMOOfdmConfig


def resolve_device(device: str) -> str:
    if device == "cuda":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def set_seed(seed: int) -> None:
    # Avoid resetting Python/NumPy RNG during repeated simulator construction.
    # Re-seeding those generators here would collapse dataset diversity for
    # workflows such as ACM label generation that instantiate many systems.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_config(cfg: MIMOOfdmConfig, **updates) -> MIMOOfdmConfig:
    cloned = copy.deepcopy(cfg)
    for key, value in updates.items():
        setattr(cloned, key, value)
    return cloned


def exponential_covariance(size: int, rho: float, device: str) -> torch.Tensor:
    index = torch.arange(size, dtype=torch.float32, device=device)
    distance = torch.abs(index[:, None] - index[None, :])
    return (rho**distance).to(torch.complex64)


class TraditionalMIMOOfdmSystem(Block):
    """Traditional coded MIMO-OFDM chain built from Sionna blocks."""

    def __init__(self, cfg: MIMOOfdmConfig):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)
        self.cfg.device = resolve_device(self.cfg.device)
        set_seed(self.cfg.seed)

        self.num_logical_tx = 1
        self.num_streams = self.cfg.total_streams

        self.resource_grid = ResourceGrid(
            num_ofdm_symbols=self.cfg.num_ofdm_symbols,
            fft_size=self.cfg.fft_size,
            subcarrier_spacing=self.cfg.subcarrier_spacing_hz,
            num_tx=self.num_logical_tx,
            num_streams_per_tx=self.num_streams,
            cyclic_prefix_length=self.cfg.cyclic_prefix_length,
            num_guard_carriers=self.cfg.num_guard_carriers,
            dc_null=self.cfg.dc_null,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=list(self.cfg.pilot_ofdm_symbol_indices),
            device=self.cfg.device,
        )
        self.stream_management = StreamManagement(
            np.ones((1, self.num_logical_tx), dtype=np.int32),
            self.num_streams,
        )

        self.n = int(self.resource_grid.num_data_symbols * self.cfg.num_bits_per_symbol)
        self.k = max(32, int(math.floor(self.n * self.cfg.coderate)))
        self.actual_coderate = self.k / self.n

        self.binary_source = BinarySource()
        self.encoder, self.decoder = self._build_coding_blocks()
        self.mapper = Mapper("qam", self.cfg.num_bits_per_symbol, device=self.cfg.device)
        self.rg_mapper = ResourceGridMapper(self.resource_grid, device=self.cfg.device)

        self.channel_model = self._build_channel_model()
        self.channel = OFDMChannel(
            self.channel_model,
            self.resource_grid,
            normalize_channel=True,
            return_channel=True,
            device=self.cfg.device,
        )

        self.channel_estimator = self._build_channel_estimator()
        self.equalizer = LMMSEEqualizer(
            self.resource_grid,
            self.stream_management,
            device=self.cfg.device,
        )
        self.demapper = Demapper(
            "app",
            "qam",
            self.cfg.num_bits_per_symbol,
            hard_out=False,
            device=self.cfg.device,
        )

    def _build_coding_blocks(self):
        code_family = self.cfg.code_family.lower()
        if code_family == "ldpc":
            encoder = LDPC5GEncoder(
                self.k,
                self.n,
                num_bits_per_symbol=self.cfg.num_bits_per_symbol,
                device=self.cfg.device,
            )
            decoder = LDPC5GDecoder(
                encoder,
                hard_out=True,
                return_infobits=True,
                num_iter=self.cfg.decoder_iterations,
                device=self.cfg.device,
            )
            return encoder, decoder
        if code_family == "polar":
            encoder = Polar5GEncoder(self.k, self.n, device=self.cfg.device)
            decoder = Polar5GDecoder(
                encoder,
                dec_type="SCL",
                list_size=8,
                device=self.cfg.device,
            )
            return encoder, decoder
        raise ValueError(f"Unsupported code family: {self.cfg.code_family}")

    def _build_channel_model(self):
        def build_array(num_ant: int, is_tx: bool):
            if num_ant == 1:
                pattern = "omni" if is_tx else "38.901"
                return Antenna(
                    polarization="single",
                    polarization_type="V",
                    antenna_pattern=pattern,
                    carrier_frequency=self.cfg.carrier_frequency_hz,
                    device=self.cfg.device,
                )
            return AntennaArray(
                num_rows=1,
                num_cols=max(1, num_ant // 2),
                polarization="dual",
                polarization_type="cross",
                antenna_pattern="38.901",
                carrier_frequency=self.cfg.carrier_frequency_hz,
                device=self.cfg.device,
            )

        if self.cfg.channel_type.lower() == "cdl":
            ut_array = build_array(self.num_streams, is_tx=True)
            bs_array = build_array(self.cfg.num_bs_ant, is_tx=False)
            return CDL(
                self.cfg.channel_profile,
                self.cfg.delay_spread_s,
                self.cfg.carrier_frequency_hz,
                ut_array=ut_array,
                bs_array=bs_array,
                direction=self.cfg.direction,
                min_speed=self.cfg.min_speed_mps,
                max_speed=self.cfg.max_speed_mps,
                device=self.cfg.device,
            )

        if self.cfg.channel_type.lower() == "tdl":
            return TDL(
                self.cfg.channel_profile,
                self.cfg.delay_spread_s,
                self.cfg.carrier_frequency_hz,
                min_speed=self.cfg.min_speed_mps,
                max_speed=max(self.cfg.min_speed_mps, self.cfg.max_speed_mps),
                num_rx_ant=self.cfg.num_bs_ant,
                num_tx_ant=self.num_streams,
                device=self.cfg.device,
            )

        raise ValueError(f"Unsupported channel type: {self.cfg.channel_type}")

    def _build_channel_estimator(self):
        estimator_type = self.cfg.estimator_type.lower()
        if estimator_type == "perfect":
            return None
        if estimator_type == "ls":
            return LSChannelEstimator(
                self.resource_grid,
                interpolation_type=self.cfg.interpolation_type,
                device=self.cfg.device,
            )
        if estimator_type == "lmmse":
            cov_time = exponential_covariance(self.cfg.num_ofdm_symbols, 0.65, self.cfg.device)
            cov_freq = exponential_covariance(self.resource_grid.num_effective_subcarriers, 0.92, self.cfg.device)
            cov_space = exponential_covariance(self.cfg.num_bs_ant, 0.70, self.cfg.device)
            interpolator = LMMSEInterpolator(
                self.resource_grid.pilot_pattern,
                cov_time,
                cov_freq,
                cov_space,
                order=self.cfg.lmmse_order,
            )
            return LSChannelEstimator(
                self.resource_grid,
                interpolator=interpolator,
                device=self.cfg.device,
            )
        raise ValueError(f"Unsupported estimator type: {self.cfg.estimator_type}")

    def compute_noise_variance(self, ebno_db: torch.Tensor | float) -> torch.Tensor:
        ebno_db = torch.as_tensor(ebno_db, dtype=torch.float32, device=self.cfg.device)
        return ebnodb2no(
            ebno_db,
            num_bits_per_symbol=self.cfg.num_bits_per_symbol,
            coderate=self.actual_coderate,
            resource_grid=self.resource_grid,
        )

    def call(
        self,
        batch_size: int,
        ebno_db: torch.Tensor | float,
        return_intermediates: bool = False,
    ):
        no = self.compute_noise_variance(ebno_db)
        bits = self.binary_source([batch_size, self.num_logical_tx, self.num_streams, self.k])
        codewords = self.encoder(bits)
        x = self.mapper(codewords)
        x_rg = self.rg_mapper(x)

        y, h_freq = self.channel(x_rg, no)

        if self.channel_estimator is None:
            h_hat = h_freq
            err_var = torch.zeros_like(h_freq.real)
        else:
            h_hat, err_var = self.channel_estimator(y, no)

        x_hat, no_eff = self.equalizer(y, h_hat, err_var, no)
        llr = self.demapper(x_hat, no_eff)
        bits_hat = self.decoder(llr)

        if not return_intermediates:
            return bits, bits_hat

        return {
            "bits": bits,
            "codewords": codewords,
            "symbols": x,
            "tx_grid": x_rg,
            "rx_grid": y,
            "channel_freq": h_freq,
            "channel_estimate": h_hat,
            "channel_estimation_error": err_var,
            "equalized_symbols": x_hat,
            "effective_noise": no_eff,
            "llr": llr,
            "decoded_bits": bits_hat,
            "noise_variance": no,
        }


@torch.no_grad()
def evaluate_single_point(
    system: TraditionalMIMOOfdmSystem,
    ebno_db: float,
    num_batches: int = 20,
    batch_size: Optional[int] = None,
) -> Dict[str, float]:
    batch_size = batch_size or system.cfg.eval_batch_size
    total_bits = 0
    total_bit_errors = 0
    total_blocks = 0
    total_block_errors = 0

    for _ in range(num_batches):
        bits, bits_hat = system(batch_size, ebno_db)
        bit_errors = torch.ne(bits, bits_hat).sum().item()
        block_errors = torch.ne(bits, bits_hat).reshape(batch_size, -1).any(dim=1).sum().item()

        total_bits += bits.numel()
        total_bit_errors += bit_errors
        total_blocks += batch_size
        total_block_errors += block_errors

    ber = total_bit_errors / max(total_bits, 1)
    bler = total_block_errors / max(total_blocks, 1)
    throughput_bits = (1.0 - bler) * system.k * system.cfg.total_streams
    spectral_efficiency = (1.0 - bler) * system.actual_coderate * system.cfg.num_bits_per_symbol * system.cfg.total_streams
    return {
        "ebno_db": float(ebno_db),
        "ber": ber,
        "bler": bler,
        "throughput_bits_per_frame": throughput_bits,
        "spectral_efficiency_bps_hz": spectral_efficiency,
    }


@torch.no_grad()
def sweep_ber(
    cfg: MIMOOfdmConfig,
    ebno_points: Iterable[float],
    num_batches: int = 20,
) -> List[Dict[str, float]]:
    system = TraditionalMIMOOfdmSystem(cfg)
    results: List[Dict[str, float]] = []
    for ebno_db in ebno_points:
        point = evaluate_single_point(system, float(ebno_db), num_batches=num_batches)
        point.update(
            {
                "modulation": cfg.modulation_name,
                "coderate": cfg.coderate,
                "code_family": cfg.code_family,
                "channel_type": cfg.channel_type,
                "channel_profile": cfg.channel_profile,
                "estimator_type": cfg.estimator_type,
                "mimo_mode": cfg.mimo_name,
            }
        )
        results.append(point)
    return results


def config_to_dict(cfg: MIMOOfdmConfig) -> Dict[str, object]:
    result = asdict(cfg)
    result["output_dir"] = str(cfg.output_dir)
    result["modulation_name"] = cfg.modulation_name
    result["mimo_name"] = cfg.mimo_name
    return result
