from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
COMSYS_DIR = ROOT / "comsys"
if str(COMSYS_DIR) not in sys.path:
    sys.path.insert(0, str(COMSYS_DIR))

from acm_controller import ACMController, mcs_table, oracle_mcs_index  # noqa: E402
from baseline_system import TraditionalMIMOOfdmSystem, clone_config, resolve_device  # noqa: E402
from config import MIMOOfdmConfig  # noqa: E402
from dl_mimo_detector import GridNeuralMIMODetector, coded_bce_loss  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint training for ACM controller and MCS-conditioned DL detector.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--snr-low", type=float, default=0.0)
    parser.add_argument("--snr-high", type=float, default=30.0)
    parser.add_argument("--snr-grid-step", type=float, default=2.0)
    parser.add_argument("--oracle-batches", type=int, default=0)
    parser.add_argument("--eval-batches", type=int, default=12)
    parser.add_argument("--channel-type", default="cdl", choices=["cdl", "tdl"])
    parser.add_argument("--channel-profile", default="A")
    parser.add_argument("--estimator-type", default="ls", choices=["ls", "lmmse", "perfect"])
    parser.add_argument("--num-tx", type=int, default=2)
    parser.add_argument("--num-bs-ant", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detector-lr", type=float, default=5e-4)
    parser.add_argument("--acm-lr", type=float, default=1e-3)
    parser.add_argument("--throughput-weight", type=float, default=0.5)
    parser.add_argument("--surrogate-scale", type=float, default=1.0)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--reliability-weight", type=float, default=0.5)
    parser.add_argument("--reliability-threshold", type=float, default=0.05)
    parser.add_argument("--reliability-power", type=float, default=2.0)
    parser.add_argument("--mcs-softmax-temperature", type=float, default=2.0)
    parser.add_argument("--acm-warmup-epochs", type=int, default=4)
    parser.add_argument("--hidden-channels", type=int, default=96)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--blocks-per-stage", type=int, default=2)
    parser.add_argument("--expansion", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--head-blocks", type=int, default=2)
    parser.add_argument("--gate-hidden-dim", type=int, default=128)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--mcs-embedding-dim", type=int, default=8)
    parser.add_argument("--bler-target", type=float, default=0.1)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def build_mcs_systems(base_cfg: MIMOOfdmConfig) -> dict[int, TraditionalMIMOOfdmSystem]:
    systems: dict[int, TraditionalMIMOOfdmSystem] = {}
    for entry in mcs_table():
        mcs_cfg = clone_config(
            base_cfg,
            num_bits_per_symbol=int(entry["num_bits_per_symbol"]),
            coderate=float(entry["coderate"]),
            code_family=str(entry["code_family"]),
        )
        systems[int(entry["mcs_index"])] = TraditionalMIMOOfdmSystem(mcs_cfg)
    return systems


def build_oracle_lookup(
    base_cfg: MIMOOfdmConfig,
    snr_points: np.ndarray,
    bler_target: float,
    num_batches: int,
):
    rows = []
    labels = {}
    for snr in snr_points:
        best = oracle_mcs_index(base_cfg, float(snr), bler_target=bler_target, num_batches=num_batches)
        rows.append(
            {
                "ebno_db": float(snr),
                "oracle_mcs_index": int(best["mcs_index"]),
                "num_bits_per_symbol": int(best["num_bits_per_symbol"]),
                "coderate": float(best["coderate"]),
                "oracle_bler": float(best["bler"]),
                "oracle_throughput_bits_per_frame": float(best["throughput_bits_per_frame"]),
            }
        )
        labels[float(snr)] = int(best["mcs_index"])
    return rows, labels


def build_pilot_mask(system: TraditionalMIMOOfdmSystem) -> torch.Tensor:
    mask = system.resource_grid.pilot_pattern.mask.float()
    return mask.reshape(
        system.cfg.total_streams,
        system.cfg.num_ofdm_symbols,
        system.cfg.fft_size,
    ).to(system.cfg.device)


def extract_llrs_for_mcs(
    llr_grid: torch.Tensor,
    data_mask: torch.Tensor,
    total_streams: int,
    bits_per_symbol: int,
    num_ofdm_symbols: int,
    fft_size: int,
) -> torch.Tensor:
    batch_size = llr_grid.shape[0]
    llr_grid = llr_grid.reshape(
        batch_size,
        1,
        total_streams,
        6,
        num_ofdm_symbols,
        fft_size,
    )

    outputs = []
    for stream_idx in range(total_streams):
        mask = data_mask[0, stream_idx].reshape(-1)
        values = llr_grid[:, 0, stream_idx, :bits_per_symbol].reshape(batch_size, bits_per_symbol, -1)
        values = values[:, :, mask].permute(0, 2, 1).reshape(batch_size, -1)
        outputs.append(values)
    return torch.stack(outputs, dim=1).unsqueeze(1)


def branch_goodput_proxy(
    detector_loss: torch.Tensor,
    system: TraditionalMIMOOfdmSystem,
    max_bits_per_frame: float,
    surrogate_scale: float,
) -> torch.Tensor:
    normalized_bits = (system.k * system.cfg.total_streams) / max(max_bits_per_frame, 1.0)
    return detector_loss.new_tensor(normalized_bits) * torch.exp(-surrogate_scale * detector_loss)


def branch_reliability_penalty(
    detector_loss: torch.Tensor,
    system: TraditionalMIMOOfdmSystem,
    max_bits_per_frame: float,
    threshold: float,
    power: float,
) -> torch.Tensor:
    normalized_bits = (system.k * system.cfg.total_streams) / max(max_bits_per_frame, 1.0)
    excess = torch.relu(detector_loss - threshold)
    if power != 1.0:
        excess = excess.pow(power)
    return detector_loss.new_tensor(normalized_bits) * excess


def save_oracle_lookup(rows, output_dir: Path) -> None:
    with (output_dir / "oracle_mcs_lookup.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_history(history, output_dir: Path) -> None:
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    plt.figure(figsize=(6.5, 4))
    plt.plot(
        [row["epoch"] for row in history],
        [row["expected_detector_loss"] for row in history],
        marker="o",
        label="Expected detector loss",
    )
    plt.plot(
        [row["epoch"] for row in history],
        [row["expected_goodput_proxy"] for row in history],
        marker="s",
        label="Expected goodput proxy",
    )
    plt.plot(
        [row["epoch"] for row in history],
        [row["acm_entropy"] for row in history],
        marker="^",
        label="ACM entropy",
    )
    plt.plot(
        [row["epoch"] for row in history],
        [row["expected_reliability_penalty"] for row in history],
        marker="d",
        label="Expected reliability penalty",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_losses.png", dpi=200)
    plt.close()


@torch.no_grad()
def evaluate_joint(
    detector: GridNeuralMIMODetector,
    acm: ACMController,
    systems: dict[int, TraditionalMIMOOfdmSystem],
    pilot_mask: torch.Tensor,
    snr_points: np.ndarray,
    eval_batches: int,
    batch_size: int,
):
    results = []
    detector.eval()
    acm.eval()
    for snr in snr_points:
        device = next(detector.parameters()).device
        features = torch.full((1, 1), float(snr), dtype=torch.float32, device=device)
        predicted_mcs = int(acm(features).argmax(dim=1).item())
        system = systems[predicted_mcs]
        data_mask = (~system.resource_grid.pilot_pattern.mask.bool()).to(system.cfg.device)
        bits_per_symbol = system.cfg.num_bits_per_symbol

        total_bits = 0
        total_bit_errors = 0
        total_blocks = 0
        total_block_errors = 0
        for _ in range(eval_batches):
            batch = system(batch_size, float(snr), return_intermediates=True)
            llr_grid = detector(
                batch["rx_grid"],
                batch["channel_estimate"],
                batch["noise_variance"],
                pilot_mask,
                mcs_index=torch.full((batch_size,), predicted_mcs, dtype=torch.long, device=system.cfg.device),
            )
            llr = extract_llrs_for_mcs(
                llr_grid,
                data_mask,
                system.cfg.total_streams,
                bits_per_symbol,
                system.cfg.num_ofdm_symbols,
                system.cfg.fft_size,
            )
            bits_hat = system.decoder(llr)
            bits = batch["bits"]
            total_bit_errors += torch.ne(bits, bits_hat).sum().item()
            total_bits += bits.numel()
            total_block_errors += torch.ne(bits, bits_hat).reshape(batch_size, -1).any(dim=1).sum().item()
            total_blocks += batch_size

        ber = total_bit_errors / max(total_bits, 1)
        bler = total_block_errors / max(total_blocks, 1)
        throughput = (1.0 - bler) * system.k * system.cfg.total_streams
        results.append(
            {
                "ebno_db": float(snr),
                "predicted_mcs_index": predicted_mcs,
                "ber": ber,
                "bler": bler,
                "throughput_bits_per_frame": throughput,
            }
        )
    return results


def save_joint_results(results, output_dir: Path) -> None:
    with (output_dir / "joint_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    ebno = [row["ebno_db"] for row in results]
    throughput = [row["throughput_bits_per_frame"] for row in results]
    plt.figure(figsize=(6.5, 4))
    plt.plot(ebno, throughput, marker="o")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Throughput (info bits / frame)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "joint_throughput.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "AI-AIDED" / "outputs" / "joint_acm_detector"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = MIMOOfdmConfig(
        channel_type=args.channel_type,
        channel_profile=args.channel_profile,
        estimator_type=args.estimator_type,
        num_tx=args.num_tx,
        num_bs_ant=args.num_bs_ant,
        device=args.device,
    )
    base_cfg.device = resolve_device(base_cfg.device)

    snr_points = np.arange(args.snr_low, args.snr_high + 1e-6, args.snr_grid_step, dtype=np.float32)
    if args.oracle_batches > 0:
        oracle_rows, _ = build_oracle_lookup(
            base_cfg,
            snr_points,
            bler_target=args.bler_target,
            num_batches=args.oracle_batches,
        )
        save_oracle_lookup(oracle_rows, output_dir)

    systems = build_mcs_systems(base_cfg)
    reference_system = systems[int(mcs_table()[-1]["mcs_index"])]
    pilot_mask = build_pilot_mask(reference_system)
    max_bits_per_frame = max(system.k * system.cfg.total_streams for system in systems.values())

    detector_cfg = clone_config(reference_system.cfg, num_bits_per_symbol=6)
    detector = GridNeuralMIMODetector(
        detector_cfg,
        hidden_channels=args.hidden_channels,
        num_stages=args.num_stages,
        blocks_per_stage=args.blocks_per_stage,
        expansion=args.expansion,
        num_heads=args.num_heads,
        head_blocks=args.head_blocks,
        gate_hidden_dim=args.gate_hidden_dim,
        gate_temperature=args.gate_temperature,
        num_mcs=len(mcs_table()),
        mcs_embedding_dim=args.mcs_embedding_dim,
    ).to(base_cfg.device)

    acm = ACMController(input_dim=1, hidden_dim=64, num_classes=len(mcs_table())).to(base_cfg.device)
    detector_optimizer = torch.optim.Adam(detector.parameters(), lr=args.detector_lr)
    acm_optimizer = torch.optim.Adam(acm.parameters(), lr=args.acm_lr)

    history = []
    for epoch in range(1, args.epochs + 1):
        detector.train()
        acm.train()
        detector_loss_sum = 0.0
        goodput_proxy_sum = 0.0
        entropy_sum = 0.0
        reliability_penalty_sum = 0.0

        for _ in range(args.steps_per_epoch):
            snr = float(np.random.uniform(args.snr_low, args.snr_high))
            snr_features = torch.full((args.batch_size, 1), snr, dtype=torch.float32, device=base_cfg.device)
            mcs_logits = acm(snr_features)
            mcs_probs = torch.softmax(mcs_logits / args.mcs_softmax_temperature, dim=1).mean(dim=0)

            branch_detector_losses = []
            branch_goodput_proxies = []
            branch_reliability_penalties = []
            for mcs_entry in mcs_table():
                mcs_index = int(mcs_entry["mcs_index"])
                system = systems[mcs_index]
                batch = system(args.batch_size, snr, return_intermediates=True)
                data_mask = (~system.resource_grid.pilot_pattern.mask.bool()).to(system.cfg.device)
                bits_per_symbol = system.cfg.num_bits_per_symbol
                mcs_batch = torch.full((args.batch_size,), mcs_index, dtype=torch.long, device=base_cfg.device)
                llr_grid = detector(
                    batch["rx_grid"],
                    batch["channel_estimate"],
                    batch["noise_variance"],
                    pilot_mask,
                    mcs_index=mcs_batch,
                )
                predicted_llr = extract_llrs_for_mcs(
                    llr_grid,
                    data_mask,
                    system.cfg.total_streams,
                    bits_per_symbol,
                    system.cfg.num_ofdm_symbols,
                    system.cfg.fft_size,
                )
                detector_loss = coded_bce_loss(predicted_llr, batch["codewords"])
                goodput_proxy = branch_goodput_proxy(
                    detector_loss,
                    system,
                    max_bits_per_frame=max_bits_per_frame,
                    surrogate_scale=args.surrogate_scale,
                )
                reliability_penalty = branch_reliability_penalty(
                    detector_loss,
                    system,
                    max_bits_per_frame=max_bits_per_frame,
                    threshold=args.reliability_threshold,
                    power=args.reliability_power,
                )
                branch_detector_losses.append(detector_loss)
                branch_goodput_proxies.append(goodput_proxy)
                branch_reliability_penalties.append(reliability_penalty)

            branch_detector_losses = torch.stack(branch_detector_losses)
            branch_goodput_proxies = torch.stack(branch_goodput_proxies)
            branch_reliability_penalties = torch.stack(branch_reliability_penalties)
            expected_detector_loss = torch.sum(mcs_probs * branch_detector_losses)
            expected_goodput_proxy = torch.sum(mcs_probs * branch_goodput_proxies)
            expected_reliability_penalty = torch.sum(mcs_probs * branch_reliability_penalties)
            acm_entropy = -(mcs_probs * torch.log(torch.clamp(mcs_probs, min=1e-8))).sum()
            if epoch <= args.acm_warmup_epochs:
                total_loss = expected_detector_loss
            else:
                total_loss = (
                    expected_detector_loss
                    - args.throughput_weight * expected_goodput_proxy
                    + args.reliability_weight * expected_reliability_penalty
                    - args.entropy_weight * acm_entropy
                )

            detector_optimizer.zero_grad(set_to_none=True)
            acm_optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            detector_optimizer.step()
            acm_optimizer.step()

            detector_loss_sum += float(expected_detector_loss.item())
            goodput_proxy_sum += float(expected_goodput_proxy.item())
            entropy_sum += float(acm_entropy.item())
            reliability_penalty_sum += float(expected_reliability_penalty.item())

        epoch_record = {
            "epoch": epoch,
            "expected_detector_loss": detector_loss_sum / args.steps_per_epoch,
            "expected_goodput_proxy": goodput_proxy_sum / args.steps_per_epoch,
            "acm_entropy": entropy_sum / args.steps_per_epoch,
            "expected_reliability_penalty": reliability_penalty_sum / args.steps_per_epoch,
        }
        history.append(epoch_record)
        print(
            f"Epoch {epoch:02d} | expected_detector_loss={epoch_record['expected_detector_loss']:.6f} | "
            f"expected_goodput_proxy={epoch_record['expected_goodput_proxy']:.6f} | "
            f"acm_entropy={epoch_record['acm_entropy']:.6f} | "
            f"expected_reliability_penalty={epoch_record['expected_reliability_penalty']:.6f}"
        )

    save_history(history, output_dir)
    torch.save(detector.state_dict(), output_dir / "joint_detector.pt")
    torch.save(acm.state_dict(), output_dir / "joint_acm.pt")

    results = evaluate_joint(
        detector,
        acm,
        systems,
        pilot_mask,
        snr_points,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
    )
    save_joint_results(results, output_dir)
    print(f"Saved joint training artifacts to {output_dir}")


if __name__ == "__main__":
    main()
