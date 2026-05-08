from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from baseline_system import MIMOOfdmConfig, clone_config, config_to_dict, sweep_ber


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the traditional coded MIMO-OFDM baseline.")
    parser.add_argument("--channel-type", default="cdl", choices=["cdl", "tdl"])
    parser.add_argument("--channel-profile", default="A")
    parser.add_argument("--code-family", default="ldpc", choices=["ldpc", "polar"])
    parser.add_argument("--estimator-type", default="ls", choices=["ls", "lmmse", "perfect"])
    parser.add_argument("--num-tx", type=int, default=2)
    parser.add_argument("--num-bs-ant", type=int, default=2)
    parser.add_argument("--bits-per-symbol", type=int, default=2, choices=[2, 4, 6])
    parser.add_argument("--coderate", type=float, default=0.5)
    parser.add_argument("--snr-start", type=float, default=0.0)
    parser.add_argument("--snr-stop", type=float, default=18.0)
    parser.add_argument("--snr-step", type=float, default=3.0)
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def save_csv(results, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def plot_results(results, output_dir: Path) -> None:
    ebno = [row["ebno_db"] for row in results]
    ber = [row["ber"] for row in results]
    bler = [row["bler"] for row in results]
    throughput = [row["throughput_bits_per_frame"] for row in results]

    plt.figure(figsize=(6, 4))
    plt.semilogy(ebno, ber, marker="o", label="BER")
    plt.semilogy(ebno, bler, marker="s", label="BLER")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Error rate")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "ber_bler_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(ebno, throughput, marker="o", color="tab:green")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Throughput (info bits / frame)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "throughput_curve.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    cfg = MIMOOfdmConfig(
        channel_type=args.channel_type,
        channel_profile=args.channel_profile,
        code_family=args.code_family,
        estimator_type=args.estimator_type,
        num_tx=args.num_tx,
        num_bs_ant=args.num_bs_ant,
        num_bits_per_symbol=args.bits_per_symbol,
        coderate=args.coderate,
        device=args.device,
    )
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)

    ebno_points = np.arange(args.snr_start, args.snr_stop + 1e-6, args.snr_step)
    results = sweep_ber(cfg, ebno_points, num_batches=args.num_batches)

    output_dir = cfg.output_dir / f"baseline_{cfg.channel_type}_{cfg.channel_profile}_{cfg.modulation_name.lower()}_{cfg.code_family}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(results, output_dir / "metrics.csv")
    plot_results(results, output_dir)

    with (output_dir / "config.txt").open("w", encoding="utf-8") as handle:
        for key, value in config_to_dict(cfg).items():
            handle.write(f"{key}: {value}\n")

    print(f"Saved results to {output_dir}")
    for row in results:
        print(
            f"Eb/N0={row['ebno_db']:>5.1f} dB | BER={row['ber']:.4e} | "
            f"BLER={row['bler']:.4e} | Throughput={row['throughput_bits_per_frame']:.1f}"
        )


if __name__ == "__main__":
    main()
