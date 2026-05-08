from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
COMSYS_DIR = ROOT / "comsys"
if str(COMSYS_DIR) not in sys.path:
    sys.path.insert(0, str(COMSYS_DIR))

from acm_controller import ACMController, evaluate_closed_loop, generate_dataset, save_dataset, train_controller  # noqa: E402
from config import MIMOOfdmConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the AI-based ACM controller.")
    parser.add_argument("--num-samples", type=int, default=40)
    parser.add_argument("--dataset-batches", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--snr-low", type=float, default=-2.0)
    parser.add_argument("--snr-high", type=float, default=20.0)
    parser.add_argument("--eval-snr-start", type=float, default=-2.0)
    parser.add_argument("--eval-snr-stop", type=float, default=20.0)
    parser.add_argument("--eval-snr-step", type=float, default=2.0)
    parser.add_argument("--bler-target", type=float, default=0.1)
    parser.add_argument("--channel-type", default="cdl", choices=["cdl", "tdl"])
    parser.add_argument("--channel-profile", default="A")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def save_history(history, output_dir: Path) -> None:
    with (output_dir / "acm_training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    plt.figure(figsize=(6, 4))
    plt.plot([row["epoch"] for row in history], [row["accuracy"] for row in history], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training accuracy")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "acm_training_accuracy.png", dpi=200)
    plt.close()


def save_closed_loop(records, output_dir: Path) -> None:
    with (output_dir / "closed_loop_throughput.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    ebno = [row["ebno_db"] for row in records]
    plt.figure(figsize=(6, 4))
    plt.plot(ebno, [row["robust_throughput"] for row in records], marker="o", label="Fixed robust MCS")
    plt.plot(ebno, [row["aggressive_throughput"] for row in records], marker="s", label="Fixed aggressive MCS")
    plt.plot(ebno, [row["ai_throughput"] for row in records], marker="^", label="AI-ACM")
    plt.plot(ebno, [row["oracle_throughput"] for row in records], marker="d", label="Oracle ACM")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Throughput (info bits / frame)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "closed_loop_throughput.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    cfg = MIMOOfdmConfig(
        channel_type=args.channel_type,
        channel_profile=args.channel_profile,
        device=args.device,
    )
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "AI-AIDED" / "outputs" / "acm"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_dataset(
        cfg,
        num_samples=args.num_samples,
        snr_low=args.snr_low,
        snr_high=args.snr_high,
        bler_target=args.bler_target,
        num_batches=args.dataset_batches,
    )
    save_dataset(dataset, output_dir)

    controller = ACMController(num_classes=6)
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    history = train_controller(
        controller,
        dataset["features"],
        dataset["labels"],
        epochs=args.epochs,
        device=device,
    )
    save_history(history, output_dir)
    torch.save(controller.state_dict(), output_dir / "acm_controller.pt")

    ebno_points = np.arange(args.eval_snr_start, args.eval_snr_stop + 1e-6, args.eval_snr_step)
    records = evaluate_closed_loop(controller, cfg, ebno_points, device=device, num_batches=args.dataset_batches)
    save_closed_loop(records, output_dir)
    print(f"Saved ACM dataset, controller, and figures to {output_dir}")


if __name__ == "__main__":
    main()
