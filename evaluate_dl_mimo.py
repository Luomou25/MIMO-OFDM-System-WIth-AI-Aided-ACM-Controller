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

from baseline_system import TraditionalMIMOOfdmSystem, evaluate_single_point  # noqa: E402
from config import MIMOOfdmConfig  # noqa: E402
from dl_mimo_detector import NeuralReceiverSystem  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DL-MIMO detector with larger Monte Carlo sample counts."
    )
    parser.add_argument("--weights", required=True, help="Path to the trained detector .pt file.")
    parser.add_argument("--channel-type", default="cdl", choices=["cdl", "tdl"])
    parser.add_argument("--channel-profile", default="A")
    parser.add_argument("--bits-per-symbol", type=int, default=2, choices=[2, 4, 6])
    parser.add_argument("--coderate", type=float, default=0.5)
    parser.add_argument("--estimator-type", default="ls", choices=["ls", "lmmse", "perfect"])
    parser.add_argument("--num-tx", type=int, default=2)
    parser.add_argument("--num-bs-ant", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-channels", type=int, default=96)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--blocks-per-stage", type=int, default=2)
    parser.add_argument("--expansion", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--head-blocks", type=int, default=1)
    parser.add_argument("--gate-hidden-dim", type=int, default=128)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--snr-start", type=float, default=-10.0)
    parser.add_argument("--snr-stop", type=float, default=20.0)
    parser.add_argument("--snr-step", type=float, default=2.0)
    parser.add_argument("--eval-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


@torch.no_grad()
def evaluate_neural(model: NeuralReceiverSystem, ebno_db: float, num_batches: int, batch_size: int):
    total_bits = 0
    total_bit_errors = 0
    total_blocks = 0
    total_block_errors = 0
    total_gate_weights = None
    for _ in range(num_batches):
        batch = model(batch_size, ebno_db, return_intermediates=True)
        bits = batch["bits"]
        bits_hat = batch["decoded_bits_ai"]
        gate_weights = batch["gate_weights"]
        total_bit_errors += torch.ne(bits, bits_hat).sum().item()
        total_bits += bits.numel()
        total_block_errors += torch.ne(bits, bits_hat).reshape(batch_size, -1).any(dim=1).sum().item()
        total_blocks += batch_size
        batch_gate_sum = gate_weights.sum(dim=0)
        if total_gate_weights is None:
            total_gate_weights = batch_gate_sum
        else:
            total_gate_weights += batch_gate_sum
    avg_gate_weights = (total_gate_weights / max(total_blocks, 1)).detach().cpu().tolist()
    return {
        "ber": total_bit_errors / max(total_bits, 1),
        "bler": total_block_errors / max(total_blocks, 1),
        "avg_gate_weights": avg_gate_weights,
    }


def save_results(curve, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ber_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)

    ebno = [row["ebno_db"] for row in curve]

    plt.figure(figsize=(6, 4))
    plt.semilogy(ebno, [row["baseline_ber"] for row in curve], marker="o", label="Traditional receiver")
    plt.semilogy(ebno, [row["ai_ber"] for row in curve], marker="s", label="DL detector")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("BER")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "ber_comparison.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.semilogy(ebno, [row["baseline_bler"] for row in curve], marker="o", label="Traditional receiver")
    plt.semilogy(ebno, [row["ai_bler"] for row in curve], marker="s", label="DL detector")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("BLER")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "bler_comparison.png", dpi=200)
    plt.close()

    gate_keys = [key for key in curve[0].keys() if key.startswith("gate_head_")]
    if gate_keys:
        with (output_dir / "gate_weights.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ebno_db", *gate_keys])
            writer.writeheader()
            writer.writerows([{key: row[key] for key in ["ebno_db", *gate_keys]} for row in curve])

        plt.figure(figsize=(6.5, 4))
        for gate_key in gate_keys:
            plt.plot(ebno, [row[gate_key] for row in curve], marker="o", label=gate_key.replace("_", " "))
        plt.xlabel("Eb/N0 (dB)")
        plt.ylabel("Average gate weight")
        plt.ylim(0.0, 1.0)
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "gate_weights.png", dpi=200)
        plt.close()

        with (output_dir / "dominant_head.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ebno_db", "dominant_head", "dominant_weight"])
            writer.writeheader()
            for row in curve:
                gate_values = [row[key] for key in gate_keys]
                dominant_head = int(np.argmax(gate_values))
                dominant_weight = float(np.max(gate_values))
                writer.writerow(
                    {
                        "ebno_db": row["ebno_db"],
                        "dominant_head": dominant_head,
                        "dominant_weight": dominant_weight,
                    }
                )

        dominant_heads = [int(np.argmax([row[key] for key in gate_keys])) for row in curve]
        dominant_weights = [float(np.max([row[key] for key in gate_keys])) for row in curve]

        plt.figure(figsize=(6.5, 4))
        plt.step(ebno, dominant_heads, where="mid", linewidth=2, label="Dominant head index")
        plt.scatter(ebno, dominant_heads, c=dominant_weights, cmap="viridis", s=60, zorder=3)
        plt.xlabel("Eb/N0 (dB)")
        plt.ylabel("Dominant head")
        plt.yticks(range(len(gate_keys)))
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / "dominant_head.png", dpi=200)
        plt.close()


def main() -> None:
    args = parse_args()
    cfg = MIMOOfdmConfig(
        channel_type=args.channel_type,
        channel_profile=args.channel_profile,
        num_bits_per_symbol=args.bits_per_symbol,
        coderate=args.coderate,
        estimator_type=args.estimator_type,
        num_tx=args.num_tx,
        num_bs_ant=args.num_bs_ant,
        device=args.device,
        eval_batch_size=args.batch_size,
    )

    model = NeuralReceiverSystem(
        cfg,
        hidden_channels=args.hidden_channels,
        num_stages=args.num_stages,
        blocks_per_stage=args.blocks_per_stage,
        expansion=args.expansion,
        num_heads=args.num_heads,
        head_blocks=args.head_blocks,
        gate_hidden_dim=args.gate_hidden_dim,
        gate_temperature=args.gate_temperature,
    )
    state_dict = torch.load(args.weights, map_location=model.reference_link.cfg.device)
    model.detector.load_state_dict(state_dict)
    model.eval()

    baseline = TraditionalMIMOOfdmSystem(cfg)

    weights_path = Path(args.weights)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else weights_path.resolve().parent / f"reeval_b{args.eval_batches}_bs{args.batch_size}"
    )

    curve = []
    for ebno_db in np.arange(args.snr_start, args.snr_stop + 1e-6, args.snr_step):
        baseline_metrics = evaluate_single_point(
            baseline,
            float(ebno_db),
            num_batches=args.eval_batches,
            batch_size=args.batch_size,
        )
        ai_metrics = evaluate_neural(
            model,
            float(ebno_db),
            num_batches=args.eval_batches,
            batch_size=args.batch_size,
        )
        record = {
            "ebno_db": float(ebno_db),
            "baseline_ber": baseline_metrics["ber"],
            "baseline_bler": baseline_metrics["bler"],
            "ai_ber": ai_metrics["ber"],
            "ai_bler": ai_metrics["bler"],
            "eval_batches": args.eval_batches,
            "batch_size": args.batch_size,
            "evaluated_frames_per_snr": args.eval_batches * args.batch_size,
        }
        for gate_idx, weight in enumerate(ai_metrics["avg_gate_weights"]):
            record[f"gate_head_{gate_idx}"] = weight
        curve.append(record)
        dominant_head = int(np.argmax(ai_metrics["avg_gate_weights"]))
        print(
            f"Eb/N0={ebno_db:>5.1f} dB | baseline BER={record['baseline_ber']:.4e} | "
            f"AI BER={record['ai_ber']:.4e} | frames={record['evaluated_frames_per_snr']} | "
            f"gate={np.round(ai_metrics['avg_gate_weights'], 4)} | dominant_head={dominant_head}"
        )

    save_results(curve, output_dir)
    print(f"Saved re-evaluation results to {output_dir}")


if __name__ == "__main__":
    main()
