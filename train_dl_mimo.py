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

from baseline_system import TraditionalMIMOOfdmSystem, evaluate_single_point  # noqa: E402
from config import MIMOOfdmConfig  # noqa: E402
from dl_mimo_detector import NeuralReceiverSystem, coded_bce_loss  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DL-based MIMO detector.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--snr-low", type=float, default=0.0)
    parser.add_argument("--snr-high", type=float, default=16.0)
    parser.add_argument("--eval-snr-start", type=float, default=0.0)
    parser.add_argument("--eval-snr-stop", type=float, default=18.0)
    parser.add_argument("--eval-snr-step", type=float, default=3.0)
    parser.add_argument("--channel-type", default="cdl", choices=["cdl", "tdl"])
    parser.add_argument("--channel-profile", default="A")
    parser.add_argument("--bits-per-symbol", type=int, default=2, choices=[2, 4, 6])
    parser.add_argument("--coderate", type=float, default=0.5)
    parser.add_argument("--estimator-type", default="ls", choices=["ls", "lmmse", "perfect"])
    parser.add_argument("--num-tx", type=int, default=2)
    parser.add_argument("--num-bs-ant", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", type=int, default=96)
    parser.add_argument("--num-stages", type=int, default=4)
    parser.add_argument("--blocks-per-stage", type=int, default=2)
    parser.add_argument("--expansion", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--head-blocks", type=int, default=2)
    parser.add_argument("--gate-hidden-dim", type=int, default=128)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument(
        "--sampling-strategy",
        default="weighted_high_db",
        choices=["uniform_db", "weighted_high_db", "uniform_linear"],
    )
    parser.add_argument("--high-snr-prob", type=float, default=0.5)
    parser.add_argument("--high-snr-threshold", type=float, default=10.0)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--resume-weights", default=None, help="Optional path to a detector checkpoint to continue training.")
    parser.add_argument("--finetune-epochs", type=int, default=0)
    parser.add_argument("--finetune-steps-per-epoch", type=int, default=300)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument(
        "--finetune-sampling-strategy",
        default="mixed_focus_db",
        choices=["uniform_db", "weighted_high_db", "uniform_linear", "mixed_focus_db", "adaptive_gap_db"],
    )
    parser.add_argument("--finetune-focus-prob", type=float, default=0.5)
    parser.add_argument("--finetune-focus-snr-low", type=float, default=14.0)
    parser.add_argument("--finetune-focus-snr-high", type=float, default=20.0)
    parser.add_argument("--adaptive-eval-batches", type=int, default=30)
    parser.add_argument("--adaptive-uniform-mix", type=float, default=0.5)
    parser.add_argument("--adaptive-temperature", type=float, default=6.0)
    parser.add_argument("--adaptive-smooth-window", type=int, default=3)
    parser.add_argument("--adaptive-eps", type=float, default=1e-8)
    parser.add_argument("--adaptive-gap-power", type=float, default=1.5)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.2)
    parser.add_argument("--gate-lr-scale", type=float, default=0.4)
    parser.add_argument("--head-lr-scale", type=float, default=1.0)
    parser.add_argument("--finetune-backbone-lr-scale", type=float, default=0.2)
    parser.add_argument("--finetune-gate-lr-scale", type=float, default=0.4)
    parser.add_argument("--finetune-head-lr-scale", type=float, default=1.0)
    parser.add_argument("--finetune-gap-loss-weight", type=float, default=0.2)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def sample_training_snr(args: argparse.Namespace) -> float:
    if args.sampling_strategy == "uniform_db":
        return torch.empty(1).uniform_(args.snr_low, args.snr_high).item()

    if args.sampling_strategy == "uniform_linear":
        snr_low_linear = 10 ** (args.snr_low / 10.0)
        snr_high_linear = 10 ** (args.snr_high / 10.0)
        sampled_linear = torch.empty(1).uniform_(snr_low_linear, snr_high_linear).item()
        return 10.0 * np.log10(sampled_linear)

    low_region_high = min(args.high_snr_threshold, args.snr_high)
    high_region_low = max(args.high_snr_threshold, args.snr_low)
    sample_high = torch.rand(1).item() < args.high_snr_prob
    if sample_high and high_region_low < args.snr_high:
        return torch.empty(1).uniform_(high_region_low, args.snr_high).item()
    if args.snr_low < low_region_high:
        return torch.empty(1).uniform_(args.snr_low, low_region_high).item()
    return torch.empty(1).uniform_(args.snr_low, args.snr_high).item()


def sample_finetune_snr(args: argparse.Namespace) -> float:
    if args.finetune_sampling_strategy == "uniform_db":
        return torch.empty(1).uniform_(args.snr_low, args.snr_high).item()

    if args.finetune_sampling_strategy == "uniform_linear":
        snr_low_linear = 10 ** (args.snr_low / 10.0)
        snr_high_linear = 10 ** (args.snr_high / 10.0)
        sampled_linear = torch.empty(1).uniform_(snr_low_linear, snr_high_linear).item()
        return 10.0 * np.log10(sampled_linear)

    if args.finetune_sampling_strategy == "weighted_high_db":
        low_region_high = min(args.high_snr_threshold, args.snr_high)
        high_region_low = max(args.high_snr_threshold, args.snr_low)
        sample_high = torch.rand(1).item() < args.high_snr_prob
        if sample_high and high_region_low < args.snr_high:
            return torch.empty(1).uniform_(high_region_low, args.snr_high).item()
        if args.snr_low < low_region_high:
            return torch.empty(1).uniform_(args.snr_low, low_region_high).item()
        return torch.empty(1).uniform_(args.snr_low, args.snr_high).item()

    # True 5:5 mixed sampling:
    # 50% from the focus interval, 50% from its complement.
    focus_low = max(args.snr_low, args.finetune_focus_snr_low)
    focus_high = min(args.snr_high, args.finetune_focus_snr_high)
    sample_focus = torch.rand(1).item() < args.finetune_focus_prob
    if sample_focus and focus_low < focus_high:
        return torch.empty(1).uniform_(focus_low, focus_high).item()

    complement_intervals: list[tuple[float, float]] = []
    if args.snr_low < focus_low:
        complement_intervals.append((args.snr_low, focus_low))
    if focus_high < args.snr_high:
        complement_intervals.append((focus_high, args.snr_high))

    if not complement_intervals:
        return torch.empty(1).uniform_(args.snr_low, args.snr_high).item()

    if len(complement_intervals) == 1:
        low, high = complement_intervals[0]
        return torch.empty(1).uniform_(low, high).item()

    lengths = [high - low for low, high in complement_intervals]
    total_length = sum(lengths)
    draw = torch.rand(1).item() * total_length
    cumulative = 0.0
    for (low, high), length in zip(complement_intervals, lengths):
        cumulative += length
        if draw <= cumulative:
            return torch.empty(1).uniform_(low, high).item()

    low, high = complement_intervals[-1]
    return torch.empty(1).uniform_(low, high).item()


def sample_weighted_snr_bin(bin_centers: np.ndarray, bin_weights: np.ndarray, bin_width: float) -> float:
    probabilities = np.asarray(bin_weights, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    index = int(np.random.choice(len(bin_centers), p=probabilities))
    center = float(bin_centers[index])
    half_width = 0.5 * bin_width
    low = center - half_width
    high = center + half_width
    return float(np.random.uniform(low, high))


@torch.no_grad()
def evaluate_neural(model: NeuralReceiverSystem, ebno_db: float, num_batches: int = 10, batch_size: int = 64):
    total_bits = 0
    total_bit_errors = 0
    total_blocks = 0
    total_block_errors = 0
    for _ in range(num_batches):
        bits, bits_hat = model(batch_size, ebno_db)
        total_bit_errors += torch.ne(bits, bits_hat).sum().item()
        total_bits += bits.numel()
        total_block_errors += torch.ne(bits, bits_hat).reshape(batch_size, -1).any(dim=1).sum().item()
        total_blocks += batch_size
    return {
        "ber": total_bit_errors / max(total_bits, 1),
        "bler": total_block_errors / max(total_blocks, 1),
    }


def save_history(history, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    plt.figure(figsize=(6, 4))
    pretrain = [row for row in history if row["phase"] == "pretrain"]
    finetune = [row for row in history if row["phase"] == "finetune"]
    if pretrain:
        plt.plot([row["epoch"] for row in pretrain], [row["train_loss"] for row in pretrain], marker="o", label="Pretrain")
    if finetune:
        plt.plot(
            [row["epoch"] for row in finetune],
            [row["train_loss"] for row in finetune],
            marker="s",
            label="Finetune",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.grid(True, linestyle="--", linewidth=0.5)
    if pretrain and finetune:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=200)
    plt.close()


def save_ber_comparison(curve, output_dir: Path) -> None:
    with (output_dir / "ber_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)

    plt.figure(figsize=(6, 4))
    ebno = [row["ebno_db"] for row in curve]
    plt.semilogy(ebno, [row["baseline_ber"] for row in curve], marker="o", label="Traditional LMMSE")
    plt.semilogy(ebno, [row["ai_ber"] for row in curve], marker="s", label="DL detector")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("BER")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "ber_comparison.png", dpi=200)
    plt.close()


def build_optimizer(
    model: NeuralReceiverSystem,
    base_lr: float,
    backbone_lr_scale: float,
    gate_lr_scale: float,
    head_lr_scale: float,
) -> torch.optim.Optimizer:
    detector = model.detector
    parameter_groups = [
        {
            "params": list(detector.input_proj.parameters())
            + list(detector.backbone.parameters())
            + list(detector.global_skip_proj.parameters()),
            "lr": base_lr * backbone_lr_scale,
        },
        {
            "params": list(detector.gate_pool.parameters()) + list(detector.gate_network.parameters()),
            "lr": base_lr * gate_lr_scale,
        },
        {
            "params": list(detector.expert_heads.parameters()),
            "lr": base_lr * head_lr_scale,
        },
    ]
    return torch.optim.Adam(parameter_groups)


def one_sided_baseline_gap_loss(
    predicted_llr: torch.Tensor,
    baseline_llr: torch.Tensor,
    codewords: torch.Tensor,
) -> torch.Tensor:
    ai_per_bit = F.binary_cross_entropy_with_logits(predicted_llr, codewords.float(), reduction="none")
    mmse_per_bit = F.binary_cross_entropy_with_logits(baseline_llr, codewords.float(), reduction="none")
    return torch.relu(ai_per_bit - mmse_per_bit).mean()


def save_adaptive_profile(records, output_dir: Path, phase_epoch: int) -> None:
    if not records:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"adaptive_profile_epoch_{phase_epoch:02d}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    ebno = [row["ebno_db"] for row in records]
    gap = [row["ber_gap_log10"] for row in records]
    prob = [row["sampling_probability"] for row in records]

    plt.figure(figsize=(6.5, 4))
    plt.plot(ebno, gap, marker="o", label="log10 BER gap")
    plt.plot(ebno, prob, marker="s", label="sampling probability")
    plt.xlabel("Eb/N0 (dB)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"adaptive_profile_epoch_{phase_epoch:02d}.png", dpi=200)
    plt.close()


def build_adaptive_sampler(
    args: argparse.Namespace,
    model: NeuralReceiverSystem,
    baseline: TraditionalMIMOOfdmSystem,
    output_dir: Path,
    phase_epoch: int,
):
    ebno_points = np.arange(args.snr_low, args.snr_high + 1e-6, args.eval_snr_step, dtype=np.float32)
    ai_ber = []
    baseline_ber = []
    for ebno_db in ebno_points:
        baseline_metrics = evaluate_single_point(
            baseline,
            float(ebno_db),
            num_batches=args.adaptive_eval_batches,
            batch_size=baseline.cfg.eval_batch_size,
        )
        ai_metrics = evaluate_neural(
            model.eval(),
            float(ebno_db),
            num_batches=args.adaptive_eval_batches,
            batch_size=baseline.cfg.eval_batch_size,
        )
        baseline_ber.append(float(baseline_metrics["ber"]))
        ai_ber.append(float(ai_metrics["ber"]))

    baseline_ber = np.asarray(baseline_ber, dtype=np.float64)
    ai_ber = np.asarray(ai_ber, dtype=np.float64)
    gap = np.maximum(
        0.0,
        np.log10(ai_ber + args.adaptive_eps) - np.log10(baseline_ber + args.adaptive_eps),
    )

    if args.adaptive_smooth_window > 1:
        kernel = np.ones(args.adaptive_smooth_window, dtype=np.float64) / args.adaptive_smooth_window
        gap = np.convolve(gap, kernel, mode="same")

    weighted_gap = np.power(np.maximum(gap, 0.0), args.adaptive_gap_power)
    gap_tensor = torch.as_tensor(weighted_gap, dtype=torch.float32)
    softmax_weights = F.softmax(args.adaptive_temperature * gap_tensor, dim=0).cpu().numpy()
    probabilities = (
        args.adaptive_uniform_mix * np.ones_like(softmax_weights) / len(softmax_weights)
        + (1.0 - args.adaptive_uniform_mix) * softmax_weights
    )
    probabilities = probabilities / probabilities.sum()

    records = []
    for snr, ber_mmse, ber_ai, gap_value, prob in zip(ebno_points, baseline_ber, ai_ber, gap, probabilities):
        records.append(
            {
                "ebno_db": float(snr),
                "baseline_ber": float(ber_mmse),
                "ai_ber": float(ber_ai),
                "ber_gap_log10": float(gap_value),
                "sampling_probability": float(prob),
            }
        )
    save_adaptive_profile(records, output_dir, phase_epoch)

    bin_width = max(float(args.eval_snr_step), 1.0)
    print(
        f"Adaptive finetune epoch {phase_epoch:02d} | "
        f"max-gap SNR={float(ebno_points[int(np.argmax(gap))]):.1f} dB"
    )
    return lambda: sample_weighted_snr_bin(ebno_points, probabilities, bin_width)


def run_training_phase(
    model: NeuralReceiverSystem,
    baseline: TraditionalMIMOOfdmSystem,
    args: argparse.Namespace,
    output_dir: Path,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    epochs: int,
    steps_per_epoch: int,
    sampler,
    phase_name: str,
    history: list[dict],
    gap_loss_weight: float = 0.0,
) -> None:
    if epochs <= 0:
        return

    starting_epoch = len(history)
    for local_epoch in range(1, epochs + 1):
        global_epoch = starting_epoch + local_epoch
        active_sampler = sampler
        if phase_name == "finetune" and args.finetune_sampling_strategy == "adaptive_gap_db":
            active_sampler = build_adaptive_sampler(args, model, baseline, output_dir, global_epoch)
        model.train()
        running_loss = 0.0
        for _ in range(steps_per_epoch):
            ebno_db = active_sampler()
            batch = model(batch_size, ebno_db, return_intermediates=True)
            primary_loss = coded_bce_loss(batch["predicted_llr"], batch["codewords"])
            gap_loss = torch.zeros((), dtype=primary_loss.dtype, device=primary_loss.device)
            if gap_loss_weight > 0.0:
                gap_loss = one_sided_baseline_gap_loss(
                    batch["predicted_llr"],
                    batch["llr"],
                    batch["codewords"],
                )
            loss = primary_loss + gap_loss_weight * gap_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_record = {
            "phase": phase_name,
            "epoch": global_epoch,
            "train_loss": running_loss / steps_per_epoch,
        }
        history.append(epoch_record)
        print(f"{phase_name.capitalize()} epoch {local_epoch:02d} | train_loss={epoch_record['train_loss']:.6f}")


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
        batch_size=args.batch_size,
    )
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "AI-AIDED" / "outputs" / "dl_mimo"
    output_dir.mkdir(parents=True, exist_ok=True)

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
    if args.resume_weights:
        state_dict = torch.load(args.resume_weights, map_location=model.reference_link.cfg.device)
        model.detector.load_state_dict(state_dict)
        print(f"Loaded detector weights from {args.resume_weights}")
    baseline = TraditionalMIMOOfdmSystem(cfg)
    history = []
    optimizer = build_optimizer(
        model,
        base_lr=args.lr,
        backbone_lr_scale=args.backbone_lr_scale,
        gate_lr_scale=args.gate_lr_scale,
        head_lr_scale=args.head_lr_scale,
    )
    run_training_phase(
        model,
        baseline,
        args,
        output_dir,
        optimizer,
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        sampler=lambda: sample_training_snr(args),
        phase_name="pretrain",
        history=history,
        gap_loss_weight=0.0,
    )

    if args.finetune_epochs > 0:
        optimizer = build_optimizer(
            model,
            base_lr=args.finetune_lr,
            backbone_lr_scale=args.finetune_backbone_lr_scale,
            gate_lr_scale=args.finetune_gate_lr_scale,
            head_lr_scale=args.finetune_head_lr_scale,
        )
        run_training_phase(
            model,
            baseline,
            args,
            output_dir,
            optimizer,
            batch_size=args.batch_size,
            epochs=args.finetune_epochs,
            steps_per_epoch=args.finetune_steps_per_epoch,
            sampler=lambda: sample_finetune_snr(args),
            phase_name="finetune",
            history=history,
            gap_loss_weight=args.finetune_gap_loss_weight,
        )

    save_history(history, output_dir)
    torch.save(model.detector.state_dict(), output_dir / "dl_mimo_detector.pt")

    curve = []
    for ebno_db in np.arange(args.eval_snr_start, args.eval_snr_stop + 1e-6, args.eval_snr_step):
        baseline_metrics = evaluate_single_point(
            baseline,
            float(ebno_db),
            num_batches=args.eval_batches,
            batch_size=cfg.eval_batch_size,
        )
        ai_metrics = evaluate_neural(
            model.eval(),
            float(ebno_db),
            num_batches=args.eval_batches,
            batch_size=cfg.eval_batch_size,
        )
        record = {
            "ebno_db": float(ebno_db),
            "baseline_ber": baseline_metrics["ber"],
            "baseline_bler": baseline_metrics["bler"],
            "ai_ber": ai_metrics["ber"],
            "ai_bler": ai_metrics["bler"],
        }
        curve.append(record)
        print(
            f"Eb/N0={ebno_db:>5.1f} dB | baseline BER={record['baseline_ber']:.4e} | "
            f"AI BER={record['ai_ber']:.4e}"
        )

    save_ber_comparison(curve, output_dir)
    print(f"Saved detector weights and figures to {output_dir}")


if __name__ == "__main__":
    main()
