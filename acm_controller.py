from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
COMSYS_DIR = ROOT / "comsys"
if str(COMSYS_DIR) not in sys.path:
    sys.path.insert(0, str(COMSYS_DIR))

from baseline_system import TraditionalMIMOOfdmSystem, clone_config, evaluate_single_point  # noqa: E402
from config import DEFAULT_MCS_TABLE, MIMOOfdmConfig  # noqa: E402


class ACMController(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 32, num_classes: int = len(DEFAULT_MCS_TABLE)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def mcs_table() -> Sequence[Dict[str, float]]:
    return DEFAULT_MCS_TABLE


def build_mcs_system(cfg: MIMOOfdmConfig, mcs_entry: Dict[str, float]) -> TraditionalMIMOOfdmSystem:
    mcs_cfg = clone_config(
        cfg,
        num_bits_per_symbol=int(mcs_entry["num_bits_per_symbol"]),
        coderate=float(mcs_entry["coderate"]),
        code_family=str(mcs_entry["code_family"]),
    )
    return TraditionalMIMOOfdmSystem(mcs_cfg)


def oracle_mcs_index(
    base_cfg: MIMOOfdmConfig,
    ebno_db: float,
    bler_target: float = 0.1,
    num_batches: int = 6,
) -> Dict[str, float]:
    candidates = []
    for entry in mcs_table():
        system = build_mcs_system(base_cfg, entry)
        metrics = evaluate_single_point(system, ebno_db, num_batches=num_batches, batch_size=base_cfg.eval_batch_size)
        candidate = dict(entry)
        candidate.update(metrics)
        candidates.append(candidate)

    feasible = [row for row in candidates if row["bler"] <= bler_target]
    if feasible:
        best = max(feasible, key=lambda row: row["throughput_bits_per_frame"])
    else:
        best = min(candidates, key=lambda row: row["bler"])
    return best


def generate_dataset(
    cfg: MIMOOfdmConfig,
    num_samples: int = 60,
    snr_low: float = -2.0,
    snr_high: float = 20.0,
    bler_target: float = 0.1,
    num_batches: int = 6,
) -> Dict[str, np.ndarray]:
    features: List[List[float]] = []
    labels: List[int] = []
    rows: List[Dict[str, float]] = []

    for _ in range(num_samples):
        ebno_db = float(np.random.uniform(snr_low, snr_high))
        best = oracle_mcs_index(cfg, ebno_db, bler_target=bler_target, num_batches=num_batches)
        features.append([ebno_db])
        labels.append(int(best["mcs_index"]))
        rows.append(
            {
                "ebno_db": ebno_db,
                "best_mcs_index": int(best["mcs_index"]),
                "best_mod_order": int(best["num_bits_per_symbol"]),
                "best_coderate": float(best["coderate"]),
                "oracle_bler": float(best["bler"]),
                "oracle_throughput_bits_per_frame": float(best["throughput_bits_per_frame"]),
            }
        )

    return {
        "features": np.asarray(features, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "rows": rows,
    }


def save_dataset(dataset: Dict[str, np.ndarray], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "acm_dataset.npz", features=dataset["features"], labels=dataset["labels"])
    with (output_dir / "acm_dataset.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dataset["rows"][0].keys()))
        writer.writeheader()
        writer.writerows(dataset["rows"])


def train_controller(
    controller: ACMController,
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
) -> List[Dict[str, float]]:
    controller.to(device)
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    y = torch.as_tensor(labels, dtype=torch.long, device=device)
    counts = np.bincount(labels, minlength=controller.net[-1].out_features).astype(np.float32)
    counts[counts == 0] = 1.0
    class_weights = torch.as_tensor(counts.sum() / counts, dtype=torch.float32, device=device)
    class_weights = class_weights / class_weights.mean()

    optimizer = torch.optim.Adam(controller.parameters(), lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        logits = controller(x)
        loss = F.cross_entropy(logits, y, weight=class_weights)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            acc = (logits.argmax(dim=1) == y).float().mean().item()
        history.append({"epoch": epoch, "loss": float(loss.item()), "accuracy": acc})
    return history


@torch.no_grad()
def evaluate_closed_loop(
    controller: ACMController,
    cfg: MIMOOfdmConfig,
    ebno_points: Sequence[float],
    device: str = "cpu",
    num_batches: int = 8,
) -> List[Dict[str, float]]:
    controller.eval().to(device)
    records = []
    for ebno_db in ebno_points:
        features = torch.tensor([[float(ebno_db)]], dtype=torch.float32, device=device)
        predicted_index = int(controller(features).argmax(dim=1).item())
        predicted_entry = mcs_table()[predicted_index]
        oracle_entry = oracle_mcs_index(cfg, float(ebno_db), num_batches=num_batches)
        robust_entry = mcs_table()[0]
        aggressive_entry = mcs_table()[-1]

        predicted_system = build_mcs_system(cfg, predicted_entry)
        robust_system = build_mcs_system(cfg, robust_entry)
        aggressive_system = build_mcs_system(cfg, aggressive_entry)

        predicted_metrics = evaluate_single_point(predicted_system, float(ebno_db), num_batches=num_batches)
        robust_metrics = evaluate_single_point(robust_system, float(ebno_db), num_batches=num_batches)
        aggressive_metrics = evaluate_single_point(aggressive_system, float(ebno_db), num_batches=num_batches)

        records.append(
            {
                "ebno_db": float(ebno_db),
                "predicted_mcs_index": predicted_index,
                "oracle_mcs_index": int(oracle_entry["mcs_index"]),
                "ai_throughput": predicted_metrics["throughput_bits_per_frame"],
                "robust_throughput": robust_metrics["throughput_bits_per_frame"],
                "aggressive_throughput": aggressive_metrics["throughput_bits_per_frame"],
                "oracle_throughput": oracle_entry["throughput_bits_per_frame"],
            }
        )
    return records
