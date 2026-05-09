# AI-Aided Modules

This folder contains the two AI modules requested by the project:

- `train_dl_mimo.py`: training and BER comparison script for the DL-based MIMO detector
- `acm_controller.py`: offline data generation and MCS classifier for AI-driven adaptive coding and modulation
- `train_acm.py`: dataset generation, ACM training, and throughput evaluation

## Example commands

```powershell
conda run -n Sionna python AI-AIDED/train_acm.py --num-samples 40 --epochs 120
```

Generated models, datasets, and figures are stored under `AI-AIDED/outputs/`.
