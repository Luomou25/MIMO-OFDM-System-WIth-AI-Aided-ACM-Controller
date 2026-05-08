# Traditional MIMO-OFDM Baseline

This folder contains the conventional end-to-end communication chain required by the project:

- binary source
- 5G NR LDPC/Polar coding and rate matching
- QPSK / 16QAM / 64QAM
- MIMO spatial multiplexing
- OFDM resource grid mapping
- 3GPP TR 38.901 CDL/TDL fading channels
- LS or LMMSE-interpolated channel estimation
- LMMSE MIMO equalization
- soft demapping and channel decoding

## Main files

- `config.py`: global simulation configuration and default MCS table
- `baseline_system.py`: reusable Sionna-based transmitter/channel/receiver chain
- `run_baseline.py`: SNR sweep script that saves BER/BLER and throughput figures

## Example

```powershell
conda run -n Sionna python comsys/run_baseline.py --channel-type cdl --channel-profile A --bits-per-symbol 2 --coderate 0.5 --estimator-type ls --num-batches 10
```

Results are saved under `comsys/outputs/`.
