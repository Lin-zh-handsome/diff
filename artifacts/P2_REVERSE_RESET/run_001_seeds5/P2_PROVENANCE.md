# P2 Reverse Reset Provenance

- Diagnostic source: `diagnose_pusht_reverse_reset.py`
- Git commit: `08d529d582fb15626bec023883434e81300c947c`
- Checkpoint: `artifacts/P0_BASELINE/checkpoints/pusht_lowdim_cnn_epoch0550.ckpt`
- Policy: EMA `DiffusionUnetLowdimPolicy`
- Common-random-number control: same clean Push-T sequence and forward epsilon for oracle, free, and reset branches.
- Deterministic sampler: `DDIMScheduler`, `eta=0.0`.

## Runs

| Run | DDIM inference steps | Seeds | Batch size | Source timesteps | Total depths | Reset fractions |
|---|---:|---|---:|---|---|---|
| `run_001_seeds5` | 100 | 0,1,2,3,4 | 64 | 20,50,90 | 4,8,16 | 0.25,0.5,0.75 |
| `run_002_ddim50_seeds5` | 50 | 0,1,2,3,4 | 64 | 40,50,90 | 4,8,16 | 0.25,0.5,0.75 |

The run-local `summary.json`, `metrics_per_seed.csv`, and curve PNG files are the primary numerical evidence. This addendum records the exact committed source that produced both runs.
