S3GRL (Scalable Simplified Subgraph Representation Learning)
===============================================================================

## Getting Started

All the requirements for getting the dev environment ready is available in `quick_install.sh`.

## Reproducing the Paper's Tabular data

### Reproducing Table 2

- All baselines (except SGRL) can be reproduced using `baseline/run_helpers/run_*.py`, where * is the respective
  baseline script.
- All SGRL baselines (except WalkPool) can be reproduced
  using `python sgrl_run_manager.py --config configs/paper/table_2.json`.
- WalkPool results can be reproduced using `bash Software/WalkPooling/bash/run_ssgrl.sh`
- S3GRL results can be reproduced using `python sgrl_run_manager.py --config configs/paper/auc_s3grl.json`.

### Reproducing Table 3

- WalkPool can be reproduced using `bash Software/WalkPooling/bash/run_ssgrl_profile_attr.sh`
  and `bash Software/WalkPooling/bash/run_ssgrl_profile_non.sh`
- All other S3GRL and SGRL models can be reproduced
  using `python sgrl_run_manager.py --config configs/paper/profiling_attr.json`
  and `python sgrl_run_manager.py --config configs/paper/profiling_non.json`

### Reproducing Table 4

- S3GRL models with ScaLed enabled can be reproduced
  using `python sgrl_run_manager.py --config configs/paper/scaled.json`.