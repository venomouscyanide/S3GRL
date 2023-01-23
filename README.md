S3GRL (Scalable Simplified Subgraph Representation Learning)
===============================================================================

## Getting Started

All the requirements for getting the dev environment ready is available in `quick_install.sh`.

## Reproducing the Paper's Tabular data

### Reproducing Table 2

- All baselines (except SGRL) can be reproduced using `baselines/run_helpers/run_*.py`, where * is the respective
  baseline script.
- All SGRL baselines (except WalkPool) can be reproduced
  using `python sgrl_run_manager.py --config configs/paper/table_2.json`.
- WalkPool results can be reproduced using `bash run_ssgrl.sh` by running from Software/WalkPooling/bash
- S3GRL results can be reproduced using `python sgrl_run_manager.py --config configs/paper/auc_s3grl.json`.

### Reproducing Table 3

- WalkPool can be reproduced using `bash run_ssgrl_profile_attr.sh`
  and `bash run_ssgrl_profile_non.sh` by running from Software/WalkPooling/bash
- All other S3GRL and SGRL models can be reproduced
  using `python sgrl_run_manager.py --config configs/paper/profiling_attr.json`
  and `python sgrl_run_manager.py --config configs/paper/profiling_non.json`

### Reproducing Table 4

- S3GRL models with ScaLed enabled can be reproduced
  using `python sgrl_run_manager.py --config configs/paper/scaled.json`.

## Acknowledgements

The code for S3GRL is based off a clone of SEAL-OGB by Zhang et al. (https://github.com/facebookresearch/SEAL_OGB) and
ScaLed by Louis et al. (https://github.com/venomouscyanide/ScaLed). The baseline softwares used are adapted from GIC by
Mavromatis et al. (https://github.com/cmavro/Graph-InfoClust-GIC) and WalkPool by Pan et
al. (https://github.com/DaDaCheng/WalkPooling). There are also some baseline model codes taken from OGB
implementations (https://github.com/snap-stanford/ogb) and other Pytorch Geometric
implementations (https://github.com/pyg-team/pytorch_geometric).