S3GRL (Scalable Simplified Subgraph Representation Learning)
===============================================================================
S3GRL (Scalable Simplified Subgraph Representation Learning) is a subgraph representation learning (SGRL) framework aimed at faster link prediction. S3GRL introduces subgraph sampling and subgraph diffusion operator pairs that allow for fast precomputation leading to faster runtimes.

Please read our paper here: https://arxiv.org/abs/2301.12562 

## S3GRL architecture
<img width="1260" alt="Screenshot 2023-01-28 at 3 19 09 PM" src="https://user-images.githubusercontent.com/14299839/215289015-e437d5d5-9df7-48b4-842a-932d4a0c7fc2.png">

Our S3GRL framework: In the preprocessing phase (shown by the shaded blue arrow), first multiple subgraphs are extracted around the target nodes $u$ and $v$ (shaded in blue) by various sampling strategies.  Diffusion matrices are then created from extracted subgraph adjacency matrices by predefined diffusion operators (e.g., powers of subgraphs in this figure). Each diffusion process involves the application of the subgraph diffusion matrix on its nodal features to create the matrix ![CodeCogsEqn](https://user-images.githubusercontent.com/14299839/215292449-7d2eb4ed-f482-4125-9a4f-6f0a0c7e7367.svg). The operator-level node representations of selected nodes (with a red border in raw data) are then aggregated for all subgraphs to form the joint ![CodeCogsEqn(4)](https://user-images.githubusercontent.com/14299839/215292690-ea451ed5-07a8-4b1c-b43a-40a868f23e3c.svg) matrix. The selected nodes in this example are the target nodes ![CodeCogsEqn(1)](https://user-images.githubusercontent.com/14299839/215292532-dd63afe8-5d7f-46bf-9a1e-a60894d4a7d6.svg)
, and their common neighbor $d$. In the learning phase (as shown by the shaded red arrow), the joint matrix ![CodeCogsEqn(4)](https://user-images.githubusercontent.com/14299839/215292690-ea451ed5-07a8-4b1c-b43a-40a868f23e3c.svg) undergoes dimensionality reduction followed by pooling using center pooling (highlighted by blue-border box) and common neighbor pooling (highlighted by purple-border box). Finally, the target representation ![CodeCogsEqn(2)](https://user-images.githubusercontent.com/14299839/215292664-86ab68a1-24f6-4f4e-9642-fcfb42f22755.svg)
 is transformed by an MLP to a link probability $P_{uv}$.




A higher quality architecture diagram can be found here: [SG3RL_arch.pdf](https://github.com/venomouscyanide/S3GRL/files/10528180/SG3RL_arch.pdf)


## Getting Started

All the requirements for getting the dev environment ready is available in `quick_install.sh`.


## Running our codes
All our codes can be run by setting a JSON configuration file. Example configuration files can be found in `configs/paper/`. Once you have the configuration JSON file setup, run our codes using `python sgrl_run_manager.py --config your_config_file_here.json --results_json your_result_file.json`. This command produces `your_result_file.json` which contains the efficacy score using the configuration on the dataset the experiments are being run on. 


## Arguments supported

Currently, our S3GRL framework support two instances, **PoS** and **SoP**. Both instances are available in `tuned_SIGN.py`. You can add your own instances by modifying this file.

Specific arguments related to our framework:
- `sign_k` - The number of diffusion operators to create (corresponds to r in our paper)
- `sign_type` - Set the type of S3GRL model. Either PoS or SoP.
- `optimize_sign` - Choose to use optimized codes for running PoS or SoP. 
- `init_features` - Initialize features for non-attributed datasets. Our S3GRL models require initial node features to work. Choose between 'degree', 'eye' or 'n2v'.
- `k_heuristic` - Choose to use `CCN` (center-common-neighbor pooling) for the nodes other than source-target nodes.
- `k_node_set_strategy` - How to choose the nodes for `CCN`. Either intersection or union. Intersection means you choose the common  neighbors. Works when `k_heuristic` is True.
- `k_pool_strategy` - How to pool the nodes other than source-target in the forward pass for `CCN` (the nodes you select using `k_node_set_strategy`).  Either mean or sum pooling. Works when `k_heuristic` is True.
- `init_representation` - Use an unsupervised model to train the initial features before running S3GRL. Choose between 'GIC', 'ARGVA', 'GAE', 'VGAE'.

## Supported Datasets
We support any PyG dataset. However, the below list covers all the datasets we use in our paper's experiments:

1) Planetoid Dataset (Cora, PubMed, CiteSeer) from "Revisiting Semi-Supervised Learning with Graph Embeddings
    <https://arxiv.org/abs/1603.08861>"
2) SEAL datasets (USAir, Yeast etc. introduced in the original paper) from "Link prediction based on graph neural networks https://arxiv.org/pdf/1802.09691.pdf"


Notably, the OGB dataset support is actively being developed in a private fork.

## Reporting Issues and Improvements
We currently don't have an issue/PR template. However, if you find an issue in our code please create an issue in GitHub. It would be great if you could give as much information regarding the issue as possible (what command was run, what are the python package versions, providing full stack trace etc.).  

If you have any further questions, you can reach out to us (the authors) via email and we will be happy to have a conversation. 
[Paul Louis](mailto:paul.louis@ontariotechu.net), [Shweta Ann Jacob](mailto:shweta.jacob@ontariotechu.net)



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

## Citing Us/BibTex
Please cite our work if you find it useful in any way.
```
@misc{https://doi.org/10.48550/arxiv.2301.12562,
  doi = {10.48550/ARXIV.2301.12562},
  url = {https://arxiv.org/abs/2301.12562},
  author = {Louis, Paul and Jacob, Shweta Ann and Salehi-Abari, Amirali},
  keywords = {Machine Learning (cs.LG), Social and Information Networks (cs.SI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Simplifying Subgraph Representation Learning for Scalable Link Prediction},
  publisher = {arXiv},
  year = {2023},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```

The PDF of our preprint is available at https://arxiv.org/pdf/2301.12562.pdf
