#!/bin/sh
conda create --name torch python==3.9
conda activate torch
conda install pytorch torchvision  -c pytorch
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
pip install matplotlib
pip install ogb
pip install networkx
pip install pytorch_memlab