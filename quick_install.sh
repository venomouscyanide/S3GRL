#!/bin/sh
conda create --name sweal_3_9 python=3.9 # don't use python 3.10 as it breaks the PyCharm debugger flow: https://youtrack.jetbrains.com/issue/PY-52137
conda activate sweal_3_9
conda install pytorch torchvision -c pytorch
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
pip install matplotlib
pip install ogb
pip install networkx
pip install pytorch_memlab
pip install class_resolver
pip install fast-pagerank
pip install sklearn
pip install graphistry
