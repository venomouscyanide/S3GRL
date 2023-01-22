#!/bin/bash
python ../src/main.py --seed $1 --log d1_$2 --data-name $3 --drnl 1 --init-attribute ones --init-representation None --embedding-dim 16 --observe-val-and-injection false --use-splitted 0