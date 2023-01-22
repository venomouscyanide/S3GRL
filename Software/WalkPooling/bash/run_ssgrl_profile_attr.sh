#!/bin/bash
for name in cora citeseer cornell wisconsin pubmed
do
    for i in 1
    do
     bash ./w_attr.sh $i $i $name
    done
done
