#!/bin/bash
for name in USAir NS Power Celegans Router PB Ecoli Yeast
do
   for i in 1 2 3 4 5 6 7 8 9 10
   do
      bash ./wo_attr.sh $i $i $name
   done
done

for name in cora citeseer pubmed cornell texas wisconsin chameleon
do
    for i in 1 2 3 4 5 6 7 8 9 10
    do
     bash ./w_attr.sh $i $i $name
    done
done
