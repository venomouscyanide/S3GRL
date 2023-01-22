#!/bin/bash
for name in Ecoli Yeast Power Router NS Celegans USAir PB
do
   for i in 1 2 3 4 5 6 7 8 9 10
   do
      bash ./wo_attr.sh $i $i $name
   done
done