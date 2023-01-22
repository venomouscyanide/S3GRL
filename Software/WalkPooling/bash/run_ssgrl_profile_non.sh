#!/bin/bash
for name in NS Power Yeast Ecoli PB
do
   for i in 1
   do
      bash ./wo_attr.sh $i $i $name
   done
done
