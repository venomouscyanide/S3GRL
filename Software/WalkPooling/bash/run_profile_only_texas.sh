#!/bin/bash
for name in texas
do
   for i in 1
   do
      bash ./wo_attr.sh $i $i $name
   done
done
