#!/bin/bash

data='jddc'
model='BERT'

for i in {0..9}
do
echo CUDA=$[ i%4 ] log/${data}_${model}_$i.log
CUDA_VISIBLE_DEVICES=$[ i%4 ] nohup python -u driver_act.py -fold=$i --data=${data} --model=${model} > log/${data}_${model}_$i.log 2>&1 &
done