#!/bin/bash

data='jddc'
model='BERT'

export CUDA_VISIBLE_DEVICES=0; nohup python -u train.py -fold=0 --data=${data} --model=${model} > pipe/${data}_${model}_0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python -u train.py -fold=1 --data=${data} --model=${model} > pipe/${data}_${model}_1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python -u train.py -fold=2 --data=${data} --model=${model} > pipe/${data}_${model}_2.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python -u train.py -fold=3 --data=${data} --model=${model} > pipe/${data}_${model}_3.log 2>&1 &
export CUDA_VISIBLE_DEVICES=4; nohup python -u train.py -fold=4 --data=${data} --model=${model} > pipe/${data}_${model}_4.log 2>&1 &
export CUDA_VISIBLE_DEVICES=0; nohup python -u train.py -fold=5 --data=${data} --model=${model} > pipe/${data}_${model}_5.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1; nohup python -u train.py -fold=6 --data=${data} --model=${model} > pipe/${data}_${model}_6.log 2>&1 &
export CUDA_VISIBLE_DEVICES=2; nohup python -u train.py -fold=7 --data=${data} --model=${model} > pipe/${data}_${model}_7.log 2>&1 &
export CUDA_VISIBLE_DEVICES=3; nohup python -u train.py -fold=8 --data=${data} --model=${model} > pipe/${data}_${model}_8.log 2>&1 &
export CUDA_VISIBLE_DEVICES=4; nohup python -u train.py -fold=9 --data=${data} --model=${model} > pipe/${data}_${model}_9.log 2>&1 &
