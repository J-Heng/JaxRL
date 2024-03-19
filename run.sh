#!/bin/bash
POLICY='iql'

SEED=(0 1 2 3 4)
LEN_SEED=${#SEED[*]}

ENV=('walker2d-medium-v2' 'ant-medium-v2')
LEN_ENV=${#ENV[*]}

CudaNum=(0 1)

for i in $(seq 1 ${LEN_ENV})
do
  for j in $(seq 1 ${LEN_SEED})
  do
	  CUDA_VISIBLE_DEVICES=${CudaNum[i]} python train.py --policy $POLICY --env_name ${ENV[i]} --seed ${SEED[j]} &
  done
done

wait
sleep 2
echo "Completed!"