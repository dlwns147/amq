#!/bin/bash

TODAY=`date +%y%m%d%H%M`

## GPU Args
CUDA_VISIBLE_DEVICES=${1}
N_PROC=1
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

## Model Args
MODEL_PATH=meta-llama
MODEL_NAME=Llama-2-7b-hf

## Quantization Args
NBITS=2
GROUP_SIZE=128

## Base Args
SAVE_PATH=/SSD/hqq/${MODEL_NAME}_${NBITS}bit_${GROUP_SIZE}gs_1axis

args=(
    --model_path ${MODEL_PATH} \
    --model_name ${MODEL_NAME} \
    --nbits ${NBITS} \
    --group_size ${GROUP_SIZE} \
    --save_path ${SAVE_PATH} \
)

echo ${args[@]}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} amq/amq_quantization_proxy.py ${args[@]}
