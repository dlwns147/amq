#!/bin/bash

TODAY=`date +%y%m%d%H%M`

## GPU Args
CUDA_VISIBLE_DEVICES=${1}
N_PROC=1
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

## Model Args
MODEL_PATH=meta-llama
MODEL_NAME=Llama-2-7b-hf
SAVE_PATH=/SSD/hqq

## Search Args
USE_FT=True
TPS=True
GEMV=False
GEMM=False
TTFT=False
MEMORY=False
PEAK_MEMORY=True

## Benchmark Args
# ARCH_PATH=/results/search/**.stats
TARGET_BITS=4
# FILE_NAME=amq_speed_benchmark_${TODAY}.json

args=(
    --model_path ${MODEL_PATH} \
    --model_name ${MODEL_NAME} \
    --save_path ${SAVE_PATH} \
    --target_bits ${TARGET_BITS} \
)

if [ ${USE_FT} == True ]; then
    args+=("--use_ft")
fi
if [ ${TPS} == True ]; then
    args+=("--tps")
fi
if [ ${GEMV} == True ]; then
    args+=("--gemv")
fi
if [ ${GEMM} == True ]; then
    args+=("--gemm")
fi
if [ ${TTFT} == True ]; then
    args+=("--ttft")
fi
if [ ${MEMORY} == True ]; then
    args+=("--memory")
fi
if [ ${PEAK_MEMORY} == True ]; then
    args+=("--peak_memory")
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} amq/amq_speed_benchmark.py ${args[@]}