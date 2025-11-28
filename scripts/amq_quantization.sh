TODAY=`date +%y%m%d%H%M`

## GPU Args
CUDA_VISIBLE_DEVICES=${1}
N_PROC=1
PORT_NUM=$(( ( RANDOM % 10000 )  + 10000 ))

## Model Args
MODEL_PATH=meta-llama
MODEL_NAME=Llama-2-7b-hf
CONFIG=amq/configs/llama.json

## Quantization Args
METHOD=awq
GROUP_SIZE=128
TARGET_BITS=3.0
LOAD=amq/results/2511280309_Llama-2-7b-hf_dataset_wikitext2/iter_50.stats

## Data Args
DATASET=wikitext2
SEQLEN=2048

## Base Args
SAVE_PATH=amq/quantization/${MODEL_NAME}_dataset_${DATASET}_method_${METHOD}_group_size_${GROUP_SIZE}_target_bits_${TARGET_BITS}
GPU_ID=${CUDA_VISIBLE_DEVICES}

args=(
    --model_path ${MODEL_PATH} \
    --model_name ${MODEL_NAME} \
    --config ${CONFIG} \
    --method ${METHOD} \
    --group_size ${GROUP_SIZE} \
    --target_bits ${TARGET_BITS} \
    --load ${LOAD} \
    --gpu_id ${GPU_ID} \
    --dataset ${DATASET} \
    --seqlen ${SEQLEN} \
)

echo ${args[@]}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} amq/amq_quantization.py ${args[@]}
