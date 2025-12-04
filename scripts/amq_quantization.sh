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
METHOD=${2}
# METHOD=awq
# METHOD=gptq
# METHOD=owq
GROUP_SIZE=128
NUM_OF_CANDIDATES=1
# TARGET_BITS=3.0
TARGET_BITS=${3}
TARGET_BITS_OFFSET=0.005
LOAD=amq/results/search/2512021742_Llama-2-7b-hf_dataset_wikitext2/iter_200.stats

## Data Args
EVAL_DATASET=(wikitext2 c4)
EVAL_SEQLEN=2048
EVAL_SEED=0

## Base Args
SAVE_PATH=amq/results/quantization/${MODEL_NAME}_${METHOD}_group_size_${GROUP_SIZE}_target_bits_${TARGET_BITS}
GPU_ID=${CUDA_VISIBLE_DEVICES}

args=(
    --gpu_id ${GPU_ID} \
    --save_path ${SAVE_PATH} \
    --model_path ${MODEL_PATH} \
    --model_name ${MODEL_NAME} \
    --config ${CONFIG} \
    --method ${METHOD} \
    --group_size ${GROUP_SIZE} \
    --num_of_candidates ${NUM_OF_CANDIDATES} \
    --target_bits ${TARGET_BITS} \
    --target_bits_offset ${TARGET_BITS_OFFSET} \
    --load ${LOAD} \
    --eval_dataset ${EVAL_DATASET} \
    --eval_seqlen ${EVAL_SEQLEN} \
    --eval_seed ${EVAL_SEED} \
)

echo ${args[@]}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --num_processes=${N_PROC} --num_machines=1 --main_process_port=${PORT_NUM} amq/amq_quantization.py ${args[@]}
