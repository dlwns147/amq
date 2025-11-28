CUDA_VISIBLE_DEVICES=${1}

## Model Args
MODEL_PATH=meta-llama
MODEL_NAME=Llama-2-7b-hf
QUANTIZATION_PROXY_PATHS=("/SSD/Woo/hqq/Llama-2-7b-hf_2bit_128gs_1axis" "/SSD/Woo/hqq/Llama-2-7b-hf_3bit_128gs_1axis" "/SSD/Woo/hqq/Llama-2-7b-hf_4bit_128gs_1axis")
GPU_ID=${CUDA_VISIBLE_DEVICES}

## Data Args
DATASET=wikitext2
SEED=0
N_SAMPLE=128
SEQLEN=2048
CONFIG=amq/configs/llama.json

## Output Args
SAVE_PATH=amq/sensitivity/${MODEL_NAME}_dataset_${DATASET}_n_sample_${N_SAMPLE}_seqlen_${SEQLEN}

## Main Args
args=(
    --gpu_id ${GPU_ID} \
    --model_path ${MODEL_PATH} \
    --model_name ${MODEL_NAME} \
    --quantization_proxy_paths ${QUANTIZATION_PROXY_PATHS[@]}
    --dataset ${DATASET} \
    --seed ${SEED} \
    --n_sample ${N_SAMPLE} \
    --seqlen ${SEQLEN} \
    --config ${CONFIG} \
    --save_path ${SAVE_PATH}
)

echo ${args[@]}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python amq/amq_sensitivity.py \
--save \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--quantization_proxy_paths ${QUANTIZATION_PROXY_PATHS[@]}