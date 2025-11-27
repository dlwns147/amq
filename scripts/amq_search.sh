CUDA_VISIBLE_DEVICES=0

## Model Args
MODEL_PATH=meta-llama
MODEL_NAME=Llama-2-7b-hf
QUANTIZATION_PROXY_PATHS=("/SSD/Woo/hqq/Llama-2-7b-hf_2bit_128gs_1axis" "/SSD/Woo/hqq/Llama-2-7b-hf_3bit_128gs_1axis" "/SSD/Woo/hqq/Llama-2-7b-hf_4bit_128gs_1axis")
GPU_ID=0

## Search Args
SENSITIVITY_THRESHOLD=2.0
SENSITIVITY_DATASETS=wikitext2
SENSITIVITY_N_SAMPLE=128
SENSITIVITY_SEQLEN=2048

PREDICTOR=rbf
ITERATIONS=200
N_DOE=250
N_ITER=50
MAX_VALUE=1.0

GA_POP_SIZE=200
CROSSOVER_PROB=0.9
MUT_PROB=0.1

SAVE_ITER=1
SAVE=True
RESUME_PATH=None

## Data Args
DATASET=wikitext2
N_SAMPLE=128
SEQLEN=2048
SEED=0

args=(
    --model_path ${MODEL_PATH}
    --model_name ${MODEL_NAME}
    --quantization_proxy_paths ${QUANTIZATION_PROXY_PATHS[@]}
    --gpu_id ${GPU_ID}
    --sensitivity_threshold ${SENSITIVITY_THRESHOLD}
    --sensitivity_datasets ${SENSITIVITY_DATASETS}
    --sensitivity_n_sample ${SENSITIVITY_N_SAMPLE}
    --sensitivity_seqlen ${SENSITIVITY_SEQLEN}
    --predictor ${PREDICTOR}
    --iterations ${ITERATIONS}
    --n_doe ${N_DOE}
    --n_iter ${N_ITER}
    --max_value ${MAX_VALUE}
    --ga_pop_size ${GA_POP_SIZE}
    --crossover_prob ${CROSSOVER_PROB}
    --mut_prob ${MUT_PROB}
    --save_iter ${SAVE_ITER}
    --resume_path ${RESUME_PATH}
    --dataset ${DATASET}
    --n_sample ${N_SAMPLE}
    --seqlen ${SEQLEN}
    --seed ${SEED}
)

if [ ${SAVE} == True ]; then
    args+=(--save)
fi

echo ${args[@]}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python amq/amq_search.py ${args[@]}
