CUDA_VISIBLE_DEVICES=0
MODEL_PATH=meta-llama
MODEL_NAME=Llama-2-7b-hf
QUANTIZATION_PROXY_PATHS=("/SSD/Woo/hqq/Llama-2-7b-hf_2bit_128gs_1axis" "/SSD/Woo/hqq/Llama-2-7b-hf_3bit_128gs_1axis" "/SSD/Woo/hqq/Llama-2-7b-hf_4bit_128gs_1axis")

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python amq/amq_sensitivity.py \
--save \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--quantization_proxy_paths ${QUANTIZATION_PROXY_PATHS[@]}