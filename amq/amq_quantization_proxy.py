import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--nbits', type=int, default=2)
parser.add_argument('--group_size', type=int, default=128)
args = parser.parse_args()

#Settings
######################################################################################
hf_auth    = None #HuggingFace token
cache_path = ''   #cache directory to store data

#Chose a model
model_path = "meta-llama"
model_name = "Llama-2-7b-hf"
model_id  = f'{model_path}/{model_name}'

nbits=args.nbits
group_size=args.group_size
axis=1

#Load model on the CPU
######################################################################################
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

#Quantize the model
######################################################################################

if nbits < 16:
    from hqq.models.hf.base import AutoHQQHFModel
    from hqq.core.quantize import *
    
    quant_config = BaseQuantizeConfig(nbits=nbits, group_size=group_size) 
    AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=torch.float16, device='cuda')

import os
save_dir = os.path.join('/SSD/hqq', f'{model_name}_{nbits}bit_{group_size}gs_{axis}axis')
from hqq.models.hf.base import AutoHQQHFModel
AutoHQQHFModel.save_quantized(model, save_dir)
