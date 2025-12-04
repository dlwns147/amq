import gc
import random

from datetime import timedelta

from utils.dispatch import simple_dispatch_model

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed as set_seed_transformers
from accelerate import Accelerator, InitProcessGroupKwargs
from hqq.models.hf.base import AutoHQQHFModel

def clean_up():
    gc.collect()
    torch.cuda.empty_cache()

def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    set_seed_transformers(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def getsubattr(obj, attr):
    attrs = attr.split('.')
    if len(attrs) > 1:
        return getsubattr(getattr(obj, attrs[0]), '.'.join(attrs[1:]))
    else:
        return getattr(obj, attr)
    
def setsubattr(obj, attr, value):
    attrs = attr.split('.')
    if len(attrs) > 1:
        setsubattr(getattr(obj, attrs[0]), '.'.join(attrs[1:]), value)
    else :
        setattr(obj, attr, value)

def delsubattr(obj, attr):
    attrs = attr.split('.')
    if len(attrs) > 1:
        return delsubattr(getattr(obj, attrs[0]), '.'.join(attrs[1:]))
    else:
        return delattr(obj, attr)
    
def hassubattr(obj, attr):
    attrs = attr.split('.')
    if len(attrs) > 1:
        return hassubattr(getattr(obj, attrs[0]), '.'.join(attrs[1:]))
    else:
        return hasattr(obj, attr)

def getblock(model, config):
    return getsubattr(model, config['layers'])

def init_accelerator(gpu_id, config):
    gpu_id = gpu_id.split(',')

    ipg_handler = InitProcessGroupKwargs(
            timeout=timedelta(seconds=5400)
            )

    accelerator = Accelerator(kwargs_handlers=[ipg_handler])
    n_proc = accelerator.num_processes
    assert len(gpu_id) % n_proc == 0, 'Total number of gpus (args.gpu_id) should be divisible by num_processes'

    gpu_start_idx = accelerator.device.index if accelerator.device.index is not None else 0
    
    gpu_per_proc = len(gpu_id) // n_proc
    n_block = int(config['n_block'])
    assert n_block % gpu_per_proc == 0, f'n_block {n_block} is not divisible by {gpu_per_proc}'

    blk_per_gpu = n_block // gpu_per_proc
    cur_gpu_id = list(range(gpu_start_idx, len(gpu_id), n_proc))

    device_map = dict()
    for pre_layer in config['pre_layer']:
        device_map[pre_layer] = cur_gpu_id[0]

    for layer_idx in range(n_block):
        device_map[f"{config['layers']}.{layer_idx}"] = cur_gpu_id[layer_idx // blk_per_gpu]
            
    for post_layer in config['post_layer']:
        device_map[post_layer] = cur_gpu_id[-1]

    return accelerator, device_map


def get_bits_usage(architecture, config, group_size=128):
    bits_usage = 0
    memory_usage = 0

    for linear_group, bits in architecture['linear'].items():
        for block_idx, bit in enumerate(bits):
            out_dim, in_dim = config['linear_shape'][linear_group]
            c_group_size = in_dim if group_size == -1 else group_size
            bit += 32 / c_group_size if bit < 16 else 0
            memory_usage += int(out_dim) * int(in_dim) * bit
                
    bits_usage = memory_usage / config['model_numel']
    
    return bits_usage
    

def get_hfmodel(model_name_or_path: str,
                device_map='auto',
                dtype='auto',
                trust_remote_code=False,
                use_cache=False,
                **kwargs
                ):

    clean_up()

    org_kaiming_uniform = torch.nn.init.kaiming_uniform_
    org_uniform = torch.nn.init.uniform_
    org_normal = torch.nn.init.normal_

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        torch_dtype=dtype,
        device_map=device_map, 
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
        use_cache=use_cache,
        **kwargs
    )
    model.config.use_cache = use_cache
    
    torch.nn.init.kaiming_uniform_ = org_kaiming_uniform
    torch.nn.init.uniform_ = org_uniform
    torch.nn.init.normal_ = org_normal

    model.eval()
    model.use_cache = False
    
    return model

def get_quantization_proxy(quant_model_paths, device_map):
    clean_up()

    org_kaiming_uniform = torch.nn.init.kaiming_uniform_
    org_uniform = torch.nn.init.uniform_
    org_normal = torch.nn.init.normal_

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    quantization_proxies = []
    for quant_model_path in quant_model_paths:
        model = AutoHQQHFModel.from_quantized(quant_model_path, device_map='cpu')
        model = simple_dispatch_model(model, device_map)
        
        quantization_proxies.append(model)
        model.config.use_cache = False

        clean_up()

        print(f'{quant_model_path} :  {torch.cuda.max_memory_reserved() / 1024 / 1024}MB')

    torch.nn.init.kaiming_uniform_ = org_kaiming_uniform
    torch.nn.init.uniform_ = org_uniform
    torch.nn.init.normal_ = org_normal

    return quantization_proxies

def get_tokenizer(model_id):
    return AutoTokenizer.from_pretrained(model_id)