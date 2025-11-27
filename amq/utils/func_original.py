import random
import numpy as np
from accelerate import Accelerator, InitProcessGroupKwargs
from hqq.models.hf.base import AutoHQQHFModel
from .dispatch import simple_dispatch_model
from accelerate import dispatch_model
import scipy.stats as stats
import torch
from transformers import AutoModelForCausalLM
from datetime import timedelta
import gc
from copy import deepcopy

# from hqq.utils.patching_woo import prepare_for_inference

def clean_up():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_correlation(prediction, target):
    rmse = np.sqrt(((prediction - target) ** 2).mean())
    rho, _ = stats.spearmanr(prediction, target)
    tau, _ = stats.kendalltau(prediction, target)

    return rmse, rho, tau

def compute_latency(arch, config, latency_table):
    # with open(config['mpe_table_json'], 'r') as f:
    #     mpe_table = json.load(f)
    if latency_table is None:
        return 0
        
    latency = 0
    if 'fp16_model' in latency_table:
        latency = latency_table['fp16_model']
        for name in arch['linear']:
            for bit in arch['linear'][name]:
                latency += latency_table[str(int(bit))][name]
        return latency
    
    elif 'etc' in latency_table:
        for layer, layer_arch in arch['layer'].items():
            # if layer_arch[blk_idx] == 1:
            latency += latency_table[layer] * sum(layer_arch)

        latency += latency_table["etc"]
        # return latency
        return latency / latency_table["full"]

def compute_bits(arch, config, group_size=-1):
    memory_usage = 0
    for linear_group, bits in arch['linear'].items():
        for blk, bit in enumerate(bits):
            for linear in linear_group.split(','):
                out_dim, in_dim = config['linear_shape'][linear]
                c_group_size = in_dim if group_size == -1 else group_size
                bit += 32 / c_group_size if bit < 16 else 0
                memory_usage += int(out_dim) * int(in_dim) * bit * (arch['layer'][config['hierarchy'][linear]][blk] if 'layer' in arch else 1)
                
    return memory_usage / config['model_numel']


# def compute_2_bits(arch, config):
#     memory_usage = 0
#     for linear_group, bits in arch['linear'].items():
#         for blk, bit in enumerate(bits):
#             if bit != 2 : continue
#             for linear in linear_group.split(','):
#                 out_dim, in_dim = config['linear_shape'][linear]
#                 memory_usage += int(out_dim) * int(in_dim) * bit * (arch['layer'][config['hierarchy'][linear]][blk] if 'layer' in arch else 1)
#     return memory_usage / config['model_numel']


# def compute_2_bits_ratio(arch):
#     concat = np.concatenate(list(arch['linear'].values()))
#     int2_cnt = np.count_nonzero(concat == 2)

#     return int2_cnt / len(concat)


def compute_sparsity(arch):
    return np.concatenate([v for v in arch['layer'].values()]).mean()

def compute_params(arch, config):
    params = 0
    total_params = 0
    for layer, layer_arch in arch['layer'].items():
        for layer_mask in layer_arch:
            total_params += config['layer_numel'][layer]
            params += config['layer_numel'][layer] * layer_mask
            
    return params / total_params

def get_net_info(arch, config, group_size=-1, latency_table=None):
    net_info = {}
    net_info['bits'] = compute_bits(arch, config, group_size) if 'linear' in arch else 0
    net_info['sparsity'] = compute_sparsity(arch) if 'layer' in arch else 0
    net_info['params'] = compute_params(arch, config) if 'layer' in arch else 0
    net_info['latency'] = compute_latency(arch, config, latency_table)
    # net_info['2bits'] = compute_2_bits(arch, config) if 'linear' in arch else 0
    # net_info['2bits_ratio'] = compute_2_bits_ratio(arch) if 'linear' in arch else 0    
    
    return net_info

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

    # print(f'cur_gpu_ids : {cur_gpu_id}, blk_per_gpu : {blk_per_gpu}, device : {accelerator.device}, device_map : {device_map}')
    # print(f'device_map : {device_map}')

    return accelerator, device_map

def load_hqq_model(model_id, device_map, use_cache=False, inference=False):
    clean_up()
    # for fast model loading
    org_kaiming_uniform = torch.nn.init.kaiming_uniform_
    org_uniform = torch.nn.init.uniform_
    org_normal = torch.nn.init.normal_
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    if model_id is not None:
        model = AutoHQQHFModel.from_quantized(model_id, device_map='cpu')
        model = simple_dispatch_model(model, device_map)
        # model = dispatch_model(model, device_map)
        model.config.use_cache = use_cache
        clean_up()
        # torch.cuda.empty_cache()
        # gc.collect()
        print(f'{model_id} :  {torch.cuda.max_memory_reserved() / 1024 / 1024}MB')
        # if inference:
        #     prepare_for_inference(model, backend='gptq')
    else :
        model = None

    torch.nn.init.kaiming_uniform_ = org_kaiming_uniform
    torch.nn.init.uniform_ = org_uniform
    torch.nn.init.normal_ = org_normal

    return model

def insert_fp16_channel_hqq(linear, outlier):
    # import pdb; pdb.set_trace()
    linear.meta['outlier'] = outlier

def remove_fp16_channel_hqq(linear):
    if 'outlier' in linear.meta:
        del linear.meta['outlier']

def get_fp16_channel(linear, idx):
    # print(f'linear.weight : {linear.weight.data.device}, idx : {idx}')
    return deepcopy(linear.weight.data[:, idx])
    # return linear.weight.data[:, idx]

def get_outlier_bits(config):
    pass


def get_hfmodel(model_name_or_path: str,
                device_map='auto',
                dtype='auto',
                trust_remote_code=False,
                use_cache=False,
                **kwargs
                ):

    # assert kwargs.get('attn_implementation') in ['hf', 'ft']        ## hf : huggingface, ft : faster transformer
    
    # for fast model loading
    org_kaiming_uniform = torch.nn.init.kaiming_uniform_
    org_uniform = torch.nn.init.uniform_
    org_normal = torch.nn.init.normal_

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    # ft = False
    # if kwargs.get('attn_implementation') == 'ft':
    #     assert 'llama' in model_name_or_path.lower() or 'vicuna' in model_name_or_path.lower()
    #     ft = True
    
    # print('attention implementaion is :', kwargs.pop('attn_implementation'))
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
    
    # if ft:
    #     convert_model_to_ft(model)
    #     replace_generate_functions()

    torch.nn.init.kaiming_uniform_ = org_kaiming_uniform
    torch.nn.init.uniform_ = org_uniform
    torch.nn.init.normal_ = org_normal
    
    return model

def load_outlier(model, outlier, config):
    outlier = dict()
    for blk_idx in range(int(config['n_block'])):
        # for linear_group in config['linear']:
        #     for linear in linear_group.split(','):
        for linear in config['linear']:
            key = f'{config["layers"]}.{blk_idx}.{linear}'
            if key in outlier:
                outlier[f'{blk_idx}.{linear}'] = [outlier[key], get_fp16_channel(getsubattr(getblock(model, config)[blk_idx], linear), outlier[key])]
    return outlier

def process_dtype(dtype):
    if dtype in ['float16', 'float', 'fp16']:
        return torch.float16
    elif dtype in ['bfloat16', 'bfloat', 'bf16']:
        return torch.bfloat16
    elif dtype == 'auto':
        return 'auto'
    else:
        raise NotImplementedError(dtype)