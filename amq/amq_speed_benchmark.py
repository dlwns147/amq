import os
import gc
import csv
import math
import json
import argparse
from copy import deepcopy

import torch
from torch import nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from hqq.utils.patching import prepare_for_inference
from hqq.models.hf.base import AutoHQQHFModel
from hqq.backends.autogptq import GPTQLinear
from hqq.core.quantize import HQQLinear
from kernel.monkeypatch.ftllama_modeling import convert_model_to_ft
from kernel.monkeypatch.ftllama_generate import replace_generate_functions
from utils.speed import benchmark_speed

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = None

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def cleanup():
    torch.cuda.empty_cache()
    gc.collect()


def get_named_linears(module, specific_cls=None):
    if specific_cls is None:
        return {name: m for name, m in module.named_modules() if (isinstance(m, nn.Linear) 
                                                              or isinstance(m, GPTQLinear)
                                                              or isinstance(m, HQQLinear))}
    else:
        return {name: m for name, m in module.named_modules() if isinstance(m, specific_cls)}
    

def get_hfmodel(model_name_or_path: str,
                dtype='auto',
                device_map='cpu',
                trust_remote_code=False,
                **kwargs
                ):

    assert kwargs.get('attn_implementation') in ['hf', 'ft']        ## hf : huggingface, ft : faster transformer
    
    # for fast model loading
    org_kaiming_uniform = torch.nn.init.kaiming_uniform_
    org_uniform = torch.nn.init.uniform_
    org_normal = torch.nn.init.normal_

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    ft = False
    if kwargs.get('attn_implementation') == 'ft':
        assert 'llama' in model_name_or_path.lower() or 'vicuna' in model_name_or_path.lower()
        ft = True
    
    print('attention implementaion is :', kwargs.pop('attn_implementation'))
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        torch_dtype=dtype,
        device_map=device_map, 
        trust_remote_code=trust_remote_code,
        **kwargs
    )
    
    if ft:
        convert_model_to_ft(model)
        replace_generate_functions()

    torch.nn.init.kaiming_uniform_ = org_kaiming_uniform
    torch.nn.init.uniform_ = org_uniform
    torch.nn.init.normal_ = org_normal
    
    return model


def get_memory_footprint(module: torch.nn.Module, return_buffers: bool = True) -> int:
    if not isinstance(module, torch.nn.Module):
        raise TypeError("Input must be a PyTorch Module")
    mem = sum([param.nelement() * param.element_size() for param in module.parameters()])
    if return_buffers:
        mem_bufs = sum([buf.nelement() * buf.element_size() for buf in module.buffers()])
        mem = mem + mem_bufs
    return mem


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, help='model path', default = 'meta-llama')
    parser.add_argument('--model_name', type=str, help='model name', default = 'Llama-2-7b-hf')
    parser.add_argument('--save_path', type=str, help='save path', default = '/SSD/hqq')
    parser.add_argument('--use_ft', action='store_true', help='use faster transformer')

    parser.add_argument('--batch_size', type=int, help='batch size', default = 1)
    parser.add_argument('--seq_length', type=int, help='sequence length', default = 64)
    parser.add_argument('--gen_length', type=int, help='generation length', default = 128)

    parser.add_argument('--tps', action='store_true', help='token per second')
    parser.add_argument('--gemm', action='store_true', help='gemm')
    parser.add_argument('--gemv', action='store_true', help='gemv')
    parser.add_argument('--ttft', action='store_true', help='ttft')
    parser.add_argument('--memory', action='store_true', help='memory')
    parser.add_argument('--peak_memory', action='store_true', help='peak memory & It only works with TPS')

    parser.add_argument('--target_bits', type=float, help='target bits', default = 4)
    parser.add_argument('--arch_path', type=str, help='arch path', default = None)
    parser.add_argument('--file_name', type=str, help='save path', default = None)

    args = parser.parse_args()

    global model_id
    model_id = f'{args.model_path}/{args.model_name}'
    target_bits = args.target_bits
    result = {}

    int2_model = AutoHQQHFModel.from_quantized(f'{args.save_path}/{args.model_name}_2bit_128gs_1axis')
    int3_model = AutoHQQHFModel.from_quantized(f'{args.save_path}/{args.model_name}_3bit_128gs_1axis')
    int4_model = AutoHQQHFModel.from_quantized(f'{args.save_path}/{args.model_name}_4bit_128gs_1axis')

    int2_model = int2_model.to(default_device)
    int3_model = int3_model.to(default_device)
    int4_model = int4_model.to(default_device)

    prepare_for_inference(int2_model, backend = 'gptq', load_path = f"{args.save_path}/{args.model_name}_2bit_128gs_1axis_GPTQLinear.pt")
    prepare_for_inference(int3_model, backend = 'gptq', load_path = f"{args.save_path}/{args.model_name}_3bit_128gs_1axis_GPTQLinear.pt")
    prepare_for_inference(int4_model, backend = 'ft', load_path = f"{args.save_path}/{args.model_name}_4bit_128gs_1axis_FTLinear.pt")
    # prepare_for_inference(int4_model, backend = 'gptq', load_path = f"{args.save_path}/{args.model_name}_4bit_128gs_1axis_GPTQLinear.pt")

    int2_layers = int2_model.model.layers
    int3_layers = int3_model.model.layers
    int4_layers = int4_model.model.layers

    int2_model = int2_model.to('cpu')
    int3_model = int3_model.to('cpu')
    int4_model = int4_model.to('cpu')

    cleanup()

    base_model = get_hfmodel(model_id, dtype='float16', attn_implementation='ft' if args.use_ft else 'hf')
    base_layers = base_model.model.layers

    base_model.eval()
    base_model = base_model.to('cuda')

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_id,
        use_fast=False,
        trust_remote_code=True
        )

    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    sizes = [args.batch_size, args.seq_length, args.gen_length]
    gemm_iteration = 20
    gemv_iteration = 5 if args.gen_length < 1024 else 2

    print(f"Get Speed of original model...")
    result['fp16'] = {}

    if args.tps:
        token_per_second = benchmark_speed(base_model, tokenizer, use_ft = args.use_ft, iteration = gemv_iteration, sizes = sizes, mode = 'TPS', get_peak_memory=args.peak_memory)
        result['fp16'].update(token_per_second)
        print('Token per second : ', token_per_second)

    if args.gemm:
        gemm = benchmark_speed(base_model, tokenizer, use_ft = args.use_ft, iteration = gemm_iteration, sizes = sizes, mode = 'GeMM', get_peak_memory=False)
        result['fp16'].update(gemm)
        print('GeMM : ', gemm)

    if args.gemv:
        gemv = benchmark_speed(base_model, tokenizer, use_ft = args.use_ft, iteration = gemv_iteration, sizes = sizes, mode = 'GeMV', get_peak_memory=False)
        result['fp16'].update(gemv)
        print('GeMV : ', gemv)

    if args.ttft:
        ttft = benchmark_speed(base_model, tokenizer, use_ft = args.use_ft, iteration = gemm_iteration, sizes = sizes, mode = 'TTFT', get_peak_memory=False)
        result['fp16'].update(ttft)
        print('TTFT : ', ttft)

    if args.memory:
        memory = get_memory_footprint(base_model) / 1024 ** 3
        result['fp16'].update({'memory' : memory})
        print(f"Base Model Memory : {memory} GB")

    if args.file_name:
        result_dir = f'benchmark/outputs'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, args.file_name)        
                       
    base_model = base_model.to('cpu')
    base_layers_length = len(base_layers)
    linears = list(get_named_linears(base_layers[0]).keys())

    if args.arch_path is not None:
        if os.path.exists(args.arch_path):
            with open(args.arch_path, 'r') as f:
                archs = json.load(f)
                archs = archs['archive'] + archs['candidates']
        else:
            raise FileNotFoundError(f"Arch file {args.arch_path} not found")

        candidates = []
        for arch in archs:
            if abs(arch[-1] - target_bits) < 0.05:
                candidates.append(arch)
        
        candidates_bits = [np.concatenate([bit for bit in arch[0]['linear'].values()]) for arch in candidates]
        count_4bit = [(bits == 4.0).sum() for bits in candidates_bits]
        candidate = candidates[np.argmax(count_4bit)]
        
        arch = candidate[0]['linear']
    else:
        assert target_bits in [2, 3, 4], "target bits should be 2, 3, 4 if arch_path is not provided"
        arch = {linear : [target_bits] * base_layers_length for linear in linears}

    model = deepcopy(base_model)

    print("Replacing...")
    for layer_idx, layer in enumerate(base_layers):
        named_linears = get_named_linears(layer)
        for name in named_linears:
            module, linear = name.split('.')

            if math.isclose(arch[name][layer_idx], 2):
                source = getattr(getattr(int2_layers[layer_idx], module), linear)
            elif math.isclose(arch[name][layer_idx], 3):
                source = getattr(getattr(int3_layers[layer_idx], module), linear)
            elif math.isclose(arch[name][layer_idx], 4):
                source = getattr(getattr(int4_layers[layer_idx], module), linear)
            else:
                raise ValueError(f'bit should be 2, 3, 4, but got {arch[name][layer_idx]}')

            if hasattr(getattr(model.model.layers[layer_idx], module), linear):
                delattr(getattr(model.model.layers[layer_idx], module), linear)

            setattr(getattr(model.model.layers[layer_idx], module), linear, source)

    cleanup()

    model.eval()
    model = model.to('cuda')

    print(f"Get Speed of {target_bits}bit model...")
    result[f'{target_bits}bit'] = {}

    if args.tps:
        token_per_second = benchmark_speed(model, tokenizer, use_ft = args.use_ft, iteration = gemv_iteration, sizes = sizes, mode = 'TPS', get_peak_memory=args.peak_memory)
        result[f'{target_bits}bit'].update(token_per_second)
        print('Token per second : ', token_per_second)

    if args.gemm:
        gemm = benchmark_speed(model, tokenizer, use_ft = args.use_ft, iteration = gemm_iteration, sizes = sizes, mode = 'GeMM', get_peak_memory=False)
        result[f'{target_bits}bit'].update(gemm)
        print('GeMM : ', gemm)
    
    if args.gemv:
        gemv = benchmark_speed(model, tokenizer, use_ft = args.use_ft, iteration = gemv_iteration, sizes = sizes, mode = 'GeMV', get_peak_memory=False)
        result[f'{target_bits}bit'].update(gemv)
        print('GeMV : ', gemv)

    if args.ttft:
        ttft = benchmark_speed(model, tokenizer, use_ft = args.use_ft, iteration = gemm_iteration, sizes = sizes, mode = 'TTFT', get_peak_memory=False)
        result[f'{target_bits}bit'].update(ttft)
        print('TTFT : ', ttft)

    if args.memory:
        memory = get_memory_footprint(model) / 1024 ** 3
        result[f'{target_bits}bit'].update({'memory' : memory})
        print(f"Quantized Model Memory : {memory} GB")

    model = model.cpu()
    del model
    cleanup()

    if args.file_name:
        with open(result_path, 'w') as f:
            result.update({'args' : vars(args)})
            json.dump(result, f, indent=4)


if __name__ == '__main__':
    main()