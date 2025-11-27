import argparse
import time
from tqdm import tqdm
import csv
import os

import numpy as np
import torch
import gc
import json
import csv
import time
from accelerate import Accelerator

from amq.evaluation.evaluator import LlamaEvaluator
from amq.utils.func import init_accelerator
from amq.utils.eval import load_and_eval_ppl, eval_zeroshot
from transformers import AutoModelForCausalLM




def linear_sensitivity(args):

    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]
    accelerator, device_map = init_accelerator(args.gpu_id, config)
    accelerator.print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    evaluator = LlamaEvaluator(
        config=config,
        accelerator=accelerator,
        device_map=device_map,
        model_id=f'{args.model_path}/{args.model_name}',
        method=args.method,
        quant_model_bits=args.quant_model_bits,
        quant_model_paths=args.quant_model_paths,
        outlier=torch.load(args.outlier_path) if args.outlier_path else None,
        seqlen=args.seqlen,
        n_sample=args.n_sample,
        loss_func=args.loss_func,
        datasets=[args.dataset]
    )
    
    n_block = config['n_block']

    ppl = 0
    loss_list = dict()
    ppl_list = dict()
    # arch = {'linear': {l: [max(args.quant_model_bits)] * n_block for lg in config['linear'] for l in lg.split(',')}, 'layer': {l: [1]* n_block for l in config['layer']}}
    arch = {'linear': {l: [max(args.quant_model_bits)] * n_block for lg in config['linear'] for l in lg.split(',')}}
    
    for linear_group in config['linear']:
        for block_idx in range(n_block):
            ppl_list[f'{block_idx}.{linear_group}'] = 0
    # accelerator.print(f'arch : {arch}')

    # ppl, complexity = evaluator.eval(arch=arch, accelerator=accelerator, metric='ppl')
    # print(f"ppl : {ppl}, complexity[bits] : {complexity['bits']}")
    # exit()

    # for linear in config['linear']:
    #     for block_idx in range(n_block):
    #         arch['linear'][linear][block_idx] = min(args.quant_model_bits)

    # ppl, complexity = evaluator.eval(arch=arch, accelerator=accelerator, metric='ppl')
    # print(f"ppl : {ppl}, complexity[bits] : {complexity['bits']}")

    # exit()

    # greedy_result_path = '/NAS/SJ/nsgaquant/csv/greedy_search/Llama-2-7b-hf_hqq_24bits_loss_desc_1axis_64_128gs_jsd.csv'
    # # last_linear = '18.self_attn.o_proj' # 7b 3.5
    # last_linear = '22.self_attn.v_proj' # 7b 3.0
    # # last_linear = '17.mlp.up_proj' # 7b 2.5
    
    # # greedy_result_path = '/NAS/SJ/nsgaquant/csv/greedy_search/Llama-2-13b-hf_hqq_loss_desc_1axis_64_128gs_jsd.csv'
    # # # last_linear = '' # 13b 3.5
    # # last_linear = '30.mlp.down_proj' # 13b 3.0
    # # # last_linear = '' # 13b 2.5
    
    # print(f'last_linear : {last_linear}, greedy_result_path : {greedy_result_path}')
    # with open(greedy_result_path, 'r') as f:
    #     selected_linears = list(csv.reader(f))[0]
    #     selected_linears = selected_linears[:selected_linears.index(last_linear) + 1]
    # for linear in selected_linears:
    #     blk_idx, layer, linear = linear.split('.')
    #     blk_idx = int(blk_idx)

    #     arch['linear'][f'{layer}.{linear}'][blk_idx] = 2

    # evaluator = LlamaEvaluator(
    #     config=config,
    #     accelerator=accelerator,
    #     device_map=device_map,
    #     model_id=f'{args.model_path}/{args.model_name}',
    #     method=args.method,
    #     quant_model_bits=args.quant_model_bits,
    #     quant_model_paths=args.quant_model_paths,
    #     outlier=torch.load(args.outlier_path) if args.outlier_path else None,
    #     seqlen=args.seqlen,
    #     n_sample=args.n_sample,
    #     loss_func=args.loss_func,
    #     datasets=['wikitext2', 'c4']
    # )

    # ppl, complexity = evaluator.eval(accelerator=accelerator, arch=arch, metric='ppl', loss_func=args.loss_func)
    # print(f'ppl :{ppl}, bits : {complexity["bits"]}')

    # model = evaluator.sample(arch)
    # del evaluator
    # gc.collect()
    # torch.cuda.empty_cache()
    # print(f'memory : {torch.cuda.memory_allocated()}')
    
    # from transformers import AutoTokenizer
    # model_id = f'{args.model_path}/{args.model_name}'
    # tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

    # results = eval_zeroshot(model, tokenizer,batch_size=64)
    # avg_acc = np.mean([task_result['acc_norm,none'] if 'acc_norm,none' in task_result else task_result['acc,none'] for task_result in results.values()])
    # print(f'avg_acc : {avg_acc}, results : {results}')
    # for task, task_result in results.items():
    #     if 'acc_norm,none' in task_result:
    #         print(f'{task} acc_norm : {task_result["acc_norm,none"]}')
    #     else:
    #         print(f'{task} acc : {task_result["acc,none"]}')
    # exit()

    

    start_point = time.time()
    
    for block_idx in range(n_block):
        for linear_group in config['linear']:
            iter_start = time.time()
            
            for linear in linear_group.split(','):
                arch['linear'][linear][block_idx] = min(args.quant_model_bits)

            key = f'{block_idx}.{linear_group}'
            loss, _ = evaluator.eval(accelerator=accelerator, arch=arch, metric='loss', loss_func=args.loss_func)
            loss_list[key] = loss[args.dataset]
            if args.eval_ppl:
                ppl, _ = evaluator.eval(accelerator=accelerator, arch=arch, metric='ppl', loss_func=args.loss_func)
                ppl_list[key] = ppl[args.dataset]
            iter_time = time.time() - iter_start
            accelerator.print(f"[{key} replaced] Loss={loss_list[key]:.4f}, PPL: {ppl_list[key]:.2f}, time: {iter_time:.2f}")
            
            for linear in linear_group.split(','):
                arch['linear'][linear][block_idx] = max(args.quant_model_bits)

    with accelerator.main_process_first():
        if args.loss_csv_file:
            with open(args.loss_csv_file, 'w', newline='') as f:
                write = csv.writer(f)
                write.writerow(list(loss_list.keys()))
                write.writerow(list(loss_list.values()))
        if args.eval_ppl and args.ppl_csv_file:
            with open(args.ppl_csv_file, 'w', newline='') as f:
                write = csv.writer(f)
                write.writerow(list(ppl_list.keys()))
                write.writerow(list(ppl_list.values()))

    finish_point = time.time()
    time_elapsed = finish_point - start_point

    accelerator.print(f"Time_Elapsed: {time_elapsed}")
    accelerator.print(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='id of available gpus')
    parser.add_argument('--model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--quant_model_bits', type=float, nargs='+', default=[], 
                        help='')
    parser.add_argument('--quant_model_paths', type=str, nargs='+', default=[], 
                        help='')
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        help='dataset')
    parser.add_argument('--seed', type=int, default=0,
                        help='test batch size for inference')
    parser.add_argument('--n_sample', type=int, default=128,
                        help='test batch size for inference')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='test batch size for inference')
    parser.add_argument('--config', type=str, default='config/llama.json',
                        help='')
    parser.add_argument('--loss_csv_file', type=str, default='',
                        help='')
    parser.add_argument('--ppl_csv_file', type=str, default='',
                        help='')
    parser.add_argument('--method', type=str, nargs='+', default=[],
                        help='')
    parser.add_argument('--eval_ppl', action='store_true')
    parser.add_argument('--loss_func', type=str, default='cross_entropy', help='')
    parser.add_argument('--outlier_bits', type=float, nargs='+', default=[], 
                        help='')
    parser.add_argument('--outlier_path', type=str, default='',
                        help='')

    args = parser.parse_args()
    linear_sensitivity(args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='id of available gpus')
    parser.add_argument('--model_path', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--model_name', type=str, default='',
                        help='file path to supernet weights')
    parser.add_argument('--quant_model_bits', type=float, nargs='+', default=[], 
                        help='')
    parser.add_argument('--quant_model_paths', type=str, nargs='+', default=[], 
                        help='')
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        help='dataset')
    parser.add_argument('--seed', type=int, default=0,
                        help='test batch size for inference')
    parser.add_argument('--n_sample', type=int, default=128,
                        help='test batch size for inference')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='test batch size for inference')
    parser.add_argument('--config', type=str, default='config/llama.json',
                        help='')
    parser.add_argument('--loss_csv_file', type=str, default='',
                        help='')
    parser.add_argument('--ppl_csv_file', type=str, default='',
                        help='')
    parser.add_argument('--method', type=str, nargs='+', default=[],
                        help='')
    parser.add_argument('--eval_ppl', action='store_true')
    parser.add_argument('--loss_func', type=str, default='cross_entropy', help='')
    parser.add_argument('--outlier_bits', type=float, nargs='+', default=[], 
                        help='')
    parser.add_argument('--outlier_path', type=str, default='',
                        help='')

    args = parser.parse_args()
    linear_sensitivity(args)

