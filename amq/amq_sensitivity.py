"""
AMQ Sensitivity Analysis Module

Linear sensitivity analysis for space pruning.
"""

import os
import time
import json
import argparse

from evaluation.evaluator import Evaluator
from utils.func import init_accelerator, set_seed
from tqdm import tqdm

def linear_sensitivity(args):
    """Run linear sensitivity analysis for space pruning."""
    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]
    
    accelerator, device_map = init_accelerator(args.gpu_id, config)
    accelerator.print("Running Linear Sensitivity Analysis")
    accelerator.print(args)

    set_seed(args.seed)
    
    evaluator = Evaluator(
        config=config,
        accelerator=accelerator,
        device_map=device_map,
        model_id=f'{args.model_path}/{args.model_name}',
        bits_range=[2, 3, 4],
        quantization_proxy_paths=args.quantization_proxy_paths,
        seqlen=args.seqlen,
        n_sample=args.n_sample,
        datasets=[args.dataset]
    )
    
    n_block = config['n_block']
    linears = config['linear']
    loss_list = {}
    
    # Initialize architecture with maximum bit-width for all layers
    architecture = {'linear': {linear_group: [4] * n_block for linear_group in linears}}

    start_time = time.time()
    
    # Measure sensitivity for each linear group in each block
    for block_idx in tqdm(range(n_block), desc="Measuring sensitivity"):
        for linear_group in linears:
            iter_start_time = time.time()
            
            key = f'{block_idx}.{linear_group}'

            architecture['linear'][linear_group][block_idx] = 2
            
            loss, _ = evaluator.eval(accelerator=accelerator, architecture=architecture)
            loss_list[key] = loss[args.dataset]
            
            iter_time = time.time() - iter_start_time
            accelerator.print(f"[{key}] Loss={loss_list[key]:.4f}, time: {iter_time:.2f}")
            
            architecture['linear'][linear_group][block_idx] = 4

    results = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "loss": loss_list,
        "time_elapsed": time.time() - start_time
    }
    
    with accelerator.main_process_first():
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        accelerator.print(f"Results saved to {args.output_file}")

    accelerator.print(f"Time Elapsed: {time.time() - start_time:.2f} seconds")
    accelerator.print("Linear sensitivity analysis completed")
    
    return results

def main():
    """Main function for sensitivity analysis"""
    parser = argparse.ArgumentParser(
        description='AMQ Linear Sensitivity Analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='id of available gpus')
    parser.add_argument('--model_path', type=str, default='meta-llama',
                        help='Path to the model directory')
    parser.add_argument('--model_name', type=str, default='Llama-2-7b-hf',
                        help='Name of the model (e.g., Llama-2-7b-hf)')
    parser.add_argument('--quantization_proxy_paths', type=str, nargs='+', default=[], 
                        help='Paths to pre-quantized models')
    
    # Data configuration
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        help='dataset')
    parser.add_argument('--seed', type=int, default=0,
                        help='test batch size for inference')
    parser.add_argument('--n_sample', type=int, default=128,
                        help='test batch size for inference')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='test batch size for inference')
    parser.add_argument('--config', type=str, default='amq/configs/llama.json',
                        help='Path to model configuration file')
    
    # Output configuration
    parser.add_argument('--save', action='store_true',
                        help='Save the output file')
    
    args = parser.parse_args()
    
    # Set default output files if not provided
    if args.save:
        print(f'Saving output file')
        args.output_file = f"amq/sensitivity/{args.model_name}_dataset_{args.dataset}_n_sample_{args.n_sample}_seqlen_{args.seqlen}.json"
    
    results = linear_sensitivity(args)
    print("Sensitivity analysis completed successfully")

    # Save results to JSON file
    if args.save:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()