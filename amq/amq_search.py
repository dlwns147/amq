#!/usr/bin/env python3
"""
AMQ: Automated Mixed-precision Quantization

Main entry point for AMQ operations:
1. Space Pruning - Reduce search space using sensitivity analysis
2. Quantization Proxy - Use quantization proxies for efficient evaluation
3. Quality Predictor - Predict model quality with surrogate models
4. Iterative Search and Update - Refine configurations through multi-objective search

Usage:
    python -m amq search --model_name Llama-2-7b-hf --config configs/llama.json ...
    python -m amq sensitivity --model_name Llama-2-7b-hf --config configs/llama.json ...
    python -m amq evaluate --expr path/to/results ...
"""

import os
import json

from utils.args import parse_args
from utils.func import init_accelerator, set_seed

def run_search(args):
    """
    Steps 2-4: Quantization Proxy + Quality Predictor + Iterative Search
    
    - Use quantized models as proxies for efficient evaluation
    - Build surrogate model to predict quality metrics
    - Perform multi-objective search using NSGA-II/III
    """
    from search.optimizer import Search
    
    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]
    
    accelerator, device_map = init_accelerator(args.gpu_id, config)
    accelerator.print("=== Running Mixed-Precision Search ===")
    accelerator.print(args)
    
    set_seed(args.seed)
    
    # Initialize search engine
    engine = Search(
        args=args,
        config=config,
        accelerator=accelerator,
        device_map=device_map,
    )
    
    # Run search
    results = engine.search(accelerator)
    
    accelerator.print(f"Search completed! Results saved to {args.save}")
    
    return results

def main():
    """Main entry point for AMQ"""

    args = parse_args(mode='search')

    # sensitivity format is linear name: sensitivity
    args.sensitivity_path = f"amq/sensitivity/{args.model_name}_dataset_{args.dataset}_n_sample_{args.n_sample}_seqlen_{args.seqlen}.json"
    if os.path.exists(args.sensitivity_path):
        with open(args.sensitivity_path, 'r') as f:
            args.sensitivity_json = json.load(f)
    else:
        raise ValueError(f"Sensitivity analysis not found. Please run sensitivity analysis first.")

    run_search(args)

if __name__ == '__main__':
    main()