import os
import json
from tqdm import tqdm

import numpy as np

from pymoo.decomposition.asf import ASF
from pymoo.core.decision_making import DecisionMaking, find_outliers_upper_tail, NeighborFinder
from pymoo.util.normalization import normalize

from evaluation.evaluator import Evaluator
from utils.args import parse_args
from utils.func import init_accelerator, clean_up, set_seed

class HighTradeoffPoints(DecisionMaking):

    def __init__(self, epsilon=0.125, n_survive=None, normalize=True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.n_survive = n_survive  # number of points to be selected
        self.normalize = normalize

    def _do(self, F, **kwargs):
        n, m = F.shape

        if self.normalize:
            F = normalize(F, estimate_bounds_if_none=True)

        neighbors_finder = NeighborFinder(F, epsilon=0.125, n_min_neigbors="auto", consider_2d=False)

        mu = np.full(n, - np.inf)

        # for each solution in the set calculate the least amount of improvement per unit deterioration
        for i in range(n):

            # for each neighbour in a specific radius of that solution
            neighbors = neighbors_finder.find(i)

            # calculate the trade-off to all neighbours
            diff = F[neighbors] - F[i]

            # calculate sacrifice and gain
            sacrifice = np.maximum(0, diff).sum(axis=1)
            gain = np.maximum(0, -diff).sum(axis=1)

            # np.warnings.filterwarnings('ignore')
            tradeoff = sacrifice / gain

            # otherwise find the one with the smalled one
            mu[i] = np.nanmin(tradeoff)
        if self.n_survive is not None:
            return np.argsort(mu)[-self.n_survive:]
        else:
            return find_outliers_upper_tail(mu)  # return points with trade-off > 2*sigma


def run_quantization(args, config):
    """    
    - Use results from search to select candidate architectures
    - Quantize the candidate architectures on target bit-width
    - Evaluate the quantized architectures
    """
    
    set_seed(args.eval_seed)

    accelerator, device_map = init_accelerator(args.gpu_id, config)
    accelerator.print("=== Running Mixed-Precision Quantization ===")
    accelerator.print(args)
    
    with open(args.load, 'r') as f:
        result_json = json.load(f)  
        archive = result_json['archive'] + result_json['candidates']

    architecture_list, metric_list, bit_usage_list = [v[0] for v in archive], list(map(float, [v[1] for v in archive])), list(map(float, [v[2] for v in archive]))
    if args.method == 'owq':
        bit_usage_list = [bit_usage + 0.1 for bit_usage in bit_usage_list]
    sort_idx = np.argsort(metric_list)
    metric_bits_stack = np.column_stack((metric_list, bit_usage_list))[sort_idx, :]

    flag = np.ones((metric_bits_stack.shape[0]), dtype=bool)
    flag = np.logical_and(flag, np.logical_and(metric_bits_stack[:, 1] > (args.target_bits - args.target_bits_offset),
                         metric_bits_stack[:, 1] < (args.target_bits + args.target_bits_offset)))
    range_idx = np.argwhere(flag).flatten()
    
    filtered_metric_bits_stack = metric_bits_stack[range_idx, :]
    filtered_architecture_list = np.array(architecture_list)[sort_idx][range_idx]
        
    # choose the architectures thats closest to the preferences
    weights = np.array([0, args.target_bits], dtype=float)
    I = ASF().do(filtered_metric_bits_stack, weights).argsort()[:args.num_of_candidates].reshape(args.num_of_candidates)

    for idx in I:
        print(f'Selected arch[{idx}], bit-usage: {filtered_metric_bits_stack[idx, 1].item():.4f}, loss: {filtered_metric_bits_stack[idx, 0].item():.4f}')

    model_id = f'{args.model_path}/{args.model_name}'
    assert args.method in ['fp16', 'awq', 'gptq', 'owq'], f'Invalid method: {args.method}'
    
    evaluator = Evaluator(
        config=config,
        accelerator=accelerator,
        model_id=model_id,
        group_size=args.group_size,
        seqlen=args.eval_seqlen,
        datasets=args.eval_dataset,
        device_map=device_map,
        search=False,
    )

    results = []

    for idx in tqdm(I, desc='Quantizing architectures & Evaluating'):
        result = {}
        arch = filtered_architecture_list[idx]

        result['arch'] = arch
        result['method'] = args.method

        accelerator.print(arch)

        if args.method != 'fp16':
            evaluator.sample(arch, method=args.method)
        ppl, bits_usage = evaluator.eval(accelerator=accelerator, architecture=arch)

        result['ppl'] = ppl
        result['loss'] = filtered_metric_bits_stack[idx, 0]
        result['bits_usage'] = bits_usage

        print(f'Selected arch[{idx}] \n \
            ppl: {ppl}, \n \
            loss: {filtered_metric_bits_stack[idx, 0]:.4f} \n \
            bits_usage: {bits_usage:.4f} \n')
            
        results.append(result)

    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, f'{args.method}_results.json'), 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    clean_up()

    print(args)


def main():
    """Main entry point for AMQ"""

    args = parse_args(mode='quantization')
    
    if not os.path.exists(args.load):
        raise ValueError(f"Search results not found. Please run search first.")
    
    with open(args.config, 'r') as f:
        config = json.load(f)[args.model_name]

    run_quantization(args, config)

    return

if __name__ == '__main__':
    main()