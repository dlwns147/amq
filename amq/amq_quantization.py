import os
import csv
import json
from tqdm import tqdm

import torch
import numpy as np

from pymoo.decomposition.asf import ASF
from pymoo.core.decision_making import DecisionMaking, find_outliers_upper_tail, NeighborFinder
from pymoo.util.normalization import normalize

from evaluation.evaluator import Evaluator
from utils.args import parse_args
from utils.func import init_accelerator, clean_up, set_seed
# from utils.eval import measure_latency, eval_zeroshot
# from utils.data import get_tokenizer

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
            # F = normalize(F, self.ideal_point, self.nadir_point, estimate_bounds_if_none=True)

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
    
    set_seed(args.seed)

    accelerator, device_map = init_accelerator(args.gpu_id, config)
    accelerator.print("=== Running Mixed-Precision Quantization ===")
    accelerator.print(args)
    
    with open(args.load, 'r') as f:
        result_json = json.load(f)
        archive = result_json['archive'] + result_json['candidates']

    architecture_list, metric_list, bit_usage_list = [v[0] for v in archive], list(map(float, [v[1] for v in archive])), list(map(float, [v[2] for v in archive]))
    sort_idx = np.argsort(metric_list)
    metric_bits_stack = np.column_stack((metric_list, bit_usage_list))[sort_idx, :]
    bit_usage_min, bit_usage_max = 2 + (32 / args.group_size), 4 + (32 / args.group_size)
    import code; code.interact('amq_quantization.py line 85', local=dict(globals(), **locals()))

    flag = np.ones((metric_bits_stack.shape[0]), dtype=bool)
    flag = np.logical_and(flag, metric_bits_stack[:, 1] > args.target_bits - args.target_bits_offset,
                         metric_bits_stack[:, 1] < args.target_bits + args.target_bits_offset)
    range_idx = np.argwhere(flag).flatten()
    
    filtered_metric_bits_stack = metric_bits_stack[range_idx, :]
    filtered_architecture_list = np.array(architecture_list)[sort_idx][range_idx]
        
    # choose the architectures thats closest to the preferences
    weights = np.array([0, args.target_bits], dtype=float)
    I = ASF().do(filtered_metric_bits_stack, weights).argsort()[:args.num_of_candidates].reshape(args.num_of_candidates)

    for idx in I:
        print(f'Selected arch[{idx}] \
            bit-usage: {filtered_metric_bits_stack[idx, 1].item():.4f}, \
            loss: {filtered_metric_bits_stack[idx, 0].item():.4f}')

    model_id = f'{args.model_path}/{args.model_name}'
    assert args.method in ['awq', 'gptq', 'owq'], f'Invalid method: {args.method}'
    
    evaluator = Evaluator(
        config=config,
        accelerator=accelerator,
        model_id=model_id,
        group_size=args.group_size,
        seqlen=args.seqlen,
        datasets=[args.dataset],
        device_map=device_map,
    )

    for idx in tqdm(I, desc='Quantizing architectures & Evaluating'):
        arch = filtered_architecture_list[idx]
        accelerator.print(arch)

        evaluator.sample(arch, method=args.method)
        ppl, bits_usage = evaluator.eval(accelerator=accelerator, architecture=arch)

        print(f'Selected arch[{idx}] \n \
            ppl: {[p for p in ppl.values()]}, \n \
            loss: {filtered_metric_bits_stack[idx, 0]:.4f} \n \
            bits_usage: {bits_usage:.4f} \n')


        # TODO: 추후 구현
        # if args.zeroshot:
        #     clean_up()
        #     evaluator.model.config.use_cache = False
            
        #     results = eval_zeroshot(model, tokenizer=get_tokenizer(model_id), task_list=args.tasks, num_fewshot=args.num_fewshot, batch_size=args.zeroshot_batch_size)
        #     acc_norm = [task_result['acc_norm,none'] if 'acc_norm,none' in task_result else task_result['acc,none'] if 'acc,none' in task_result else 0 for task_result in results.values()]
        #     acc = [task_result['acc,none'] if 'acc,none' in task_result else 0 for task_result in results.values()]
        #     em_strict = [task_result['exact_match,strict-match'] if 'exact_match,strict-match' in task_result else 0 for task_result in results.values()]
        #     em_flexible = [task_result['exact_match,flexible-extract'] if 'exact_match,flexible-extract' in task_result else 0 for task_result in results.values()]
        #     em = em_strict + em_flexible
            
        #     task = list(results.keys())
        #     avg_acc_norm = np.mean(acc_norm)
        #     avg_acc = np.mean(acc)
        #     print(f'avg_acc_norm : {avg_acc_norm}, avg_acc : {avg_acc}')
        #     print(f'task : {task}')
        #     print(f'acc_norm : {acc_norm}')
        #     print(f'em : {em}')
        #     # print(F'results: {results}')
        #     # for task, task_result in results.items():
        #     #     if 'acc_norm,none' in task_result:
        #     #         print(f'{task} acc_norm : {task_result["acc_norm,none"]}')
        #     #     else:
        #     #         print(f'{task} acc : {task_result["acc,none"]}')

        os.makedirs(args.save, exist_ok=True)
        with open(os.path.join(args.save, args.results_csv_file), 'w') as f:
            writer = csv.writer(f)
            for c in args.comp_obj:
                writer.writerow(comp_list[c])
            for d in args.datasets:
                writer.writerow(metric_list[d])

        clean_up()

    print(args)
    return

    sentences = []
    for k, v in vars(args).items():
        sentences.append(f"{k}: {v}\n")
    sentences.append("\n")
    for a, c, p in zip(arch_list, complexity_list, ppl_list):
        sentences.append(f"arch: {a}, bits: {c:.4f}, ppl: {p}\n")

    with open(os.path.join(args.save, args.results_file), 'w') as f:
        for sentence in sentences:
            f.write(sentence)

    with open(os.path.join(args.save, args.results_csv_file), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['arch', 'bits', 'params', 'sparsity', 'metric', 'latency'] + args.datasets)
        for a, b, p, s, m, l, ppl in zip(arch_list, bits_list, param_list, sparsity_list, metric_list, latency_list, ppl_list):
            writer.writerow([a, b, p, s, m, l] + list(ppl.values()))

    with open(os.path.join(args.save, args.results_arch_file), 'w') as f:
        json.dump({'archive': [[a, c, p] for a, c, p in zip(arch_list, complexity_list, ppl_list)]}, f, ensure_ascii=False, indent=4)

    return



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