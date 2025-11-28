import os
import json
import argparse
import torch
import numpy as np
from pymoo.decomposition.asf import ASF
from pymoo.visualization.scatter import Scatter
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
# from pymoo.model.decision_making import DecisionMaking, normalize, find_outliers_upper_tail, NeighborFinder
from pymoo.core.decision_making import DecisionMaking, find_outliers_upper_tail, NeighborFinder
from pymoo.util.normalization import normalize

from evaluator import LlamaEvaluator
from tqdm import tqdm
import csv
from matplotlib import pyplot as plt
from utils.func import init_accelerator, get_net_info, clean_up, set_seed, process_dtype
from utils.eval import measure_latency, eval_zeroshot
from utils.data import get_tokenizer

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

    n_comp_obj = len(args.comp_obj)
    # assert n_comp_obj == len(archive[0][2:])
    # subnets, metric, sec_obj = [v[0] for v in archive], [v[1] for v in archive], [v[2] for v in archive]
    subnets, metric = [v[0] for v in archive], [v[1] for v in archive]
    # sec_obj = [get_net_info(n, config, latency_table)[args.sec_obj] for n in subnets]
    comp_obj = [[get_net_info(n, config, args.group_size, latency_table)[obj] for n in subnets] for obj in args.comp_obj]
    # sec_objs = [[get_net_info(n, config, latency_table)[o] for n in subnets] for o in args.sec_obj]
    sort_idx = np.argsort(metric)
    F = np.column_stack((metric, *comp_obj))[sort_idx, :]
    # F = np.column_stack((metric, sec_obj))[sort_idx, :]
    n_comp_obj_min, n_comp_obj_max = len(args.comp_obj_min), len(args.comp_obj_max)
    assert n_comp_obj == n_comp_obj_min and n_comp_obj_min == n_comp_obj_max
    
    if args.only_front:
        assert 0 <= args.front_sample and args.front_sample <= 1
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        if args.front_sample < 1:
            front = np.random.choice(front, size=int(len(front) * args.front_sample), replace=False)
            front.sort()
        # sort_idx = np.argsort(F[front, 0])
        pf = F[front, :]
        ps = np.array(subnets)[sort_idx][front]
        comp_list = {c: [] for c in args.comp_obj}
        metric_list = {d: [] for d in args.datasets}
    
    elif n_comp_obj_min > 0:
        # assert args.sec_obj_range[0] >= min(args.quant_model_bits) and args.sec_obj_range[1] <= max(args.quant_model_bits), f'Target bits range should be in [small model bits, large model bits]'
        # range_idx = np.argwhere(np.logical_and(F[:, 1] > args.sec_obj_range[0], F[:, 1] < args.sec_obj_range[1])).flatten()
        flag = np.ones((F.shape[0]), dtype=bool)
        for i, obj in enumerate(args.comp_obj):
            flag = np.logical_and(flag, np.logical_and(F[:, 1+i] > args.comp_obj_min[i], F[:, 1+i] < args.comp_obj_max[i]))
        range_idx = np.argwhere(flag).flatten()
        
        pf = F[range_idx, :]
        ps = np.array(subnets)[sort_idx][range_idx]

    else:
        pf = F
        ps = np.array(subnets)[sort_idx]
        
    if args.high_tradeoff:
        
        I = NonDominatedSorting().do(pf, only_non_dominated_front=True)
        # # choose the architectures with highest trade-off
        # dm = HighTradeoffPoints(n_survive=args.n)

        # I = dm.do(np.column_stack([pf[:, 0], *[pf[:, 1+args.comp_obj.index(obj)] for obj in args.high_tradeoff]]))
        # temp = np.column_stack([pf[:, 0], *[pf[:, 1+args.comp_obj.index(obj)] for obj in args.high_tradeoff]])
        
        # temp_norm = normalize(temp, estimate_bounds_if_none=True)
        # import matplotlib.pyplot as plt
        # plt.scatter(temp[:, 1], temp[:, 0], c='b', s=5, alpha=0.8, facecolor=None, label='candidates')
        # plt.scatter(temp[I, 1], temp[I, 0], c='r', s=5, label='selected points')
        # # plt.scatter(temp_norm[:, 1], temp_norm[:, 0], c='b', s=5, alpha=0.8, facecolor=None, label='candidates')
        # # plt.scatter(temp_norm[I, 1], temp_norm[I, 0], c='r', s=5, label='selected points')
        # plt.xlabel('latency')
        # plt.ylabel('metric')
        # plt.grid()
        # plt.legend()
        # plt.savefig('test2.png')
        # exit()
        
        # I = dm.do(pf[idx])

    elif args.prefer:
        # preferences
        preferences = {}
        # for p in args.prefer.split("+"):
        for p in args.prefer:
            k, v = p.split("#")
            preferences[k] = float(v)
        weights = np.fromiter(preferences.values(), dtype=float)

        # choose the architectures thats closest to the preferences
        I = ASF().do(pf, weights).argsort()[:args.n].reshape(args.n)
    else:
        I = range(len(pf))

    # always add most accurate architectures
    # I = np.append(I, 0)

    for idx in I:
        # print(f'Selected arch[{idx}] {args.sec_obj}: {pf[idx, 1]:.4f}, metric: {pf[idx, 0]:.4f}, arch: {ps[idx]}')
        # print(f'arch : {ps[idx]}')
        print(f'Selected arch[{idx}] {args.comp_obj}: {pf[idx, 1:].tolist()}, metric: {pf[idx, 0].item():.4f}')
                
    latency_table = None
    if args.latency_table_file:
        with open(args.latency_table_file, 'r') as f:
            latency_table = json.load(f)

    model_id = f'{args.model_path}/{args.model_name}'
    awq_gptq_qeft = 'awq' in args.method or 'gptq' in args.method or 'qeft' in args.method
    
    evaluator = LlamaEvaluator(
        config=config,
        accelerator=accelerator,
        dtype=dtype,
        device_map=device_map,
        model_id=model_id,
        method=args.method,
        quant_model_bits=args.quant_model_bits,
        quant_model_paths=args.quant_model_paths,
        group_size=args.group_size,
        outlier=torch.load(args.outlier_path) if args.outlier_path else None,
        seqlen=args.seqlen,
        n_sample=args.n_sample,
        datasets=args.datasets,
        latency_table=latency_table,
        clip_asym=args.clip_asym
    )

    for idx in tqdm(I):

        arch = ps[idx]
        # arch = dict()
        # arch['linear'] = {linear: [4] * config['n_block'] for linear in config['linear']}
        accelerator.print(arch)
        
        # linear_bits = np.concatenate(list(arch['linear'].values()))
        # do_qeft = ((linear_bits - linear_bits.astype(int)).sum() != 0)
        # print(f'do_qeft : {do_qeft}, awq_gptq_qeft : {awq_gptq_qeft}')
        # if awq_gptq_qeft:
        #     method = 'awq' if 'awq' in args.method else 'gptq' if 'gptq' in args.method else 'qeft' if 'qeft' in args.method else None
        #     model = get_quantized_model(method, arch, model_id, device_map, group_size=args.group_size, config=config, prune='layer_prune' in args.method, do_owq=do_qeft, outlier_path=args.outlier_path)
        # else:
        model = evaluator.sample(arch)
        metric, complexity = evaluator.eval(arch=arch, metric='ppl', model=model, accelerator=accelerator)
        latency = measure_latency(model, generation=True, device=model.device) if args.latency else 0
        print(f'Selected arch[{idx}] {args.comp_obj}: {pf[idx, 1:]}, ppl: {[p for p in metric.values()]}, metric: {pf[idx, 0]:.4f} complexity: {complexity}, latency: {latency}')
        if args.only_front and args.save and args.results_csv_file:
            for c in args.comp_obj:
                comp_list[c].append(complexity[c])
            for d in args.datasets:
                metric_list[d].append(metric[d])

            os.makedirs(args.save, exist_ok=True)
            with open(os.path.join(args.save, args.results_csv_file), 'w') as f:
                writer = csv.writer(f)
                for c in args.comp_obj:
                    writer.writerow(comp_list[c])
                for d in args.datasets:
                    writer.writerow(metric_list[d])
        
        if args.zeroshot:
            clean_up()
            # model.use_cache = False
            model.config.use_cache = False
            
            results = eval_zeroshot(model, tokenizer=get_tokenizer(model_id), task_list=args.tasks, num_fewshot=args.num_fewshot, batch_size=args.zeroshot_batch_size)
            acc_norm = [task_result['acc_norm,none'] if 'acc_norm,none' in task_result else task_result['acc,none'] if 'acc,none' in task_result else 0 for task_result in results.values()]
            acc = [task_result['acc,none'] if 'acc,none' in task_result else 0 for task_result in results.values()]
            em_strict = [task_result['exact_match,strict-match'] if 'exact_match,strict-match' in task_result else 0 for task_result in results.values()]
            em_flexible = [task_result['exact_match,flexible-extract'] if 'exact_match,flexible-extract' in task_result else 0 for task_result in results.values()]
            em = em_strict + em_flexible
            
            task = list(results.keys())
            avg_acc_norm = np.mean(acc_norm)
            avg_acc = np.mean(acc)
            print(f'avg_acc_norm : {avg_acc_norm}, avg_acc : {avg_acc}')
            print(f'task : {task}')
            print(f'acc_norm : {acc_norm}')
            print(f'em : {em}')
            # print(F'results: {results}')
            # for task, task_result in results.items():
            #     if 'acc_norm,none' in task_result:
            #         print(f'{task} acc_norm : {task_result["acc_norm,none"]}')
            #     else:
            #         print(f'{task} acc : {task_result["acc,none"]}')
            
        if awq_gptq_qeft:
            del model, evaluator.model
            clean_up()

    print(args)
    return

    # if args.debug:
    #     # print(ps[I])
    #     # plot = Scatter()
    #     # plot.add(pf, alpha=0.2)
    #     # plot.add(pf[I, :], color="blue", s=10)
    #     # plot.add(gs_data, color="red", s=10)
    #     # plot.show()
    #     # plot.save(os.path.join(args.save, "best_trade_off_line.png"))
    #     os.makedirs(args.save, exist_ok=True)
        
    #     plt.scatter(complexity_list, [p[args.datasets[0]] for p in ppl_list], color='b', s=5, label='NSGA2')
    #     if args.greedy_search_result_path:
    #         with open(args.greedy_search_result_path, 'r') as f:
    #             gs_data = list(csv.reader(f))
    #             gs_bits = list(map(float, gs_data[1]))[:-3]
    #             gs_metric = list(map(float, gs_data[2]))[:-3]
    #             plt.scatter(gs_bits, gs_metric, color='r', s=5, label='Greedy Search')
        
    #     plt.xlabel(f'{args.sec_obj}')
    #     plt.ylabel('PPL')
    #     plt.legend()
    #     plt.show()
    #     plt.savefig(os.path.join(args.save, "best_trade_off_line.png"), dpi=300)

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