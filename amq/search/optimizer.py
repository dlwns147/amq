import os
import json
import numpy as np
from scipy import stats
from tqdm import tqdm
from time import time
from matplotlib import pyplot as plt

from pymoo.optimize import minimize
from pymoo.indicators.hv import Hypervolume
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.crossover.binx import BinomialCrossover
from pymoo.algorithms.moo.nsga2 import NSGA2

from evaluation.evaluator import Evaluator
from search.space import SearchSpace
from search.problem import AuxiliarySingleLevelProblem, SubsetProblem
from predictor.factory import get_predictor
from utils.ga import MySampling, BinaryCrossover, MyMutation, IntMutation

class Search:
    def __init__(self, args, config, accelerator, device_map):
        self.args = args
        self.quantization_proxy_paths = args.quantization_proxy_paths
        self.resume_path = args.resume_path
        self.iterations = args.iterations
        self.n_doe = args.n_doe
        self.n_iter = args.n_iter
        self.save_iter = args.save_iter
        self.crossover_prob = args.crossover_prob
        self.mut_prob = args.mut_prob
        self.ga_pop_size = args.ga_pop_size
        self.subset_pop_size = args.subset_pop_size
        self.predictor = args.predictor
        self.save_path = args.save_path
        self.result_file = args.result_file
        self.max_value = args.max_value
        self.dataset = args.dataset
        self.seqlen = args.seqlen
        self.n_sample = args.n_sample

        self.group_size = 128 # default group size
        self.config = config
        self.device_map = device_map

        model_id = f'{args.model_path}/{args.model_name}'
        
        self.sensitivity_json = args.sensitivity_json
        self.sensitivity_threshold = args.sensitivity_threshold

        linear_list = list(self.sensitivity_json['loss'].keys())
        medium = np.median(list(map(float, self.sensitivity_json['loss'].values())))
        pass_linear_list = [linear for linear in linear_list if self.sensitivity_json['loss'][linear] > medium * self.sensitivity_threshold]
        
        self.evaluator = Evaluator(
            config=self.config,
            accelerator=accelerator,
            model_id=model_id,
            quantization_proxy_paths=self.quantization_proxy_paths,
            bits_range=[2, 3, 4],
            group_size=self.group_size,
            seqlen=self.seqlen,
            n_sample=self.n_sample,
            datasets=[self.dataset],
            device_map=device_map,
        )
        # self.evaluator = None

        self.search_space = SearchSpace(
            config=self.config,
            n_block=self.config['n_block'],
            n_linear=len(self.config['linear']),
            group_size=self.group_size,
            pass_linear_list=pass_linear_list,
            bits_range=[2, 3, 4],
        )

        accelerator.wait_for_everyone()
        
    def search(self, accelerator):
        total_start = time()
        start_it = 1
        
        if self.resume_path:
            archive, start_it = self._resume_from_dir()
        else:
            archive = []

            # Design Of Experiment
            if accelerator.is_main_process:
                if self.iterations < 1:
                    # Generate random architectures
                    architectures_doe = self.search_space.sample(
                        n_samples=self.n_doe,
                        pool=[x[0] for x in archive])
                else:
                    # Generate architectures for design of experiment
                    architectures_doe = self.search_space.initialize(self.n_doe, pool=[x[0] for x in archive])
            else:
                architectures_doe = list()
            architectures_doe = accelerator.gather_for_metrics(architectures_doe, use_gather_object=True)
            accelerator.wait_for_everyone()

            # Parallel evaluation of arch_doe
            metric_list, bits_usage_list = self._evaluate(architectures=architectures_doe, accelerator=accelerator)

            if accelerator.is_main_process:
                for architecture, metric, bits_usage in zip(architectures_doe, metric_list, bits_usage_list):
                    archive.append((architecture, metric, bits_usage))

        if accelerator.is_main_process:
            # reference point (nadir point) for calculating hypervolume
            if archive:
                ref_pt = np.array([np.max([x[1] for x in archive]), np.max([x[2] for x in archive])])
            else:
                ref_pt = np.array([np.max(metric_list), np.max(bits_usage_list)])
            accelerator.print(f'data preparation time : {time() - total_start:.2f}s')
        accelerator.wait_for_everyone()

        # main loop of the search
        for it in range(start_it, self.iterations + 1):
            if accelerator.is_main_process:
                accelerator.print(self.args)
                iter_start = time()

                # construct accuracy predictor surrogate model from archive
                predictor_start = time()
                quality_predictor, archive_pred = self._fit_predictor(archive, device=accelerator.device)
                predictor_time = time() - predictor_start

                # search for the next set of candidates for high-fidelity evaluation (lower level)
                next_start = time()
                candidates, candidate_pred = self._next(archive, quality_predictor, self.n_iter)
                next_time = time() - next_start
            else:
                candidates = list()
            accelerator.wait_for_everyone()
            candidates = accelerator.gather_for_metrics(candidates, use_gather_object=True)

            # high-fidelity evaluation (lower level)
            candidate_metric_list, candidate_bits_usage_list = self._evaluate(architectures=candidates, accelerator=accelerator) 

            if accelerator.is_main_process:
                # check for accuracy predictor's performance
                rmse, rho, tau = self._get_correlation(
                    np.vstack((archive_pred, candidate_pred)), np.array([x[1] for x in archive] + candidate_metric_list))

                # add to archive
                for candidate, metric, bits_usage in zip(candidates, candidate_metric_list, candidate_bits_usage_list):
                    archive.append((candidate, metric, bits_usage))

                # calculate hypervolume
                hv = self._calc_hv(
                    ref_pt, np.column_stack(([x[1] for x in archive], [x[2] for x in archive])))

                iter_time = time() - iter_start
                # print iteration-wise statistics
                accelerator.print(f"Iter {it}: hv = {hv:.2f}, iter time : {(time() - iter_start):.2f}s, predictor_time : {predictor_time:.2f}, next_time : {next_time:.2f}")
                accelerator.print(f"fitting {self.predictor}: RMSE = {rmse:.4f}, Spearman's Rho = {rho:.4f}, Kendall’s Tau = {tau:.4f}")
                accelerator.print(f'iteration time : {iter_time:.2f}s')

                # dump the statistics
                if it % self.save_iter == 0:
                    os.makedirs(self.save_path, exist_ok=True)
                    with open(os.path.join(self.save_path, "iter_{}.stats".format(it)), "w") as handle:
                        json.dump({'archive': archive, 'candidates': archive[-self.n_iter:], 'hv': hv,
                                'surrogate': {
                                    'model': self.predictor, 'name': quality_predictor.name,
                                    'winner': quality_predictor.winner if self.predictor == 'as' else quality_predictor.name,
                                    'rmse': rmse, 'rho': rho, 'tau': tau, 'total_time': iter_time}, 'iteration' : it}, handle)

                        cand_bits_np = np.array(candidate_bits_usage_list)
                        fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
                        bits_usage = np.array([x[2] for x in archive])  # bits usage
                        metric = np.array([x[1] for x in archive])  # performance metric
                        axe.scatter(bits_usage, metric, s=5, facecolors='none', edgecolors='b', label='archive')
                        cand_perf = np.array(candidate_metric_list)
                        axe.scatter(cand_bits_np, cand_perf, s=10, color='r', label='candidates evaluated')
                        cand_pred_perf = candidate_pred[:, 0]
                        axe.scatter(cand_bits_np, cand_pred_perf, s=10, facecolors='none', edgecolors='g', label='candidates predicted')
                        axe.legend()
                        axe.grid(c='0.8') 
                        axe.set_xlabel('f1')
                        axe.set_ylabel('f2')
                        fig.tight_layout() 
                        plt.savefig(os.path.join(self.save_path, 'iter_{}.png'.format(it)))
            accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            total_time_elapsed = time() - total_start
            accelerator.print(f'total time elapsed : {total_time_elapsed:.2f}s')

            sentences = []
            for k, v in self.args.__dict__.items():
                sentences.append(f"{k}: {v}\n")
            sentences.append(f'Total time: {total_time_elapsed:.2f}s')

            with open(os.path.join(self.save_path, self.result_file), 'w') as f:
                for sentence in sentences:
                    f.write(sentence)

            accelerator.print(self.args)
        return

    def _resume_from_dir(self):
        """ resume search from a previous iteration """

        with open(self.resume_path, 'r') as f:
            resume_file = json.load(f)
            archive = resume_file['archive'] + resume_file['candidates']
            it = resume_file['iteration']

        return archive, it + 1

    def _evaluate(self, architectures, accelerator):
        metric_list, bits_usage_list = [], []
        for architecture in tqdm(architectures, desc='Eval Arch'):
            metric, bits_usage = self.evaluator.eval(accelerator=accelerator, architecture=architecture)
            metric_list.append(min(self.max_value, np.nan_to_num(metric[self.dataset], nan=self.max_value)))
            bits_usage_list.append(bits_usage)

        return metric_list, bits_usage_list

    def _fit_predictor(self, archive, device='cpu'):
        inputs = np.array([self.search_space.encode_predictor(x[0]) for x in archive])
        targets = np.array([x[1] for x in archive])

        kwargs = {}
        if self.predictor == 'rbf':
            n_block = self.config['n_block']
            n_linear = self.config['n_linear']
            lb = np.zeros((n_linear, n_block))
            ub = np.ones((n_linear, n_block))
            
            for linear_idx, linear in enumerate(self.config['linear']):
                ub[linear_idx] = len(self.search_space.bits_range) - 1
            
            lb = np.delete(lb.flatten(), self.search_space.pass_linear_idx_list, axis=-1)
            ub = np.delete(ub.flatten(), self.search_space.pass_linear_idx_list, axis=-1)

            kwargs = {'lb': lb, 'ub': ub}

        quality_predictor = get_predictor(self.predictor, inputs, targets, device=device, **kwargs)

        return quality_predictor, quality_predictor.predict(inputs)
    
    def _next(self, archive, predictor, K):
        """ searching for next K candidate for high-fidelity evaluation (lower level) """

        # get non-dominated architectures from archive
        F = np.column_stack(([x[1] for x in archive], [x[2] for x in archive]))
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        # non-dominated arch bit-strings
        nd_X = np.array([self.search_space.encode(x[0]) for x in archive])[front]

        # initiate a multi-objective solver to optimize the problem
        method = NSGA2(pop_size=self.ga_pop_size, sampling=nd_X,  # initialize with current nd archs
            crossover=BinomialCrossover(prob=self.crossover_prob, n_offsprings=1),
            mutation=IntMutation(prob=self.mut_prob),
            eliminate_duplicates=True)
        
        # initialize the candidate finding optimization problem
        problem = AuxiliarySingleLevelProblem(self.search_space, predictor, self.config, self.group_size)
        
        # kick-off the search
        res = minimize(problem, method, termination=('n_gen', 20), save_history=True, verbose=True)
        
        # check for duplicates
        not_duplicate = np.logical_not(
            [x in [x[0] for x in archive] for x in [self.search_space.decode(x) for x in res.pop.get("X")]])
        print(f'not_duplicate : {sum(not_duplicate)}')

        pop = res.pop[not_duplicate]
        if sum(not_duplicate) >= K:
            indices = self._subset_selection(pop, F[front, 1], K, self.subset_pop_size)
            pop = pop[indices]

        candidates = []
        for x in pop.get("X"):
            candidates.append(self.search_space.decode(x))

        # decode integer bit-string to config and also return predicted top1_err
        return candidates, predictor.predict(self.search_space.decode_encode_predictor(pop.get("X")))

    # @staticmethod
    def _subset_selection(self, pop, nd_F, K, pop_size):
        problem = SubsetProblem(pop.get("F")[:, 1], nd_F, K)
        algorithm = GA(
            pop_size=pop_size, sampling=MySampling(), crossover=BinaryCrossover(),
            mutation=MyMutation(), eliminate_duplicates=True)

        res = minimize(
            problem, algorithm, ('n_gen', 60), verbose=False)

        return res.X

    @staticmethod
    def _calc_hv(ref_pt, F, normalized=True):
        # calculate hypervolume on the non-dominated set of F
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        nd_F = F[front, :]
        ref_point = 1.01 * ref_pt
        hv = Hypervolume(ref_point=ref_point).do(nd_F)
        if normalized:
            hv = hv / np.prod(ref_point)
        return hv

    def _get_correlation(self, prediction, target):
        rmse = np.sqrt(((prediction - target) ** 2).mean())
        rho, _ = stats.spearmanr(prediction, target)
        tau, _ = stats.kendalltau(prediction, target)

        return rmse, rho, tau