import time
from functools import partial
from statistics import median
from tqdm import tqdm
import gc

import torch
import torch.nn as nn
from .data import *
from .loss import JSD
# import glog
from .func import clean_up


# Function to evaluate perplexity (ppl) on a specified model and tokenizer
@torch.no_grad()
def load_and_eval_ppl(model, model_name='', device=torch.device("cuda:0"), dataset='wikitext2', seqlen=2048, testloader=None, tokenizer=None):
    # Print status
    print(f"Evaluating on {dataset}")

    # Get the test loader
    if testloader is None:
        if tokenizer is None:
            tokenizer = get_tokenizer(model_name)

        testloader = get_loader(name=dataset, train=False, seed=0, seqlen=seqlen, tokenizer=tokenizer)
        print(f"Dataset Loaded.")

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl(model, testloader, seqlen=seqlen, device=device)
    return ppl_test 

@torch.no_grad()
def eval_ppl(model, accelerator, loader, seqlen=2048):
    # # Get input IDs
    # testenc = testenc.input_ids

    # # Calculate number of samples
    # n_sample = testenc.numel() // seqlen

    # List to store negative log likelihoods
    nlls = []
    # print(f"n_sample {n_sample}")
    
    # Loop through each batch
    for inputs in tqdm(loader, desc='Eval PPL'):

        # Forward pass through the model
        outputs = model(inputs)
        lm_logits = outputs.logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_logits = shift_logits.reshape(-1, shift_logits.size(-1))
        shift_labels = inputs[:, 1:].reshape(-1)
        
        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits, shift_labels)

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * seqlen * lm_logits.shape[0]

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # # Loop through each batch
    # for i in tqdm(range(0,n_sample,bs), desc='Eval PPL'):

    #     # Calculate end index
    #     j = min(i+bs, n_sample)

    #     # Prepare inputs and move to device
    #     inputs = testenc[:,(i * seqlen):(j * seqlen)].to(device)
    #     inputs = inputs.reshape(j-i, seqlen)

    #     # Forward pass through the model
    #     lm_logits = model(inputs).logits

    #     # Shift logits and labels for next token prediction
    #     shift_logits = lm_logits[:, :-1, :].contiguous()
    #     shift_labels = inputs[:, 1:]

    #     # Compute loss
    #     loss_fct = nn.CrossEntropyLoss()
    #     loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

    #     # Calculate negative log likelihood
    #     neg_log_likelihood = loss.float() * seqlen * (j-i)

    #     # Append to list of negative log likelihoods
    #     nlls.append(neg_log_likelihood)

    # print(f'{accelerator.device} nlls : {len(nlls)}')
    # nlls = accelerator.gather_for_metrics(nlls)
    # print(f'{accelerator.device} gathered nlls : {len(nlls)}')
    # nlls = torch.cat(nlls)
    # print(f'{accelerator.device} torch nlls : {nlls.shape}')
    nlls = torch.stack(accelerator.gather_for_metrics(nlls)).flatten()

    # Compute perplexity
    # ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * seqlen))
    ppl = torch.exp(nlls.sum() / (len(nlls) * seqlen))

    # Empty CUDA cache to save memory
    # torch.cuda.empty_cache()

    return ppl.item()

@torch.no_grad()
def get_logits(model, loader):    
    # List to store negative log likelihoods
    logits = []
    for inputs in loader:

        outputs = model(inputs)
        lm_logits = outputs.logits
        logits.append(lm_logits)

    dense_logits_list = torch.cat(logits, dim=0).detach()

    return dense_logits_list


@torch.no_grad()
def eval_loss(model, accelerator, loader, seqlen=2048, loss_func='cross_entropy', dense_logits_list=None):
    # Get input IDs
    # testenc = testenc.input_ids

    # Calculate number of samples
    # n_sample = testenc.numel() // seqlen
  
    # List to store negative log likelihoods
    losses = []
    
    # Loop through each batch
    for i, inputs in enumerate(loader):

        outputs = model(inputs)
        lm_logits = outputs.logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_logits = shift_logits.reshape(-1, shift_logits.size(-1))
        shift_labels = inputs[:, 1:].reshape(-1)
        
        # Compute loss
        if loss_func == 'cross_entropy':
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)
        elif loss_func == 'jsd':
            dense_logits = dense_logits_list[i]
            dense_logits = dense_logits[:-1, :].reshape(-1, shift_logits.size(-1)).contiguous()
            loss_fct = JSD()
            loss = loss_fct(shift_logits, dense_logits)
        else:
            raise NotImplementedError(f'{loss_func} is not implemented')

        # Calculate negative log likelihood
        loss = loss.float() * seqlen * lm_logits.shape[0]

        # Append to list of negative log likelihoods
        losses.append(loss)

    # for i in range(0,n_sample,bs):

    #     # Calculate end index
    #     j = min(i+bs, n_sample)

    #     # Prepare inputs and move to device
    #     inputs = testenc[:,(i * seqlen):(j * seqlen)].to(device)
    #     inputs = inputs.reshape(j-i, seqlen)

    #     # Forward pass through the model
    #     outputs = model(inputs)
    #     lm_logits = outputs.logits

    #     # Shift logits and labels for next token prediction
    #     shift_logits = lm_logits[:, :-1, :]
    #     shift_logits = shift_logits.reshape(-1, shift_logits.size(-1)).contiguous()
    #     shift_labels = inputs[:, 1:]

    #     # Compute loss
    #     if loss_func == 'cross_entropy':
    #         loss_fct = nn.CrossEntropyLoss()
    #         loss = loss_fct(shift_logits, shift_labels.reshape(-1))
    #     elif loss_func == 'jsd':
    #         dense_logits = dense_logits_list[i: j]
    #         dense_logits = dense_logits[:, :-1, :].reshape(-1, dense_logits.size(-1)).contiguous()
    #         loss_fct = JSD()
    #         loss = loss_fct(shift_logits, dense_logits)
    #     else:
    #         raise NotImplementedError(f'{loss_func} is not implemented')

        # # Calculate negative log likelihood
        # loss = loss.float() * seqlen * (j-i)
        # loss = accelerator.gather_for_metrics(loss)

        # # Append to list of negative log likelihoods
        # losses.append(loss)
    
    # Compute sum of negative log_likelihood
    # losses = accelerator.gather_for_metrics(losses)
    # print(f'losses: {losses}, {len(losses)}')
    # losses = torch.cat(losses)
    losses = torch.stack(accelerator.gather_for_metrics(losses)).flatten()
    loss_sum = losses.sum() / (len(losses) * seqlen)
    # loss_sum = torch.stack(losses).sum() / seqlen

    return loss_sum.item()


def eval_metric(model, accelerator, metric, loader, seqlen, loss_func='cross_entropy', dense_logits_list=None):
    # accelerator.wait_for_everyone()
    if metric == 'ppl':
        return eval_ppl(model, accelerator, loader, seqlen=seqlen)
    elif metric == 'loss':
        return eval_loss(model, accelerator, loader, seqlen=seqlen, loss_func=loss_func, dense_logits_list=dense_logits_list)
    else:
        raise NotImplementedError(f'{metric} is not supported')



def get_graph_wrapper(cls, device=0):

    class GraphWrapper(cls):

        def __init__(self, *args, **kwargs):
            super(GraphWrapper, self).__init__(*args, **kwargs)
            self.built_graph = False
            self.graph_device = device

        def forward(self, *args, **kwargs):
            with torch.cuda.device(self.graph_device):
                if not self.built_graph:
                    self.static_args = args
                    self.static_kwargs = kwargs

                    s = torch.cuda.Stream(device=self.graph_device)
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        super(GraphWrapper,
                              self).forward(*self.static_args,
                                            **self.static_kwargs)
                    torch.cuda.current_stream().wait_stream(s)

                    self.graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self.graph, stream=s):
                        self.static_output = super(GraphWrapper, self).forward(
                            *self.static_args, **self.static_kwargs)

                    self.built_graph = True
                    print("Built CUDA graph of model.")

                # these two loops take < 1e-4 seconds for llama2
                for i in range(len(args)):
                    if isinstance(args[i], torch.Tensor):
                        self.static_args[i].copy_(args[i])
                for kw in kwargs:
                    if isinstance(kwargs[kw], torch.Tensor):
                        self.static_kwargs[kw].copy_(kwargs[kw])

                self.graph.replay()
                return self.static_output

        def reset(self):
            if self.built_graph:
                del self.static_args, self.static_kwargs
                del self.graph
                del self.static_output
                self.built_graph = False

    return GraphWrapper

@torch.inference_mode()
def device_warmup(device: str):
    warm_up = torch.randn((4096, 4096)).to(device)
    for i in range(100):
        torch.mm(warm_up, warm_up)


@torch.inference_mode()
def measure_latency_v2(model, tokenizer=None, use_ft=False, use_cuda_graph=False, iteration=1, sizes=(1, 64, 128), mode='gemv', get_peak_memory=True, device='cuda'):

    vocab_size = model.config.vocab_size if hasattr(model.config, "vocab_size") else 32000

    ori_use_cache = model.config.use_cache
    ori_generation_use_cache = model.generation_config.use_cache
    ori_pad_token_id = model.generation_config.pad_token_id

    wrapped_ft = use_ft
    wrapped_cuda_graph = use_cuda_graph
    benchmark_iteration = iteration

    model.eval()
    model = model.to('cuda')

    data = {}
    data[mode.lower()] = {}

    if get_peak_memory:
        clean_up()

        torch.cuda.reset_peak_memory_stats(device = device)

        data['peak_memory'] = {}

    batch_size, input_seq_len, gen_seq_len = sizes

    input_ids = torch.randint(0, vocab_size - 1, (batch_size, input_seq_len), dtype=torch.long).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    device_warmup(device)
    clean_up()

    if get_peak_memory:
        torch.cuda.reset_peak_memory_stats(device = device)

    ## calculate average token per second by GeMV with prefill
    model.config.use_cache = False
    model.generation_config.use_cache = False

    time_list = []

    for _ in range(benchmark_iteration):
        clean_up()

        if wrapped_ft:
            start_pos = 0

            if mode.lower() == 'gemm':
                torch.cuda.synchronize()
                start = time.perf_counter()

            out = model(input_ids, start_pos=start_pos, use_cache=False)

            if mode.lower() == 'gemm':
                torch.cuda.synchronize()
                end = time.perf_counter()

                time_list.append(end - start)

            start_pos += out.logits.shape[1]
            max_logit = out.logits[:, -1].max(1)[1].unsqueeze(1)
            gemv_input_ids = torch.as_tensor([[max_logit]], device=device)

            if mode.lower() == 'gemv':
                for _ in range(gen_seq_len):
                    torch.cuda.synchronize()
                    start = time.perf_counter()

                    out = model(gemv_input_ids, start_pos=start_pos, use_cache=False)

                    torch.cuda.synchronize()
                    end = time.perf_counter()

                    start_pos += out.logits.shape[1]
                    max_logit = out.logits[:, -1].max(1)[1].unsqueeze(1)
                    gemv_input_ids = torch.as_tensor([[max_logit]], device=device)

                    time_list.append(end - start)
        else:
            last_key_values = None

            if mode.lower() == 'gemm':
                torch.cuda.synchronize()
                start = time.perf_counter()

            if wrapped_cuda_graph:
                model.reset()

            out = model(input_ids, past_key_values=last_key_values)

            if mode.lower() == 'gemm':
                torch.cuda.synchronize()
                end = time.perf_counter()

                time_list.append(end - start)

            logits, last_key_values = out.logits, out.past_key_values
            max_logit = logits[:, -1].max(1)[1].unsqueeze(1)
            gemv_input_ids = torch.as_tensor([[max_logit]], device=device)

            if wrapped_cuda_graph:
                model.reset()
                model(gemv_input_ids, past_key_values=last_key_values)

            if mode.lower() == 'gemv':
                for _ in range(gen_seq_len):
                    torch.cuda.synchronize()
                    start = time.perf_counter()

                    # nvtx.push_range("gemv")

                    out = model(gemv_input_ids, past_key_values=last_key_values)

                    # nvtx.pop_range()

                    torch.cuda.synchronize()
                    end = time.perf_counter()

                    logits, last_key_values = out.logits, out.past_key_values
                    max_logit = logits[:, -1].max(1)[1].unsqueeze(1)
                    gemv_input_ids = torch.as_tensor([[max_logit]], device=device)

                    time_list.append(end - start)

    return 1 / np.median(time_list)



@torch.no_grad()
def measure_latency(model, generation, device, batch_size=64, prompt_length=64, generation_length=128, iteration=10, max_time=1e9) :

    def cuda_timestamp(sync=False, device=None):
        if sync:
            torch.cuda.synchronize(device=device)
        return time.perf_counter()

    time_fn = partial(cuda_timestamp, device=device)

    def _step(input):
        t_step_start = time_fn()
        model(input)
        t_step_end = time_fn(True)
        return t_step_end - t_step_start

    def _step_gen(input, generation_length):
        t_step_start = time_fn()
        model.generate(input,min_new_tokens=generation_length, max_new_tokens=generation_length)
        t_step_end = time_fn(sync=True)
        return t_step_end - t_step_start
    
    latency = []
    if (generation) :
        # setting for token generation
        max_length = prompt_length + generation_length
        model.config.max_length = max_length
        config_use_cache = model.config.use_cache
        generation_config_use_cache = model.generation_config.use_cache
        model.config.use_cache = True
        model.generation_config.use_cache = True

        # make dummy input
        random_input = torch.randint(0, 31999, (batch_size, prompt_length), dtype=torch.long)
        random_input = random_input.to(device).contiguous()

        # dummy inference
        model.generate(random_input,min_new_tokens=generation_length, max_new_tokens=generation_length)

        # latency for 10 iterations
        # starter,ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        for i in tqdm(range(iteration)):
            # starter.record()
            # model.generate(random_input,min_new_tokens=generation_length, max_new_tokens=generation_length)
            # ender.record()
            # torch.cuda.synchronize()
            # cur_time = starter.elapsed_time(ender)
            cur_time = max_time
            try:
                cur_time = _step_gen(random_input, generation_length)
            except RuntimeError:
                pass
            latency.append(cur_time)

    else :
        # setting for prompt processing
        # batch_size = 1
        config_use_cache = model.config.use_cache
        generation_config_use_cache = model.generation_config.use_cache
        model.config.use_cache = False
        model.generation_config.use_cache = False
        # iteration = 50

        # make dummy input for module.weight shape
        random_input = torch.randint(0, 31999, (batch_size, 2048), dtype=torch.long)
        random_input = random_input.to(device).contiguous()
        
        # dummy inference
        model(random_input)

        # latency for 50 iterations
        # starter,ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        for i in tqdm(range(iteration)):
            # starter.record()
            # model(random_input)
            # ender.record()
            # torch.cuda.synchronize()
            # cur_time = starter.elapsed_time(ender)
            cur_time = _step(random_input)
            latency.append(cur_time)

    # curr_time = starter.elapsed_time(ender)
    median_latency = median(latency)
    # mean_latency = curr_time/iteration

    model.config.use_cache = config_use_cache
    model.generation_config.use_cache = generation_config_use_cache
    
    gc.collect()
    torch.cuda.empty_cache()

    return median_latency

torch.no_grad()
def eval_zeroshot(model, tokenizer, task_list=['piqa','winogrande','hellaswag','arc_challenge','arc_easy'], 
        num_fewshot=None, batch_size=None):
    
    import os
    from lm_eval.models.huggingface import HFLM
    from lm_eval import tasks, evaluator, utils
    import datasets
    datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

    # task_manager = tasks.TaskManager()
    # task_manager = tasks.TaskManager(include_path='')
    # task_manager = tasks.TaskManager(include_path='lm-evaluation-harness/lm_eval/tasks')
    # task_manager = tasks.TaskManager(include_path='/NAS/SJ/lm-evaluation-harness/lm_eval/tasks')
    # task_manager = tasks.TaskManager(include_path='/NAS/SJ/sleb/lm-evaluation-harness/lm_eval/tasks')
 
    # task_names = task_manager.match_tasks(task_list)
    # for task in [task for task in task_list if task not in task_names]:
    #             if os.path.isfile(task):
    #                 config = utils.load_yaml_config(task)
    #                 task_names.append(config)
    # task_missing = [
    #     task
    #     for task in task_list
    #     if task not in task_names and "*" not in task
    #     ]  # we don't want errors if a wildcard ("*") task name was used
    
    # model.tie_weights = lambda: None
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size if batch_size is not None else 1)# , batch_size='auto')
    
    results = evaluator.simple_evaluate(
        model=hflm,
        tasks=task_list,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        # batch_size='auto',
        # max_batch_size=None,
        # device='cuda:0',
        # use_cache=None,
        # limit=None,
        # check_integrity=False,
        # write_out=False,
        # gen_kwargs=None,
        # task_manager=task_manager,
        # decontamination_ngrams_path=None,
    )

    return results['results']