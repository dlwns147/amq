import gc
import time
import torch
import numpy as np

def device_warmup(device: str):
    warm_up = torch.randn((4096, 4096)).to(device)
    for i in range(100):
        torch.mm(warm_up, warm_up)

        
@torch.inference_mode()
def measure_tps(model, device = 'cuda', context_length = 64, generation_length = 256, iteration = 2, use_ft = False):
    input_ids = [1 for _ in range(context_length)]
    
    torch.cuda.reset_peak_memory_stats()

    time_list = []

    for _ in range(iteration):
        torch.cuda.reset_peak_memory_stats(device = device)
        torch.cuda.empty_cache()
        gc.collect()

        if use_ft:
            ## prefill
            start_pos = 0
            inputs = torch.as_tensor([input_ids], device=device)
            out = model(inputs, start_pos=start_pos, use_cache=False)
            start_pos += out.logits.shape[1]
            token = out.logits[:, -1].max(1)[1].unsqueeze(1)

            ## first GeMV
            inputs = torch.as_tensor([[token]], device=device)
            out = model(inputs, start_pos=start_pos, use_cache=False)
            start_pos += out.logits.shape[1]
            token = out.logits[:, -1].max(1)[1].unsqueeze(1)

            torch.cuda.synchronize()

            for i in range(generation_length):
                inputs = torch.as_tensor([[token]], device=device)

                torch.cuda.synchronize()
                start = time.perf_counter()
                out = model(inputs, start_pos=start_pos, use_cache=False)
                torch.cuda.synchronize()
                end = time.perf_counter()

                start_pos += out.logits.shape[1]                    
                token = out.logits[:, -1].max(1)[1].unsqueeze(1)

                time_list.append(end - start)
        else:
            ## prefill
            last_key_values = None
            inputs = torch.as_tensor([input_ids], device=device)
            out = model(inputs, past_key_values=last_key_values)
            out, last_key_values = out.logits, out.past_key_values
            token = out[:, -1].max(1)[1].unsqueeze(1)

            ## first GeMV
            inputs = torch.as_tensor([[token]], device=device)
            out = model(inputs, past_key_values=last_key_values)
            out, last_key_values = out.logits, out.past_key_values
            token = out[:, -1].max(1)[1].unsqueeze(1)

            torch.cuda.synchronize()

            for i in range(generation_length):
                inputs = torch.as_tensor([[token]], device=device)

                torch.cuda.synchronize()
                start = time.perf_counter()
                out = model(inputs, past_key_values=last_key_values)
                torch.cuda.synchronize()
                end = time.perf_counter()

                out, last_key_values = out.logits, out.past_key_values                    
                token = out[:, -1].max(1)[1].unsqueeze(1)

                time_list.append(end - start)

    return {
        'context_length' : context_length,
        'generation_length' : generation_length,
        'mean' : 1 / np.mean(time_list),
        'median' : 1 / np.median(time_list)

        }


@torch.inference_mode()
def measure_ttft(model, tokenizer, device = 'cuda', context_length = 64, iteration = 10, use_ft = False):
    """Time To First Token (TTFT)를 측정하는 함수"""
    input_ids = [1 for _ in range(context_length - 1)]
    context = tokenizer.decode(input_ids)

    time_list = []

    device_warmup(device)

    for _ in range(iteration):
        torch.cuda.reset_peak_memory_stats(device = device)
        torch.cuda.empty_cache()
        gc.collect()

        start_pos = 0
        last_key_values = None

        if use_ft:
            torch.cuda.synchronize()
            start = time.perf_counter()  # TTFT 시작 시간 측정

            input_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)
            start_pos = 0
            out = model(input_ids, start_pos=start_pos, use_cache=False)
            token = out.logits[:, -1].max(1)[1].unsqueeze(1)
            
            context = tokenizer.decode(token)

            torch.cuda.synchronize()
            end = time.perf_counter()  # TTFT 종료 시간 측정
        else:
            torch.cuda.synchronize()
            start = time.perf_counter()

            input_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)
            out = model(input_ids, past_key_values=last_key_values).logits
            token = out[:, -1].max(1)[1].unsqueeze(1)

            context = tokenizer.decode(token)

            torch.cuda.synchronize()
            end = time.perf_counter()

        time_list.append(end - start)

    return {
            'context_length' : context_length,
            'mean' : np.mean(time_list),
            'median' : np.median(time_list)
            }

    