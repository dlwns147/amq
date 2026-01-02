import gc
import time
import torch
import numpy as np

device = 'cuda'
wrapped_ft = True
benchmark_iteration = 1

def cleanup():
    torch.cuda.empty_cache()
    gc.collect()


@torch.inference_mode()
def device_warmup(device: str):
    warm_up = torch.randn((4096, 4096)).to(device)
    for i in range(100):
        torch.mm(warm_up, warm_up)


@torch.inference_mode()
def benchmark_tps(model, input_ids, attention_mask, gen_seq_len):
    global device, wrapped_ft, benchmark_iteration

    time_list = []

    for _ in range(benchmark_iteration):
        cleanup()

        torch.cuda.synchronize()
        start = time.perf_counter()

        _ = model.generate(input_ids, 
            min_new_tokens = gen_seq_len,
            max_new_tokens = gen_seq_len,
            do_sample=False,
            num_beams=1,
            attention_mask = attention_mask)

        torch.cuda.synchronize()
        end = time.perf_counter()

        time_list.append(end - start)

    return gen_seq_len / np.median(time_list)


@torch.inference_mode()
def benchmark_gemv_gemm(model, input_ids, gen_seq_len, mode = 'gemv'):
    global device, wrapped_ft, benchmark_iteration

    time_list = []

    for _ in range(benchmark_iteration):
        cleanup()

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

            out = model(input_ids, past_key_values=last_key_values)

            if mode.lower() == 'gemm':
                torch.cuda.synchronize()
                end = time.perf_counter()

                time_list.append(end - start)

            logits, last_key_values = out.logits, out.past_key_values
            max_logit = logits[:, -1].max(1)[1].unsqueeze(1)
            gemv_input_ids = torch.as_tensor([[max_logit]], device=device)

            if mode.lower() == 'gemv':
                for _ in range(gen_seq_len):
                    torch.cuda.synchronize()
                    start = time.perf_counter()

                    out = model(gemv_input_ids, past_key_values=last_key_values)

                    torch.cuda.synchronize()
                    end = time.perf_counter()

                    logits, last_key_values = out.logits, out.past_key_values
                    max_logit = logits[:, -1].max(1)[1].unsqueeze(1)
                    gemv_input_ids = torch.as_tensor([[max_logit]], device=device)

                    time_list.append(end - start)

    return 1 / np.median(time_list)


@torch.inference_mode()
def benchmark_speed(model, tokenizer = None, use_ft = True, iteration = 1, sizes = (1, 128, 128), mode = 'TPS', get_peak_memory = True):
    assert mode.lower() in ['tps', 'gemv', 'gemm', 'ttft'], "speed benchmark mode should be one of ['TPS', 'GeMV', 'GeMM', 'TTFT']"

    vocab_size = model.config.vocab_size if hasattr(model.config, "vocab_size") else 32000

    ori_use_cache = model.config.use_cache
    ori_generation_use_cache = model.generation_config.use_cache
    ori_pad_token_id = model.generation_config.pad_token_id

    global device, wrapped_ft, benchmark_iteration
    device = model.device
    wrapped_ft = use_ft
    benchmark_iteration = iteration

    model.eval()
    model = model.to('cuda')

    data = {}
    data[mode.lower()] = {}

    if get_peak_memory:
        cleanup()

        torch.cuda.reset_peak_memory_stats(device = device)

        data['peak_memory'] = {}

    ## TODO : all 구현
    if mode.lower() in ['tps', 'gemv', 'gemm']:
        batch_size, input_seq_len, gen_seq_len = sizes

        input_ids = torch.randint(0, vocab_size - 1, (batch_size, input_seq_len), dtype=torch.long).to(model.device)
        attention_mask = torch.ones_like(input_ids).to(model.device)

        device_warmup(device)
        cleanup()

        if get_peak_memory:
            torch.cuda.reset_peak_memory_stats(device = device)

        if mode.lower() == 'tps':
            ## calculate token per second by generating sequences(GeMM + GeMV)
            model.generation_config.pad_token_id = model.generation_config.eos_token_id
            model.config.use_cache = True
            model.generation_config.use_cache = True

            speed = benchmark_tps(model, input_ids, attention_mask, gen_seq_len)

        elif mode.lower() in ['gemv', 'gemm']:
            ## calculate average token per second by GeMV with prefill
            model.config.use_cache = False
            model.generation_config.use_cache = False

            speed = benchmark_gemv_gemm(model, input_ids, gen_seq_len, mode)

    else:
        assert tokenizer is not None, "Tokenizer should be provided for TTFT benchmark"

        model.config.use_cache = False
        model.generation_config.use_cache = False

        batch_size, input_seq_len, gen_seq_len = sizes

        input_ids = torch.randint(0, vocab_size - 1, (batch_size, input_seq_len), dtype=torch.long).to(model.device)
        text = tokenizer.decode(input_ids[0])

        time_list = []

        device_warmup(device)
        cleanup()

        if get_peak_memory:
            torch.cuda.reset_peak_memory_stats(device = device)

        torch.cuda.synchronize()

        for _ in range(benchmark_iteration):
            cleanup()

            if wrapped_ft:
                start_pos = 0

                torch.cuda.synchronize()
                start = time.perf_counter()

                input_ids = tokenizer(text, return_tensors='pt', truncation = True, max_length=input_seq_len).input_ids.to(model.device)
                out = model(input_ids, start_pos=start_pos, use_cache=False)
                max_logit = out.logits[:, -1].max(1)[1].unsqueeze(1)
                _ = tokenizer.decode(max_logit[0])

                torch.cuda.synchronize()
                end = time.perf_counter()
            else:
                last_key_values = None

                torch.cuda.synchronize()
                start = time.perf_counter()

                input_ids = tokenizer(text, return_tensors='pt', truncation = True, max_length=input_seq_len).input_ids.to(model.device)
                out = model(input_ids, past_key_values=last_key_values)
                max_logit = out.logits[:, -1].max(1)[1].unsqueeze(1)
                _ = tokenizer.decode(max_logit[0])
                
                torch.cuda.synchronize()
                end = time.perf_counter()

            time_list.append((end - start) * 1000)      ## ms 단위로 변환
            
            speed = np.median(time_list)

    data[mode.lower()][f'{batch_size}.{input_seq_len}.{gen_seq_len}'] = speed

    if get_peak_memory:
        data['peak_memory'][f'{batch_size}.{input_seq_len}.{gen_seq_len}'] = torch.cuda.max_memory_allocated(device = device) / 1024 ** 3


    model.config.use_cache = ori_use_cache
    model.generation_config.use_cache = ori_generation_use_cache
    model.generation_config.pad_token_id = ori_pad_token_id
    model = model.to(device)

    torch.cuda.reset_peak_memory_stats(device = device)
    cleanup()

    return data


all = ["benchmark_speed"]