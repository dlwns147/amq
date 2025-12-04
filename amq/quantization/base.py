import random

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from datasets import load_dataset
from utils.func import clean_up

from model import skip_llama

def get_awq_calib_dataset(data="pileval", tokenizer=None, n_samples=512, block_size=512):
    if data == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    else:
        raise NotImplementedError
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]


def get_gptq_calib_dataset(data="c4", tokenizer=None, n_samples=128, seed=0, seqlen=2048):
    if data == "c4":
        traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    else:
        raise NotImplementedError

    random.seed(seed)
    trainloader = []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader

def get_owq_calib_dataset(data="c4", tokenizer=None, n_samples=128, seed=0, seqlen=2048):
    
    random.seed(seed)
    if 'wikitext2' in data:
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
        
        trainloader = []
        for _ in range(n_samples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
            
    elif 'c4' in data:
        traindata = load_dataset(
            'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
        )
        trainloader = []
        for _ in range(n_samples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] > seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
    else:
        raise NotImplementedError(data)
    
    return trainloader

class BASE:
    def __init__(self, model, method, avg_bits, arch, group_size=128, dev='cuda', **kwargs):
        self.model = model
        self.method = method
        self.avg_bits = avg_bits
        self.arch = arch
        self.group_size = group_size
        self.dev = dev
        
        print(f'Quantization options: \n \
                method: {method}, \n \
                arch: {arch}, \n \
                avg_bits: {avg_bits:.4f}, \n \
                group_size: {group_size}, \n \
                dev: {dev}')

    def append_str_prefix(self, x, prefix):
        if isinstance(x, str):
            return prefix + x
        elif isinstance(x, tuple):
            return tuple([self.append_str_prefix(y, prefix) for y in x])
        elif isinstance(x, list):
            return [self.append_str_prefix(y, prefix) for y in x]
        else:
            return x

    @staticmethod
    def get_named_linears(module):
        return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}
    

    @staticmethod    
    def get_op_by_name(module, op_name):
        # get the op by its name relative to the module
        for name, m in module.named_modules():
            if name == op_name:
                return m
        raise ValueError(f"Cannot find op {op_name} in module {module}")


    @staticmethod
    def set_op_by_name(layer, name, new_module):
        levels = name.split(".")
        if len(levels) > 1:
            mod_ = layer
            for l_idx in range(len(levels) - 1):
                if levels[l_idx].isdigit():
                    mod_ = mod_[int(levels[l_idx])]
                else:
                    mod_ = getattr(mod_, levels[l_idx])
            setattr(mod_, levels[-1], new_module)
        else:
            setattr(layer, name, new_module)


    @staticmethod
    def get_op_name(module, op):
        # get the name of the op relative to the module
        for name, m in module.named_modules():
            if m is op:
                return name
        raise ValueError(f"Cannot find op {op} in module {module}")
