# Import necessary modules
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer
from torch.utils.data import DataLoader

class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

def get_tokenizer(model, cache_dir=None):
    # if "llama" in model.lower():
    #     tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False, cache_dir=cache_dir)
    #     # fix for transformer 4.28.0.dev0 compatibility
    #     if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
    #         try:
    #             tokenizer.bos_token_id = 1
    #             tokenizer.eos_token_id = 2
    #         except AttributeError:
    #             pass
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, cache_dir=cache_dir)
    return tokenizer

def get_wikitext2(tokenizer, seqlen=2048, batch_size=1, cache_dir=None):
    
    # traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', cache_dir=cache_dir)
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', cache_dir=cache_dir)

    # trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt').input_ids
    n_sample = testenc.numel() // seqlen
    testenc = testenc[:, :n_sample * seqlen].reshape(n_sample, seqlen)

    return DataLoader(testenc, batch_size=batch_size, drop_last=False)

def get_c4(tokenizer, seqlen=2048, batch_size=1, cache_dir=None):
   
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation', cache_dir=cache_dir)

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    # valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    n_sample = valenc.numel() // seqlen
    valenc = valenc[:, :n_sample * seqlen].reshape(n_sample, seqlen)

    return DataLoader(valenc, batch_size=batch_size, drop_last=False)

def get_wikitext2_trainenc(seed, n_sample, tokenizer, batch_size=1, seqlen=2048, cache_dir=None):
    
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', cache_dir=cache_dir)
    traindata = traindata.shuffle(seed=seed)
    trainenc = tokenizer("\n\n".join(traindata[:n_sample]['text']), return_tensors='pt').input_ids
    n_sample = trainenc.numel() // seqlen
    trainenc = trainenc[:, :n_sample * seqlen].reshape(n_sample, seqlen)

    return DataLoader(trainenc, batch_size=batch_size)

def get_c4_trainenc(seed, n_sample, tokenizer, batch_size=1, seqlen=2048, cache_dir=None):
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', cache_dir=cache_dir
    )
    traindata = traindata.shuffle(seed=seed)
    
    trainenc = tokenizer(' '.join(traindata[:n_sample]['text']), return_tensors='pt').input_ids
    n_sample = trainenc.numel() // seqlen
    trainenc = trainenc[:, :n_sample * seqlen].reshape(n_sample, seqlen)

    return DataLoader(trainenc, batch_size=batch_size, drop_last=True)

def get_trainloaders(name, n_sample=128, seed=0, seqlen=2048, model='', batch_size=1, cache_dir=None):
    tokenizer = get_tokenizer(model)
    if 'wikitext2' in name:
        return get_wikitext2_trainenc(seed, n_sample, seqlen, model, tokenizer, batch_size, cache_dir=cache_dir)
    if 'c4' in name:
        return get_c4_trainenc(seed, n_sample, seqlen, model, tokenizer, batch_size, cache_dir=cache_dir)

def get_loader(name, n_sample=128, train=True, seed=0, seqlen=2048, tokenizer=None, model='', cache_dir=None):
    if tokenizer is None:
        tokenizer = get_tokenizer(model, cache_dir=cache_dir)
    if train:
        if 'wikitext2' in name:
            return get_wikitext2_trainenc(seed=seed, n_sample=n_sample, seqlen=seqlen, tokenizer=tokenizer, cache_dir=cache_dir)
        if 'c4' in name:
            return get_c4_trainenc(seed=seed, n_sample=n_sample, seqlen=seqlen, tokenizer=tokenizer, cache_dir=cache_dir)
    else:
        if 'wikitext2' in name:
            return get_wikitext2(tokenizer=tokenizer, seqlen=seqlen, cache_dir=cache_dir)
        if 'c4' in name:
            return get_c4(seqlen=seqlen, tokenizer=tokenizer, cache_dir=cache_dir)
