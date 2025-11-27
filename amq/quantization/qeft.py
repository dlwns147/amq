import time

import torch
import torch.nn as nn

import argparse
from tqdm import tqdm

from .qeft_utils.recon import GPTQ_OWQ
from .qeft_utils.quant import *
from .qeft_utils.misc import *
from .qeft_utils.reorder import *

import torch
import torch.nn as nn
import math
import json
import transformers

from .base import BASE, get_owq_calib_dataset
    
class QEFT(BASE):
    def __init__(self, model_name, config, arch, device_map, dtype='auto', group_size=128, dev='cuda', prune=False, do_owq=False, outlier_path=None, **kwargs):
        super().__init__(model_name, config, arch, device_map=device_map, dtype=dtype, group_size=group_size, dev=dev, prune=prune, do_owq=do_owq, outlier_path=outlier_path)
        self.method = 'qeft'

    @torch.no_grad()
    def run(
        self,
        samples=None,
        n_samples=128,
        seqlen=2048,
        dataset='c4',
        # dataset='wikitext2',
        reorder=True,
        # reorder=False,
        # target_rank=32,
        seed=42,
        sym=False,
        true_sequential=False,
        percdamp=.01,
        act_order=False,
        no_frob_norm=False,
        tuning='mse',
        layers=None,
        nearest_owq=False
    ):
        
        if samples is None:
            dataloader = get_owq_calib_dataset(dataset, tokenizer=self.tokenizer, n_samples=n_samples, seqlen=seqlen, seed=seed)
            
        # assert args.no_frob_norm == True
        meta = get_meta(self.model_name, layers)
        print('Starting ...')

        use_cache = self.model.config.use_cache
        layers, pre_layers, post_layers = parsing_layers(self.model, meta)
        
        self.model.config.use_cache = False
        
        for pre_layer in pre_layers:
            pre_layer = pre_layer.to(self.dev)
        
        layers[0] = layers[0].to(self.dev)
        if hasattr(self.model.model, 'rotary_emb'):
            self.model.model.rotary_emb = self.model.model.rotary_emb.to(self.dev)

        dtype = next(iter(self.model.parameters())).dtype
        inps = torch.zeros(
            (n_samples, seqlen, self.model.config.hidden_size), dtype=dtype, device=self.dev
        )

        cache = {kw:None for kw in meta['inp_kwargs']}
        cache['i'] = 0
        
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp
                for key in cache:
                    if key == 'i':
                        cache['i'] += 1
                    else:
                        cache[key] = kwargs[key]
                raise ValueError
        
        layers[0] = Catcher(layers[0])
        for batch in dataloader:
            try:
                self.model(batch[0].to(self.dev))
            except ValueError:
                pass
        
        layers[0] = layers[0].module.cpu()
        if hasattr(self.model.model, 'rotary_emb'):
            self.model.model.rotary_emb = self.model.model.rotary_emb.cpu()
        
        for pre_layer in pre_layers:
            pre_layer = pre_layer.cpu()
        torch.cuda.empty_cache()

        outs = torch.zeros_like(inps)
        del cache['i']
        inp_kwargs = cache

        # print('Ready.')
        owq_layers = meta['owq_layers']
        ratios = meta['ratios']
        n_out_dict = {l:[0] * len(layers) for l in owq_layers.keys()}
        target_rank = self.outlier['n_outlier']
        if target_rank is not None:
            for l, owq in owq_layers.items():
                if owq:
                    n_out_dict[l] = [target_rank if bits != math.ceil(bits) else 0 for bits in self.arch['linear'][l]]
                    # n_out_dict[l] = target_rank
        print(f'n_out_dict : {n_out_dict}')
        print(f'self.arch : {self.arch}')
        
        quantizers = {}
        for i in tqdm(range(len(layers)), "Reconstruction Blocks..."):
            layer = layers[i].to(self.dev)
            block_layers = find_layers(layer, layers=[nn.Linear])

            if true_sequential:
                sequential = meta['sequential']
            else:
                sequential = [list(block_layers.keys())]
                
            for names in sequential:
                subset = {n: block_layers[n] for n in names}

                gptq_owq = {}
                for name in subset:
                    wbits = self.arch['linear'][name][i]
                    gptq_owq[name] = GPTQ_OWQ(subset[name], n_out=n_out_dict[name][i])
                    gptq_owq[name].quantizer = Quantizer(
                        math.floor(wbits), perchannel=True, sym=sym, mse=(tuning == 'mse'), group_size=self.group_size
                    )
                    gptq_owq[name].quantizer.n_out = n_out_dict[name][i]
                    
                def add_batch(name):
                    def tmp(_, inp, out):
                        gptq_owq[name].add_batch(inp[0].data, out.data)
                    return tmp
                handles = []
                for name in subset:
                    handles.append(subset[name].register_forward_hook(add_batch(name)))
                for j in range(n_samples):
                    layer(inps[j].unsqueeze(0), **inp_kwargs)
                for h in handles:
                    h.remove()
                
                for name in subset:
                    wbits = self.arch['linear'][name][i]
                    if not no_frob_norm and (not reorder or (reorder and name in meta['sequential'][1] + meta['sequential'][3])):
                        W = subset[name].weight.data.clone().to(torch.float)
                        temp_quantizer = Quantizer(
                            math.floor(wbits), perchannel=True, sym=sym, mse=(tuning == 'mse'), group_size=self.group_size
                        )
                        temp_quantizer.find_params(W, weight=True, num=40)
                        W_quant = temp_quantizer.quantize(W)
                        frob_norm_error = (W - W_quant).pow(2).sum(dim=0)
                    else:
                        frob_norm_error = None

                    key = f"{meta['prefix']}.{i}.{name}"
                    outidx = torch.tensor(self.outlier[key]) if key in self.outlier and wbits != math.floor(wbits) else None
                    # print(f'key : {key}, outidx : {outidx}, wbits : {wbits}, key in self.outlier : {key in self.outlier}')
                    out_ids = gptq_owq[name].hessian_sorting(
                        actorder=act_order, 
                        frob_norm=frob_norm_error, 
                        # outidx=torch.tensor(self.outlier[key]) if key in self.outlier and name not in meta['sequential'][1] and wbits != math.floor(wbits) else None, 
                        outidx=torch.tensor(self.outlier[key]) if key in self.outlier and wbits != math.floor(wbits) else None, 
                        # outidx=outidx if name not in meta['sequential'][1] + meta['sequential'][3] and wbits != math.floor(wbits) else None, 
                        )
                    gptq_owq[name].quantizer.out_ids = out_ids
                    gptq_owq[name].quantizer.n_out = out_ids.numel()
                    gptq_owq[name].quantizer.reorder = reorder # if name not in meta['sequential'][1] else False
                    # print(f'n_out_dict[name][i] : {n_out_dict[name][i]}, n_out : {out_ids.numel()}, self.outlier[key] : {self.outlier[key] if key in self.outlier else 0}')

                if not no_frob_norm:
                    del W
                    del W_quant
                    del temp_quantizer
                    torch.cuda.empty_cache()
                
                for name in subset:
                    key = f"{meta['prefix']}.{i}.{name}"
                    # print(f"Quantizing {key}")
                    if name not in meta['sequential'][1] and name not in meta['sequential'][3]:
                        global_ids = torch.tensor(self.outlier[key])
                    # print(f'out_ids : {gptq_owq[name].quantizer.out_ids.tolist()}')
                    # print(f'self.outlier[key] : {self.outlier[key] if key in self.outlier else 0}') 
                    # print(f'global_ids : {global_ids.tolist()}')
                    # if key in self.outlier:
                    #     print(f'(gptq_owq[name].quantizer.out_ids == self.outlier[key]).sum() : {(gptq_owq[name].quantizer.out_ids == self.outlier[key]).sum()}')
                    # print('=' * 20)
                    if nearest_owq:
                        if gptq_owq[name].quantizer.reorder:
                            gptq_owq[name].fasterquant_nearest_owq_reorder(groupsize=self.group_size, actorder=act_order)
                        else:
                            gptq_owq[name].fasterquant_nearest_owq(groupsize=self.group_size, actorder=act_order)
                    else:
                        if gptq_owq[name].quantizer.reorder:
                            gptq_owq[name].fasterquant_reorder(percdamp=percdamp, groupsize=self.group_size, actorder=act_order)
                        else:
                            gptq_owq[name].fasterquant(percdamp=percdamp, groupsize=self.group_size, actorder=act_order)
                    quantizers[f"{meta['prefix']}.{i}.{name}"] = gptq_owq[name].quantizer
                    gptq_owq[name].free()
            
            for j in range(n_samples):
                outs[j] = layer(inps[j].unsqueeze(0), **inp_kwargs)[0]
                
            for name in list(block_layers.keys()):
                quantizers[f"{meta['prefix']}.{i}.{name}"] = quantizers[f"{meta['prefix']}.{i}.{name}"].cpu()

            layers[i] = layer.cpu()
            del layer
            del gptq_owq 
            torch.cuda.empty_cache()

            inps, outs = outs, inps
                
        if reorder:
            global_ids = global_ids.cpu()
            make_reorder(self.model, quantizers, global_ids, meta)
        self.model.config.use_cache = use_cache
