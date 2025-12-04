import json
import math
import os
import time

import torch
import torch.nn as nn
import transformers

from .base import BASE, get_owq_calib_dataset
from utils.func import clean_up

DEBUG = False

def quantize(x, scale, zero, minq, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, minq, maxq)
    return scale * (q - zero)

def quantize_efficient(x_round, scale, zero, minq, maxq):
    q = torch.clamp(x_round + zero, minq, maxq)
    return scale * (q - zero)

def processing_meta(model_name):
    model_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model_config.json')
    with open(model_config) as f:
        metas = json.load(f)
    
    # model config
    if 'opt' in model_name:
        meta = metas['opt']
        if '350m' in model_name:
            meta['pre_layers'].append('model.model.decoder.project_in')
            meta['post_layers'].append('model.model.decoder.project_out')
        else:
            meta['post_layers'].append('model.model.decoder.final_layer_norm')
    elif 'llama' in model_name or 'vicuna' in model_name:
        meta = metas['llama']
    elif 'bloom' in model_name:
        meta = metas['bloom']
    else:
        raise NotImplementedError(f"{model_name} model is not implemented.")
    
    map_layer = meta['map_layer']
    layers_owq = {l:False for l in map_layer.values()}
    for l in layers_owq:
        layers_owq[l] = True
    for l in layers_owq:
        if not layers_owq[l]:
            meta['ratios'][l] = 0.0
    
    meta['owq_layers'] = layers_owq

    return meta


class OWQ(BASE):
    def __init__(self, model, tokenizer, method, arch, avg_bits, group_size=128, config=None, dev='cuda', **kwargs):
        super().__init__(model=model, tokenizer=tokenizer, method=method, arch=arch, avg_bits=avg_bits, group_size=group_size, config=config, dev=dev)
        self.method = 'owq'

    @torch.no_grad
    def run(
        self,
        samples=None,
        n_samples=128,
        target_bits=0.1,
        true_sequential=True,
        percdamp=.01,
        act_order=False,
        static_groups=False, 
        no_frob_norm=False,
        nsamples=128,
        seqlen=2048,
        **kwargs
    ):

        sym = kwargs.get('sym', False)

        print('Running OWQ...')

        if self.group_size > 0:
            self.avg_bits -= 32 / self.group_size        ## OWQ 연산 방식에서 avg_bit로 연산하기 때문에 group_size로 추가되는 비트를 제거
        
        model_name = self.model.model.config.name_or_path.lower()
        meta = processing_meta(model_name)
        
        assert self.arch is not None, "arch is not provided"

        if samples is None:
            # Using GPTQ calibration dataset for OWQ
            samples = get_owq_calib_dataset(data='wikitext2', tokenizer=self.tokenizer, n_samples=nsamples, seqlen=seqlen)

        print('Starting ...')

        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        if 'llama' in model_name or 'qwen2.5' in model_name or 'mistral' in model_name:
            layers = self.model.model.layers
        elif 'opt' in model_name:
            layers = self.model.model.decoder.layers
        else:
            raise NotImplementedError(f"{model_name} model is not implemented.")

        layers[0] = layers[0].to(self.dev)

        if 'llama' in model_name or 'qwen2.5' in model_name or 'mistral' in model_name:
            self.model.model.embed_tokens = self.model.model.embed_tokens.to(self.dev)
            self.model.model.norm = self.model.model.norm.to(self.dev)
            if hasattr(self.model.model, 'rotary_emb'):
                self.model.model.rotary_emb = self.model.model.rotary_emb.to(self.dev)
        elif 'opt' in model_name:
            self.model.model.decoder.embed_tokens = self.model.model.decoder.embed_tokens.to(self.dev)
            self.model.model.decoder.final_layer_norm = self.model.model.decoder.final_layer_norm.to(self.dev)
            self.model.model.decoder.embed_positions = self.model.model.decoder.embed_positions.to(self.dev)
        else:
            raise NotImplementedError(f"{model_name} model is not implemented.")

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
        for batch in samples:
            try:
                self.model(batch[0].to(self.dev))
            except ValueError:
                pass
        layers[0] = layers[0].module

        layers[0] = layers[0].cpu()
        if 'llama' in model_name or 'qwen2.5' in model_name or 'mistral' in model_name:
            self.model.model.embed_tokens = self.model.model.embed_tokens.cpu()
            self.model.model.norm = self.model.model.norm.cpu()
            if hasattr(self.model.model, 'rotary_emb'):
                self.model.model.rotary_emb = self.model.model.rotary_emb.cpu()
        elif 'opt' in model_name:
            self.model.model.decoder.embed_tokens = self.model.model.decoder.embed_tokens.cpu()
            self.model.model.decoder.final_layer_norm = self.model.model.decoder.final_layer_norm.cpu()
            self.model.model.decoder.embed_positions = self.model.model.decoder.embed_positions.cpu()
        torch.cuda.empty_cache()

        outs = torch.zeros_like(inps)
        del cache['i']
        inp_kwargs = cache

        print('Ready.')

        owq_layers = meta['owq_layers']
        ratios = meta['ratios']
        n_out_dict = {l:0 for l in owq_layers.keys()}
        n_owq_layers = sum(owq_layers.values())
        r = (12 / (16 - self.avg_bits)) * 0.1
        r /= n_owq_layers
        layer = self.get_named_linears(layers[0])

        for l in owq_layers:
            n_out = round(layer[l].weight.data.shape[1] * r * ratios[l])
            if n_out % 2 == 1: n_out += 1
            n_out_dict[l] = n_out

        quantizers = {}
        for i in range(len(layers)):
            layer = layers[i].to(self.dev)
            full = self.get_named_linears(layer)

            if true_sequential:
                if 'llama' in model_name or 'qwen2.5' in model_name or 'mistral' in model_name:
                        sequential = [
                            ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                            ['self_attn.o_proj'],
                            ['mlp.up_proj', 'mlp.gate_proj'],
                            ['mlp.down_proj']
                        ]
                elif 'opt' in model_name:
                    sequential = [
                        ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                        ['self_attn.out_proj'],
                        ['fc1'],
                        ['fc2']
                    ]
                else:
                    raise NotImplementedError
            else:
                sequential = [list(full.keys())]
        
            for names in sequential:
                subset = {n: full[n] for n in names}

                gptq_owq = {}
                for name in subset:
                    gptq_owq[name] = GPTQ_OWQ(subset[name], n_out=n_out_dict[name])
                    gptq_owq[name].quantizer = Quantizer(
                        round(self.arch['linear'][name][i]), perchannel=True, sym=sym, mse=True
                    )
                    gptq_owq[name].quantizer.n_out = n_out_dict[name]

                def add_batch(name):
                    def tmp(_, inp, out):
                        gptq_owq[name].add_batch(inp[0].data, out.data)
                    return tmp
                handles = []
                for name in subset:
                    handles.append(subset[name].register_forward_hook(add_batch(name)))
                for j in range(n_samples):
                    outs[j] = layer(inps[j].unsqueeze(0), **inp_kwargs)[0]
                for h in handles:
                    h.remove()

                for name in subset:
                    if not no_frob_norm:        ## defualt: False
                        W = subset[name].weight.data.clone().to(torch.float)
                        temp_quantizer = Quantizer(
                            round(self.arch['linear'][name][i]), perchannel=True, sym=sym, mse=True
                        )
                        temp_quantizer.find_params(W, weight=True, num=40)
                        W_quant = temp_quantizer.quantize(W)
                        frob_norm_error = (W - W_quant).pow(2).sum(dim=0)
                    else:
                        frob_norm_error = None
                    out_ids = gptq_owq[name].hessian_sorting(actorder=act_order, frob_norm=frob_norm_error)
                    gptq_owq[name].quantizer.out_ids = out_ids
                
                if not no_frob_norm:
                    del W
                    del W_quant
                    del temp_quantizer
                    torch.cuda.empty_cache()

                for name in subset:
                    print(f"Quantizing {meta['prefix']}.{i}.{name}")
                    gptq_owq[name].fasterquant(percdamp=percdamp, groupsize=self.group_size, actorder=act_order)
                    quantizers[f"{meta['prefix']}.{i}.{name}"] = gptq_owq[name].quantizer
                    gptq_owq[name].free()

            for name in list(full.keys()):
                quantizers[f"{meta['prefix']}.{i}.{name}"] = quantizers[f"{meta['prefix']}.{i}.{name}"].cpu()

            for j in range(n_samples):
                outs[j] = layer(inps[j].unsqueeze(0), **inp_kwargs)[0]

            layers[i] = layer.cpu()
            del layer
            del gptq_owq 
            torch.cuda.empty_cache()

            inps, outs = outs, inps

        self.model.config.use_cache = use_cache
        
        return quantizers


class GPTQ_OWQ:
    def __init__(self, layer, n_out):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()

        if isinstance(self.layer, nn.Conv2d): 
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        
        self.rows = W.shape[0]
        self.columns = W.shape[1] 
        self.H = torch.zeros((self.columns, self.columns), device=self.dev) 
        self.nsamples = 0 

        self.n_out = n_out
        self.n_nonout = W.shape[1] - n_out
        self.owq = n_out > 0
        self.out_quantizer = None
        self.ids = None
    
    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0] 
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t() 
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        
        
    def hessian_sorting(self, actorder=False, frob_norm=None):
        H = self.H

        if not self.owq:
            if actorder:
                self.ids = torch.argsort(torch.diag(H), descending=True)
            return torch.tensor([])
        
        temp_mask = torch.full([self.columns], True, device=self.dev)
        
        H_diag = torch.diag(H)
        if frob_norm is not None:
            H_diag *= frob_norm
        descending_ids = torch.argsort(H_diag, descending=True)
        
        temp_mask[descending_ids[:self.n_out]] = False
        if actorder:
            ids = torch.cat([descending_ids[self.n_out:],descending_ids[:self.n_out]])
        else:
            ids = torch.cat([torch.arange(self.columns, device=self.dev)[temp_mask], descending_ids[:self.n_out]])
        
        self.ids = ids
        return torch.sort(descending_ids[:self.n_out])[0].to(torch.int32)

    
    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        
        tick = time.time()
        
        if actorder or self.owq:
            W = W[:, self.ids]
            self.H = self.H[self.ids][:,self.ids]
        
        self.quantizer.find_params(W[:,:self.n_nonout], weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0 

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)
        
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp 
        
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        
        for i1 in range(0, self.n_nonout, blocksize):
            i2 = min(i1 + blocksize, self.n_nonout)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(W[:, (i1 + i):min((i1 + i + groupsize),(self.columns-self.n_out))], weight=True, num=40)

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d       
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        # print('time %.2f' % (time.time() - tick))
        # print('error', torch.sum(Losses).item())
               
        if actorder or self.owq:
            Q[:,self.n_nonout:] = W[:,self.n_nonout:]
            invids = torch.argsort(self.ids)
            Q = Q[:, invids]
        
        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)


    def free(self):
        self.H = None
        self.Losses = None
        self.ids = None
        torch.cuda.empty_cache()


class Quantizer(nn.Module):
    def __init__(
            self,
            bits, perchannel=False, sym=False, 
            mse=False, norm=2.4, 
        ):
        super(Quantizer, self).__init__()
        self.register_buffer('scale', torch.zeros(1))
        self.register_buffer('zero', torch.zeros(1))
        self.register_buffer('out_ids', torch.zeros(1))
        
        self.bits = bits
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.perchannel = perchannel
        self.n_levels = 2 ** bits
        
        if self.sym:
            self.minq, self.maxq = -((self.n_levels - 1) // 2 + 1), (self.n_levels - 1) // 2
        else:
            self.minq, self.maxq = 0, self.n_levels - 1
        
        self.num = 100
        self.eps = torch.tensor(1e-8)
        
    def lp_loss(self, pred, tgt, p=2.0):
        x = (pred - tgt).abs().pow(p)
        if not self.perchannel:
            return x.mean()
        else:
            y = torch.flatten(x, 1)
            return y.mean(1)
    
    def find_params(self, x, weight=False, num=100):
        self.num = num
        dev = x.device
        minq, maxq = self.minq, self.maxq
        
        shape = x.shape
        if self.perchannel: # row-wise
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)
        
        if self.mse:
            if self.perchannel:
                new_shape = [-1] + [1] * (len(x.shape) -  1)
            
            best_score = torch.zeros_like(xmin) + (1e+10)
            best_min = xmin.clone()
            best_max = xmax.clone()
            
            if self.sym:
                xrange = torch.max(xmin.abs(), xmax)
                zero = torch.zeros_like(xmin)
                if self.perchannel:
                    zero = zero.reshape(new_shape)
                for i in range(1, self.num + 1):
                    tmp_max = xrange / self.num * i
                    scale = torch.max(tmp_max / -minq, self.eps)
                    if self.perchannel:
                        scale = scale.reshape(new_shape)
                    x_round = torch.round(x / scale)
                    x_q = quantize_efficient(x_round, scale, zero, minq, maxq)
                    score = self.lp_loss(x, x_q, 2.4)
                    best_max = torch.where(score < best_score, tmp_max, best_max)
                    best_score = torch.min(score, best_score)
                
                max_val = torch.max(best_max, torch.zeros_like(best_max))

                self.scale = torch.max(max_val / -minq, self.eps)
                self.zero = torch.zeros_like(self.scale)
            else:
                xrange = xmax - xmin
                tmp_min = torch.zeros_like(xmin)
                for i in range(1, self.num + 1):
                    tmp_max = xrange / self.num * i
                    scale = torch.max((tmp_max - tmp_min) / (maxq - minq), self.eps)
                    delta = scale.clone()
                    if self.perchannel:
                        scale = scale.reshape(new_shape)
                    x_round = torch.round(x / scale)
                    for zp in range(0, self.n_levels):
                        new_min = tmp_min - zp * delta
                        new_max = tmp_max - zp * delta
                        zero = torch.clamp(minq - torch.round(new_min / delta), minq, maxq)
                        if self.perchannel:
                            zero = zero.reshape(new_shape)
                        x_q = quantize_efficient(x_round, scale, zero, minq, maxq)
                        score = self.lp_loss(x, x_q, 2.4)
                        best_min = torch.where(score < best_score, new_min, best_min)
                        best_max = torch.where(score < best_score, new_max, best_max)
                        best_score = torch.min(best_score, score)
            
                min_val_neg = torch.min(best_min, torch.zeros_like(best_min))
                max_val_pos = torch.max(best_max, torch.zeros_like(best_max))

                self.scale = torch.max((max_val_pos - min_val_neg) / (maxq - minq), self.eps)
                self.zero = torch.clamp(minq - torch.round(min_val_neg / self.scale), minq, maxq)
        else:
            if self.sym:
                xmax = torch.maximum(torch.abs(xmin), xmax)
                tmp = xmin < 0
                if torch.any(tmp):
                    xmin[tmp] = -xmax[tmp]

            tmp = (xmin == 0) & (xmax == 0) 
            xmin[tmp] = -1
            xmax[tmp] = +1

            if self.sym:
                self.scale = xmax / -minq
                self.zero = torch.zeros_like(self.scale)
            else:
                self.scale = (xmax - xmin) / maxq
                self.zero = torch.round(-xmin / self.scale)
        
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1)) 
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.minq, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)