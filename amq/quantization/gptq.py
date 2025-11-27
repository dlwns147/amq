import math
import time

from .base import BASE, get_gptq_calib_dataset

import torch
import torch.nn as nn
import numpy as np
import transformers
from accelerate import dispatch_model

DEBUG = False

# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


class GPTQ(BASE):
    def __init__(self, model_name, config, arch, device_map, group_size, dtype='auto', dev='cuda', prune=False, do_owq=False, outlier_path=None, **kwargs):
        super().__init__(model_name, config, arch, device_map=device_map, dtype=dtype, group_size=group_size, dev=dev, prune=prune, do_owq=do_owq, outlier_path=outlier_path)
        self.method = 'gptq'

        if do_owq:
            print('GPTQ is not supported owq.')
            exit()

    @torch.no_grad
    def run(
        self,
        samples=None,
        sym=False,
        nsamples=128,
        true_sequential=False,
        percdamp=.01,
        act_order=False,
        static_groups=False,
        seqlen=2048
    ):
        
        assert self.arch is not None, "arch is not provided"
       
        if samples is None:
            samples = get_gptq_calib_dataset(tokenizer=self.tokenizer, n_samples=nsamples, seqlen=seqlen)

        print('Starting ...')

        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        layers = self.model.model.layers

        layers[0] = layers[0].to(self.dev)
        self.model.model.embed_tokens = self.model.model.embed_tokens.to(self.dev)
        self.model.model.norm = self.model.model.norm.to(self.dev)
        if hasattr(self.model.model, 'rotary_emb'):
            self.model.model.rotary_emb = self.model.model.rotary_emb.to(self.dev)

        dtype = next(iter(self.model.parameters())).dtype
        inps = torch.zeros(
            (nsamples, seqlen, self.model.config.hidden_size), dtype=dtype, device=self.dev
        )
        cache = {'i': 0, 'attention_mask': None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp
                cache['i'] += 1
                cache['attention_mask'] = kwargs['attention_mask']
                cache['position_ids'] = kwargs['position_ids']
                raise ValueError
        layers[0] = Catcher(layers[0])
        for batch in samples:
            try:
                self.model(batch[0].to(self.dev))
            except ValueError:
                pass
        layers[0] = layers[0].module

        layers[0] = layers[0].cpu()
        self.model.model.embed_tokens = self.model.model.embed_tokens.cpu()
        self.model.model.norm = self.model.model.norm.cpu()
        if hasattr(self.model.model, 'rotary_emb'):
            self.model.model.rotary_emb = self.model.model.rotary_emb.cpu()
        torch.cuda.empty_cache()

        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']
        position_ids = cache['position_ids']

        print('Ready.')

        quantizers = {}
        for i in range(len(layers)):
            layer = layers[i].to(self.dev)
            full = self.get_named_linears(layer)

            if true_sequential:
                sequential = [
                    ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                    ['self_attn.o_proj'],
                    ['mlp.up_proj', 'mlp.gate_proj'],
                    ['mlp.down_proj']
                ]
            else:
                sequential = [list(full.keys())]
        
            for names in sequential:
                subset = {n: full[n] for n in names}

                gptq = {}
                for name in subset:
                    gptq[name] = GPTQ_METHOD(subset[name])
                    gptq[name].quantizer = Quantizer()
                    # gptq[name].quantizer.configure(
                    #     args.wbits, perchannel=True, sym=args.sym, mse=False
                    # )
                    gptq[name].quantizer.configure(
                        round(self.arch['linear'][name][i]), perchannel=True, sym=sym, mse=False
                    )

                def add_batch(name):
                    def tmp(_, inp, out):
                        gptq[name].add_batch(inp[0].data, out.data)
                    return tmp
                handles = []
                for name in subset:
                    handles.append(subset[name].register_forward_hook(add_batch(name)))
                for j in range(nsamples):
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                for h in handles:
                    h.remove()

                for name in subset:
                    print(i, name)
                    print('Quantizing ...')
                    # gptq[name].fasterquant(
                    #     percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
                    # )
                    gptq[name].fasterquant(
                        percdamp=percdamp, groupsize=self.group_size, actorder=act_order, static_groups=static_groups
                    )
                    quantizers['self.model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                    gptq[name].free()

            for j in range(nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            # outs = layer(inps, attention_mask=attention_mask, position_ids=position_ids)[0]

            layers[i] = layer.cpu()
            del layer
            del gptq 
            torch.cuda.empty_cache()

            inps, outs = outs, inps

        self.model.config.use_cache = use_cache
        
        return quantizers


class GPTQ_METHOD:
    def __init__(self, layer):
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

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
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
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        assert groupsize != 0, 'Groupsize must be non-zero'
        print(f'gptq groupsize : {groupsize}')

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
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
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = quantize(
                    w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                ).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if actorder:
            Q = Q[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()


class Quantizer(nn.Module):
    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True, 
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        trits=False
    ):
        # print(bits)
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink 
        if trits:
            self.maxq = torch.tensor(-1) 

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
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

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
          self.scale = xmax
          self.zero = xmin
        else:
          self.scale = (xmax - xmin) / self.maxq
          if self.sym:
              self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
          else:
              self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid 
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
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
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)