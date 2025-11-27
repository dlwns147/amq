import numpy as np
import torch
import torch.nn as nn
import os

# from qeft.qlinear import QuantLinear
from .qlinear import QuantLinear

def quantize(x, scale, zero, minq, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, minq, maxq)
    return scale * (q - zero)

def quantize_efficient(x_round, scale, zero, minq, maxq):
    q = torch.clamp(x_round + zero, minq, maxq)
    return scale * (q - zero)

class Quantizer(nn.Module):
    def __init__(
            self,
            bits, perchannel=False, sym=False, 
            mse=False, norm=2.4, group_size=-1,
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
        self.group_size = group_size
        
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
        
    def append_params(self):
        if not hasattr(self, 'scale_group'):
            #self.scale.size() : torch.Size([4096, 1])
            self.register_buffer('scale_group', self.scale)
            self.register_buffer('zero_group', self.zero)
        else:
            self.scale_group = torch.cat((self.scale_group, self.scale), 1)
            self.zero_group = torch.cat((self.zero_group, self.zero), 1)
    
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

def make_quant(module, quantinfos, name=''):
    if isinstance(module, (QuantLinear)):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in quantinfos:
            setattr(
                module, attr, 
                QuantLinear(quantinfos[name1].bits, 
                            tmp.in_features, 
                            tmp.out_features, 
                            tmp.bias is not None, 
                            tmp.weight.dtype,
                            getattr(quantinfos[name1],'n_out', 0),
                            getattr(quantinfos[name1],'group_size', -1),
                            getattr(quantinfos[name1],'reorder', False),
                            name1).to(tmp.weight.device)
            )
    for name1, child in module.named_children():
        make_quant(child, quantinfos, name + '.' + name1 if name != '' else name1)

def lm_pack(model, quantinfos, linears=[nn.Linear]):
    from qeft.utils.misc import find_layers
    from tqdm import tqdm
    
    layers = find_layers(model, linears)
    layers = {n: layers[n] for n in quantinfos}
    make_quant(model, quantinfos)
    qlayers = find_layers(model, [QuantLinear])
    for name in tqdm(qlayers, f"Packing ..."):
        quantinfos[name] = quantinfos[name].cpu()
        qlayers[name].pack(
            layers[name], 
            scales = getattr(quantinfos[name], 'scale_group', quantinfos[name].scale), 
            zeros = getattr(quantinfos[name], 'zero_group', quantinfos[name].zero), 
            outlieridx = getattr(quantinfos[name], 'out_ids', None), 
            sym = getattr(quantinfos[name], 'sym', False)
        )
    print('Done.')
    return model