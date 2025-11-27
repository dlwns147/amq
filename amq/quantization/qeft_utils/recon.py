import math
import time

import torch
import torch.nn as nn
import transformers

# from qeft.quant import *
from .quant import *

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class GPTQ_OWQ:
    def __init__(self, layer, n_out) :
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
        
        
    def hessian_sorting(self, actorder=False, frob_norm=None, outidx=None):
        H = self.H

        if not self.owq:
            if actorder:
                self.ids = torch.argsort(torch.diag(H), descending=True)
            else:
                self.ids = torch.arange(self.columns, device=self.dev)
                
            return torch.tensor([])
        
        temp_mask = torch.full([self.columns], True, device=self.dev)
        
        H_diag = torch.diag(H)
        if frob_norm is not None:
            H_diag *= frob_norm
            
        descending_ids = torch.argsort(H_diag, descending=True)
            
        self.H_diag = H_diag
        if outidx == None:
            temp_mask[descending_ids[:self.n_out]] = False
            if actorder:
                ids = torch.cat([descending_ids[self.n_out:],descending_ids[:self.n_out]])
            else:
                ids = torch.cat([torch.arange(self.columns, device=self.dev)[temp_mask], descending_ids[:self.n_out]])
            
            self.ids = ids
            # return descending_ids[:self.n_out].to(torch.int32)
            return torch.sort(descending_ids[:self.n_out])[0].to(torch.int32)
        else:
            outidx = outidx.to(device=self.dev)
            temp_mask[outidx] = False
            if actorder:
                ids = torch.cat([descending_ids[self.n_out:],descending_ids[:self.n_out]])
            else:
                ids = torch.cat([torch.arange(self.columns, device=self.dev)[temp_mask], outidx])
            
            self.ids = ids
            # return torch.sort(outidx)[0].to(torch.int32)
            return outidx.to(torch.int32)

    def hessian_sorting_qkv_frob_norm(self, actorder=False, frob_norm_k=None, frob_norm_q=None, frob_norm_v=None):
        # print("Using hessian_sorting_qkv_frob_norm")
        H = self.H

        if not self.owq:
            if actorder:
                self.ids = torch.argsort(torch.diag(H), descending=True)
            return torch.tensor([])

        temp_mask = torch.full([self.columns], True, device=self.dev)
        H_diag_q = torch.diag(H) * frob_norm_q
        H_diag_k = torch.diag(H) * frob_norm_k
        H_diag_v = torch.diag(H) * frob_norm_v
        descending_ids_q = torch.argsort(H_diag_q, descending=True)
        descending_ids_k = torch.argsort(H_diag_k, descending=True)
        descending_ids_v = torch.argsort(H_diag_v, descending=True)
        descending_ids_all = torch.argsort(torch.cat((H_diag_k, H_diag_q, H_diag_v)), descending=True)

        true_index = []
        index = 0
        while len(true_index) < self.n_out:
            if (descending_ids_all[index] % self.columns) not in true_index:
                true_index.append(descending_ids_all[index] % self.columns)
            index += 1

        #import code; code.interact("temp_mask", local=locals())
        true_index = torch.tensor(true_index, device=self.dev)
        temp_mask[true_index] = False
        if actorder:
            for ele in true_index:
                descending_ids_k = descending_ids_k[descending_ids_k!=ele]
                descending_ids_q = descending_ids_q[descending_ids_q!=ele]
                descending_ids_v = descending_ids_v[descending_ids_v!=ele]
            ids_k = torch.cat([descending_ids_k, true_index])
            ids_q = torch.cat([descending_ids_q, true_index])
            ids_v = torch.cat([descending_ids_v, true_index])
        else:
            ids_k = ids_q = ids_v = torch.cat([torch.arange(self.columns, device=self.dev)[temp_mask], true_index])
        
        return ids_k, ids_q, ids_v, true_index.to(torch.int32)
    
    def hessian_sorting_upgate_frob_norm(self, actorder=False, frob_norm_gate=None, frob_norm_up=None):
        # print("Using hessian_sorting_upgate_frob_norm")
        H = self.H

        if not self.owq:
            if actorder:
                self.ids = torch.argsort(torch.diag(H), descending=True)
            return torch.tensor([])

        temp_mask = torch.full([self.columns], True, device=self.dev)
        H_diag_gate = torch.diag(H) * frob_norm_gate
        H_diag_up = torch.diag(H) * frob_norm_up
        descending_ids_gate = torch.argsort(H_diag_gate, descending=True)
        descending_ids_up = torch.argsort(H_diag_up, descending=True)
        descending_ids_all = torch.argsort(torch.cat((H_diag_gate, H_diag_up)), descending=True)

        true_index = []
        index = 0
        while len(true_index) < self.n_out:
            if (descending_ids_all[index] % self.columns) not in true_index:
                true_index.append(descending_ids_all[index] % self.columns)
            index += 1

        #import code; code.interact("temp_mask", local=locals())
        true_index = torch.tensor(true_index, device=self.dev)
        temp_mask[true_index] = False
        if actorder:
            for ele in true_index:
                descending_ids_gate = descending_ids_gate[descending_ids_gate!=ele]
                descending_ids_up = descending_ids_up[descending_ids_up!=ele]
            ids_gate = torch.cat([descending_ids_gate, true_index])
            ids_up = torch.cat([descending_ids_up, true_index])
        else:
            ids_gate = ids_up = torch.cat([torch.arange(self.columns, device=self.dev)[temp_mask], true_index])
        
        return ids_gate, ids_up, true_index.to(torch.int32)
    
    def lora_reconstruction(self, lora_full, blocksize=128, percdamp=.01):
        W = self.layer.weight.data.clone()
        L = lora_full.clone()

        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if self.owq:
            W = W[:, self.ids]
            L = L[:, self.ids]
            self.H = self.H[self.ids][:,self.ids]

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
            L1 = L[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                l = L1[:, i]

                #q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                q = w - l
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
               
        if self.owq:
            Q[:,self.n_nonout:] = W[:,self.n_nonout:]
            invids = torch.argsort(self.ids)
            Q = Q[:, invids]
        
        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def fasterquant_nearest_owq(
        self, blocksize=128, groupsize=-1, actorder=False,
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

        self.quantizer.find_params(W[:,:self.n_nonout], weight=True)
        num = 40 if self.quantizer.mse else 1
        Q = torch.zeros_like(W)

        group_count = 0
        start_group = 0
        end_group = 0

        if groupsize == -1:
            W1 = W[:, :self.n_nonout].clone()
            Q1 = torch.zeros_like(W1)
            Q1 = self.quantizer.quantize(W1)
            Q[:, :self.n_nonout] = Q1
        else:
            groupnum = self.columns // groupsize
            if self.n_out > 0:
                group_divide = [torch.sum((self.ids[-self.n_out:].sort()[0] // groupsize) == i).item() for i in range(groupnum)]
            else:
                group_divide = [0] * groupnum
        
            while group_count < len(group_divide):
                end_group = start_group + groupsize - group_divide[group_count]
                if start_group == end_group:
                    self.quantizer.find_params(W[:, start_group-1:end_group], weight=True, num=40)
                    self.quantizer.append_params()
                    group_count += 1
                    end_group = start_group + groupsize - group_divide[group_count]

                W1 = W[:, start_group:end_group].clone()
                Q1 = torch.zeros_like(W1)

                self.quantizer.find_params(W1, weight=True, num=num)
                self.quantizer.append_params()

                Q1 = self.quantizer.quantize(W1)
                Q[:, start_group:end_group] = Q1
                
                start_group = end_group
                group_count += 1
            
        torch.cuda.synchronize()
        # print('time %.2f' % (time.time() - tick))

        if actorder or self.owq:
            Q[:,self.n_nonout:] = W[:,self.n_nonout:]
            invids = torch.argsort(self.ids)
            Q = Q[:, invids]
        
        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def fasterquant_nearest_owq_reorder(
        self, blocksize=128, groupsize=-1, actorder=False,
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

        self.quantizer.find_params(W[:,:self.n_nonout], weight=True)
        Q = torch.zeros_like(W)

        group_count = 0
        start_group = 0
        end_group = 0
        
        if groupsize == -1:
            W1 = W[:, :self.n_nonout].clone()
            Q1 = torch.zeros_like(W1)
            Q1 = self.quantizer.quantize(W1)
            Q[:, :self.n_nonout] = Q1
        else:
            groupnum = self.columns // groupsize
        
            while group_count < groupnum:
                end_group = min(start_group + groupsize, self.n_nonout)
                if start_group == end_group:
                    self.quantizer.append_params()
                    break
                W1 = W[:, start_group:end_group].clone()
                self.quantizer.find_params(W1, weight=True, num=1)

                self.quantizer.append_params()

                Q1 = self.quantizer.quantize(W1)
                Q[:, start_group:end_group] = Q1
                
                start_group = end_group
                group_count += 1
            
        torch.cuda.synchronize()
        # print('time %.2f' % (time.time() - tick))

        if actorder or self.owq:
            Q[:,self.n_nonout:] = W[:,self.n_nonout:]
            invids = torch.argsort(self.ids)
            Q = Q[:, invids]
        
        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)


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

        group_count = 0
        start_group = 0
        end_group = 0
        if groupsize != -1:
            groupnum = self.columns // groupsize
            if self.n_out > 0:
                group_divide = [torch.sum((self.ids[-self.n_out:].sort()[0] // groupsize) == i).item() for i in range(groupnum)]
            else:
                group_divide = [0] * groupnum
        
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
                    if (i1 + i == start_group):
                        end_group = start_group + groupsize - group_divide[group_count]
                        if start_group == end_group:
                            group_count += 1
                            end_group = start_group + groupsize - group_divide[group_count]
                            self.quantizer.find_params(W[:, start_group:end_group], weight=True, num=40)
                            self.quantizer.append_params()
                        self.quantizer.find_params(W[:, start_group:end_group], weight=True, num=40)
                        self.quantizer.append_params()
                        start_group = end_group
                        group_count += 1

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


    def fasterquant_reorder(
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
                        self.quantizer.find_params(W[:, i1 + i: min(i1 + i + groupsize, self.n_nonout)], weight=True, num=40)
                        self.quantizer.append_params()

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
        
        temp = W.shape[1] // groupsize - self.quantizer.zero_group.shape[-1]
        if temp > 0:
            for _ in range(temp):
                self.quantizer.append_params()

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
