import numpy as np
import torch
import torch.nn as nn
import os
# import nvtx
# from qeft_utils.reorder import sparse_to_dense_ids
from .reorder import sparse_to_dense_ids

try:
    import qeft_cuda
except:
    print('QEFT CUDA kernel extension is not installed.')

class QuantMatMulQEFT(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x, oweight, qweight, scales, scaled_zeros, n_out, bias, name):
        dtype = scales.dtype
        x = x.to(dtype)
        x_outlier = x[...,-n_out:]
        y = qeft_cuda.gemm_4bit(x, qweight, scales, scaled_zeros)
        y += torch.nn.functional.linear(x_outlier, oweight.to(dtype), bias)
        
        ctx.dequant_params = [oweight, qweight, scales, scaled_zeros, n_out, dtype, name]
        ctx.tensors = x_outlier
        
        return y

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        x_outlier = ctx.tensors
        oweight, qweight, scales, scaled_zeros, n_out, dtype, name = ctx.dequant_params

        grad_input = None
        grad_oweight = None
        
        if ctx.needs_input_grad[0]:
            grad_input = qeft_cuda.gemm_4bit(grad_output.to(dtype), qweight, scales, scaled_zeros)
            grad_input += torch.nn.functional.linear(grad_output[...,-n_out:].to(dtype), oweight.to(dtype))
        if ctx.needs_input_grad[1]:
            grad_oweight = torch.matmul(grad_output.transpose(-2,-1), x_outlier.to(grad_output.dtype))
            grad_oweight = grad_oweight.transpose(-1, -2)

        return grad_input, grad_oweight, None, None, None, None, None, None

class QuantMatMul(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x, qweight, scales, scaled_zeros, n_out, bias, name):
        dtype = scales.dtype
        y = qeft_cuda.gemm_4bit(x.to(dtype), qweight, scales, scaled_zeros)
        y = y + bias if bias is not None else y
        
        ctx.dequant_params = [qweight, scales, scaled_zeros, n_out, dtype, name]
        
        return y

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        qweight, scales, scaled_zeros, n_out, dtype, name = ctx.dequant_params

        grad_input = None
        
        if ctx.needs_input_grad[0]:
            grad_input = qeft_cuda.gemm_4bit(grad_output.to(dtype), qweight, scales, scaled_zeros)

        return grad_input, None, None, None, None, None, None

def pack_oweight(oweight, interleave=4):
    new_oweight = []
    for i in range(0, oweight.shape[0], 2*interleave):
        for j in range(interleave):
            new_row = []
            for k in range(0, oweight.shape[1], 32): # 128
                new_row.append(torch.stack([oweight[i+j,k:k+32], oweight[i+j+interleave,k:k+32]], dim=0).t().flatten())
            new_oweight.append(torch.cat(new_row, dim=0))

    return torch.stack(new_oweight, dim=0)
    
def pack_intweight(unpacked_qweight, interleave, kstride):
    # unpacked_qweight: [N, K]
    N = unpacked_qweight.shape[0]
    K = unpacked_qweight.shape[1]

    Packed_Kernel = unpacked_qweight.cpu().numpy().reshape(N, K // 32, 32)
    # np.arange(32).reshape(4, 4, 2).transpose(1, 0, 2) => [0, 1, 8, 9, 16, 17, 24, 25, ...]
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 3, 2, 4)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 32)

    # reorder each 8 weights for fast dequantization
    # [0, 1, 2, 3, 4, 5, 6, 7] => [0, 2, 4, 6, 1, 3, 5, 7]
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 8)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 2, 4, 3)
    Packed_Kernel = Packed_Kernel.reshape(N, K)

    # interleaving every four rows
    Packed_Kernel = Packed_Kernel.reshape(
        N // interleave, interleave, K // kstride, kstride
    )
    # N // 4, K // 64, 4, 64
    Packed_Kernel = Packed_Kernel.transpose(0, 2, 1, 3)
    Packed_Kernel = Packed_Kernel.reshape(
        N // interleave, K // kstride, kstride, interleave
    )
    # Packing -> (N // 4, K // 64, 64)

    Packed_Kernel = (
        Packed_Kernel[..., 0]
        | (Packed_Kernel[..., 1] << 4)
        | (Packed_Kernel[..., 2] << 8)
        | (Packed_Kernel[..., 3] << 12)
    )
    # reshape to (N // 4, K), FP16 format
    Packed_Kernel = Packed_Kernel.reshape(N // interleave, K)
    qweight = (
        torch.tensor(Packed_Kernel.astype("int16"))
        .to(unpacked_qweight.device)
        .contiguous()
    )
    return qweight

class QuantLinear(nn.Module):

    def __init__(self, bits, infeatures, outfeatures, bias, dtype, outlierfeatures, group_size, reorder, name): # TODO
        super().__init__()
        assert bits in [4], "Only 4 bits is supported."
        assert dtype == torch.float16, "Only fp16 is supported."
        # assert group_size == 128, "Only group 128 is supported."
        
        self.bits = bits
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.outlierfeatures = outlierfeatures
        
        self.group_size = group_size if group_size != -1 else infeatures
        self.interleave = 4
        
        assert infeatures % group_size == 0
        assert outfeatures % (32 // self.bits) == 0
        int16_pack_num = 16 // self.bits # 4
        
        self.register_buffer(
            'qweight', torch.empty(
                (
                    outfeatures // self.interleave,
                    infeatures // int16_pack_num * self.interleave,
                ),
                dtype=torch.int16
            ),
        )
        
        numgroup = infeatures // group_size if group_size > 0 else 1
        
        self.register_buffer('scales', torch.empty((numgroup, outfeatures), dtype=dtype))
        self.register_buffer('scaled_zeros', torch.empty((numgroup, outfeatures), dtype=dtype))

        if bias:
            self.register_buffer("bias", torch.empty((outfeatures), dtype=torch.float16))
        else:
            self.bias = None
        
        # for weak columns
        if outlierfeatures > 0:
            self.register_buffer(
                'oweight', torch.empty((outfeatures, outlierfeatures), dtype=dtype)
            )
            self.register_buffer(
                'oweight_interleaved', torch.empty((outfeatures // 2, outlierfeatures * 2), dtype=dtype)
            )
            self.register_buffer(
                'outlieridx', torch.zeros((outlierfeatures), dtype=torch.int)
            )
        
        self.faster = True
        self.dtype = dtype
        self.name = name
        self.training = False
        
    def pack(self, linear, scales, zeros, outlieridx:torch.Tensor, sym:bool=False):
        dtype = self.dtype
        
        self.sym = sym
        if sym:
            zeros += 2**(self.bits - 1)
            
        if linear.bias is not None:
            self.bias = linear.bias.to(dtype)
        
        # [OC, IC // g]
        # 1. pack qweight
        scale_zeros = zeros * scales
        num_interleave = 1 if self.group_size == self.infeatures else self.group_size
        scales_interleave = torch.repeat_interleave(scales, num_interleave, dim=1)
        scale_zeros_interleave = torch.repeat_interleave(scale_zeros, num_interleave, dim=1)
        
        intweight = torch.round((linear.weight.data + scale_zeros_interleave) / scales_interleave).to(torch.int)
        intweight = intweight.to(dtype=torch.int32)
        
        if self.outlierfeatures > 0:
            for i in range(self.infeatures - self.outlierfeatures, self.infeatures):
                intweight[:, i] = zeros[:, i // self.group_size].to(torch.int32)
        self.qweight = pack_intweight(intweight, interleave=4, kstride=64)
        
        # [IC // g, OC]
        # 2. save scales, scaled_zeros
        self.scales = scales.t().contiguous().to(dtype)
        self.scaled_zeros = -scale_zeros.t().contiguous().to(dtype)
        
        # 3. save oweight if format == qeft
        if self.outlierfeatures > 0:
            oweight = linear.weight.data[:,-self.outlierfeatures:].clone()
            self.oweight = oweight
            self.oweight_interleaved = pack_oweight(oweight, interleave=4)
            self.outlieridx = outlieridx

    def set_kernel(self, training=False):
        # set kernel for corresponding format
        self.training = training
        if self.outlierfeatures > 0:
            if self.oweight.shape[1] % 64 > 0: # padding oweight
                self.oweight = torch.cat([torch.zeros((self.oweight.shape[0],64 - self.oweight.shape[1] % 64), dtype=self.oweight.dtype, device=self.oweight.device), self.oweight], dim=-1)
            self.gemv = qeft_cuda.gemv_4bit_qeft
            self.gemm = qeft_cuda.gemm_4bit
            self.forward = self.forward_outlier
            # 5. if name == out_proj, use dynamic reordering
            if 'o_proj' in self.name or 'out_proj' in self.name:
                self.register_buffer('reorder_ids', sparse_to_dense_ids(self.outlieridx, self.infeatures))
                self.forward = self.forward_outlier_out_proj
            if training:
                self.matmul = QuantMatMulQEFT.apply
        else:
            self.gemv = qeft_cuda.gemv_4bit
            self.gemm = qeft_cuda.gemm_4bit
            self.forward = self.forward_normal
            if training:
                self.matmul = QuantMatMul.apply
    
    def set_for_wct(self):
        self.qweight = torch.nn.Parameter(self.qweight, requires_grad=False)
        if self.outlierfeatures > 0:
            self.oweight = torch.nn.Parameter(self.oweight.to(dtype=torch.float), requires_grad=True)
    
    def forward_outlier(self, x):
        if self.training:
            y = self.matmul(x, self.oweight, self.qweight, 
                        self.scales, self.scaled_zeros, 
                        self.outlierfeatures, self.bias, 
                        self.name)
        else:
            seq_len = x.numel() // x.shape[-1]
            if seq_len < 8:
                y = self.gemv(
                    x,
                    self.qweight,
                    self.scales,
                    self.scaled_zeros,
                    self.oweight_interleaved,
                    seq_len,
                    self.outfeatures,
                    self.infeatures,
                    self.group_size,
                )
            else:
                y = self.gemm(x, self.qweight, self.scales, self.scaled_zeros)
                y += torch.nn.functional.linear(x[...,-self.outlierfeatures:], self.oweight)
            
            y = y + self.bias if self.bias is not None else y
        # import code; code.interact(f'{self.name} forward_outlier', local=dict(globals(), **locals()))
        # nvtx.pop_range()
        return y
    
    def forward_outlier_out_proj(self, x):
        # dynamic reordering
        inputs = torch.index_select(x, -1, self.reorder_ids)
        # nvtx.push_range(f'{self.name} forward seq_len : {seq_len}')
        if self.training:
            y = self.matmul(inputs, self.oweight, self.qweight, 
                        self.scales, self.scaled_zeros, 
                        self.outlierfeatures, self.bias, 
                        self.name)
        else:
            seq_len = inputs.numel() // inputs.shape[-1]
            if seq_len < 8:
                y = self.gemv(
                    inputs,
                    self.qweight,
                    self.scales,
                    self.scaled_zeros,
                    self.oweight_interleaved,
                    seq_len,
                    self.outfeatures,
                    self.infeatures,
                    self.group_size,
                )
            else:
                y = self.gemm(inputs, self.qweight, self.scales, self.scaled_zeros)
                y += torch.nn.functional.linear(inputs[...,-self.outlierfeatures:], self.oweight)
            
            y = y + self.bias if self.bias is not None else y
        # nvtx.pop_range()
        # import code; code.interact(f'{self.name} forward_outlier_out_proj', local=dict(globals(), **locals()))
        return y
    
    def forward_normal(self, x):
        if self.training:
            y = self.matmul(x, self.qweight, self.scales, 
                        self.scaled_zeros, self.outlierfeatures, 
                        self.bias, self.name)
        else:
            seq_len = x.numel() // x.shape[-1]
            # nvtx.push_range(f'{self.name} forward seq_len : {seq_len}')
            if seq_len < 8:
                y = self.gemv(
                    x,
                    self.qweight,
                    self.scales,
                    self.scaled_zeros,
                    seq_len,
                    self.outfeatures,
                    self.infeatures,
                    self.group_size,
                )
            else:
                y = self.gemm(x, self.qweight, self.scales, self.scaled_zeros)
            
            y = y + self.bias if self.bias is not None else y
        # import code; code.interact(f'{self.name} forward_normal', local=dict(globals(), **locals()))
        # nvtx.pop_range()
        return y
