import numpy as np
import torch
import torch.nn as nn
import os
import time

from ..core.quantize import HQQLinear, Quantizer
from ..core.peft import HQQLinearLoRA

try:
    import faster_transformer as ft
except:
    print('FT CUDA kernel extension is not installed.')
    
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

class FT_QuantLinear(nn.Module):

    def __init__(self, bits, infeatures, outfeatures, bias, dtype, group_size, name): # TODO
        super().__init__()
        assert bits in [4], "Only 4 bits is supported."
        assert dtype == torch.float16, "Only fp16 is supported."
        # assert group_size == 128, "Only group 128 is supported."
        
        self.bits = bits
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        
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
        
        self.dtype = dtype
        self.name = name

        self.gemv = ft.gemv_4bit
        self.gemm = ft.gemm_4bit
        self.forward = self.forward_normal
        
    def pack(self, weight, scales, zeros, sym:bool=False):
        dtype = self.dtype
        
        self.sym = sym
        if sym:
            zeros += 2**(self.bits - 1)

        
        # [OC, IC // g]
        # 1. pack qweight
        scale_zeros = zeros * scales
        num_interleave = 1 if self.group_size == self.infeatures else self.group_size
        scales_interleave = torch.repeat_interleave(scales, num_interleave, dim=1)
        scale_zeros_interleave = torch.repeat_interleave(scale_zeros, num_interleave, dim=1)
        
        intweight = torch.round((weight.data + scale_zeros_interleave) / scales_interleave).to(torch.int)
        intweight = intweight.to(dtype=torch.int32)
        
        self.qweight = pack_intweight(intweight, interleave=4, kstride=64)
        
        # [IC // g, OC]
        # 2. save scales, scaled_zeros
        self.scales = scales.t().contiguous().to(dtype)
        self.scaled_zeros = -scale_zeros.t().contiguous().to(dtype)
        
    def forward_normal(self, x):
        seq_len = x.numel() // x.shape[-1]
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
        return y


def patch_hqq_to_ft(layer, patch_params, load = False):
    hqq_layer = None
    if type(layer) is HQQLinear:
        hqq_layer = layer
    if type(layer) is HQQLinearLoRA:
        hqq_layer = layer.linear_layer

    if hqq_layer is None:
        return layer

    # hqq_layer = layer.linear_layer if hasattr(layer, "linear_layer") else layer

    device = hqq_layer.device
    nbits = hqq_layer.meta['nbits']
    group_size = hqq_layer.meta['group_size']
    outfeatures, infeatures = hqq_layer.meta['shape']
    bias = hqq_layer.bias

    ft_layer = FT_QuantLinear(nbits, 
                            infeatures, 
                            outfeatures, 
                            bias,
                            torch.float16,
                            group_size,
                            hqq_layer.name).to(device) 

    if not load:
        # TODO: The below process is not optimal because it involves depacking + dequantize.
        W_deq = Quantizer.dequantize(hqq_layer.W_q, hqq_layer.meta)
        outfeatures = hqq_layer.meta['shape'][0]
        scales = hqq_layer.meta['scale'].reshape(outfeatures, -1)
        zeros = hqq_layer.meta['zero'].reshape(outfeatures, -1)

        ft_layer.pack(W_deq, scales, zeros, False)

    del hqq_layer.W_q
    del hqq_layer.meta      
    del hqq_layer.bias
    del hqq_layer
    torch.cuda.empty_cache()

    if isinstance(layer, HQQLinear):
        return ft_layer

    if isinstance(layer, HQQLinearLoRA):
        layer.linear_layer = ft_layer

    torch.cuda.empty_cache()

    return layer


def patch_hqq_to_ft_load(layer, patch_params):
    return patch_hqq_to_ft(layer, patch_params, load = True)