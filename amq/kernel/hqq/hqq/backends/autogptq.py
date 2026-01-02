import math
import numpy as np
import torch
import torch.nn as nn
import transformers
from torch import Tensor

from ..core.quantize import HQQLinear, Quantizer
from ..core.peft import HQQLinearLoRA

# import hqq.backends.custom_ops
# from hqq.backends.custom_ops import vecquant3matmul

import time

try:
    # from autogptq_cuda_256 import vecquant2matmul, vecquant3matmul, vecquant4matmul, vecquant2matmul_faster_old, vecquant3matmul_faster_old, vecquant4matmul_faster_old
    import auto_gptq

    _autogptq_cuda_available = True
except ImportError:
    # logger.warning("CUDA extension not installed.")
    print("GPTQ : CUDA extension not installed.")
    auto_gptq = None
    _autogptq_cuda_available = False

class GPTQLinear(nn.Module):
    QUANT_TYPE = "cuda-old"

    def __init__(
        self,
        bits,
        group_size,
        infeatures,
        outfeatures,
        bias,
        use_cuda_fp16=True,
        kernel_switch_threshold=128,
        trainable=False,
        weight_dtype=torch.float16,
        outlierfeatures = 32
    ):
        super().__init__()
        global _autogptq_cuda_available
        if bits not in [2, 3, 4, 8]:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        if trainable:
            _autogptq_cuda_available = False
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2**self.bits - 1

        self.register_buffer(
            "qweight",
            torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32),
        )
        self.register_buffer(
            "zeros",
            torch.zeros(
                (
                    math.ceil(infeatures / self.group_size),
                    outfeatures,
                ),
                dtype=torch.float,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (math.ceil(infeatures / self.group_size), outfeatures),
                dtype=torch.float,
            ),
        )

        self.name = None

        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures), dtype=weight_dtype))
        else:
            self.bias = None
        self.half_indim = self.infeatures // 2

        self.use_cuda_fp16 = use_cuda_fp16 if bits != 8 else False

        # is performed by unpacking the weights and using torch.matmul
        if self.bits in [2, 4, 8]:
            self.wf = torch.tensor(list(range(0, 32, self.bits)), dtype=torch.int32).unsqueeze(0)
        elif self.bits == 3:
            self.wf = torch.tensor(
                [
                    [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
                    [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
                    [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0],
                ],
                dtype=torch.int32,
            ).reshape(1, 3, 12)

        self.kernel_switch_threshold = kernel_switch_threshold
        self.autogptq_cuda_available = _autogptq_cuda_available
        self.autogptq_cuda = auto_gptq

        self.auto_gptq = auto_gptq
        self.trainable = trainable

    def post_init(self):
        pass

    ## 수정본
    def pack(self, W, scales, zeros):
        scale_zeros = zeros * scales
        self.scales = scales.t().contiguous().clone().to(dtype=torch.float)
        self.zeros = scale_zeros.t().contiguous().clone().to(dtype=torch.float)

        num_interleave = 1 if self.group_size == self.infeatures else self.group_size
        scales_interleave = torch.repeat_interleave(scales, num_interleave, dim=1)
        scale_zeros_interleave = torch.repeat_interleave(scale_zeros, num_interleave, dim=1)
        
        intweight = torch.round((W + scale_zeros_interleave) / scales_interleave).to(torch.int)
        intweight = intweight.t().contiguous()
        intweight = intweight.cpu().numpy().astype(np.uint32)

        i = 0
        row = 0
        qweight = np.zeros((intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32)
        while row < qweight.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i))
                i += 10
                qweight[row] |= intweight[i] << 30
                row += 1
                qweight[row] |= (intweight[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row] |= intweight[i] << 31
                row += 1
                qweight[row] |= (intweight[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 2)
                i += 10
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight).to(scales.device)


    def forward(self, x):
        x_dtype = x.dtype
        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.reshape(-1, x.shape[-1])
        if x.shape[0] < self.kernel_switch_threshold:
            out = torch.zeros(x.shape[0], out_shape[-1], dtype=torch.float, device=x.device)
            if self.use_cuda_fp16:
                if x_dtype != torch.float16:
                    print(
                        f"The cuda-old kernel for GPTQ with use_cuda_fp16=True requires a float16 input activation, while {x_dtype} was passed. Casting to float16.\nMake sure you loaded your model with torch_dtype=torch.float16, that the model definition does not inadvertently cast to float32, or disable AMP Autocast that may produce float32 intermediate activations in the model."
                    )

                if self.bits == 2:
                    self.auto_gptq.vecquant2matmul_faster_old(
                        x,
                        self.qweight,
                        out,
                        self.scales,
                        self.zeros,
                        self.group_size,
                        self.half_indim,
                    )
                elif self.bits == 3:
                    self.auto_gptq.vecquant3matmul_faster_old(
                        x,
                        self.qweight,
                        out,
                        self.scales,
                        self.zeros,
                        self.group_size,
                        self.half_indim,
                    )
                elif self.bits == 4:
                    self.auto_gptq.vecquant4matmul_faster_old(
                        x,
                        self.qweight,
                        out,
                        self.scales,
                        self.zeros,
                        self.group_size,
                        self.half_indim,
                    )

                else:
                    raise NotImplementedError("Only 2,3,4 bits are supported.")
            else:
                raise NotImplementedError("Only use_cuda_fp16=True is supported.")
                x = x.to(torch.float32)  # This is required for autocast compatibility.
                if self.bits == 2:
                    self.autogptq_cuda.vecquant2matmul_old(
                        x,
                        self.qweight,
                        out,
                        self.scales.float(),
                        self.qzeros,
                        self.group_size,
                    )
                elif self.bits == 3:
                    self.autogptq_cuda.vecquant3matmul_old(
                        x,
                        self.qweight,
                        out,
                        self.scales.float(),
                        self.qzeros,
                        self.group_size,
                    )
                elif self.bits == 4:
                    self.autogptq_cuda.vecquant4matmul_old(
                        x,
                        self.qweight,
                        out,
                        self.scales.float(),
                        self.qzeros,
                        self.group_size,
                    )
                elif self.bits == 8:
                    self.autogptq_cuda.vecquant8matmul_old(
                        x,
                        self.qweight,
                        out,
                        self.scales.float(),
                        self.qzeros,
                        self.group_size,
                    )
                else:
                    raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        else:
            if self.wf.device != self.zeros.device:
                self.wf = self.wf.to(self.zeros.device)

            if self.bits in [2, 4, 8]:
                zeros = self.zeros.half()
                zeros = zeros.reshape(-1, 1, zeros.shape[1])

                scales = self.scales.half()
                scales = scales.reshape(-1, 1, scales.shape[-1])

                weight = torch.bitwise_right_shift(
                    torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1),
                    self.wf.unsqueeze(-1),
                ).to(torch.int16 if self.bits == 8 else torch.int8)
                weight = torch.bitwise_and(weight, (2**self.bits) - 1)
                weight = weight.reshape(-1, self.group_size, weight.shape[2])
            elif self.bits == 3:
                zeros = self.zeros.half()
                zeros = zeros.reshape(-1, 1, zeros.shape[-1])

                scales = self.scales.half()
                scales = scales.reshape(-1, 1, scales.shape[-1])

                weight = self.qweight.reshape(self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]).expand(
                    -1, -1, 12, -1
                )
                weight = (weight >> self.wf.unsqueeze(-1)) & 0x7
                weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
                weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
                weight = weight & 0x7
                weight = torch.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
                weight = weight.reshape(-1, self.group_size, weight.shape[2])
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

            weight = scales * weight - zeros
            weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
            out = torch.matmul(x, weight)
        out = out.to(dtype=x_dtype).reshape(
            out_shape
        )  # A cast is needed here as for some reason the vecquant2matmul_faster_old still allocate a float32 output.
        out = out + self.bias if self.bias is not None else out
        return out


def patch_hqq_to_gptq(layer, patch_params, load = False):
    # import code; code.interact('patch_hqq_to_gptq', local=dict(globals(), **locals()))
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

    gptq_layer = GPTQLinear(nbits, 
                            group_size, 
                            infeatures, 
                            outfeatures, 
                            bias).to(device)
    
    gptq_layer.name = hqq_layer.name

    if not load:
        # TODO: The below process is not optimal because it involves depacking + dequantize.
        W_deq = Quantizer.dequantize(hqq_layer.W_q, hqq_layer.meta)
        outfeatures = hqq_layer.meta['shape'][0]
        scales = hqq_layer.meta['scale'].reshape(outfeatures, -1)
        zeros = hqq_layer.meta['zero'].reshape(outfeatures, -1)

        gptq_layer.pack(W_deq, scales, zeros)

    del hqq_layer.W_q
    del hqq_layer.meta      
    del hqq_layer.bias
    del hqq_layer
    torch.cuda.empty_cache()

    if isinstance(layer, HQQLinear):
        return gptq_layer

    if isinstance(layer, HQQLinearLoRA):
        layer.linear_layer = gptq_layer

    torch.cuda.empty_cache()

    return layer


def patch_hqq_to_gptq_load(layer, patch_params):
    return patch_hqq_to_gptq(layer, patch_params, load = True)