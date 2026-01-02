/*
 * Modified from NVIDIA [TRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/d37b507f41a87457fe9f10f7459d08f5db235745/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv)
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
@article{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Dang, Xingyu and Han, Song},
  journal={arXiv},
  year={2023}
}
*/

#include <cuda_fp16.h>
#include <stdio.h>
#include <torch/extension.h>
#include "dequantize_cuda.h"
#include "../dequantize.cuh"
#define PACK_FACTOR 8
#define WARP_SIZE 32
#define MEM_ACCESS_SIZE 128
#define HALF_SIZE 16

template <int NPerBlock, int BlockSize>
__global__ void dequant_kernel(
  const uint32_t* weight, const half* scales, const half* zeros, half* outputs, 
  const int IC, const int OC, int GroupSize)
{
    const int kStride = 64;
    const int kElemsPerThread = MEM_ACCESS_SIZE / 4;
    const int kThreadsNumPerTile = kStride / kElemsPerThread;
    const int Elt_per_write = MEM_ACCESS_SIZE / HALF_SIZE; // 8
    // assert(MEM_ACCESS_SIZE == 128);

    static constexpr int kShuffleSize = 32;
    static constexpr int kShuffleBasicTile = 2;
    static constexpr int kShuffleContinous = 4;
    static constexpr int kShuffleStrided = 4;

    constexpr int kInterleave = 4;

    uint32_t local_qweights[MEM_ACCESS_SIZE / 32];
    half half_weight_buffer[kElemsPerThread]; 
    half2 local_scale[NPerBlock];
    half2 local_scaled_zeros[NPerBlock];
    
    const int blk_row_offset = blockIdx.x * NPerBlock * kInterleave;
    const int thd_row_offset = (threadIdx.x / kThreadsNumPerTile) % kInterleave;
    const int act_k_offset = threadIdx.x / (kThreadsNumPerTile * kInterleave) * kStride
                               + (threadIdx.x % kThreadsNumPerTile) * kElemsPerThread;
    const int group_offset = act_k_offset / GroupSize;
    
    const uint32_t* blk_weight_ptr = weight + blk_row_offset * IC / PACK_FACTOR;
    const half* scale_ptr = scales + blk_row_offset + thd_row_offset + group_offset * OC;
    const half* zeros_ptr = zeros + blk_row_offset + thd_row_offset + group_offset * OC;

    const int act_forward_step = BlockSize * kElemsPerThread / kInterleave;
    const int scale_forward_step = act_forward_step / GroupSize * OC;

    // Main loop iteration, each block completes the outputs for several OCs
    for (int kk = threadIdx.x * kElemsPerThread; kk < IC * kInterleave; kk += BlockSize * kElemsPerThread)
    {
        // Load qweight, scales and scaled_zeros
        #pragma unroll
        for (int idx = 0; idx < NPerBlock; ++idx)
        {
            // use float4 to load weights, each thread load 32 int4 numbers (1 x float4, 128 bit)
            *((float4*)(local_qweights)) = 
                *((float4*)(blk_weight_ptr + (idx * kInterleave * IC + kk)/ PACK_FACTOR));
            local_scale[idx] = __half2half2(*(scale_ptr + idx * kInterleave));
            local_scaled_zeros[idx] = __half2half2(*(zeros_ptr + idx * kInterleave));
            
            // Map int4 qweight to fp format 
            #pragma unroll
            for (int i = 0; i < MEM_ACCESS_SIZE / 32; ++i)
            {
                // Converts 32 bits (8 x int4) to 8 fp16
                dequantize_s4_to_fp16x2(*reinterpret_cast<half2 *>(local_qweights + i), reinterpret_cast<uint4 *>(half_weight_buffer + i * PACK_FACTOR));
            }

            // Dequantize (apply s/z) and shuffle elements to match the weight packing format
            #pragma unroll
            for (int i = 0; i < kElemsPerThread / 2; ++i)
            {
                *reinterpret_cast<half2*>(half_weight_buffer + i * 2) 
                    = __hfma2(*reinterpret_cast<half2*>(half_weight_buffer + i * 2), 
                        local_scale[idx], 
                        local_scaled_zeros[idx]);
            }

            // Write to DRAM Tensor
            #pragma unroll
            for (int i = 0; i < kElemsPerThread / Elt_per_write; ++i)
            {
                *((float4*)(outputs + (idx * kInterleave + blk_row_offset + thd_row_offset) * IC \
                                    + act_k_offset + i * Elt_per_write))
                                    = *((float4*)(half_weight_buffer + i * Elt_per_write));
            }

        }
        outputs += act_forward_step;
        scale_ptr += scale_forward_step;
        zeros_ptr += scale_forward_step;
    }
}


/*
Args:
  _in_feats: tensor of shape [B, IC];
  _kernel: int tensor of shape [OC, IC // 8];
  _zeros: int tensor of shape [OC, IC // G // 8];
  _scaling_factors: tensor of shape [OC, IC // G];
  blockDim_x: size of thread block, dimension x, where blockDim_x * workload_per_thread = IC;
  blockDim_y: size of thread block, dimension y, where blockDim_y * gridDim_y = OC;

Returns:
  out_feats: tensor of shape [B, OC];
*/
torch::Tensor dequantize_weight_4bit_qeft(
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int n,
    int k,
    int group_size)
{
    auto kernel = reinterpret_cast<uint32_t*>(_kernel.data_ptr());
    auto zeros = reinterpret_cast<half*>(_zeros.data_ptr<at::Half>());
    auto scaling_factors = reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());

    auto options = torch::TensorOptions().dtype(_scaling_factors.dtype()).device(_scaling_factors.device());
    torch::Tensor _half_weight = torch::empty({n,k}, options);
    half * half_weight = reinterpret_cast<half *>(_half_weight.data_ptr());
    
    static constexpr int N_PER_BLOCK = 2;
    static constexpr int K_INTERLEAVE = 4;
    static constexpr int BLOCK_SIZE = 256;

    dim3 num_blocks(n / N_PER_BLOCK / K_INTERLEAVE);
    dim3 num_threads(BLOCK_SIZE);

    dequant_kernel<N_PER_BLOCK, BLOCK_SIZE><<<num_blocks, num_threads>>>(
      kernel, scaling_factors, zeros, half_weight, k, n, group_size
    );
    return _half_weight;
}
