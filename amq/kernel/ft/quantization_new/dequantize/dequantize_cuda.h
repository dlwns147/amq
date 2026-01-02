#pragma once
#include <torch/all.h>

const int BLOCKWIDTH  = 256;
const int BLOCKHEIGHT =  24;
const int BLOCKHEIGHT4B =  32;

const int BLOCKWIDTHG128  = 128;
const int BLOCKHEIGHTG128 =  12;
const int BLOCKHEIGHT4BG128 = 16;

template <typename T1, typename T2>
__global__ void MatQuant4DequantKernelFaster(
    const      int* __restrict__ mat,
                T1* __restrict__ out,
    const       T1* __restrict__ scales,
    const  uint8_t* __restrict__ zeros,
    int height,
    int width
);

template <typename T1, typename T2>
__global__ void MatQuant4DequantKernelFasterGroup(
    const      int* __restrict__ mat,
                T1* __restrict__ out,
    const       T1* __restrict__ scales,
    const  uint8_t* __restrict__ zeros,
    int height,
    int width
);

void dequantize_weight_4bit(
  torch::Tensor mat,
  torch::Tensor out,
  torch::Tensor scales,
  torch::Tensor zeros
);

void dequantize_weight_4bit_group(
  torch::Tensor mat,
  torch::Tensor out,
  torch::Tensor scales,
  torch::Tensor zeros
);

torch::Tensor dequantize_weight_4bit_qeft(
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int n,
    int k,
    int group_size);