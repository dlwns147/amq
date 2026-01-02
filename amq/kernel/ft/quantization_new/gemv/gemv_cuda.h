#pragma once
#include <torch/extension.h>

torch::Tensor gemv_4bit(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int m,
    int n,
    int k,
    int group_size);

torch::Tensor gemv_4bit_qeft(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    torch::Tensor _oweight,
    int m,
    int n,
    int k,
    int group_size);