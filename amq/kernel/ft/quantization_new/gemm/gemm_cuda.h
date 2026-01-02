#include <torch/extension.h>

torch::Tensor gemm_4bit(torch::Tensor _in_feats, torch::Tensor _kernel, torch::Tensor _scales, torch::Tensor _zeros);
torch::Tensor gemm_4bit_qeft(torch::Tensor _in_feats, torch::Tensor _kernel, torch::Tensor _scales, torch::Tensor _zeros, torch::Tensor _oweights);
