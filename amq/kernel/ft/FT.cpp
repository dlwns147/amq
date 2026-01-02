#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include "attention/ft_attention.h"
#include "layernorm/layernorm.h"
#include "quantization_new/gemv/gemv_cuda.h"
#include "quantization_new/gemm/gemm_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // faster transformer
  m.def("layernorm_forward_cuda", &layernorm_forward_cuda, "FasterTransformer layernorm kernel");
  m.def("single_query_attention", &single_query_attention, "Attention with a single query",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("k_cache"), py::arg("v_cache"),
        py::arg("length_per_sample_"), py::arg("alibi_slopes_"), py::arg("timestep"), py::arg("rotary_embedding_dim")=0,
        py::arg("rotary_base")=10000.0f, py::arg("neox_rotary_style")=true);
  // gemv kernel
  m.def("gemv_4bit", &gemv_4bit, "GEMV 4-bit kernel");
  m.def("gemm_4bit", &gemm_4bit, "GEMM 4-bit kernel");
}
