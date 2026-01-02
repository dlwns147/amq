#include "cuda_utils.h"
#include "dequantize_cuda.h"

template <typename T1, typename T2>
__global__ void MatQuant4DequantKernelFaster(
    const      int* __restrict__ mat,
                T1* __restrict__ out,
    const       T1* __restrict__ scales,
    const  uint8_t* __restrict__ zeros,
    int height,
    int width
) {
  const int mmblockwidth2 = BLOCKWIDTH / 2;

  int row = BLOCKHEIGHT4B * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  int new_row = (BLOCKHEIGHT4B * 8) * blockIdx.x;
  int bwidth = ((height - row) < BLOCKHEIGHT4B) ? ((height - row) * 4) : mmblockwidth2;

  __shared__ T1 out_temp[8][BLOCKWIDTH];

  if (col < width) {
    T2 scale = TtoT2(scales[col]);
    T2 zero = threadIdx.x % 2 ? \
              TtoT2(hmul(int2T<T1>(zeros[col / 2] >> 4), hneg(scale.x))) : \
              TtoT2(hmul(int2T<T1>(zeros[col / 2] & 0xf), hneg(scale.x)));

    int i = width * row + col;
    int k = 0;

    T2 res;
    T2 temp;

    unsigned int tmp;

    while (k < bwidth) {
      tmp = as_unsigned(mat[i]);
      for (int a = 0; a < 4; a++){
        temp = pair2pack(
          int2T<T1>(((tmp >> (a * 8))) & 0x0f),
          int2T<T1>(((tmp >> (a * 8 + 4))) & 0x0f)
        );
        res = hfma2(temp, scale, zero);
        out_temp[2*a][threadIdx.x] = res.x;
        out_temp[2*a+1][threadIdx.x] = res.y;
      }
      i += width;
      k += 4;
    
      for (int a = 0; a < 8; a++){
        out[(new_row + (k - 4) * 2 + a) * width + col] = out_temp[a][threadIdx.x];
      }
    }
    __syncthreads();
  }
}

template <typename T1, typename T2, int G>
__global__ void MatQuant4DequantKernelFasterGroup(
    const      int* __restrict__ mat,
                T1* __restrict__ out,
    const       T1* __restrict__ scales,
    const  uint8_t* __restrict__ zeros,
    int height,
    int width
) {
  const int mmblockwidth2 = BLOCKWIDTH / 2;

  int row = BLOCKHEIGHT4B * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  int new_row = (BLOCKHEIGHT4B * 8) * blockIdx.x;
  int bwidth = ((height - row) < BLOCKHEIGHT4B) ? ((height - row) * 4) : mmblockwidth2;

  __shared__ T1 out_temp[8][BLOCKWIDTH];

  if (col < width) {
    int group_idx = col + width * (new_row / G);
    T2 scale;
    T2 zero;

    int i = width * row + col;
    int k = 0;

    T2 res;
    T2 temp;

    unsigned int tmp;

    for (int g = 0; g < (bwidth*2 + G - 1)/G; g++){
      int gidx = group_idx + g * width;
      int loop_end = min((g + 1) * G / 2, bwidth);
      scale = TtoT2(scales[gidx]);
      zero = threadIdx.x % 2 ? \
              TtoT2(hmul(int2T<T1>(zeros[gidx / 2] >> 4), hneg(scale.x))) : \
              TtoT2(hmul(int2T<T1>(zeros[gidx / 2] & 0xf), hneg(scale.x)));

      while (k < loop_end) {
        tmp = as_unsigned(mat[i]);
        for (int a = 0; a < 4; a++){
          temp = pair2pack(
            int2T<T1>(((tmp >> (a * 8))) & 0x0f),
            int2T<T1>(((tmp >> (a * 8 + 4))) & 0x0f)
          );
          res = hfma2(temp, scale, zero);
          out_temp[2*a][threadIdx.x] = res.x;
          out_temp[2*a+1][threadIdx.x] = res.y;
        }
        i += width;
        k += 4;

        for (int a = 0; a < 8; a++){
          out[(new_row + (k - 4) * 2 + a) * width + col] = out_temp[a][threadIdx.x];
        }
      }
    }
    __syncthreads();
  }
}

void dequantize_weight_4bit(
  torch::Tensor mat,
  torch::Tensor out,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4B - 1) / BLOCKHEIGHT4B,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  if (scales.dtype() == torch::kBFloat16){
    MatQuant4DequantKernelFaster<nv_bfloat16, nv_bfloat162><<<blocks, threads>>>(
      mat.data_ptr<int>(),
      (nv_bfloat16*) out.data_ptr(),
      (nv_bfloat16*) scales.data_ptr(),
      zeros.data_ptr<uint8_t>(),
      height, width
    );
  }
  else{
    MatQuant4DequantKernelFaster<half, half2><<<blocks, threads>>>(
      mat.data_ptr<int>(),
      (half*) out.data_ptr(),
      (half*) scales.data_ptr(),
      zeros.data_ptr<uint8_t>(),
      height, width
    );
  }
}

void dequantize_weight_4bit_group(
  torch::Tensor mat,
  torch::Tensor out,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int height = mat.size(0);
  int width = mat.size(1);

  const int group_size = out.size(0) / scales.size(0);

  dim3 blocks(
    (height + BLOCKHEIGHT4B - 1) / BLOCKHEIGHT4B,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  if (group_size == 128){
    if (scales.dtype() == torch::kBFloat16){
      MatQuant4DequantKernelFasterGroup<nv_bfloat16, nv_bfloat162, 128><<<blocks, threads>>>(
        mat.data_ptr<int>(),
        (nv_bfloat16*) out.data_ptr(),
        (nv_bfloat16*) scales.data_ptr(),
        zeros.data_ptr<uint8_t>(),
        height, width
      );
    }
    else{
      MatQuant4DequantKernelFasterGroup<half, half2, 128><<<blocks, threads>>>(
        mat.data_ptr<int>(),
        (half*) out.data_ptr(),
        (half*) scales.data_ptr(),
        zeros.data_ptr<uint8_t>(),
        height, width
      );
    }
  }
  else if (group_size == 64){
    if (scales.dtype() == torch::kBFloat16){
      MatQuant4DequantKernelFasterGroup<nv_bfloat16, nv_bfloat162, 64><<<blocks, threads>>>(
        mat.data_ptr<int>(),
        (nv_bfloat16*) out.data_ptr(),
        (nv_bfloat16*) scales.data_ptr(),
        zeros.data_ptr<uint8_t>(),
        height, width
      );
    }
    else{
      MatQuant4DequantKernelFasterGroup<half, half2, 64><<<blocks, threads>>>(
        mat.data_ptr<int>(),
        (half*) out.data_ptr(),
        (half*) scales.data_ptr(),
        zeros.data_ptr<uint8_t>(),
        height, width
      );
    }
  }
  else if (group_size == 32){
    if (scales.dtype() == torch::kBFloat16){
      MatQuant4DequantKernelFasterGroup<nv_bfloat16, nv_bfloat162, 32><<<blocks, threads>>>(
        mat.data_ptr<int>(),
        (nv_bfloat16*) out.data_ptr(),
        (nv_bfloat16*) scales.data_ptr(),
        zeros.data_ptr<uint8_t>(),
        height, width
      );
    }
    else{
      MatQuant4DequantKernelFasterGroup<half, half2, 32><<<blocks, threads>>>(
        mat.data_ptr<int>(),
        (half*) out.data_ptr(),
        (half*) scales.data_ptr(),
        zeros.data_ptr<uint8_t>(),
        height, width
      );
    }
  }
}