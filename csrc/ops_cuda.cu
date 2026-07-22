#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <tuple>

namespace qkl {

namespace {

template <typename scalar_t, bool write_residual>
__global__ void add_rms_norm_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ residual_output,
    int hidden_size,
    float eps) {
  extern __shared__ float shared[];
  const int row = blockIdx.x;
  const int base = row * hidden_size;
  float sum_sq = 0.0f;

  for (int col = threadIdx.x; col < hidden_size; col += blockDim.x) {
    const float value = static_cast<float>(x[base + col]) +
                        static_cast<float>(residual[base + col]);
    sum_sq += value * value;
  }

  shared[threadIdx.x] = sum_sq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inv_rms = rsqrtf(shared[0] / static_cast<float>(hidden_size) + eps);
  for (int col = threadIdx.x; col < hidden_size; col += blockDim.x) {
    const float value = static_cast<float>(x[base + col]) +
                        static_cast<float>(residual[base + col]);
    if constexpr (write_residual) {
      residual_output[base + col] = static_cast<scalar_t>(value);
    }
    output[base + col] = static_cast<scalar_t>(
        value * inv_rms * static_cast<float>(weight[col]));
  }
}

template <typename scalar_t>
__global__ void silu_mul_kernel(
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ up,
    scalar_t* __restrict__ output,
    int64_t numel) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < numel) {
    const float value = static_cast<float>(gate[index]);
    const float silu = value / (1.0f + expf(-value));
    output[index] = static_cast<scalar_t>(silu * static_cast<float>(up[index]));
  }
}

void check_cuda_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(
      tensor.scalar_type() == torch::kFloat32 || tensor.scalar_type() == torch::kFloat16,
      name,
      " must be float32 or float16");
}

}  // namespace

torch::Tensor add_rms_norm_cuda(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    double eps) {
  check_cuda_tensor(x, "x");
  check_cuda_tensor(residual, "residual");
  check_cuda_tensor(weight, "weight");
  TORCH_CHECK(x.sizes() == residual.sizes(), "x and residual must have equal shapes");
  TORCH_CHECK(weight.dim() == 1 && weight.numel() == x.size(-1),
              "weight must match hidden size");
  TORCH_CHECK(x.scalar_type() == residual.scalar_type() &&
                  x.scalar_type() == weight.scalar_type(),
              "all inputs must have the same dtype");

  c10::cuda::CUDAGuard device_guard(x.device());
  auto output = torch::empty_like(x);
  const int hidden_size = static_cast<int>(x.size(-1));
  const int rows = static_cast<int>(x.numel() / hidden_size);
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "add_rms_norm_cuda", [&] {
    add_rms_norm_kernel<scalar_t, false><<<rows, threads, threads * sizeof(float), stream>>>(
        x.data_ptr<scalar_t>(),
        residual.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        nullptr,
        hidden_size,
        static_cast<float>(eps));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> add_rms_norm_residual_cuda(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    double eps) {
  check_cuda_tensor(x, "x");
  check_cuda_tensor(residual, "residual");
  check_cuda_tensor(weight, "weight");
  TORCH_CHECK(x.sizes() == residual.sizes(), "x and residual must have equal shapes");
  TORCH_CHECK(weight.dim() == 1 && weight.numel() == x.size(-1),
              "weight must match hidden size");
  TORCH_CHECK(x.scalar_type() == residual.scalar_type() &&
                  x.scalar_type() == weight.scalar_type(),
              "all inputs must have the same dtype");

  c10::cuda::CUDAGuard device_guard(x.device());
  auto output = torch::empty_like(x);
  auto residual_output = torch::empty_like(x);
  const int hidden_size = static_cast<int>(x.size(-1));
  const int rows = static_cast<int>(x.numel() / hidden_size);
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      x.scalar_type(), "add_rms_norm_residual_cuda", [&] {
        add_rms_norm_kernel<scalar_t, true>
            <<<rows, threads, threads * sizeof(float), stream>>>(
                x.data_ptr<scalar_t>(),
                residual.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                residual_output.data_ptr<scalar_t>(),
                hidden_size,
                static_cast<float>(eps));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, residual_output};
}

torch::Tensor silu_mul_cuda(const torch::Tensor& gate, const torch::Tensor& up) {
  check_cuda_tensor(gate, "gate");
  check_cuda_tensor(up, "up");
  TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up must have equal shapes");
  TORCH_CHECK(gate.scalar_type() == up.scalar_type(), "gate and up must share dtype");

  c10::cuda::CUDAGuard device_guard(gate.device());
  auto output = torch::empty_like(gate);
  constexpr int threads = 256;
  const int blocks = static_cast<int>((gate.numel() + threads - 1) / threads);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(gate.scalar_type(), "silu_mul_cuda", [&] {
    silu_mul_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        gate.data_ptr<scalar_t>(),
        up.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        gate.numel());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace qkl
