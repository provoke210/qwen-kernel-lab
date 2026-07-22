#include <torch/extension.h>
#include <tuple>

namespace qkl {

void check_rms_inputs(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& weight) {
  TORCH_CHECK(x.sizes() == residual.sizes(), "x and residual must have equal shapes");
  TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");
  TORCH_CHECK(weight.dim() == 1, "weight must be one-dimensional");
  TORCH_CHECK(weight.numel() == x.size(-1), "weight size must equal hidden size");
  TORCH_CHECK(x.device() == residual.device() && x.device() == weight.device(),
              "all inputs must be on the same device");
  TORCH_CHECK(x.scalar_type() == residual.scalar_type() &&
                  x.scalar_type() == weight.scalar_type(),
              "all inputs must have the same dtype");
}

torch::Tensor add_rms_norm_cpu(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    double eps) {
  check_rms_inputs(x, residual, weight);
  auto summed = x + residual;
  auto summed_fp32 = summed.to(torch::kFloat32);
  auto variance = summed_fp32.square().mean({-1}, true);
  auto output = summed_fp32 * torch::rsqrt(variance + eps) * weight.to(torch::kFloat32);
  return output.to(x.scalar_type());
}
std::tuple<torch::Tensor, torch::Tensor> add_rms_norm_residual_cpu(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    double eps) {
  check_rms_inputs(x, residual, weight);
  auto summed = x + residual;
  auto summed_fp32 = summed.to(torch::kFloat32);
  auto variance = summed_fp32.square().mean({-1}, true);
  auto normalized =
      summed_fp32 * torch::rsqrt(variance + eps) * weight.to(torch::kFloat32);
  return {normalized.to(x.scalar_type()), summed};
}


torch::Tensor silu_mul_cpu(const torch::Tensor& gate, const torch::Tensor& up) {
  TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up must have equal shapes");
  TORCH_CHECK(gate.device() == up.device(), "gate and up must share a device");
  TORCH_CHECK(gate.scalar_type() == up.scalar_type(), "gate and up must share a dtype");
  return at::silu(gate) * up;
}

#ifdef WITH_CUDA
torch::Tensor add_rms_norm_cuda(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    double eps);
torch::Tensor silu_mul_cuda(const torch::Tensor& gate, const torch::Tensor& up);
std::tuple<torch::Tensor, torch::Tensor> add_rms_norm_residual_cuda(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    double eps);
#endif

}  // namespace

TORCH_LIBRARY(qwen_kernel_lab, m) {
  m.def("add_rms_norm(Tensor x, Tensor residual, Tensor weight, float eps=1e-6) -> Tensor");
  m.def("add_rms_norm_residual(Tensor x, Tensor residual, Tensor weight, float eps=1e-6) -> (Tensor, Tensor)");
  m.def("silu_mul(Tensor gate, Tensor up) -> Tensor");
}

TORCH_LIBRARY_IMPL(qwen_kernel_lab, CPU, m) {
  m.impl("add_rms_norm", &qkl::add_rms_norm_cpu);
  m.impl("add_rms_norm_residual", &qkl::add_rms_norm_residual_cpu);
  m.impl("silu_mul", &qkl::silu_mul_cpu);
}

#ifdef WITH_CUDA
TORCH_LIBRARY_IMPL(qwen_kernel_lab, CUDA, m) {
  m.impl("add_rms_norm", &qkl::add_rms_norm_cuda);
  m.impl("add_rms_norm_residual", &qkl::add_rms_norm_residual_cuda);
  m.impl("silu_mul", &qkl::silu_mul_cuda);
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

