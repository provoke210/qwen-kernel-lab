# Qwen Kernel Lab

面向 Qwen 推理的正确性优先 C++/CUDA 算子项目。项目实现：

- Fused Add + RMSNorm
- Fused SiLU + Mul（Qwen gated MLP）
- PyTorch Reference、CPU C++、CUDA 三层后端
- Prefill/Decode 典型 shape 的正确性与性能测试
- Hugging Face Qwen RMSNorm 数值集成验证
- Qwen2.5 Decoder 模块注入与端到端配对 Benchmark
- MLP-only、RMS-only 和组合融合消融实验

> 仓库保留 RTX 3070 Ti Laptop GPU 的实测 JSON；不同硬件与软件版本应重新运行配对测试。

## 1. 为什么做这个项目

Qwen 推理中的 RMSNorm、残差连接与 gated MLP 包含大量逐元素运算。若分别执行，会产生额外的 kernel launch 和中间 Tensor 显存读写。本项目用融合算子建立一条可复现的工程链路：

```text
PyTorch 数学参考
        ↓ correctness oracle
C++/CUDA 自定义算子
        ↓ torch dispatcher
Qwen 模块数值对齐
        ↓
Prefill / Decode benchmark
```

这不是一个完整推理引擎，而是模型迁移和硬件后端适配中可独立验证的算子子项目。

## 2. 环境

基础环境：

- Python 3.9+
- PyTorch 2.4+

CUDA 构建额外需要：

- Linux 或 WSL2（推荐）
- 与 PyTorch 匹配的 CUDA Toolkit
- 
vcc`、支持 C++17 的编译器
- NVIDIA GPU

## 3. 快速开始

### 只运行 Python Reference

无需编译：

```bash
python -m unittest discover -s tests -v
python benchmarks/benchmark_ops.py --device cpu --dtype float32 --runs 20
```

### 构建 CPU C++ 扩展

```bash
python -m pip install -e . --no-build-isolation
python -m unittest discover -s tests -v
```

### 构建 CUDA 扩展

Linux/WSL2：

```bash
export QKL_BUILD_CUDA=1
python -m pip install -e . --no-build-isolation
python benchmarks/benchmark_ops.py \
  --device cuda \
  --dtype float16 \
  --warmup 100 \
  --runs 1000 \
  --output benchmark_results/rtx3090_fp16.json
```

PowerShell：

```powershell
$env:QKL_BUILD_CUDA = "1"
python -m pip install -e . --no-build-isolation
```

## 4. 正确性设计

`add_rms_norm_reference` 使用 FP32 计算方差，再转换回输入 dtype：

```python
summed = x + residual
variance = summed.float().square().mean(-1, keepdim=True)
output = summed.float() * torch.rsqrt(variance + eps) * weight.float()
```

测试覆盖：

- Decode：`[1, 1, 896]`、`[8, 1, 896]`
- Prefill：`[1, 128, 896]`、`[1, 512, 896]`
- shape/dtype/device 参数检查
- Reference autograd 回退
- CUDA FP16 与 FP32 Reference 对齐

Native Kernel 定位为 inference-only；输入需要梯度时，Python API 自动回退到 PyTorch Reference。

## 5. CUDA 实现

### Add-RMSNorm

- 一个 CUDA block 处理一个 token row。
- 每个线程以 stride 方式处理 hidden dimension。
- 使用 shared memory 完成平方和归约。
- 平方和与归一化因子采用 FP32。
- 同一 kernel 内完成残差相加、归一化和权重缩放。

### SiLU-Mul

- 一维 grid 覆盖全部元素。
- 同一 kernel 内计算 `silu(gate) * up`。
- 避免生成独立 SiLU 中间 Tensor。

当前实现是清晰、可验证的 baseline，不宣称达到生产级最优性能。后续优化方向包括 warp shuffle reduction、向量化 load/store、BF16、不同 hidden size 的模板特化与 persistent kernel。

## 6. Qwen 集成验证

安装依赖：

```bash
python -m pip install -e ".[qwen]"
python examples/validate_qwen.py --model Qwen/Qwen2.5-0.5B --device cuda
```

脚本替换所有类名以 `RMSNorm` 结尾且包含 `weight` 的模块，并比较：

- 最大绝对 logits 误差
- 平均绝对 logits 误差
- Top-1 token 是否一致

注意：Hugging Face 的残差相加位于 RMSNorm 模块外，因此这个替换只用于模型级数值验证。要获得 Add-RMSNorm 的端到端融合收益，需要对 decoder layer 做模型特定的 residual graph rewrite。

## 7. 实测结果

环境：RTX 3070 Ti Laptop GPU、CUDA 12.6、PyTorch 2.13.0+cu126、FP16、Qwen2.5-0.5B、batch size 1。

| Shape | Add-RMSNorm 加速 | SiLU-Mul 加速 |
| --- | ---: | ---: |
| Decode `[1, 1, 896]` | 4.08x | 1.88x |
| Decode `[8, 1, 896]` | 4.57x | 1.60x |
| Prefill `[1, 128, 896]` | 6.04x | 1.05x |
| Prefill `[1, 512, 896]` | 4.95x | 1.24x |

MLP-only 端到端配对实验覆盖 128/512/1024 prompt length、生成 128 tokens：Decode 吞吐提升 1.71%-4.42%，端到端吞吐提升 1.89%-4.26%，所有生成 Token 完全一致。Residual + RMSNorm 的 Prefill 融合作为实验选项保留，默认使用收益更稳定的 MLP-only 策略。

原始结果见 [`benchmark_results/`](benchmark_results/)。

## 8. Benchmark 解释

Benchmark 输出 JSON，记录环境、shape、Reference 延迟、Native 延迟与加速比。分析时应分别讨论：

- Decode：元素少，kernel launch 与调度开销明显。
- Prefill：元素多，更容易体现减少中间 Tensor 读写的收益。
- 算子级与端到端：单个算子加速不会等比例转化为模型吞吐提升。
- 同步开销：脚本为保证测量正确，每次 sample 都同步；提交正式报告时可增加 CUDA Event 批量测量。

## 9. Nsight 实验建议

```bash
nsys profile -o benchmark_results/qkl \
  python benchmarks/benchmark_ops.py --device cuda --dtype float16 --runs 100
```

重点观察：

- kernel launch 数量
- DRAM throughput
- achieved occupancy
- shared memory 使用量
- prefill/decode 下的执行时间占比

## 10. 下一步

1. BF16 与更多 hidden size 模板特化。
2. RoPE + KV Cache Write 融合。
3. 接入 vLLM CustomOp 路径并运行 serving benchmark。
4. 使用 Nsight Systems/Compute 分析 launch、访存与 occupancy。
