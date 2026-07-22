import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME


def build_extension():
    requested_cuda = os.getenv("QKL_BUILD_CUDA", "0") == "1"
    sources = ["csrc/ops.cpp"]
    define_macros = []
    extra_compile_args = {"cxx": ["-O3"]}
    extension_cls = CppExtension

    if os.name == "nt":
        extra_compile_args = {"cxx": ["/O2"]}

    if requested_cuda:
        if CUDA_HOME is None:
            raise RuntimeError(
                "QKL_BUILD_CUDA=1, but CUDA_HOME was not found. Install the CUDA "
                "Toolkit or unset QKL_BUILD_CUDA to build the CPU extension."
            )
        extension_cls = CUDAExtension
        sources.append("csrc/ops_cuda.cu")
        define_macros.append(("WITH_CUDA", None))
        extra_compile_args["nvcc"] = ["-O3", "--use_fast_math", "-lineinfo"]

    return extension_cls(
        name="qwen_kernel_lab._C",
        sources=sources,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


setup(
    name="qwen-kernel-lab",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[build_extension()],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)

