from __future__ import annotations

import shutil

import torch
from torch.utils.cpp_extension import CUDA_HOME

from qwen_kernel_lab import native_extension_error, native_extension_loaded


def main() -> None:
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_home={CUDA_HOME}")
    print(f"nvcc={shutil.which('nvcc')}")
    print(f"cxx={shutil.which('c++') or shutil.which('g++') or shutil.which('cl')}")
    print(f"native_extension={native_extension_loaded()}")
    if native_extension_error() is not None:
        print(f"native_extension_error={native_extension_error()}")
    if torch.cuda.is_available():
        print(f"device={torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()

