from pathlib import Path
from setuptools import setup

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
include_dirs = [
    os.path.join(ROOT, "mast3r_slam/backend/include"),
    os.path.join(ROOT, "thirdparty/eigen"),
]

sources = [
    "mast3r_slam/backend/src/gn.cpp",
]
extra_compile_args = {
    "cores": ["j8"],
    "cxx": ["-O3"],
}

if CUDA_HOME is None:
    raise RuntimeError(
        "A CUDA toolkit is required to build mast3r_slam_backends. "
        "Install a toolkit compatible with your PyTorch build and retry."
    )

sources.extend(
    [
        "mast3r_slam/backend/src/gn_kernels.cu",
        "mast3r_slam/backend/src/matching_kernels.cu",
    ]
)
extra_compile_args["nvcc"] = ["-O3"]
ext_modules = [
    CUDAExtension(
        "mast3r_slam_backends",
        include_dirs=include_dirs,
        sources=sources,
        extra_compile_args=extra_compile_args,
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
