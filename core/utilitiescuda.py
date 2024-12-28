#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    various conversion methods
"""

import os
import cupy as cp


def read_kernel_by_name(
    kernel_name: str,
    kernel_class: str = "limiter",
    kernel_identifier: str = "limiter_kernel",
) -> cp.RawKernel:
    this_directory = os.path.dirname(os.path.abspath(__file__))
    kernel_relativepath = f"kernels/{kernel_class}/{kernel_name}.cu"
    kernel_filepath = os.path.join(this_directory, kernel_relativepath)
    print(f"Loading CUDA kernel... {kernel_relativepath}")

    if not os.path.exists(kernel_filepath):
        raise FileNotFoundError(f"Kernel file not found: {kernel_filepath}")

    with open(kernel_filepath, "r", encoding="utf-8") as file:  # Open as text
        code = file.read()  # Read kernel source code as string
        # Pass code to RawKernel
        return cp.RawKernel(code=code, name=kernel_identifier, backend="nvrtc")
