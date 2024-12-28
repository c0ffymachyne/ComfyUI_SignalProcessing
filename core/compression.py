#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    various compression methods
"""


import cupy as cp
import torch
import numpy as np

from typing import Tuple, Any
from ..core.utilitiescuda import read_kernel_by_name

compressor_kernel = read_kernel_by_name(
    "compressor", kernel_class="compressor", kernel_identifier="compexp_kernel"
)


def compressor(
    audio_in: torch.Tensor,
    sample_rate: int,
    comp: float = -0.3,  # Compression/expansion factor
    attack: float = 0.1,  # Attack time in ms
    release: float = 60.0,  # Release time in ms
    a: float = 0.3,  # Filter parameter < 1
    device: str = "cuda",
) -> Tuple[torch.Tensor, Any]:
    """
    Compresses or expands stereo audio using an optimized CUDA kernel.

    Parameters:
        audio_in (torch.Tensor or np.ndarray): Input stereo audio signal with shape (n_samples, 2).
        sample_rate (int): Sampling rate in Hz.
        comp (float): Compression/expansion factor.
        attack (float): Attack time in milliseconds.
        release (float): Release time in milliseconds.
        a (float): Filter parameter (< 1) for envelope smoothing.
        device (str): Device to place the output tensor ('cuda' or 'cpu').

    Returns:
        torch.Tensor: Compressed stereo audio with shape (n_samples, 2).
    """
    # Convert input to NumPy array if necessary
    audio_in = audio_in.T
    if isinstance(audio_in, torch.Tensor):
        audio_in = audio_in.detach().cpu().numpy()
    else:
        audio_in = np.asarray(audio_in, dtype=np.float64)

    # 2. Ensure the audio is in shape (n_samples, 2)
    if audio_in.ndim != 2 or audio_in.shape[1] != 2:
        raise ValueError(
            f"Input audio must have shape (n_samples, 2), but got {audio_in.shape}"
        )

    n_samples, n_channels = audio_in.shape

    # Flatten the audio for kernel processing
    wav_in_flat = audio_in.flatten()

    # Move data to GPU
    wav_in_gpu = cp.asarray(wav_in_flat, dtype=cp.float64)
    wav_out_gpu = cp.zeros_like(wav_in_gpu)

    # Define grid and block dimensions
    block_size = 256  # Number of threads per block
    grid_size = n_channels  # One block per channel

    # Launch the CUDA kernel
    compressor_kernel(
        (grid_size,),  # Grid dimensions
        (block_size,),  # Block dimensions
        (
            wav_in_gpu,
            wav_out_gpu,
            np.int32(n_channels),
            np.int32(n_samples),
            np.float64(comp),
            np.float64(release),
            np.float64(attack),
            np.float64(a),
            np.float64(sample_rate),
        ),
    )

    # Retrieve results from GPU
    wav_out_host = wav_out_gpu.get().astype(np.float64)

    # Reshape the output
    wav_out_stereo = wav_out_host.reshape((n_samples, n_channels))

    # Convert back to Torch tensor if desired
    if device == "cuda":
        out_tensor = torch.from_numpy(wav_out_stereo).to("cuda")
    else:
        out_tensor = torch.from_numpy(wav_out_stereo).to("cpu")

    return out_tensor.T, None
