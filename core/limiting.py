#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    various conversion methods
"""

import cupy as cp
import torch
import numpy as np
from typing import List

from ..core.utilitiescuda import read_kernel_by_name

limiter_updown_parallel = read_kernel_by_name(
    "limiter_updown_parallel",
    kernel_class="limiter",
    kernel_identifier="limiter_kernel",
)
limiter_updown_weighted_parallel = read_kernel_by_name(
    "limiter_updown_weighted_parallel",
    kernel_class="limiter",
    kernel_identifier="limiter_kernel",
)
limiter_down_parallel = read_kernel_by_name(
    "limiter_down_parallel", kernel_class="limiter", kernel_identifier="limiter_kernel"
)
limiter_soft_clipper = read_kernel_by_name(
    "limiter_soft_clipper", kernel_class="limiter", kernel_identifier="limiter_kernel"
)
limiter_hard_clipper = read_kernel_by_name(
    "limiter_hard_clipper", kernel_class="limiter", kernel_identifier="limiter_kernel"
)
limiter_dev = read_kernel_by_name(
    "limiter_dev", kernel_class="limiter", kernel_identifier="limiter_kernel"
)

_limiter_kernel_map = {
    "downward-upward": limiter_updown_weighted_parallel,
    "downward": limiter_down_parallel,
    "soft-clipper": limiter_soft_clipper,
    "hard-clipper": limiter_hard_clipper,
}


def limiter_get_modes() -> List[str]:
    return list(_limiter_kernel_map.keys())


def limiter(
    audio_in: torch.Tensor | np.ndarray,
    mode: str = "downward",
    sample_rate: int = 44100,
    threshold: float = 0.5,  # Threshold in percents
    slope: float = 1.0,  # Slope in percents
    attack_ms: float = 0.008,  # Attack time in ms
    release_ms: float = 100.0,  # Release time in ms
) -> torch.Tensor:
    """
    Compresses stereo audio using an optimized CUDA kernel with running sum RMS calculation.

    Parameters:
        audio_in (torch.Tensor or np.ndarray): Input stereo audio signal with shape (n_samples, 2).
        sample_rate (int): Sampling rate in Hz.
        threshold (float): Threshold in percents (0-100).
        slope (float): Slope angle in percents (0-100).
        rms_window_ms (float): RMS window width in milliseconds.
        attack_ms (float): Attack time in milliseconds.
        release_ms (float): Release time in milliseconds.
        chunk_size (int): Number of samples per chunk.
        device (str): Device to place the output tensor ('cuda' or 'cpu').

    Returns:
        torch.Tensor: Compressed stereo audio with shape (n_samples, 2).
        np.ndarray: Debug information (envelope and gain) with shape (n_samples, 2).
    """
    # Convert input to CPU double-precision NumPy array if necessary

    device = audio_in.device

    audio_in = audio_in.T
    if isinstance(audio_in, torch.Tensor):
        audio_in = audio_in.detach().cpu()
        audio_in = audio_in.numpy()
    else:
        audio_in = np.asarray(audio_in, dtype=np.float64)

    # Ensure the audio is in shape (n_samples, 2)
    if audio_in.ndim != 2 or audio_in.shape[1] != 2:
        raise ValueError(
            f"Input audio must have shape (n_samples, 2), but got {audio_in.shape}"
        )

    n_samples = audio_in.shape[0]
    n_channels = audio_in.shape[1]

    wav_in_flat = audio_in.flatten()

    wav_in_gpu = cp.asarray(wav_in_flat, dtype=cp.float64)
    wav_out_gpu = cp.zeros_like(wav_in_gpu)

    # Define grid and block dimensions
    block_size = 64
    grid_size = n_channels  # (n_channels + block_size - 1) // block_size
    shared_mem_size = n_channels * cp.float64().nbytes  # Shared memory for envelopes

    # attack_coeff = math.exp(-1.0 / (sample_rate * (attack_ms * 1e-3)))
    # release_coeff = math.exp(-1.0 / (sample_rate * (release_ms * 1e-3)))

    # print('attack_coeff',attack_coeff)
    # print('attack_coeff',release_coeff)

    if mode not in limiter_get_modes():
        raise Exception(f"Limiter Kernel '{mode}' Not Found")

    kernel = _limiter_kernel_map[mode]

    kernel(
        (grid_size,),
        (block_size,),
        (
            wav_in_gpu,
            wav_out_gpu,
            np.int32(n_channels),
            np.int32(n_samples),
            np.float64(threshold),
            np.float64(slope),
            np.float64(sample_rate),
            np.float64(attack_ms),
            np.float64(release_ms),
        ),
        shared_mem=shared_mem_size,
    )

    # Retrieve results
    wav_out_host = wav_out_gpu.get().astype(np.float64)
    # Reshape the output
    wav_out_stereo = wav_out_host.reshape((n_samples, n_channels))

    if device == "cuda":
        out_tensor = torch.from_numpy(wav_out_stereo).to(device)
    else:
        out_tensor = torch.from_numpy(wav_out_stereo).to("cpu")

    return out_tensor.T
