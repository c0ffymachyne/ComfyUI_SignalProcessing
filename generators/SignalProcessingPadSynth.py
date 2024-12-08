#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    Pad Synthesiser port of code from this article https://zynaddsubfx.sourceforge.io/doc/PADsynth/PADsynth.htm#c_implementation
"""


import torch
import math, json
from typing import Tuple, List, Dict

from ..core.io import audio_to_comfy_3d

class SignalProcessingPadSynth:
    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        return {
            "required": {
                "sample_rate": ("INT", {"default": 44100, "min": 8000, "max": 96000, "step": 1}),
                "fundamental_freq": ("FLOAT", {"default": 261.0, "min": 20.0, "max": 2000.0, "step": 1.0}),
                "bandwidth_cents": ("FLOAT", {"default": 40.0, "min": 10.0, "max": 100.0, "step": 1.0}),
                "number_harmonics": ("INT", {"default": 64, "min": 1, "max": 128, "step": 1})
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    CATEGORY = "Signal Processing"
    FUNCTION = "process"

    def process(
        self,
        sample_rate: int,
        fundamental_freq: float,
        bandwidth_cents: float,
        number_harmonics: int,
    ) -> Tuple[Dict[str, torch.Tensor]]:
        """
        Apply PADsynth algorithm to generate audio.

        Parameters:
            samplerate (int): Sampling rate in Hz.
            fundamental_freq (float): Fundamental frequency in Hz.
            bandwidth_cents (float): Bandwidth in cents for Gaussian profile.
            number_harmonics (int): Number of harmonics to generate.

        Returns:
            Tuple[Dict[str, torch.Tensor]]: Generated audio with waveform and sample rate.
        """

        # Define FFT size
        N = 262144  # As per C++ code

        # Use default amplitude distribution
        A = torch.zeros(number_harmonics, dtype=torch.double)
        A[0] = 0.0  # A[0] is not used
        for i in range(1, number_harmonics):
            A[i] = 1.0 / i
            if (i % 2) == 0:
                A[i] *= 2.0

        # Initialize frequency amplitude and phase arrays
        freq_amp = torch.zeros(N // 2, dtype=torch.double)
        freq_phase = torch.rand(N // 2, dtype=torch.double) * 2.0 * math.pi  # Random phases between 0 and 2pi

        # Define Gaussian profile function
        def profile(fi: torch.Tensor, bwi: torch.Tensor) -> torch.Tensor:
            x = fi / bwi
            x_sq = x ** 2
            # Avoid computing exp(-x^2) for x_sq > 14.71280603
            mask = x_sq <= 14.71280603
            result = torch.zeros_like(x_sq)
            result[mask] = torch.exp(-x_sq[mask]) / bwi[mask]
            return result

        # Convert bandwidth from cents to Hz
        # bw_Hz = (2^(bw/1200) -1) * f * nh
        # Convert bandwidth_cents to multiplier
        bw_multiplier = 2.0 ** (bandwidth_cents / 1200.0) - 1.0

        # Populate frequency amplitude array
        for nh in range(1, number_harmonics):
            f_nh = fundamental_freq * nh
            bw_Hz = bw_multiplier * f_nh
            bwi = bw_Hz / (2.0 * sample_rate)
            fi = f_nh / sample_rate  # Normalized frequency

            # Create tensors for frequency bins
            i = torch.arange(N // 2, dtype=torch.double)
            # Normalized frequency for each bin
            normalized_freq = i / N  # Equivalent to i * (sample_rate / N) / sample_rate = i / N

            # Compute profile
            fi_tensor = torch.full_like(i, fi)
            bwi_tensor = torch.full_like(i, bwi)
            profile_values = profile(normalized_freq - fi_tensor, bwi_tensor)

            # Update frequency amplitude
            freq_amp += profile_values * A[nh]

        # Construct complex frequency domain tensor
        real = freq_amp * torch.cos(freq_phase)
        imag = freq_amp * torch.sin(freq_phase)
        freq_complex = torch.complex(real, imag)  # Shape: (N//2,)

        # Perform IFFT using torch.fft.irfft
        smp = torch.fft.irfft(freq_complex, n=N)  # Shape: (N,)

        # Normalize the signal to prevent clipping
        max_val = torch.max(torch.abs(smp))
        if max_val < 1e-5:
            max_val = 1e-5  # Prevent division by zero
        smp = smp / (max_val * math.sqrt(2))  # Normalize to 1/sqrt(2) as in C++ code

        # Convert to float32 for saving
        smp = smp.float()

        # Prepare waveform tensor: (C, N)
        waveform_out = smp.unsqueeze(0)  # Mono audio

        return audio_to_comfy_3d(waveform_out,sample_rate)