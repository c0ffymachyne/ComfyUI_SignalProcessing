#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    This is a port of Paul's Extreme Sound Stretch (Paulstretch) - by Nasca Octavian PAUL
    http://www.paulnasca.com/
    http://hypermammut.sourceforge.net/paulstretch/
    https://github.com/paulnasca/paulstretch_python
    https://github.com/paulnasca/paulstretch_python/blob/master/paulstretch_stereo.py
"""

import torch

import math
from typing import Tuple, Dict, Any, Union

from ..core.utilities import comfy_root_to_syspath
from ..core.io import audio_from_comfy_2d, audio_to_comfy_3d
from ..core.loudness import lufs_normalization, get_loudness

comfy_root_to_syspath()  # add comfy to sys path for dev


class SignalProcessingPaulStretch:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "audio_input": ("AUDIO", {"forceInput": True}),
                "stretch_factor": (
                    "FLOAT",
                    {"default": 8.0, "min": 1.0, "max": 100.0, "step": 0.1},
                ),
                "window_size_seconds": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.05, "max": 10.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    CATEGORY = "Signal Processing"
    FUNCTION = "process"

    def process(
        self,
        audio_input: Dict[str, Union[torch.Tensor, int]],
        stretch_factor: float,
        window_size_seconds: float,
    ) -> Tuple[Dict[str, Union[torch.Tensor, int]]]:

        # Conditional processing: If stretch_factor is 1.0, return original audio
        if stretch_factor == 1.0:
            return audio_to_comfy_3d(
                audio_input["waveform"], audio_input["sample_rate"]
            )

        # Extract waveform and sample_rate
        waveform, sample_rate = audio_from_comfy_2d(
            audio_input, repeat=True, try_gpu=True
        )
        loudness = get_loudness(waveform, sample_rate)

        nchannels, nsamples = waveform.shape

        # Optimize window size to be divisible by 2, 3, and 5
        window_size = int(window_size_seconds * sample_rate)
        if window_size < 16:
            window_size = 16
        window_size = self.optimize_windowsize(window_size)
        window_size = int(window_size / 2) * 2  # Ensure even window size
        half_window_size = int(window_size / 2)

        # Correct the end of the waveform by applying a fade-out
        end_size = int(sample_rate * 0.05)
        if end_size < 16:
            end_size = 16
        fade_out = torch.linspace(
            1.0, 0.0, end_size, device=waveform.device, dtype=waveform.dtype
        )
        waveform[:, -end_size:] = waveform[:, -end_size:] * fade_out

        # Compute displacement
        start_pos = 0.0
        displace_pos = (window_size * 0.5) / stretch_factor

        # Create custom window function as in original code
        window = torch.pow(
            1.0
            - torch.pow(
                torch.linspace(
                    -1.0, 1.0, window_size, device=waveform.device, dtype=waveform.dtype
                ),
                2.0,
            ),
            1.25,
        )

        # Initialize old windowed buffer
        old_windowed_buf = torch.zeros(
            (nchannels, window_size), device=waveform.device, dtype=waveform.dtype
        )

        # Initialize list to store output frames
        output_frames = []

        # Processing loop
        frame_count = 0
        while True:
            # Get the windowed buffer
            istart_pos = int(math.floor(start_pos))
            buf = waveform[:, istart_pos : istart_pos + window_size]
            if buf.shape[1] < window_size:
                padding = window_size - buf.shape[1]
                buf = torch.nn.functional.pad(buf, (0, padding), "constant", 0.0)
            buf = buf * window

            # FFT: Real FFT since the input is real
            freqs = torch.fft.rfft(buf, dim=1)

            # Get amplitudes and randomize phases
            amplitudes = freqs.abs()
            phases = (
                torch.rand(freqs.shape, device=waveform.device, dtype=waveform.dtype)
                * 2
                * math.pi
            )
            freqs = amplitudes * torch.exp(1j * phases)

            # Inverse FFT
            buf_ifft = torch.fft.irfft(freqs, n=window_size, dim=1)

            # Window again the output buffer
            buf_ifft = buf_ifft * window

            # Overlap-add the output
            output = (
                buf_ifft[:, :half_window_size] + old_windowed_buf[:, half_window_size:]
            )
            old_windowed_buf = buf_ifft

            # Append to output_frames
            output_frames.append(output)

            # Increment start_pos
            start_pos += displace_pos
            frame_count += 1

            # Check if we have reached the end of the input
            if start_pos >= nsamples:
                break

        # Concatenate all output frames horizontally
        output_array = torch.cat(output_frames, dim=1)

        # LUFS Normalization
        output_tensor = lufs_normalization(output_array, sample_rate, loudness)

        # Return as audio dictionary
        return audio_to_comfy_3d(output_tensor, sample_rate)

    @staticmethod
    def optimize_windowsize(n: int) -> int:

        orig_n = n
        while True:
            n = orig_n
            while (n % 2) == 0:
                n //= 2
            while (n % 3) == 0:
                n //= 3
            while (n % 5) == 0:
                n //= 5

            if n < 2:
                break
            orig_n += 1
        return orig_n
