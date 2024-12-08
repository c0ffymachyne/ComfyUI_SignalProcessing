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
from typing import Tuple, List, Dict

from ..core.utilities import comfy_root_to_syspath

comfy_root_to_syspath()  # add comfy to sys path for dev

from ..core.io import audio_from_comfy_2d, audio_to_comfy_3d
from ..core.loudness import lufs_normalization, get_loudness


class SignalProcessingPaulStretch:
    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        return {
            "required": {
                "audio": ("AUDIO", {"forceInput": True}),
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
        audio: Dict[str, torch.Tensor],
        stretch_factor: float,
        window_size_seconds: float,
    ) -> Tuple[Dict[str, torch.Tensor]]:

        # Conditional processing: If stretch_factor is 1.0, return original audio
        if stretch_factor == 1.0:
            return audio_to_comfy_3d(audio["waveform"], audio["sample_rate"])

        # Extract waveform and sample_rate
        waveform, sample_rate = audio_from_comfy_2d(audio, repeat=True, try_gpu=True)
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
                n /= 2
            while (n % 3) == 0:
                n /= 3
            while (n % 5) == 0:
                n /= 5

            if n < 2:
                break
            orig_n += 1
        return orig_n


if __name__ == "__main__":

    import torchaudio
    from pathlib import Path
    from ..core.io import audio_from_comfy_2d, audio_to_comfy_3d, from_disk_as_raw_2d
    from ..core.mixing import combine_audio_files

    node = SignalProcessingPaulStretch()
    types = node.INPUT_TYPES()

    samples_path = Path("ComfyUI_SignalProcessing/audio/samples")

    samples = list(samples_path.rglob("*.*"))

    print(samples)

    source_path = samples[1].absolute()
    source_audio, source_audio_sample_rate = from_disk_as_raw_2d(source_path)

    input = audio_to_comfy_3d(source_audio, source_audio_sample_rate)[0]

    window_size_seconds = 0.25
    stretch_factor = 2.0

    result = node.process(
        input, stretch_factor=stretch_factor, window_size_seconds=window_size_seconds
    )[0]

    output_audio, sample_rate_audio = audio_from_comfy_2d(result)

    combined = combine_audio_files(
        source_audio.cpu(), output_audio.cpu(), sample_rate_audio, chunk_duration=4.0
    )

    torchaudio.save(
        "ComfyUI_SignalProcessing/audio/tests/paulstretch.wav",
        combined.cpu(),
        sample_rate_audio,
    )

    # set console to comfy ComfyUI-0.2.4/custom_nodes and run below command
    # export coffy_local_dev=1
    # python3 -m ComfyUI_SignalProcessing.effects.SignalProcessingPitchShifter
