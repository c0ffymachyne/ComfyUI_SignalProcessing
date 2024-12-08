#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    Baxandall shelf EQ
    references used : # reference https://webaudio.github.io/Audio-EQ-Cookbook/Audio-EQ-Cookbook.txt
"""

import torch
import torchaudio
import math

from ..core.io import audio_to_comfy_3d, audio_from_comfy_3d
from ..core.loudness import lufs_normalization, get_loudness

import torch
import math


class SignalProcessingBaxandallEQ:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_input": ("AUDIO",),
                "bass_gain_db": (
                    "FLOAT",
                    {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.1},
                ),
                "treble_gain_db": (
                    "FLOAT",
                    {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.1},
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("processed_audio",)
    CATEGORY = "Audio Processing"
    FUNCTION = "process"

    def process(self, audio_input, bass_gain_db=0.0, treble_gain_db=0.0):
        waveform, sample_rate = audio_from_comfy_3d(audio_input, try_gpu=True)
        loudness = get_loudness(waveform, sample_rate)

        # Apply Bass Shelf (low shelf) using RBJ formula
        b_bass, a_bass = self.design_rbj_shelf(
            sample_rate, freq=100.0, gain_db=bass_gain_db, shelf_type="low"
        )
        waveform = torchaudio.functional.lfilter(
            waveform,
            a_bass.to(waveform.device),
            b_bass.to(waveform.device),
            clamp=False,
        )

        # Apply Treble Shelf (high shelf) using RBJ formula
        b_treble, a_treble = self.design_rbj_shelf(
            sample_rate, freq=10000.0, gain_db=treble_gain_db, shelf_type="high"
        )
        waveform = torchaudio.functional.lfilter(
            waveform,
            a_treble.to(waveform.device),
            b_treble.to(waveform.device),
            clamp=False,
        )

        waveform = lufs_normalization(waveform, sample_rate, loudness)
        return audio_to_comfy_3d(waveform, sample_rate)

    def design_rbj_shelf(self, sr, freq, gain_db, shelf_type="low"):
        # RBJ Audio EQ Cookbook shelf filters
        A = 10.0 ** (gain_db / 40.0)
        w0 = 2 * math.pi * freq / sr
        alpha = (
            math.sin(w0) / 2.0 * math.sqrt((A + 1 / A) * (1.0 / 1.0 - 1) + 2.0)
        )  # S=1.0

        cosw0 = math.cos(w0)
        if shelf_type == "low":
            b0 = A * ((A + 1) - (A - 1) * cosw0 + 2 * math.sqrt(A) * alpha)
            b1 = 2 * A * ((A - 1) - (A + 1) * cosw0)
            b2 = A * ((A + 1) - (A - 1) * cosw0 - 2 * math.sqrt(A) * alpha)
            a0 = (A + 1) + (A - 1) * cosw0 + 2 * math.sqrt(A) * alpha
            a1 = -2 * ((A - 1) + (A + 1) * cosw0)
            a2 = (A + 1) + (A - 1) * cosw0 - 2 * math.sqrt(A) * alpha
        else:  # high shelf
            b0 = A * ((A + 1) + (A - 1) * cosw0 + 2 * math.sqrt(A) * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * cosw0)
            b2 = A * ((A + 1) + (A - 1) * cosw0 - 2 * math.sqrt(A) * alpha)
            a0 = (A + 1) - (A - 1) * cosw0 + 2 * math.sqrt(A) * alpha
            a1 = 2 * ((A - 1) - (A + 1) * cosw0)
            a2 = (A + 1) - (A - 1) * cosw0 - 2 * math.sqrt(A) * alpha

        b = torch.tensor([b0 / a0, b1 / a0, b2 / a0], dtype=torch.float64)
        a = torch.tensor([1.0, a1 / a0, a2 / a0], dtype=torch.float64)
        return b, a


class SignalProcessingBaxandall3BandEQ:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_input": ("AUDIO",),
                "bass_gain_db": (
                    "FLOAT",
                    {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.1},
                ),
                "mid_gain_db": (
                    "FLOAT",
                    {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1},
                ),
                "treble_gain_db": (
                    "FLOAT",
                    {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.1},
                ),
                "low_freq": (
                    "FLOAT",
                    {"default": 100.0, "min": 20.0, "max": 500.0, "step": 1.0},
                ),
                "mid_freq": (
                    "FLOAT",
                    {"default": 1000.0, "min": 200.0, "max": 5000.0, "step": 10.0},
                ),
                "high_freq": (
                    "FLOAT",
                    {"default": 10000.0, "min": 2000.0, "max": 20000.0, "step": 100.0},
                ),
                "mid_q": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.1, "max": 10.0, "step": 0.1},
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("processed_audio",)
    CATEGORY = "Audio Processing"
    FUNCTION = "process"

    def process(
        self,
        audio_input,
        bass_gain_db: float = 0.0,
        mid_gain_db: float = 0.0,
        treble_gain_db: float = 0.0,
        low_freq: float = 100.0,
        mid_freq: float = 1000.0,
        high_freq: float = 10000.0,
        mid_q: float = 0.7,
    ):

        waveform, sample_rate = audio_from_comfy_3d(audio_input, try_gpu=True)
        device = waveform.device
        dtype = waveform.dtype
        loudness = get_loudness(waveform, sample_rate)

        # Low shelf filter
        b_low, a_low = self.design_rbj_shelf(
            sample_rate, low_freq, bass_gain_db, shelf_type="low"
        )
        b_low = b_low.to(device=device, dtype=dtype)
        a_low = a_low.to(device=device, dtype=dtype)
        waveform = torchaudio.functional.lfilter(waveform, a_low, b_low, clamp=False)

        # Mid peaking filter
        b_mid, a_mid = self.design_rbj_peak(sample_rate, mid_freq, mid_gain_db, Q=mid_q)
        b_mid = b_mid.to(device=device, dtype=dtype)
        a_mid = a_mid.to(device=device, dtype=dtype)
        waveform = torchaudio.functional.lfilter(waveform, a_mid, b_mid, clamp=False)

        # High shelf filter
        b_high, a_high = self.design_rbj_shelf(
            sample_rate, high_freq, treble_gain_db, shelf_type="high"
        )
        b_high = b_high.to(device=device, dtype=dtype)
        a_high = a_high.to(device=device, dtype=dtype)
        waveform = torchaudio.functional.lfilter(waveform, a_high, b_high, clamp=False)

        # Normalize loudness after EQ
        waveform = lufs_normalization(waveform, sample_rate, loudness)

        return audio_to_comfy_3d(waveform, sample_rate)

    def design_rbj_shelf(self, sr, freq, gain_db, shelf_type="low"):
        # RBJ audio EQ cookbook formula for shelving filters
        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * math.pi * freq / sr
        # Slope S=1.0 (Baxandall-like gentle slope)
        S = 1.0
        alpha = math.sin(w0) / 2.0 * math.sqrt((A + 1.0 / A) * (1.0 / S - 1.0) + 2.0)
        cosw0 = math.cos(w0)

        if shelf_type == "low":
            b0 = A * ((A + 1.0) - (A - 1.0) * cosw0 + 2.0 * math.sqrt(A) * alpha)
            b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cosw0)
            b2 = A * ((A + 1.0) - (A - 1.0) * cosw0 - 2.0 * math.sqrt(A) * alpha)
            a0 = (A + 1.0) + (A - 1.0) * cosw0 + 2.0 * math.sqrt(A) * alpha
            a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cosw0)
            a2 = (A + 1.0) + (A - 1.0) * cosw0 - 2.0 * math.sqrt(A) * alpha
        else:
            # high shelf
            b0 = A * ((A + 1.0) + (A - 1.0) * cosw0 + 2.0 * math.sqrt(A) * alpha)
            b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cosw0)
            b2 = A * ((A + 1.0) + (A - 1.0) * cosw0 - 2.0 * math.sqrt(A) * alpha)
            a0 = (A + 1.0) - (A - 1.0) * cosw0 + 2.0 * math.sqrt(A) * alpha
            a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cosw0)
            a2 = (A + 1.0) - (A - 1.0) * cosw0 - 2.0 * math.sqrt(A) * alpha

        b = torch.tensor([b0 / a0, b1 / a0, b2 / a0], dtype=torch.float64)
        a = torch.tensor([1.0, a1 / a0, a2 / a0], dtype=torch.float64)
        return b, a

    def design_rbj_peak(self, sr, freq, gain_db, Q=0.7):
        # RBJ audio EQ cookbook peak filter
        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * math.pi * freq / sr
        alpha = math.sin(w0) / (2.0 * Q)
        cosw0 = math.cos(w0)

        b0 = 1.0 + alpha * A
        b1 = -2.0 * cosw0
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * cosw0
        a2 = 1.0 - alpha / A

        b = torch.tensor([b0 / a0, b1 / a0, b2 / a0], dtype=torch.float64)
        a = torch.tensor([1.0, a1 / a0, a2 / a0], dtype=torch.float64)
        return b, a


if __name__ == "__main__":
    import torchaudio
    from pathlib import Path
    from ..core.io import from_disk_as_raw_2d, audio_to_comfy_3d, audio_from_comfy_2d

    node = SignalProcessingBaxandallEQ()
    samples_path = Path("ComfyUI_SignalProcessing/audio/inputs/song.mp4")

    source_path = samples_path.absolute()
    source_audio, source_audio_sample_rate = from_disk_as_raw_2d(source_path)
    input = audio_to_comfy_3d(source_audio, source_audio_sample_rate)[0]

    # Test with some gain settings
    result = node.process(input, bass_gain_db=5.0, treble_gain_db=-3.0)[0]

    output_audio, sample_rate_audio = audio_from_comfy_2d(result)

    # Save output for analysis
    torchaudio.save(
        "ComfyUI_SignalProcessing/audio/tests/baxandall_eq.wav",
        output_audio.cpu(),
        sample_rate_audio,
    )

    node = SignalProcessingBaxandall3BandEQ()

    # Example usage:
    # Provide a test audio file path
    test_audio_path = Path("ComfyUI_SignalProcessing/audio/samples/test_audio.wav")
    source_audio, source_audio_sample_rate = from_disk_as_raw_2d(test_audio_path)
    input_tensors = audio_to_comfy_3d(source_audio, source_audio_sample_rate)[0]

    # Apply EQ with some settings:
    result = node.process(
        input_tensors,
        bass_gain_db=4.0,
        mid_gain_db=-2.0,
        treble_gain_db=3.0,
        low_freq=100,
        mid_freq=1000,
        high_freq=10000,
        mid_q=0.7,
    )[0]

    output_audio, out_sr = audio_from_comfy_2d(result)
    torchaudio.save(
        "ComfyUI_SignalProcessing/audio/tests/3band_baxandall_eq_output.wav",
        output_audio.cpu(),
        out_sr,
    )
