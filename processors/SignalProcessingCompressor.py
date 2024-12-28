#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
  This file contains a description of a compressor node that utilizes a CUDA-optimized kernel.
"""
import torch
from typing import Dict, Any, Tuple, Union

from ..core.io import audio_to_comfy_3d, audio_from_comfy_2d
from ..core.compression import compressor
from ..core.loudness import get_loudness, lufs_normalization


class SignalProcessingCompressor:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "audio_input": ("AUDIO",),
                "comp": (
                    "FLOAT",
                    {"default": -0.3, "min": -2.0, "max": 2.0, "step": 0.01},
                ),
                "attack": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.01, "max": 100.0, "step": 0.01},
                ),
                "release": (
                    "FLOAT",
                    {"default": 60.0, "min": 0.01, "max": 1000.0, "step": 0.1},
                ),
                "filter_param": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01},
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
        comp: float = -0.3,  # Compression/expansion factor
        attack: float = 0.1,  # Attack time in ms
        release: float = 60.0,  # Release time in ms
        filter_param: float = 0.3,  # Filter parameter < 1
    ) -> Tuple[Dict[str, Union[torch.Tensor, int]]]:
        """
        Apply compression or expansion to the audio input using CUDA.

        Parameters:
            audio_input (Dict[str, Union[torch.Tensor, int]]): Input audio waveform and sample rate.
            comp (float): Compression/expansion factor.
            attack (float): Attack time in milliseconds.
            release (float): Release time in milliseconds.
            filter_param (float): Filter parameter for envelope smoothing.

        Returns:
            Tuple[Dict[str, Union[torch.Tensor, int]]]: Compressed audio and sample rate.
        """
        # Extract waveform and sample rate
        waveform, sample_rate = audio_from_comfy_2d(audio_input, try_gpu=True)

        loudness = get_loudness(waveform, sample_rate=sample_rate)

        # Apply the compressor kernel
        filtered_waveform, _ = compressor(
            waveform,
            sample_rate,
            comp=comp,
            attack=attack,
            release=release,
            a=filter_param,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        filtered_waveform = lufs_normalization(
            filtered_waveform, sample_rate=sample_rate, target_lufs=loudness
        )

        # Return the processed audio
        return audio_to_comfy_3d(filtered_waveform, sample_rate)
