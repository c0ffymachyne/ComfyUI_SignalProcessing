#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    Audio widening node
"""

import os
import sys
import math
import torch

from typing import Dict, Any, Tuple, Union
from ..core.io import audio_from_comfy_2d, audio_to_comfy_3d
from ..core.loudness import lufs_normalization, get_loudness
from ..core.widening import (
    StereoWidenerFrequencyBased,
    DecorrelationType,
    FilterbankType,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))


def interpolate(t: float, a: float, b: float) -> float:
    if not 0.0 <= t <= 1.0:
        raise ValueError("t must be in the range [0.0, 1.0]")
    return a + t * (b - a)


class SignalProcessingStereoWidening:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "mode": (["decorrelation", "simple"],),
                "audio_input": ("AUDIO",),
            },
            "optional": {
                "width": (
                    "FLOAT",
                    {"default": 6.0, "min": 1.0, "max": 8.0, "step": 0.1},
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("widened_audio",)
    CATEGORY = "Signal Processing"
    FUNCTION = "process"

    def process(
        self,
        mode: str,
        audio_input: Dict[str, Union[torch.Tensor, int]],
        width: float = 1.2,
    ) -> Tuple[Dict[str, Union[torch.Tensor, int]]]:
        """
        Widen stereo audio or convert mono audio to wide stereo
        using the provided widening algorithm.

        Parameters:
            audio_input (Dict): Dictionary containing 'waveform' and 'sample_rate'.
            width (float): Width factor (>1.0).

        Returns:
            Tuple[Dict[str, torch.Tensor]]: Dictionary with widened 'waveform' and 'sample_rate'.
        """

        waveform, sample_rate = audio_from_comfy_2d(
            audio_input, repeat=False, try_gpu=True
        )
        channels, num_samples = waveform.shape

        loudness = get_loudness(waveform, sample_rate)

        if mode == "simple":

            if channels not in [1, 2]:
                raise ValueError(
                    f"Unsupported number of channels: {channels}. \
                        Only mono and stereo are supported."
                )

            # Calculate coefficients based on the provided width parameter
            width_coeff = 1.0 / max(1.0 + width, 2.0)  # Scalar

            coef_mid = 1.0 * width_coeff  # Coefficient for mid
            coef_sides = width * width_coeff  # Coefficient for sides

            if channels == 2:
                # Stereo to Widened Stereo
                L = waveform[0, :]  # Left channel
                R = waveform[1, :]  # Right channel

                # Apply the widening algorithm
                mid = (L + R) * coef_mid  # Mid signal
                sides = (R - L) * coef_sides  # Side signal

                widened_L = mid - sides  # New Left channel
                widened_R = mid + sides  # New Right channel

                # Stack the widened channels back into a stereo waveform
                widened_waveform = torch.stack(
                    (widened_L, widened_R), dim=0
                )  # [2, samples]

            elif channels == 1:
                # Mono to Wide Stereo
                L = waveform[0, :].clone()  # Duplicate mono channel to Left
                R = waveform[0, :].clone()  # Duplicate mono channel to Right

                # Apply the widening algorithm
                mid = (L + R) * coef_mid  # Mid signal
                sides = (R - L) * coef_sides  # Side signal

                widened_L = mid - sides  # New Left channel
                widened_R = mid + sides  # New Right channel

                # Stack the widened channels into a stereo waveform
                widened_waveform = torch.stack(
                    (widened_L, widened_R), dim=0
                )  # [2, samples]

            widened_waveform = lufs_normalization(
                widened_waveform, sample_rate, loudness
            )

            return audio_to_comfy_3d(widened_waveform, sample_rate)

        if mode == "decorrelation":

            waveform = waveform.cpu()

            decorellation_type = DecorrelationType.VELVET
            filterbank_type = FilterbankType.ENERGY_PRESERVE
            start_value = 0.0
            end_value = math.pi / 2

            if width > 1.0:
                width = 1.0

            beta = interpolate(width, start_value, end_value)
            cutoff_frequency_hz = 22000  # sample_rate//2 # max possible
            cutoff_frequency_hz = (sample_rate // 2) - 10  # max possible

            stereoWidener = StereoWidenerFrequencyBased(
                waveform,
                sample_rate,
                filterbank_type,
                decorellation_type,
                [beta, beta],
                cutoff_frequency_hz,
            )
            widener_result = stereoWidener.process()
            widened_waveform = torch.from_numpy(widener_result)
            widened_waveform = widened_waveform.T

            widened_waveform = lufs_normalization(
                widened_waveform, sample_rate, loudness
            )

            return audio_to_comfy_3d(widened_waveform, sample_rate)

        return audio_to_comfy_3d(waveform, sample_rate)
