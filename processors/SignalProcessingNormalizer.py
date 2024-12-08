#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    Varios normalizatin techqniues node
"""

from ..core.io import audio_to_comfy_3d, audio_from_comfy_2d
from ..core.loudness import (
    rms_normalization,
    lufs_normalization,
    peak_normalization,
    automatic_gain_control,
)


class SignalProcessingNormalizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_input": ("AUDIO",),
                "mode": (["lufs", "rms", "peak", "auto"],),
                "target_rms": (
                    "FLOAT",
                    {"default": 0.1, "min": 0, "max": 10.0, "step": 0.1},
                ),
                "target_lufs_db": (
                    "FLOAT",
                    {"default": -14.0, "min": -100, "max": 100.0, "step": 0.1},
                ),
                "target_peak": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
                "target_auto": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
                "target_auto_alpha": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 10.0, "step": 0.1},
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
        mode: str,
        target_rms: float,
        target_lufs_db: float,
        target_peak: float,
        target_auto: float,
        target_auto_alpha: float,
    ):

        waveform, sample_rate = audio_from_comfy_2d(audio_input, try_gpu=True)

        if mode == "rms":
            processed_waveform = rms_normalization(waveform, target_rms)
        elif mode == "lufs":
            processed_waveform = lufs_normalization(
                waveform, sample_rate, target_lufs_db
            )
        elif mode == "peak":
            processed_waveform = peak_normalization(waveform, target_peak)
        elif mode == "auto":
            processed_waveform = automatic_gain_control(
                waveform, target_auto, target_auto_alpha
            )
        else:
            processed_waveform = waveform

        return audio_to_comfy_3d(processed_waveform, sample_rate)
