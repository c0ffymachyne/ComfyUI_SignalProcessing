#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    Classic Audio filter set

Reference:
    https://ccrma.stanford.edu/~jatin/ComplexNonlinearities/Hysteresis.html
    https://viennatalk.mdw.ac.at/papers/Pap_01_79_Tronchin.pdf
    https://jatinchowdhury18.medium.com/complex-nonlinearities-episode-3-hysteresis-fdeb2cd3e3f6
    https://ccrma.stanford.edu/~dtyeh/papers/yeh07_dafx_clipode.pdf
"""

import torch
from typing import Dict, Any, Tuple, Union

from ..core.io import audio_to_comfy_3d, audio_from_comfy_2d
from ..core.saturation import saturator, saturator_get_modes
from ..core.loudness import get_loudness, lufs_normalization


class SignalProcessingSaturation:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "audio_input": ("AUDIO",),
                "mode": (saturator_get_modes(),),
                "drive": (
                    "FLOAT",
                    {"default": 50.0, "min": 0.0, "max": 200.0, "step": 0.1},
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
        mode: str = "poly",
        drive: float = 50.0,
    ) -> Tuple[Dict[str, Union[torch.Tensor, int]]]:
        waveform, sample_rate = audio_from_comfy_2d(audio_input, try_gpu=True)

        loudness = get_loudness(waveform, sample_rate)

        filtered_waveform = saturator(
            waveform, mode=mode, sample_rate=sample_rate, drive=drive
        )

        filtered_waveform = lufs_normalization(filtered_waveform, sample_rate, loudness)

        return audio_to_comfy_3d(filtered_waveform, sample_rate)
