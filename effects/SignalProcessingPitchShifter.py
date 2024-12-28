#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    Pitch shifting node
"""

import torch
from typing import Tuple, Dict, Any, Union

import torchaudio.functional as F

from ..core.utilities import comfy_root_to_syspath
from ..core.io import audio_from_comfy_3d, audio_to_comfy_3d
from ..core.loudness import lufs_normalization, get_loudness

comfy_root_to_syspath()  # add comfy to sys path for dev


class SignalProcessingPitchShifter:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "audio_input": ("AUDIO",),  # Input audio
                "pitch_shift_factor": (
                    "INT",
                    {"default": 2, "min": -12 * 4, "max": 12 * 4, "step": 1},
                ),
            },
            "optional": {},
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("output_audio",)
    CATEGORY = "Signal Processing"
    FUNCTION = "process"

    def process(
        self,
        audio_input: Dict[str, Union[torch.Tensor, int]],
        pitch_shift_factor: int = 2,
    ) -> Tuple[Dict[str, Union[torch.Tensor, int]]]:

        try_gpu: bool = True
        waveform, sample_rate = audio_from_comfy_3d(audio_input, try_gpu=try_gpu)

        loudness = get_loudness(waveform, sample_rate)

        pitch_shifted_waveform = F.pitch_shift(
            waveform, sample_rate, pitch_shift_factor
        )
        pitch_shifted_waveform = lufs_normalization(
            pitch_shifted_waveform, sample_rate, loudness
        )

        return audio_to_comfy_3d(pitch_shifted_waveform, sample_rate)
