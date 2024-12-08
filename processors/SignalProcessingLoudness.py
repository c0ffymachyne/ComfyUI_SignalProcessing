#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    Loudness node
"""

from ..core.io import audio_from_comfy_2d
from ..core.loudness import get_loudness


class SignalProcessingLoudness:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_input": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("loudness",)
    CATEGORY = "Audio Processing"
    FUNCTION = "process"

    def process(self, audio_input):
        waveform, sample_rate = audio_from_comfy_2d(audio_input, try_gpu=True)

        loudness: float = get_loudness(waveform, sample_rate)

        return (loudness,)
