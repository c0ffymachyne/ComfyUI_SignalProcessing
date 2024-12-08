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
from typing import Tuple, Dict

import torchaudio.functional as F

from ..core.utilities import comfy_root_to_syspath
comfy_root_to_syspath() # add comfy to sys path for dev

from ..core.io import audio_from_comfy_3d, audio_to_comfy_3d
from ..core.loudness import lufs_normalization, get_loudness


class SignalProcessingPitchShifter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_audio": ("AUDIO",),  # Input audio
                "pitch_shift_factor": ("INT", {"default": 2, "min": -12*4, "max": 12*4, "step": 1}),
            },
            "optional": {},
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("output_audio",)
    CATEGORY = "Audio Processing"
    FUNCTION = "process"

    def process(
        self,
        input_audio: Dict[str, torch.Tensor],
        pitch_shift_factor: int = 2,
    ) -> Tuple[Dict[str, torch.Tensor]]:

        try_gpu : bool = True
        waveform, sample_rate = audio_from_comfy_3d(input_audio,try_gpu=try_gpu)

        loudness = get_loudness(waveform,sample_rate)

        pitch_shifted_waveform = F.pitch_shift(waveform,sample_rate,pitch_shift_factor)
        pitch_shifted_waveform = lufs_normalization(pitch_shifted_waveform,sample_rate,loudness)

        return audio_to_comfy_3d(pitch_shifted_waveform,sample_rate)


if __name__ == "__main__":

    import torchaudio
    from pathlib import Path
    from ..core.io import audio_from_comfy_2d, audio_to_comfy_3d, from_disk_as_raw_2d
    from ..core.mixing import combine_audio_files

    node = SignalProcessingPitchShifter()
    types = node.INPUT_TYPES()

    samples_path = Path('ComfyUI_SignalProcessing/audio/samples')

    samples = list(samples_path.rglob('*.*'))

    source_path = samples[1].absolute()
    source_audio,source_audio_sample_rate = from_disk_as_raw_2d(source_path)

    input = audio_to_comfy_3d(source_audio,source_audio_sample_rate)[0]

    pitch_shift_factor=-4

    result = node.process(input,pitch_shift_factor=pitch_shift_factor)[0]

    output_audio, sample_rate_audio = audio_from_comfy_2d(result)

    combined = combine_audio_files( source_audio.cpu(),
                                    output_audio.cpu(),
                                    sample_rate_audio,
                                    chunk_duration=4.0
                                    )
    torchaudio.save('ComfyUI_SignalProcessing/audio/tests/pitchshifter.wav',combined.cpu(),sample_rate_audio)