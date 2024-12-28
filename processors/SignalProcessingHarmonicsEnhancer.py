#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    Classic Audio filter set
"""
from ast import literal_eval
import torch
import torchaudio
from typing import Dict, Any, List, Tuple, Union

from ..core.utilities import comfy_root_to_syspath
from ..core.io import audio_to_comfy_3d, audio_from_comfy_2d
from ..core.loudness import lufs_normalization, get_loudness

comfy_root_to_syspath()  # add comfy to sys path for dev


class SignalProcessingHarmonicsEnhancer:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "audio_input": ("AUDIO",),
                "harmonics": ("STRING", {"default": "1,3,5,7,9"}),
                "mode": (["detect base frequency", "use base frequency"],),
                "base_frequency": ("FLOAT", {"default": 440, "min": 0, "max": 20000}),
                "gain_db": ("INT", {"default": 5, "min": 0, "max": 500, "step": 1}),
                "Q": ("FLOAT", {"default": 0.707, "min": 0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    CATEGORY = "Signal Processing"
    FUNCTION = "process"

    def process(
        self,
        audio_input: Dict[str, Union[torch.Tensor, int]],
        harmonics: str = "1,3,5,7,9",
        mode: str = "detect base frequency",
        base_frequency: int = 440,
        gain_db: int = 5,
        Q: float = 0.707,
    ) -> Tuple[Dict[str, Union[torch.Tensor, int]]]:
        waveform, sample_rate = audio_from_comfy_2d(audio_input, try_gpu=True)
        loudness = get_loudness(waveform, sample_rate)

        try:
            harmonics_list: List[int] = [literal_eval(x) for x in harmonics.split(",")]
        except Exception:
            raise RuntimeWarning(
                "Invalid Harmonics Format. Please delimit integers by a comma \
                    ',' like this 1,,3,5,7,9 "
            )
        if mode == "detect base frequency":
            filtered_waveform = self.enhance_harmonics(
                waveform, sample_rate, harmonics=harmonics_list, gain_db=gain_db, Q=Q
            )
        elif mode == "use base frequency":
            filtered_waveform = self.enhance_harmonics(
                waveform,
                sample_rate,
                harmonics=harmonics_list,
                gain_db=gain_db,
                base_frequency=base_frequency,
                Q=Q,
            )

        filtered_waveform = lufs_normalization(filtered_waveform, sample_rate, loudness)
        return audio_to_comfy_3d(filtered_waveform, sample_rate)

    def add_harmonics(self, audio: torch.Tensor, gain: float = 1.2) -> torch.Tensor:
        # Apply saturation using a tanh curve
        harmonic_audio = torch.tanh(audio * gain)
        return harmonic_audio

    def detect_fundamental(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        # Estimate the fundamental frequency using a pitch detection method
        pitch = torchaudio.functional.detect_pitch_frequency(audio, sample_rate)

        return pitch

    def detect_fundamental_mean(self, audio: torch.Tensor, sample_rate: int) -> int:
        # Estimate the fundamental frequency using a pitch detection method
        pitch = torchaudio.functional.detect_pitch_frequency(audio, sample_rate)

        return int(pitch.mean().item())

    def enhance_harmonics(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        harmonics: List[int] = [1, 3, 5, 7, 9, 11],
        gain_db: float = 5,
        base_frequency: float = 0,
        Q: float = 0.707,
    ) -> torch.Tensor:
        # Detect the base frequency
        if base_frequency == 0:
            base_frequency = self.detect_fundamental_mean(audio, sample_rate)
            if base_frequency <= 0:  # Fallback if pitch detection fails
                base_frequency = 440  # Use a default base frequency

        # Apply EQ boosts to specific harmonic frequencies
        for harmonic in harmonics:
            freq = base_frequency * harmonic
            if freq < sample_rate / 2:  # Ensure it's within the Nyquist frequency
                audio = torchaudio.functional.equalizer_biquad(
                    audio, sample_rate, center_freq=freq, gain=gain_db, Q=Q
                )

        return audio
