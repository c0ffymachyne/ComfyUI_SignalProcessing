#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    Mixdown node for pad synths
"""

import torch

from typing import Tuple, List, Dict, Union
import torchaudio

from ..core.io import audio_to_comfy_3d
from ..core.loudness import lufs_normalization


class SignalProcessingMixdown:
    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        return {
            "required": {
                "audio_inputs": ("AUDIO_LIST", {"default": []}),
            },
            "optional": {
                "gain_factors": (
                    "FLOAT_LIST",
                    {"default": [], "min": 0.0, "max": 2.0, "step": 0.1},
                ),
                # If empty, default to [1.0] * num_audios
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("mixed_audio",)
    CATEGORY = "Audio Processing"
    FUNCTION = "process"

    def process(
        self,
        audio_inputs: List[Dict[str, Union[torch.Tensor, int]]],
        gain_factors: List[float] = [],
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Mix down multiple audio inputs into a single audio output with optional individual volume controls.

        Parameters:
            audio_inputs (List[Dict]): List of audio inputs, each containing 'waveform' and 'sample_rate'.
            output_normalization (float): Normalization factor for the mixed audio (0.0 to 1.0).
            gain_factors (List[float], optional): List of gain factors for each audio input.

        Returns:
            Tuple[Dict[str, torch.Tensor], int]: Mixed audio with waveform and sample rate.
        """

        if not audio_inputs:
            raise ValueError("No audio inputs provided for mixing.")

        num_audios = len(audio_inputs)

        # Handle gain_factors
        if not gain_factors:
            gain_factors = [1.0] * num_audios
        elif len(gain_factors) != num_audios:
            raise ValueError(
                f"Number of gain factors ({len(gain_factors)}) does not match number of audio inputs ({num_audios})."
            )

        # Extract sample rates and verify consistency
        sample_rates = [audio["sample_rate"] for audio in audio_inputs]
        target_sample_rate = sample_rates[0]

        for idx, sr in enumerate(sample_rates):
            if sr != target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=target_sample_rate
                )
                resampler.to(
                    device=audio_inputs[idx]["waveform"].device,
                    dtype=audio_inputs[idx]["waveform"].dtype,
                )
                audio_inputs[idx]["waveform"] = resampler(audio_inputs[idx]["waveform"])
                audio_inputs[idx]["sample_rate"] = target_sample_rate

        # Determine the maximum length among all audio inputs
        lengths = [audio["waveform"].shape[-1] for audio in audio_inputs]
        max_length = max(lengths)

        # Pad or truncate each audio to match the maximum length and apply gain
        for idx, audio in enumerate(audio_inputs):
            waveform = audio["waveform"]
            current_length = waveform.shape[-1]
            gain = gain_factors[idx]

            if current_length < max_length:
                padding = max_length - current_length
                # Pad with zeros (silence) at the end
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            elif current_length > max_length:
                # Truncate the waveform to max_length
                waveform = waveform[:, :, :max_length]

            # Apply gain
            waveform = waveform * gain

            audio["waveform"] = waveform

        # Sum all waveforms to create the mix
        mixed_waveform = torch.zeros_like(audio_inputs[0]["waveform"])
        for idx, audio in enumerate(audio_inputs):
            mixed_waveform += audio["waveform"]

        mixed_waveform = lufs_normalization(mixed_waveform, target_sample_rate)

        return audio_to_comfy_3d(mixed_waveform, target_sample_rate)
