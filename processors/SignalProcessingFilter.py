#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
     The code defines a classic audio filter set for performing various audio processing tasks such as filtering out unwanted frequencies
"""

import torch
import torchaudio
from typing import Dict, Any, Tuple, Union

from ..core.io import audio_to_comfy_3d, audio_from_comfy_3d
from ..core.loudness import lufs_normalization, get_loudness


class SignalProcessingFilter:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "audio_input": ("AUDIO", {"forceInput": True}),
                "cutoff": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "filter_type": (
                    ["lowpass", "highpass", "bandpass", "bandstop"],
                    {"default": "lowpass"},
                ),
                "q_factor": (
                    "FLOAT",
                    {"default": 0.707, "min": 0.1, "max": 5.0, "step": 0.01},
                ),  # For resonance/bandwidth
            }
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("audio", "sample_rate")
    CATEGORY = "Signal Processing"
    FUNCTION = "process"

    def process(
        self,
        audio_input: Dict[str, Union[torch.Tensor, int]],
        cutoff: float,
        filter_type: str,
        q_factor: float,
    ) -> Tuple[Dict[str, Union[torch.Tensor, int]]]:
        """
        Apply a specified filter to the input audio.

        Parameters:
            audio (Dict[str, torch.Tensor]): Input audio with 'waveform' and 'sample_rate'.
            cutoff (float): Normalized cutoff frequency (0.0 to 1.0).
            filter_type (str): Type of filter ('lowpass', 'highpass', 'bandpass', 'bandstop').
            q_factor (float): Quality factor determining the filter's bandwidth.

        Returns:
            Tuple[Dict[str, torch.Tensor]]: Filtered audio.
        """

        waveform, sample_rate = audio_from_comfy_3d(audio_input)

        loudness = get_loudness(waveform, sample_rate)

        nyquist = sample_rate / 2.0

        # Define minimum and maximum frequencies for mapping
        log_min = 20.0  # 20 Hz, typical lower bound of human hearing
        log_max = nyquist - 100.0  # Slightly below Nyquist to prevent instability

        # Avoid log(0) by ensuring cutoff is within (0,1)
        cutoff = min(max(cutoff, 1e-6), 1.0 - 1e-6)

        # Logarithmic mapping
        log_min = torch.log(torch.tensor(log_min))
        log_max = torch.log(torch.tensor(log_max))
        log_cutoff = log_min + cutoff * (log_max - log_min)
        cutoff_freq = torch.exp(log_cutoff).item()

        # Choose filter type
        if filter_type == "lowpass":
            filtered_waveform = torchaudio.functional.lowpass_biquad(
                waveform, sample_rate, cutoff_freq, Q=q_factor
            )
        elif filter_type == "highpass":
            filtered_waveform = torchaudio.functional.highpass_biquad(
                waveform, sample_rate, cutoff_freq, Q=q_factor
            )
        elif filter_type in ["bandpass", "bandstop"]:
            center_freq = cutoff_freq
            # Ensure that the bandwidth does not exceed the Nyquist frequency

            if filter_type == "bandpass":
                filtered_waveform = torchaudio.functional.bandpass_biquad(
                    waveform, sample_rate, center_freq, Q=q_factor
                )
            else:  # bandstop
                filtered_waveform = torchaudio.functional.band_biquad(
                    waveform, sample_rate, center_freq, Q=q_factor
                )
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

        filtered_waveform = lufs_normalization(filtered_waveform, sample_rate, loudness)

        return audio_to_comfy_3d(filtered_waveform, sample_rate)
