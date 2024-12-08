#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    various conversion methods
"""

import math
import torch
from torch import nn

class TubeValveSimulator:
    def __init__(self, sample_rate, coefficients=None):
        """
        Initialize the Tube/Valve simulator using Volterra Series Expansion.

        Args:
            sample_rate (int): The sample rate of the audio signal.
            coefficients (dict): Coefficients for Volterra series terms.
                                Example: {'linear': 1.0, 'quadratic': 0.5, 'cubic': 0.2}
        """
        self.sample_rate = sample_rate
        self.coefficients = coefficients if coefficients else {
            'linear': 1.0,
            'quadratic': 0.5,
            'cubic': 0.2,
        }

    def process(self, audio_input):
        """
        Process the audio signal to simulate tube/valve characteristics.

        Args:
            audio_input (torch.Tensor): The input audio signal (1D or 2D tensor).

        Returns:
            torch.Tensor: The processed audio signal.
        """
        # Ensure input is a float tensor
        #audio_input = audio_input.to(torch.float32)

        # Volterra series expansion
        linear_term = self.coefficients['linear'] * audio_input
        quadratic_term = self.coefficients['quadratic'] * audio_input**2
        cubic_term = self.coefficients['cubic'] * audio_input**3

        # Combine the terms
        processed_audio = linear_term + quadratic_term + cubic_term

        # Optional soft clipping to simulate saturation
        #processed_audio = self.soft_clip(processed_audio)

        return processed_audio

    @staticmethod
    def soft_clip(audio_signal, threshold=0.9):
        """
        Apply soft clipping to the audio signal to limit extreme values.

        Args:
            audio_signal (torch.Tensor): The input audio signal.
            threshold (float): The clipping threshold.

        Returns:
            torch.Tensor: The clipped audio signal.
        """
        return torch.tanh(audio_signal / threshold) * threshold

########################################
# Example Usage
########################################
if __name__ == "__main__":

    from pathlib import Path
    import torchaudio
    from core.io import from_disk_as_raw_2d, audio_to_comfy_3d, audio_from_comfy_2d

    samples_path = Path("/media/broot/zyzx/git/ComfyUI-0.2.4/custom_nodes/ComfyUI_SignalProcessing/audio/inputs/baxandall-normalizer_00003_.flac")

    source_path = samples_path.absolute()
    source_audio, source_audio_sample_rate = from_disk_as_raw_2d(source_path,try_gpu=True)
    input = audio_to_comfy_3d(source_audio, source_audio_sample_rate)[0]


    # Initialize the simulator
    tube_simulator = TubeValveSimulator(source_audio_sample_rate, coefficients={
        'linear': 1.0,
        'quadratic': 0.3,
        'cubic': 0.4,
    })

    compressed = tube_simulator.process(source_audio)
    print("Compressed shape:", compressed.shape)
    print("Device:", compressed.device)

    torchaudio.save(
        "audio/outputs/valve.wav",
        compressed.cpu(),
        source_audio_sample_rate,
    )
