#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    various conversion methods
"""

import torch
import torch.nn as nn
from torch.nn import Module
from ..core.sampling import oversample, downsample
from ..core.harmonics import enahnce_harmonics_23

from typing import List


def sigmoid_saturation(audio: torch.Tensor, drive: float = 50.0) -> torch.Tensor:
    """
    Apply sigmoid saturation with drive control.
    Parameters:
        audio: [channels, samples] input audio signal in the range [-1, 1].
        drive: Controls the steepness of the sigmoid in range [0, 100.0].
    Returns:
        Saturated audio signal.
    """
    k = 0.1 + (drive / 100.0) * 10  # Map drive to steepness control
    normalized = (audio + 1) / 2
    saturated = 1 / (1 + torch.exp(-k * (normalized - 0.5)))
    return 2 * saturated - 1


def tanh_saturation(audio: torch.Tensor, drive: float = 50.0) -> torch.Tensor:
    """
    Apply tanh saturation with drive control.
    Parameters:
        audio: Input audio signal (Tensor).
        drive: Controls the strength of tanh effect in range [0, 100.0].
    Returns:
        Saturated audio signal.
    """

    audio = audio / torch.max(torch.abs(audio))
    k = 0.1 + (drive / 100.0) * 10  # Map drive to scaling factor
    k_tensor = torch.tensor(
        k, dtype=audio.dtype, device=audio.device
    )  # Convert k to Tensor
    return torch.tanh(
        k_tensor * audio
    )  # / torch.tanh(k_tensor)  # Normalize output range


def poly_saturation(audio: torch.Tensor, drive: float = 50.0) -> torch.Tensor:
    """
    Apply cubic polynomial saturation with drive control.
    Parameters:
        audio: Input audio signal.
        drive: Controls the strength of the cubic term in range [0, 100.0].
    Returns:
        Saturated audio signal.
    """
    c3 = 0.01 + (drive / 100.0) * 0.3  # Map drive to nonlinearity strength
    return audio - c3 * audio**3


def logarithmic_mapping(audio: torch.Tensor, drive: float = 50.0) -> torch.Tensor:
    """
    Apply logarithmic mapping with drive control.
    Parameters:
        audio: Input audio signal.
        drive: Controls the scaling of the logarithmic mapping.
    Returns:
        Saturated audio signal.
    """
    max_value = 0.1 + (drive / 100.0) * 10  # Map drive to maximum scaling
    return (
        torch.sign(audio)
        * torch.log1p(torch.abs(audio * max_value))
        / torch.log1p(torch.tensor(max_value))
    )


class Saturator(Module):
    @staticmethod
    def get_modes() -> List[str]:
        return ["poly", "soft", "tanh", "sig", "log"]

    def __init__(
        self,
        drive: float = 0.5,
        order: int = 3,
        sample_rate: int = 48000,
        mode: str = "poly",
        oversample_factor: int = 4,
    ):
        super(Saturator, self).__init__()
        self.order: int = order
        self.sample_rate: int = sample_rate
        self.mode: str = mode
        self.drive: float = drive
        self.oversample_factor: int = oversample_factor
        self.harmonics_level: int = 0

        order = order  # third order polynomial approximation

        # Input scaling and output gain (adjust as needed)
        self.input_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.output_gain = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y, sample_rate = oversample(x, self.sample_rate, factor=self.oversample_factor)

        y = enahnce_harmonics_23(y, sample_rate, gain_db_base=self.harmonics_level)

        if self.mode == "poly":
            y = poly_saturation(y, drive=self.drive)
        elif self.mode == "tanh":
            y = tanh_saturation(y, drive=self.drive)
        elif self.mode == "sig":
            y = sigmoid_saturation(y, drive=self.drive)
        elif self.mode == "log":
            y = logarithmic_mapping(y, drive=self.drive)

        y = downsample(y, sample_rate, factor=self.oversample_factor)

        return y


def saturator_get_modes() -> List[str]:
    return ["poly", "tanh", "sig", "log"]


def saturator(
    audio_in: torch.Tensor,
    mode: str = "poly",
    sample_rate: int = 44100,
    drive: float = 1.5,  # Removed lookahead
    oversample_factor: int = 4,
    harmonics_level: float = 1.2,
) -> torch.Tensor:
    y = audio_in.clone()
    # loudness = get_loudness(audio_in, sample_rate)
    # y = automatic_gain_control(audio_in)
    # y = y*drive_pre
    y, _sample_rate = oversample(y, sample_rate, factor=oversample_factor)

    # y = enahnce_harmonics_23(y, _sample_rate, gain_db_base=harmonics_level)

    if mode == "poly":
        drive = drive * 2
        y = poly_saturation(y, drive=drive)
    elif mode == "tanh":
        drive = drive / 3
        y = tanh_saturation(y, drive=drive)
    elif mode == "sig":
        y = sigmoid_saturation(y, drive=drive)
    elif mode == "log":
        y = logarithmic_mapping(y, drive=drive)

    y = downsample(y, sample_rate, factor=oversample_factor)
    # y = lufs_normalization(y, sample_rate, loudness)

    return y
