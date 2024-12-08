#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    various normalization methods
"""

import torch
import pyloudnorm as pyln  # pip install pyloudnorm
import torch


def rms_normalization(audio: torch.Tensor, target_rms: float = 0.1):
    rms = torch.sqrt(torch.mean(audio**2))
    scaling_factor = target_rms / rms
    normalized_audio = audio * scaling_factor
    return normalized_audio


def lufs_normalization(
    audio: torch.Tensor, sample_rate: float, target_lufs: float = -14.0
):
    meter = pyln.Meter(sample_rate)  # Create a loudness meter
    audio = audio.T.cpu().numpy()
    loudness = meter.integrated_loudness(audio)  # Current LUFS

    loudness_offset = target_lufs - loudness
    normalized_audio = audio * (10 ** (loudness_offset / 20.0))

    result = torch.from_numpy(normalized_audio)
    result = result.T

    return result


def peak_normalization(audio, target_peak=0.9):
    peak = torch.max(torch.abs(audio))
    scaling_factor = target_peak / peak
    normalized_audio = audio * scaling_factor
    return normalized_audio


def get_loudness(audio: torch.Tensor, sample_rate: float) -> float:
    meter = pyln.Meter(sample_rate)  # Create a loudness meter
    audio = audio.T
    audio = audio.cpu()
    loudness = meter.integrated_loudness(audio.numpy())  # Current LUFS
    return loudness


def automatic_gain_control(audio, target_level=0.7, alpha=0.1):
    current_level = torch.mean(torch.abs(audio))
    gain = target_level / (current_level + 1e-6)
    smoothed_gain = alpha * gain + (1 - alpha) * 1.0
    agc_audio = audio * smoothed_gain
    return agc_audio
