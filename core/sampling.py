#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    utility functions for sampling
"""

import torch
import torchaudio


def oversample(
    audio_signal: torch.Tensor, sample_rate: int, factor: int = 8
) -> torch.Tensor:

    resampler = torchaudio.transforms.Resample(sample_rate, sample_rate * factor).to(
        device=audio_signal.device, dtype=audio_signal.dtype
    )
    return resampler(audio_signal), sample_rate * factor


def downsample(
    audio_signal: torch.Tensor, sample_rate: int, factor: int = 8
) -> torch.Tensor:

    resampler = torchaudio.transforms.Resample(sample_rate, sample_rate // factor).to(
        device=audio_signal.device, dtype=audio_signal.dtype
    )
    return resampler(audio_signal)
