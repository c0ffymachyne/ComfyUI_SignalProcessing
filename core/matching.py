#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    loudness matching
"""

import torch
import torchaudio.transforms as T
from torch.optim import Adam


def approximate_loudness_matching(audio, target_loudness=-14.0):
    rms = torch.sqrt(torch.mean(audio**2))
    current_loudness = 20 * torch.log10(rms + 1e-6)
    gain = 10 ** ((target_loudness - current_loudness) / 20.0)
    return audio * gain


def waveform_matching(audio, target_audio, lr=0.01, steps=100):
    audio = audio.clone().requires_grad_(True)
    optimizer = Adam([audio], lr=lr)

    for _ in range(steps):
        loss = torch.nn.functional.mse_loss(audio, target_audio)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return audio.detach()
