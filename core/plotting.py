#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    various plotting methods for debugging and visualization
"""

import torch
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def _figure_to_image(figure, format="png", dpi=96):
    buf = BytesIO()
    figure.savefig(buf, format=format, dpi=96)
    plt.close(figure)

    buf.seek(0)
    waveform_image = Image.open(buf).convert("RGB")

    return waveform_image


def get_wave(
    waveform: torch.Tensor,
    sample_rate: int,
    title: str = "Waveform",
    xlim=None,
    ylim=None,
):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)

    if num_channels > 1:
        axes[c].set_ylabel(f"Channel {c+1}")
    if xlim:
        axes[c].set_xlim(xlim)
    if ylim:
        axes[c].set_ylim(ylim)

    figure.suptitle(title)

    waveform_image = _figure_to_image(figure)

    return waveform_image


def get_spectogram(
    waveform: torch.Tensor, sample_rate: int, title="Spectrogram", xlim=None
):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
    if num_channels > 1:
        axes[c].set_ylabel(f"Channel {c+1}")
    if xlim:
        axes[c].set_xlim(xlim)
    figure.suptitle(title)

    waveform_image = _figure_to_image(figure)

    return waveform_image


def save_image(filepath: str, image: Image):
    image.save(filepath, format="JPEG", quality=95)
