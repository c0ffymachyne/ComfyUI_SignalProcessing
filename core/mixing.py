#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    audio mixing and combining methods
"""

import torch

def combine_audio_files(waveform_a, waveform_b, sample_rate, chunk_duration=2.0):
    """
    Combine two audio files by alternating 2-second chunks, cropping to the shorter audio.

    Args:
        waveform_a (torch.Tensor): Tensor of the first audio file (channels x samples).
        waveform_b (torch.Tensor): Tensor of the second audio file (channels x samples).
        sample_rate (int): Sample rate of the audio files.
        chunk_duration (float): Duration of each chunk in seconds (default is 2 seconds).

    Returns:
        torch.Tensor: Combined waveform.
    """
    # Crop to the shorter length
    min_length = min(waveform_a.shape[1], waveform_b.shape[1])
    waveform_a = waveform_a[:, :min_length]
    waveform_b = waveform_b[:, :min_length]

    # Calculate chunk size in samples
    chunk_size = int(chunk_duration * sample_rate)

    # Determine the total number of samples
    total_samples = waveform_a.shape[1]

    # Initialize the output waveform
    combined_waveform = []

    # Alternate chunks between the two audio files
    for start in range(0, total_samples, chunk_size):
        end = min(start + chunk_size, total_samples)
        combined_waveform.append(waveform_a[:, start:end])
        combined_waveform.append(waveform_b[:, start:end])

    # Concatenate the combined waveform
    combined_waveform = torch.cat(combined_waveform, dim=1)

    return combined_waveform
