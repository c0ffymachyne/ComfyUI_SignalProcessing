#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    Spectogram image node
"""

import torch

from PIL import Image

from typing import Dict, Tuple, Any, Type, List

import numpy as np
import torchaudio

import matplotlib.pyplot as plt


class SignalProcessingSpectrogram:
    @classmethod
    def INPUT_TYPES(cls: Type["SignalProcessingSpectrogram"]) -> Dict[str, Any]:
        cmaps: List[str] = ["viridis", "plasma", "inferno", "magma", "cividis"]
        return {
            "required": {
                "audio_input": ("AUDIO",),
                "color_map": (cmaps,),
            },
            "optional": {
                "n_fft": (
                    "INT",
                    {"default": 4096, "min": 512, "max": 8192, "step": 256},
                ),
                "hop_length": (
                    "INT",
                    {"default": 128, "min": 64, "max": 4096, "step": 128},
                ),
                "n_mels": ("INT", {"default": 512, "min": 32, "max": 2048, "step": 32}),
                "top_db": (
                    "FLOAT",
                    {"default": 80.0, "min": 10.0, "max": 100.0, "step": 5.0},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("spectrogram_image",)
    CATEGORY = "Signal Processing"
    FUNCTION = "process"

    def process(
        self,
        audio_input: Dict[str, torch.Tensor],
        color_map: str = "viridis",
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        top_db: float = 80.0,
    ) -> Tuple[torch.Tensor]:
        waveform = audio_input.get("waveform")
        sample_rate = audio_input.get("sample_rate")

        # Validate that waveform and sample_rate are not None
        if waveform is None:
            raise ValueError("The 'waveform' key is missing or None in 'audio_input'.")
        if not isinstance(waveform, torch.Tensor):
            raise TypeError(
                f"Expected 'waveform' to be a torch.Tensor, got {type(waveform)}."
            )
        if sample_rate is None:
            raise ValueError(
                "The 'sample_rate' key is missing or None in 'audio_input'."
            )
        if not isinstance(sample_rate, int):
            raise TypeError(
                f"Expected 'sample_rate' to be an int, got {type(sample_rate)}."
            )

        # waveform, sample_rate = audio_from_comfy_2d(audio_input)

        # Convert to mono by averaging channels
        if waveform.ndim == 3:
            # [batch, channels, samples]
            waveform = waveform.mean(dim=1, keepdim=True)  # [batch, 1, samples]
            waveform = waveform.squeeze(0)  # [1, samples]
        elif waveform.ndim == 2:
            # [channels, samples]
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)  # [1, samples]
            else:
                waveform = waveform.unsqueeze(0)  # [1, samples]
        elif waveform.ndim == 1:
            # [samples]
            waveform = waveform.unsqueeze(0)  # [1, samples]
        else:
            raise ValueError(f"Unsupported waveform shape: {waveform.shape}")

        # Generate Mel Spectrogram
        spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
            norm="slaney",
            mel_scale="htk",
        ).to(waveform.device, dtype=waveform.dtype)
        spectrogram = spectrogram_transform(waveform)  # [1, n_mels, time_frames]

        # Convert to decibel scale
        amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=top_db)
        spectrogram_db = amplitude_to_db(spectrogram)  # [1, n_mels, time_frames]

        # Convert to numpy
        spectrogram_db = (
            spectrogram_db.squeeze().detach().cpu().numpy()
        )  # [n_mels, time_frames]

        # Clip spectrogram to a range for better contrast
        spectrogram_db = np.clip(spectrogram_db, -top_db, 0.0)

        # Normalize spectrogram to [0,1]
        spectrogram_normalized = (spectrogram_db + top_db) / top_db  # [0,1]

        # Apply a colormap (e.g., 'inferno') using matplotlib
        cmap = plt.get_cmap(color_map)
        spectrogram_colored = cmap(
            spectrogram_normalized
        )  # [n_mels, time_frames, 4] RGBA

        # Convert to RGB by removing alpha channel
        spectrogram_rgb = (spectrogram_colored[:, :, :3] * 255).astype(
            np.uint8
        )  # [n_mels, time_frames, 3]
        spectrogram_rgb = np.squeeze(spectrogram_rgb)

        # Check the shape and adjust if necessary
        if len(spectrogram_rgb.shape) == 3 and spectrogram_rgb.shape[-1] == 3:
            # Ensure the array is in uint8 format (0-255 range)
            spectrogram_rgb = np.clip(spectrogram_rgb, 0, 255).astype(np.uint8)
        else:
            raise ValueError(f"Unexpected spectrogram shape: {spectrogram_rgb.shape}")

        # Convert to RGB image
        spectrogram_image = Image.fromarray(spectrogram_rgb).convert("RGB")

        # Optionally resize for better resolution
        spectrogram_image = spectrogram_image.resize(
            (spectrogram_image.width * 2, spectrogram_image.height * 2), Image.BILINEAR
        )

        # Convert to numpy array and normalize to [0,1]
        image_np = np.array(spectrogram_image).astype(np.float32) / 255.0  # [H, W, 3]

        # Convert to torch tensor and add batch dimension
        # image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

        image = torch.from_numpy(image_np)[None,]

        return (image,)
