#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    io for audio files and comfy
"""

import torch
import torchaudio
from pathlib import Path
from typing import Dict, Tuple, Any, Union
import numpy as np

debug_print = False


def audio_to_comfy_3d(
    waveform: torch.Tensor, sample_rate: int, cpu: bool = True, clip: bool = False
) -> Tuple[Dict[str, Union[torch.Tensor, int]]]:

    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)

    if waveform.ndim != 3:
        raise RuntimeError(
            f"returning to_comfy failed \
                {waveform.shape}@{waveform.device}[{waveform.ndim}]  \
            has incorrect dimentions, it should be of dimension [batch,channels,audio]"
        )

    if cpu:
        if waveform.device != "cpu":
            waveform = waveform.cpu()

    if waveform.dtype != torch.float64:
        waveform = waveform.to(dtype=torch.float64)

    if clip:
        waveform = np.clip(waveform, -1.0, 1.0)

    return_value = ({"waveform": waveform, "sample_rate": sample_rate},)

    if debug_print:
        print(
            f"audio_to_comfy_3d '{return_value}' with shape \
                {waveform.shape}@{waveform.device}dim[{waveform.ndim}] \
                    and sample rate {sample_rate} Hz."
        )

    return return_value


def audio_to_comfy_3dp1(
    waveform: torch.Tensor,
    sample_rate: int,
    value0: Any,
    cpu: bool = True,
    clip: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Any]:

    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)

    if waveform.ndim != 3:
        raise RuntimeError(
            f"returning to_comfy failed \
                 {waveform.shape}@{waveform.device}dim[{waveform.ndim}]\
                     has incorrect dimentions, it should be of dimension [batch,channels,audio]"
        )

    if cpu:
        if waveform.device != "cpu":
            waveform = waveform.cpu()

    if waveform.dtype != torch.float64:
        waveform = waveform.to(dtype=torch.float64)

    if clip:
        waveform = np.clip(waveform, -1.0, 1.0)

    return_value = (
        {"waveform": waveform, "sample_rate": sample_rate},
        value0,
    )

    if debug_print:
        print(
            f"audio_to_comfy_3d '{return_value}' with shape \
            {waveform.shape}@{waveform.device}dim[{waveform.ndim}] \
                and sample rate {sample_rate} Hz."
        )

    return return_value


def audio_to_comfy_3dp2(
    waveform: torch.Tensor,
    sample_rate: int,
    value0: Any,
    value1: Any,
    cpu: bool = True,
    clip: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Any, Any]:

    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)

    if waveform.ndim != 3:
        raise RuntimeError(
            f"returning to_comfy failed \
                {waveform.shape}@{waveform.device}dim[{waveform.ndim}] \
                    has incorrect dimentions, it should be of dimension [batch,channels,audio]"
        )

    if cpu:
        if waveform.device != "cpu":
            waveform = waveform.cpu()

    if waveform.dtype != torch.float64:
        waveform = waveform.to(dtype=torch.float64)

    if clip:
        waveform = np.clip(waveform, -1.0, 1.0)

    return_value = (
        {"waveform": waveform, "sample_rate": sample_rate},
        value0,
        value1,
    )

    if debug_print:
        print(
            f"audio_to_comfy_3d '{return_value}' with shape \
            {waveform.shape}@{waveform.device}dim[{waveform.ndim}] \
                and sample rate {sample_rate} Hz."
        )

    return return_value


def audio_from_comfy_2d(
    audio: Dict[str, Union[torch.Tensor, int]],
    repeat: bool = True,
    try_gpu: bool = True,
) -> Tuple[torch.Tensor, int]:

    if audio is None:
        raise ValueError("Input audio must be provided.")

    waveform: torch.Tensor = audio.get("waveform")  # [batch, channels, samples]
    sample_rate: int | Any | None = audio.get("sample_rate")

    if waveform.dtype != torch.float64:
        waveform = waveform.to(dtype=torch.float64)

    if waveform is None or sample_rate is None:
        raise ValueError("Input audio must contain 'waveform' and 'sample_rate'.")

    if not isinstance(waveform, torch.Tensor):
        raise TypeError("Waveform must be a torch.Tensor.")

    if debug_print:
        print(
            f"audio_from_comfy_2d with shape \
            {waveform.shape}@{waveform.device}dim[{waveform.ndim}] \
                and sample rate {sample_rate} Hz."
        )

    if waveform.ndim != 3:
        raise ValueError(
            f"Waveform must be a 3D tensor with shape (batch, channels, samples) not \
                {waveform.shape}@{waveform.device}"
        )

    if repeat:
        if waveform.ndim == 1:  # add extra channel in case of mono
            print("audio_from_comfy_2d repeat")
            waveform = waveform.unsqueeze(0)  # Add channel dimension
            waveform = waveform.repeat(2, 1)  # copy mono to the new channel

    waveform = waveform.squeeze(0)
    # waveform = waveform.contiguous()

    if try_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        waveform = waveform.to(device)

    if debug_print:
        print(
            f"audio_from_comfy_2d with shape \
            {waveform.shape}@{waveform.device}dim[{waveform.ndim}] \
                and sample rate {sample_rate} Hz."
        )

    return waveform, sample_rate


def audio_from_comfy_3d_to_disk(
    audio: Dict[str, Union[torch.Tensor, int]],
    filepath: str,
    repeat: bool = True,
    try_gpu: bool = True,
) -> None:
    waveform: torch.Tensor = audio.get("waveform")  # [batch, channels, samples]
    sample_rate: int | Any | None = audio.get("sample_rate")
    waveform = waveform.squeeze(0)
    waveform = waveform.cpu()
    torchaudio.save(filepath, waveform, sample_rate, buffer_size=4096 * 16)


def audio_from_comfy_3d(
    audio: Dict[str, Union[torch.Tensor, int]],
    repeat: bool = True,
    try_gpu: bool = True,
) -> Tuple[torch.Tensor, int]:

    if audio is None:
        raise ValueError("Input audio must be provided.")

    waveform: torch.Tensor = audio.get("waveform")  # [batch, channels, samples]
    sample_rate: Union[torch.Tensor, int] = audio.get("sample_rate")

    if waveform.dtype != torch.float64:
        waveform = waveform.to(dtype=torch.float64)

    if waveform is None or sample_rate is None:
        raise ValueError("Input audio must contain 'waveform' and 'sample_rate'.")

    if not isinstance(waveform, torch.Tensor):
        raise TypeError("Waveform must be a torch.Tensor.")

    if debug_print:
        print(
            f"audio_from_comfy_2d with shape \
            {waveform.shape}@{waveform.device}dim[{waveform.ndim}] \
                and sample rate {sample_rate} Hz."
        )

    if waveform.ndim != 3:
        raise ValueError(
            f"Waveform must be a 3D tensor with shape \
                (batch, channels, samples) not \
                    {waveform.shape}@{waveform.device}"
        )

    if repeat:
        if waveform.ndim == 1:  # add extra channel in case of mono
            print("audio_from_comfy_2d repeat")
            waveform = waveform.unsqueeze(0)  # Add channel dimension
            waveform = waveform.repeat(2, 1)  # copy mono to the new channel

    # waveform = waveform.squeeze(0).contiguous()

    if try_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        waveform = waveform.to(device)

    if debug_print:
        print(
            f"audio_from_comfy_3d with shape \
            {waveform.shape}@{waveform.device}dim[{waveform.ndim}] \
                and sample rate {sample_rate} Hz."
        )

    return waveform, sample_rate


def from_disk_as_dict_3d(
    audio_file: str,
    gain: float = 1.0,
    repeat: bool = True,
    start_time: float = 0.0,
    end_time: float = 0.0,
) -> Tuple[Dict[str, Union[torch.Tensor, int]]]:

    if not Path(audio_file).exists():
        raise ValueError(f"File {audio_file} Does Not Exist")

    info = torchaudio.info(audio_file)
    sample_rate = info.sample_rate

    # cut sudio segment
    if (start_time > 0.0) and (end_time > start_time):
        frame_offset = int(start_time * sample_rate)  # Start frame
        num_frames = int((end_time - start_time) * sample_rate)  # Total frames to read
        waveform, sample_rate = torchaudio.load(
            audio_file, frame_offset=frame_offset, num_frames=num_frames
        )
    else:
        waveform, sample_rate = torchaudio.load(audio_file)

    if waveform.dtype != torch.float64:
        waveform = waveform.to(dtype=torch.float64)
    waveform = waveform * gain

    if repeat:
        if waveform.ndim == 1:  # add extra channel in case of mono
            waveform = waveform.repeat(2, 1)  # copy mono to the new channel

    # add batch dimension at the front
    waveform = waveform.unsqueeze(0).contiguous()

    if debug_print:
        print(
            f"from_disk_as_dict_3d loaded audio file: '{audio_file}' with shape: \
            {waveform.shape}@{waveform.device}dim[{waveform.ndim}] \
                and sample rate: {sample_rate} Hz."
        )

    return audio_to_comfy_3d(waveform, sample_rate)


def from_disk_as_raw_3d(
    audio_file: str,
    repeat: bool = True,
    try_gpu: bool = False,
    start_seconds: float = 0.0,
    duration_seconds: float = 0.0,
) -> Tuple[torch.Tensor, int]:

    waveform, sample_rate = torchaudio.load(audio_file, buffer_size=4096 * 16)
    # info = torchaudio.info(audio_file)
    if debug_print:
        print(
            f"from_disk_as_raw_3d loaded audio file: '{audio_file}' with shape \
            {waveform.shape}@{waveform.device}dim[{waveform.ndim}] \
                and sample rate {sample_rate} Hz."
        )

    if duration_seconds != 0.0:
        start_sample = int(start_seconds * sample_rate)
        end_sample = None
        if duration_seconds is not None:
            end_sample = start_sample + int(duration_seconds * sample_rate)
        print(f"slicing from_disk_as_raw_3d: [{start_seconds}:{duration_seconds}]")
        # Slice the waveform
        waveform = waveform[:, start_sample:end_sample]

    if waveform.dtype != torch.float64:
        waveform = waveform.to(dtype=torch.float64)

    if repeat:
        # if info.num_channels == 1:  # add extra channel in case of mono
        if waveform.ndim == 1:  # add extra channel in case of mono
            waveform = waveform.repeat(2, 1)  # copy mono to the new channel

    waveform = waveform.unsqueeze(0)

    if try_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        waveform = waveform.to(device)

    if debug_print:
        print(
            f"from_disk_as_raw_3d loaded audio file: '{audio_file}' with shape \
            {waveform.shape}@{waveform.device}dim[{waveform.ndim}] \
                and sample rate {sample_rate} Hz."
        )

    return waveform, sample_rate


def from_disk_as_raw_2d(
    audio_file: str, repeat: bool = True, try_gpu: bool = False
) -> Tuple[torch.Tensor, int]:
    # all loaded audio comes in 3 dimensions
    # [batch,channels,audio] 3 dimensions

    waveform, sample_rate = torchaudio.load(audio_file)
    info = torchaudio.info(audio_file)

    if debug_print:
        print(
            f"from_disk_as_raw_2d loaded audio file: '{audio_file}' with shape \
            {waveform.shape}@{waveform.device}dim[{waveform.ndim}] \
                and sample rate {sample_rate} Hz."
        )

    if waveform.dtype != torch.float64:
        waveform = waveform.to(dtype=torch.float64)

    if repeat:
        if info.num_channels == 1:  # add extra channel in case of mono
            waveform = waveform.repeat(2, 1)  # copy mono to the new channel
            # waveform = waveform.unsqueeze(0) # Add channel dimension

    if try_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        waveform = waveform.to(device)

    if debug_print:
        print(
            f"from_disk_as_raw_2d loaded audio file: '{audio_file}' with shape \
            {waveform.shape}@{waveform.device}dim[{waveform.ndim}] \
                and sample rate {sample_rate} Hz."
        )

    return waveform, sample_rate
