#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    Convolution Reverb
"""

import os 
import torch, torchaudio
import torch.nn.functional as F
from pathlib import Path

from ..core.utilities import comfy_root_to_syspath
comfy_root_to_syspath() # add comfy to sys path for dev

from ..core.io import audio_from_comfy_2d, audio_to_comfy_3d, from_disk_as_raw_2d
from ..core.loudness import lufs_normalization, get_loudness


import folder_paths


class SignalProcessingConvolutionReverb():
    supported_formats = ['.wav','.mp3','.ogg','.m4a','.flac','.mp4']
    this_directory = os.path.dirname(os.path.realpath(__file__))
    ir_directory = os.path.join(os.path.split(this_directory)[0],'audio','ir')

    @classmethod
    def INPUT_TYPES(s):
        
        files, _ = folder_paths.recursive_search(SignalProcessingConvolutionReverb.ir_directory)
        
        ir_files = []
        for file in files:
            try:
                _, ext = os.path.splitext(file)
                if ext in SignalProcessingConvolutionReverb.supported_formats:
                    ir_files.append(file)
            except:
                pass

        return {
            "required":  {
                "impulse_response":  ( sorted(ir_files), ),
                "audio_input": ("AUDIO", ),
                "wet_dry":("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    CATEGORY = "Signal Processing"
    FUNCTION = "process"

    def process(self, impulse_response, audio_input, wet_dry):

        try_gpu : bool = True
        repeat : bool = True

        print(impulse_response)
        print(audio_input)

        waveform, sample_rate = audio_from_comfy_2d(audio_input,repeat=repeat,try_gpu=try_gpu)

        loudness = get_loudness(waveform,sample_rate)

        it_filepath = os.path.join(SignalProcessingConvolutionReverb.ir_directory,impulse_response)
        ir,ir_sr = from_disk_as_raw_2d(it_filepath,repeat=repeat,try_gpu=try_gpu)

        # Resample IR if sampling rates do not match
        if ir_sr != sample_rate:
            print(f"Resampling impulse response from {ir_sr} Hz to {sample_rate} Hz.")
            resampler = torchaudio.transforms.Resample(orig_freq=ir_sr, new_freq=sample_rate).to(ir.device,dtype=waveform.dtype)
            ir = resampler(ir)
            ir_sr = sample_rate

        print('ensure channels')
        #if wave is mono and ir is not mono
        if waveform.shape[0] == 1 and ir.shape[0] == 2:
            ir = ir.mean(dim=0, keepdim=True) 
        if waveform.shape[0] == 2 and ir.shape[0] == 1:
            ir = ir.repeat(2,1)

        print('apply reverb')
        processed_audio = self.apply_reverb(waveform, sample_rate, ir, wet_dry=wet_dry)
        processed_audio = lufs_normalization(processed_audio,sample_rate,loudness)

        return audio_to_comfy_3d(processed_audio,sample_rate,cpu=True)


    def apply_reverb(self, audio: torch.Tensor, sr: int, ir: torch.Tensor, wet_dry: float = 0.5) -> torch.Tensor:

        num_audio_channels, audio_length = audio.shape
        num_ir_channels, ir_length = ir.shape

        # Normalize IR to prevent amplification
        ir = ir / torch.max(torch.abs(ir)) if torch.max(torch.abs(ir)) > 0 else ir

        # Initialize list to hold processed channels
        processed_channels = []

        # Apply convolution per channel
        for channel in range(num_audio_channels):
            # Get the current audio and IR channel
            audio_channel = audio[channel].unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, N]
            ir_channel = ir[channel].flip(0).unsqueeze(0).unsqueeze(0)  # Reverse IR, Shape: [1, 1, M]

            # Perform convolution
            convolved = F.conv1d(audio_channel, ir_channel, padding=ir_length - 1)  # Shape: [1, 1, N + M -1]

            # Remove batch and channel dimensions
            convolved = convolved.squeeze(0).squeeze(0)  # Shape: [N + M -1]

            # Trim convolved signal to original audio length
            convolved = convolved[:audio_length]

            # Normalize convolved signal to prevent clipping
            max_val = torch.max(torch.abs(convolved))
            if max_val > 0:
                convolved = convolved / max_val

            # Apply wet/dry mix
            dry = (1 - wet_dry)
            wet = wet_dry
            processed = dry * audio[channel] + wet * convolved

            # Prevent clipping by normalizing if necessary
            processed_max = torch.max(torch.abs(processed))
            if processed_max > 1.0:
                processed = processed / processed_max

            # Append processed channel
            processed_channels.append(processed)

        # Stack channels back into a tensor
        processed_audio = torch.stack(processed_channels)  # Shape: [2, N]

        return processed_audio


if __name__ == "__main__":

    import torchaudio
    from pathlib import Path
    from ..core.io import audio_from_comfy_2d, audio_to_comfy_3d, from_disk_as_raw_2d

    node = SignalProcessingConvolutionReverb()
    types = node.INPUT_TYPES()

    samples_path = Path('ComfyUI_SignalProcessing/audio/samples')
    impulses_path = Path('ComfyUI_SignalProcessing/audio/ir')

    samples = list(samples_path.rglob('*.*'))
    impulses = list(impulses_path.rglob('*.*'))

    print(samples)

    s = samples[1].absolute()
    i = impulses[0].absolute()
    s,sr= from_disk_as_raw_2d(s)

    sound = audio_to_comfy_3d(s,sr)[0]

    result = node.process(i,sound,wet_dry=.5)[0]
    audio, sr = audio_from_comfy_2d(result)
    torchaudio.save('ComfyUI_SignalProcessing/audio/tests/reverb.wav',audio.cpu(),sr)

    #set console to comfy ComfyUI-0.2.4/custom_nodes and run below command
    #export coffy_local_dev=1
    #python3 -m ComfyUI_SingalProcessing.effects.SignalProcessingConvolutionReverb