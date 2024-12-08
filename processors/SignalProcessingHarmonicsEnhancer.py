#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    Classic Audio filter set 
"""
from ast import literal_eval
import torch, torchaudio
import torch
from typing import Dict

from ..core.utilities import comfy_root_to_syspath

comfy_root_to_syspath()  # add comfy to sys path for dev
from ..core.io import from_disk_as_raw_2d, audio_to_comfy_3d, audio_from_comfy_2d
from ..core.loudness import lufs_normalization, get_loudness


class SignalProcessingHarmonicsEnhancer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_input": ("AUDIO",),
                "harmonics": ("STRING", {"default": "1,3,5,7,9"}),
                "mode": (["detect base frequency", "use base frequency"],),
                "base_frequency" : ("FLOAT", {"default" : 440, "min" : 0, "max" : 20000 }),
                "gain_db" : ("INT", {"default": 5,"min" : 0, "max" : 50, "step" : 1}),
                "Q" : ("FLOAT", {"default" : 0.707, "min" : 0, "max" : 1.0, "step" : 0.01 }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    CATEGORY = "Signal Processing"
    FUNCTION = "process"

    def process(
        self,
        audio_input: Dict[str, torch.Tensor],
        harmonics : str = '1,3,5,7,9',
        mode : str = "detect base frequency",
        base_frequency : int = 440,
        gain_db : int = 5,
        Q: float = 0.707,
    ):
        waveform, sample_rate = audio_from_comfy_2d(audio_input,try_gpu=True)
        loudness = get_loudness(waveform, sample_rate)
        try:
            harmonics = [literal_eval(x) for x in harmonics.split(',')]
        except:
            raise RuntimeWarning("Invalid Harmonics Format. Please delimit integers by a comma ',' like this 1,,3,5,7,9 ")
        if mode == "detect base frequency":
            filtered_waveform = self.enhance_harmonics(waveform,sample_rate,harmonics=harmonics,gain_db=gain_db,Q=Q)
        elif mode == "use base frequency":
            filtered_waveform = self.enhance_harmonics(waveform,sample_rate,harmonics=harmonics,gain_db=gain_db,base_frequency=base_frequency,Q=Q)

        filtered_waveform = lufs_normalization(filtered_waveform, sample_rate, loudness)
        return audio_to_comfy_3d(filtered_waveform, sample_rate)

    def add_harmonics(self, audio : torch.Tensor, gain=1.2):
        # Apply saturation using a tanh curve
        harmonic_audio = torch.tanh(audio * gain)
        return harmonic_audio

    def detect_fundamental(self,audio, sample_rate):
        # Estimate the fundamental frequency using a pitch detection method
        pitch = torchaudio.functional.detect_pitch_frequency(audio, sample_rate)

        return pitch

    def detect_fundamental_mean(self,audio, sample_rate):
        # Estimate the fundamental frequency using a pitch detection method
        pitch = torchaudio.functional.detect_pitch_frequency(audio, sample_rate)

        return pitch.mean().item()

    def enhance_harmonics(self,audio, sample_rate, harmonics=[1,3,5,7,9,11], gain_db=5, base_frequency = None,Q = 0.707):
        # Detect the base frequency
        if base_frequency == None:
            base_frequency = self.detect_fundamental_mean(audio, sample_rate)
            if base_frequency <= 0:  # Fallback if pitch detection fails
                base_frequency = 440  # Use a default base frequency

        # Apply EQ boosts to specific harmonic frequencies

        print('base_frequency',base_frequency)
        for harmonic in harmonics:
            freq = base_frequency * harmonic
            if freq < sample_rate / 2:  # Ensure it's within the Nyquist frequency
                print('center_freq',freq)
                audio = torchaudio.functional.equalizer_biquad(audio, sample_rate, center_freq=freq, gain=gain_db,Q=Q)
        
        return audio

if __name__ == "__main__":
    import torchaudio
    from pathlib import Path
    from ..core.io import from_disk_as_raw_2d, audio_to_comfy_3d, audio_from_comfy_2d

    node = SignalProcessingHarmonicsEnhancer()
    samples_path = Path("ComfyUI_SignalProcessing/audio/inputs/baxandall-normalizer_00003_.flac")

    source_path = samples_path.absolute()
    source_audio, source_audio_sample_rate = from_disk_as_raw_2d(source_path)
    input = audio_to_comfy_3d(source_audio, source_audio_sample_rate)[0]

    # Test with some gain settings
    result = node.process(input)[0]

    output_audio, sample_rate_audio = audio_from_comfy_2d(result)

    # Save output for analysis
    torchaudio.save(
        "ComfyUI_SignalProcessing/audio/outputs/harmonics.wav",
        output_audio.cpu(),
        sample_rate_audio,
    )

    # set console to comfy ComfyUI-0.2.4/custom_nodes and run below command
    # export coffy_local_dev=1
    # python3 -m ComfyUI_SignalProcessing.processors.SignalProcessingHarmonicsEnhancer