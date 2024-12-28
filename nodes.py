#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    node definitions
"""

# generators
from .generators.SignalProcessingPadSynth import SignalProcessingPadSynth
from .generators.SignalProcessingPadSynthChoir import SignalProcessingPadSynthChoir

# effects
from .effects.SignalProcessingStereoWidening import SignalProcessingStereoWidening
from .effects.SignalProcessingPaulStretch import SignalProcessingPaulStretch
from .effects.SignalProcessingPitchShifter import SignalProcessingPitchShifter
from .effects.SignalProcessingConvolutionReverb import SignalProcessingConvolutionReverb

# processors
from .processors.SignalProcessingFilter import SignalProcessingFilter
from .processors.SignalProcessingMixdown import SignalProcessingMixdown
from .processors.SignalProcessingLoadAudio import SignalProcessingLoadAudio
from .processors.SignalProcessingNormalizer import SignalProcessingNormalizer
from .processors.SignalProcessingLoudness import SignalProcessingLoudness
from .processors.SignalProcessingBaxandallEQ import (
    SignalProcessingBaxandallEQ,
    SignalProcessingBaxandall3BandEQ,
)
from .processors.SignalProcessingHarmonicsEnhancer import (
    SignalProcessingHarmonicsEnhancer,
)
from .processors.SignalProcessingSaturation import SignalProcessingSaturation
from .processors.SignalProcessingLimiter import SignalProcessingLimiter

# from .processors.SignalProcessingCompressor import SignalProcessingCompressor

# visuals
from .visuals.SignalProcessingSpectrogram import SignalProcessingSpectrogram
from .visuals.SignalProcessingWaveform import SignalProcessingWaveform

NODE_CLASS_MAPPINGS = {
    "SignalProcessingLoadAudio": SignalProcessingLoadAudio,
    "SignalProcessingFilter": SignalProcessingFilter,
    "SignalProcessingPaulStretch": SignalProcessingPaulStretch,
    "SignalProcessingPadSynth": SignalProcessingPadSynth,
    "SignalProcessingPadSynthChoir": SignalProcessingPadSynthChoir,
    "SignalProcessingMixdown": SignalProcessingMixdown,
    "SignalProcessingSpectrogram": SignalProcessingSpectrogram,
    "SignalProcessingWaveform": SignalProcessingWaveform,
    "SignalProcessingStereoWidening": SignalProcessingStereoWidening,
    "SignalProcessingPitchShifter": SignalProcessingPitchShifter,
    "SignalProcessingConvolutionReverb": SignalProcessingConvolutionReverb,
    "SignalProcessingNormalizer": SignalProcessingNormalizer,
    "SignalProcessingLoudness": SignalProcessingLoudness,
    "SignalProcessingBaxandallEQ": SignalProcessingBaxandallEQ,
    "SignalProcessingBaxandall3BandEQ": SignalProcessingBaxandall3BandEQ,
    "SignalProcessingHarmonicsEnhancer": SignalProcessingHarmonicsEnhancer,
    "SignalProcessingSaturation": SignalProcessingSaturation,
    "SignalProcessingLimiter": SignalProcessingLimiter,
    # "SignalProcessingCompressor": SignalProcessingCompressor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SignalProcessingLoadAudio": "(SP) Load Audio",
    "SignalProcessingFilter": "(SP) Filter",
    "SignalProcessingPaulStretch": "(SP) PaulStretch",
    "SignalProcessingPadSynth": "(SP) PadSynth",
    "SignalProcessingPadSynthChoir": "(SP) PadSynth Choir",
    "SignalProcessingMixdown": "(SP) Mix Down",
    "SignalProcessingSpectrogram": "(SP) Spectogram",
    "SignalProcessingWaveform": "(SP) Waveform",
    "SignalProcessingStereoWidening": "(SP) Stereo Width",
    "SignalProcessingPitchShifter": "(SP) PitchShift",
    "SignalProcessingConvolutionReverb": "(SP) Convolution Reverb",
    "SignalProcessingNormalizer": "(SP) Normalizer",
    "SignalProcessingLoudness": "(SP) Loudness",
    "SignalProcessingBaxandallEQ": "(SP) Baxandall EQ",
    "SignalProcessingBaxandall3BandEQ": "(SP) Baxandall 3 Band EQ",
    "SignalProcessingHarmonicsEnhancer": "(SP) Enhance Harmonics",
    "SignalProcessingSaturation": "(SP) Saturation",
    "SignalProcessingLimiter": "(SP) Limiter",
    # "SignalProcessingCompressor": "(SP) Compressor",
}
