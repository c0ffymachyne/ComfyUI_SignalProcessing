import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

from .SignalProcessingLoadAudio import SignalProcessingLoadAudio
from .SignalProcessingFilter import SignalProcessingFilter
from .SignalProcessingMultiBandEQ import SignalProcessingMultibandEQ
from .SignalProcessingPaulStretch import SignalProcessingPaulStretch
from .SignalProcessingPadSynth import SignalProcessingPadSynth
from .SignalProcessingPadSynthChoir import SignalProcessingPadSynthChoir
from .SignalProcessingMixdown import SignalProcessingMixdown
from .SignalProcessingPadSynthChoir2 import SignalProcessingPadSynthChoir2
from .SignalProcessingVolumeControl import SignalProcessingVolumeControl
from .SignalProcessingSpectrogram import SignalProcessingSpectrogram
from .SignalProcessingWaveform import SignalProcessingWaveform
from .SignalProcessingStereoWidening import SignalProcessingStereoWidening
from .SignalProcessingMultiBandEQExperimental import SignalProcessingMultiBandEQExperimental
        
NODE_CLASS_MAPPINGS = {
    "SignalProcessingLoadAudio": SignalProcessingLoadAudio,
    "SignalProcessingFilter": SignalProcessingFilter,
    "SignalProcessingPaulStretch": SignalProcessingPaulStretch,
    "SignalProcessingPadSynth": SignalProcessingPadSynth,
    "SignalProcessingPadSynthChoir": SignalProcessingPadSynthChoir,
    "SignalProcessingPadSynthChoir2":SignalProcessingPadSynthChoir2,
    "SignalProcessingMixdown": SignalProcessingMixdown,
    "SignalProcessingVolumeControl" : SignalProcessingVolumeControl,
    "SignalProcessingSpectrogram" : SignalProcessingSpectrogram,
    "SignalProcessingWaveform" : SignalProcessingWaveform,
    "SignalProcessingStereoWidening" : SignalProcessingStereoWidening,
    "SignalProcessingMultibandEQ": SignalProcessingMultibandEQ,
    "SignalProcessingMultiBandEQExperimental": SignalProcessingMultiBandEQExperimental
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SignalProcessingLoadAudio": "(SP) Load Audio",
    "SignalProcessingFilter": "(SP) Filter",
    "SignalProcessingPaulStretch" : "(SP) PaulStretch",
    "SignalProcessingPadSynth": "(SP) PadSynth",
    "SignalProcessingPadSynthChoir": "(SP) PadSynth Choir",
    "SignalProcessingPadSynthChoir2":"(SP) PadSynth Choir 2",
    "SignalProcessingMixdown": "(SP) Mix Down",
    "SignalProcessingVolumeControl" : "(SP) Volume",
    "SignalProcessingSpectrogram" : "(SP) Spectogram",
    "SignalProcessingWaveform" : "(SP) Waveform",
    "SignalProcessingStereoWidening" : "(SP) Stereo Widening",
    "SignalProcessingMultibandEQ" : "(SP) Equalizer",
    "SignalProcessingMultiBandEQExperimental": "(SP) Equalizer Experimental"
}
