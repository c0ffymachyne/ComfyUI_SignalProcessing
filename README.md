# ComfyUI Signal Processing

This repo contains 4 nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that allows for audio filtering

## Nodes

### SignalProcessingPaulStretch :
This node lets you stretch audio to about 100x it's original speed whil mainting pitch. it's great for making pad sounds. It's a port of 
fantastic algorithm of by by Nasca Octavian PAUL, his repository here : https://github.com/paulnasca/paulstretch_python
- **audio**: input audio
- **stretch_factor**: the stretch factor from 0 to 100
- **window_size_seconds**: window length ins seconds. higher value create smoother stretching effect.

### SignalProcessingPadSynth :
This node is a synthesiser "PadSynth" based on an algorithm by Nasca Octavian PAUL 
Original Source Code Here: https://zynaddsubfx.sourceforge.io/doc/PADsynth/PADsynth.htm
- **samplerate**: samplerate
- **fundamental_freq**: fundamental frequency for the sounds generation
- **bandwidth_cents**: bandwidth centers
- **number_harmonics**: number of harmonics
- **amplitude_per_harmonic**: amplitude per harmonic as a json, takes as list of amplitudes [0,1,4,...], it's count must match number of harmonics

### SignalProcessingFilter :
This node lets you stretch audio to about 100x it's original speed whil mainting pitch. it's great for making pad sounds:
- **audio**: input audio
- **cutoff**: filter cutoff
- **filter_type**: filter type - "lowpass", "highpass", "bandpass", "bandstop"
- **q_factor**: weidth of the filter

### SignalProcessingLoadAudio :
This node lets you stretch audio to about 100x it's original speed whil mainting pitch. it's great for making pad sounds:
- **audio_file**: input audio file
- **seek_seconds**: when to start audio from