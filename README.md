# ComfyUI Signal Processing

## THIS IS WORK IN PROGRESS REPOSITORY

This repo contains signal processing nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that allows for audio filtering

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
- **audios**: audio output one channel per note - use "SignalProcessingMixdown" node after to get single audio with monot channel copies to L and R

### SignalProcessingPadSynthChoir/SignalProcessingPadSynthChoir2
This node is a synthesiser "PadSynth" based on an algorithm by Nasca Octavian PAUL 
Original Source Code Here: https://zynaddsubfx.sourceforge.io/doc/PADsynth/PADsynth.htm

### SignalProcessingFilter :
This node lets you stretch audio to about 100x it's original speed whil mainting pitch. it's great for making pad sounds:
- **audio**: input audio
- **cutoff**: filter cutoff
- **filter_type**: filter type - "lowpass", "highpass", "bandpass", "bandstop"
- **q_factor**: weidth of the filter

### SignalProcessingStereoWidening: 
Simple Stereo Widening Algorithm. May be inspired by this implementation : https://github.com/AudioEmerge/Widener

### SignalProcessingMixdown
mixdown outputs from PadSynths with volume control per note
- **audios**: audios input
- **audio**: audio output

### SignalProcessingMultiBandEQ
multi bans EQ with different methods 
- **audio**: audio input
- **method**: hann window smoothing, rfft simple fft based eq, subcomp with bass compression
- **sub_bass**: gain db
- **bass**: gain db
- **low_mid**: gain db
- **mid**: gain db
- **upper_mid**: gain db
- **presence**: gain db
- **brilliance**: gain db
- **audio**: audio output

### SignalProcessingSpectrogram
Renders Mel Spectrum Into An Image
- **audio**: audio input
- **image**: image output

### SignalProcessingWaveform
Render Wave Shape Into An Image
- **audio**: audio input
- **image**: image output

### SignalProcessingLoadAudio :
This node lets you stretch audio to about 100x it's original speed whil mainting pitch. it's great for making pad sounds:
- **audio_file**: input audio file
- **seek_seconds**: when to start audio from

### SignalProcessingMultiBandEQExperimental
just a scratch pad for EQs dev