# ComfyUI Signal Processing


## THIS IS WORK IN PROGRESS REPOSITORY


This repo contains signal processing nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) allowing for audio manipulation. 

[DeepWiki Documentation](https://deepwiki.com/c0ffymachyne/ComfyUI_SignalProcessing)


#### Licensing And Attribution

- **`LICENSE-GPL-V3`** The source code in this repository is licensed under "GNU General Public License Version 3".
- **`LICENSE-APACHE-2`** Some components are built from parts of code licensed under "Apache License Version 2".
- **`LICENSE-CCA-ANY`** Some components are built from parts of code licensed under "Creative Commons Zero v1.0 Universal"

---

### Latests Updates

- **`Gpu limiter - experimental`** - Limiter optimized for parallel GPU execution
- **`Saturatation - experimental`** - Added basic saturation node with couple algorithms
- **`Tests suite`** - Added basic tests suite for most of the nodes

---

### Mastering Nodes

### Baxandall EQ/Baxandall 3 Band EQ

The Baxandall EQ is a smooth, wide-band tone control circuit widely used in audio systems, offering gentle boost or cut for bass and treble frequencies. 
Its simple design and musical response make it ideal for achieving natural tonal adjustments. 
Implementation is using the standard shelf filter equations from the Audio EQ Cookbook by Robert Bristow-Johnson

#### Parameters:

- **`audio`**  self explanatory... 
- **`bass_gain_db`** Bass gain in decibels
- **`mid_gain_db`** Mid gain in decibels
- **`treble_gain_db`** Treble gain in decibels
- **`low_freq`** : The corner frequency for the low shelf (e.g. ~100 Hz).
- **`mid_freq`** : The center frequency for the mid peaking filter (e.g. ~1 kHz).
- **`high_freq`** : The corner frequency for the high shelf (e.g. ~10 kHz).
- **`mid_q`** : Quality factor for the mid peaking band. Adjusting Q controls the bandwidth of the mid peak. A typical Q might be 0.7 for a broad bell.

---

### Enhance Harmonics

Harmonic enhancer boosts selected harmonics to enrich the sound. 

#### Parameters:

- **`harmonincs`**  comma separated numbers of harmonics to boost
- **`mode`**  whether to automatically detect base frequency based on an audio or use manual setting
- **`base_frequency`** base frequency to use in manual mode
- **`gain_db`** How much to boost harmonics
- **`gain_db`** Width of the filters

---

### Normalizer

Normalizer is an amalgamate of multiple normalization approaches, including Loudness Units Full Scale (LUFS) with standard default set to -14db.

### Parametes

- **`audio_input`**: audio input
- **`mode`**: "lufs","rms","peak","auto"
- **`target_rms`**: The desired RMS value for the audio signal. Default is 0.1, which corresponds to a moderate average signal level.
- **`target_lufs_db`**: The desired loudness level in LUFS. Default is -14.0, which is a common loudness target for streaming platforms like Spotify.
- **`target_peak`**: The desired peak amplitude for the audio signal. Default is 0.9, meaning the loudest sample will be scaled to 90% of the maximum possible amplitude.
- **`target_auto`**: The desired amplitude level for the audio signal. The algorithm scales the audio to match this level. Default is 0.7 (normalized scale from 0 to 1).
- **`target_auto_alpha`**: The smoothing factor for the gain adjustment. A smaller value of alpha makes the gain adjustment slower and smoother (avoiding sudden jumps). A larger value makes the gain adjustment faster but potentially introduces abrupt changes.

---

### Loudness

The get_loudness function calculates the integrated loudness of an audio signal in LUFS (Loudness Units relative to Full Scale). This is a perceptual measure of loudness, taking into account the human ear's sensitivity to different frequencies and the entire audio signal's duration.

---

### SignalProcessingStereoWidening: 
Open Source Stere Widening Plugin. The implementation is a direct copy of parts of the source code corresponding to this [paper](https://www.dafx.de/paper-archive/2024/papers/DAFx24_paper_92.pdf) developed by Orchisama Das. The code is distributed under `**CC0 1.0 Universal**` license.
[Original Source Code](https://github.com/orchidas/StereoWidener)


#### Parameters:

- **`audio`**: input audio
- **`mode`**: "decorrelation" and "simple" - "decorrelation" is based on "Open Source Stere Widening Plugin" as described above
- **`gain`**: post width gain
- **`width`**: width of the stereo effect

---

### Effects Nodes

### Convolution Reverb

Convolution reverb simulates realistic acoustic spaces by applying the impulse response of a physical environment to an audio signal. 
It captures the natural reverberation characteristics, providing authentic spatial depth and ambience.


### How do I use it ? 

I recommend downloading impulse response files from this location [Voxengo-IR](https://oramics.github.io/sampled/IR/Voxengo/) and [Greg Hopkins EMT 140 Plate Reverb Impulse Response](https://oramics.github.io/sampled/IR/EMT140-Plate/). They sound absolutely fantastic and have great licensing. In order for the files to show up for selection in the convolution reverb please download the files and organize them like this : 
- `comfyui_signalprocessing/audio/ir/Voxengo/` <- copy wave files into this directory
- `comfyui_signalprocessing/audio/ir/EMT-140-Plate/` <- copy wav files into this directory

#### Parameters:

- **`impulse_response`**: impulse response file selected
- **`audio_input`**:  audio to apply reverb to
- **`wet_dry`**: mix amount of the effect

---

### SignalProcessingPaulStretch

PaulStretch excels at extreme time-stretching with high-quality results, preserving the pitch and tonal characteristics of the original audio. 
This node contains a port of algorithm developed by Nasca Octavian Paul.  
[Original Source Code](https://github.com/paulnasca/paulstretch_python)

#### Parameters:

- **`audio`**  
  The input audio signal to be stretched

- **`stretch_factor`**  
  Determines the amount of stretching applied to the audio.  
  **Range**: `0` (no stretch) to `100` (maximum stretch).  
  **Example**: A `stretch_factor = 10` stretches the audio to 10 times its original length.

- **`window_size_seconds`**  
  Specifies the window length for the stretching algorithm, in seconds. Larger values produce smoother and more ambient results by averaging the time-domain samples over a longer period.  
  **Example**: `window_size_seconds = 1.0` provides smooth stretching for most applications, while smaller values retain more transient detail.

---

### SignalProcessingPadSynth :

This node is a synthesiser "PadSynth" based on a PADSynth algorithm
This node contains a port of algorithm developed by Nasca Octavian Paul
[Original Source Code](https://zynaddsubfx.sourceforge.io/doc/PADsynth/PADsynth.htm)

#### Parameters:

- **`samplerate`**: samplerate
- **`fundamental_freq`**: fundamental frequency for the sounds generation
- **`bandwidth_cents`**: bandwidth centers
- **`number_harmonics`**: number of harmonics
- **`amplitude_per_harmonic`**: amplitude per harmonic as a json, takes as list of amplitudes [0,1,4,...], it's count must match number of harmonics
- **`audios`**: audio output one channel per note - use "SignalProcessingMixdown" node after to get single audio with monot channel copies to L and R

---

### SignalProcessingPadSynthChoir
This node is a synthesiser "PadSynth" emulating choirs
[Original Source Code](https://zynaddsubfx.sourceforge.io/doc/PADsynth/PADsynth.htm)

#### Parameters:

- **`samplerate`**: samplerate
- **`base_freq`**: base frequency
- **`step_size`**: step size
- **`num_notes`**: number of notes 
- **`bandwidth_cents`**:  bandwidth cents
- **`number_harmonics`**: number of harmonics to produce

---

### SignalProcessingFilter :

Classic filters

#### Parameters:

- **`audio`**: input audio
- **`cutoff`**: filter cutoff
- **`filter_type`**: filter type - "lowpass", "highpass", "bandpass", "bandstop"
- **`q_factor`**: width of the filter

---

### SignalProcessingMixdown
mixdown outputs from PadSynths with volume control per note

#### Parameters:

- **`audios`**: audios input
- **`audio`**: audio output

---

### Testing/Visualization Nodes

This section contains nodes enabling basic analysis and development of other nodes

---

### SignalProcessingSpectrogram
Renders Mel Spectrum Into An Image

#### Parameters:

- **`audio`**: audio input
- **`image`**: image output

---

### SignalProcessingWaveform

Renders Wave Shape Into An Image

#### Parameters:

- **`audio`**: audio input
- **`image`**: image output

---

### SignalProcessingLoadAudio :

This node lets you stretch audio to about 100x it's original speed whil mainting pitch. it's great for making pad sounds:

#### Parameters:

- **`audio_file`**: input audio file
- **`gain`**: when to start audio from


### Development and Testing/Profiling

### Dependnecies:

The development dependencies are specified in the pyproject.toml file and are not included in the requirements.txt. 
To run tests and contribute to development, you'll need the following tools and modules installed. 
My primary development and testing environment is Ubuntu Linux with an RTX 3090 GPU and CUDA 11.8.

## Required Tools and Libraries:

-**`scalene`**: High-performance CPU and GPU profiler.
-**`pytest`**: Comprehensive testing framework.
-**`nox`**: Automation tool for managing tasks.
-**`ruff/flake8`**: Linter for ensuring clean and consistent code.

### Contributing

Contributions are always welcome! Feel free to fork the repository, make changes, and create a pull request. 
I'm open to collaboration and willing to make adjustments to improve the project. 
Let's build something great together! 🚀
