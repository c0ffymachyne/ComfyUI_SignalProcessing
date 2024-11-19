import torch

import os, sys, re, shutil, subprocess
import hashlib
import numpy as np
import math
from typing import Tuple, List, Dict
import torchaudio.functional as F
from scipy.signal import get_window

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import torchaudio
import folder_paths

class SignalProcessingLoadAudio():
    supported_formats = ['wav','mp3','ogg','m4a','flac']
    @classmethod
    def INPUT_TYPES(s):
        supported_extensions = tuple(f".{fmt.lower()}" for fmt in SignalProcessingLoadAudio.supported_formats)
        
        input_dir = folder_paths.get_input_directory()
        all_items = os.listdir(input_dir)
        filtered_files = [
            x for x in all_items
            if x.lower().endswith(supported_extensions)
        ]
        files = [os.path.join(input_dir,x) for x in filtered_files]

        return {
            "required":  {"audio_file": (sorted(files), {"image_upload": True})},
            "optional" : {"seek_seconds": ("FLOAT", {"default": 0, "min": 0})}
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    CATEGORY = "Signal Processing"
    FUNCTION = "process"

    def process(self, audio_file, seek_seconds):

        audio_file_path = folder_paths.get_annotated_filepath(audio_file)

        waveform, sample_rate = torchaudio.load(audio_file_path)
        waveform = waveform.unsqueeze(0).contiguous()
        return ({'waveform': waveform, 'sample_rate': sample_rate},)

    @classmethod
    def IS_CHANGED(s, audio_file, seek_seconds):
        audio_file_path = folder_paths.get_annotated_filepath(audio_file)
        m = hashlib.sha256()
        with open(audio_file_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, audio_file, seek_seconds):
        if not folder_paths.exists_annotated_filepath(audio_file):
            return "Invalid image file: {}".format(audio_file)

        return True

class SignalProcessingFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"forceInput": True}),
                "cutoff": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "filter_type": (["lowpass", "highpass", "bandpass", "bandstop"], {"default": "lowpass"}),
                "q_factor": ("FLOAT", {"default": 0.707, "min": 0.1, "max": 5.0, "step": 0.01}),  # For resonance/bandwidth
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    CATEGORY = "Signal Processing"
    FUNCTION = "apply_filter"

    def apply_filter(self, audio: Dict[str, torch.Tensor], cutoff: float, filter_type: str, q_factor: float):
        """
        Apply a specified filter to the input audio.

        Parameters:
            audio (Dict[str, torch.Tensor]): Input audio with 'waveform' and 'sample_rate'.
            cutoff (float): Normalized cutoff frequency (0.0 to 1.0).
            filter_type (str): Type of filter ('lowpass', 'highpass', 'bandpass', 'bandstop').
            q_factor (float): Quality factor determining the filter's bandwidth.

        Returns:
            Tuple[Dict[str, torch.Tensor]]: Filtered audio.
        """
        waveform = audio['waveform'].squeeze(0)
        sample_rate = audio['sample_rate']
        nyquist = sample_rate / 2.0

        # Define minimum and maximum frequencies for mapping
        log_min = 20.0        # 20 Hz, typical lower bound of human hearing
        log_max = nyquist - 100.0  # Slightly below Nyquist to prevent instability

        # Avoid log(0) by ensuring cutoff is within (0,1)
        cutoff = min(max(cutoff, 1e-6), 1.0 - 1e-6)

        # Logarithmic mapping
        log_min = torch.log(torch.tensor(log_min))
        log_max = torch.log(torch.tensor(log_max))
        log_cutoff = log_min + cutoff * (log_max - log_min)
        cutoff_freq = torch.exp(log_cutoff).item()

        # Debug: Print mapped cutoff frequency
        print(f"Normalized cutoff: {cutoff} mapped to frequency: {cutoff_freq:.2f} Hz")

        # Choose filter type
        if filter_type == "lowpass":
            filtered_waveform = torchaudio.functional.lowpass_biquad(
                waveform, sample_rate, cutoff_freq, Q=q_factor
            )
        elif filter_type == "highpass":
            filtered_waveform = torchaudio.functional.highpass_biquad(
                waveform, sample_rate, cutoff_freq, Q=q_factor
            )
        elif filter_type in ["bandpass", "bandstop"]:
            center_freq = cutoff_freq
            # Ensure that the bandwidth does not exceed the Nyquist frequency
            bandwidth = center_freq / q_factor
            lower_freq = max(center_freq - bandwidth / 2.0, 20.0)  # Prevent dropping below 20 Hz
            upper_freq = min(center_freq + bandwidth / 2.0, nyquist - 100.0)  # Prevent exceeding Nyquist

            # Debug: Print bandwidth details
            print(f"Band filter with center frequency: {center_freq:.2f} Hz, "
                  f"lower: {lower_freq:.2f} Hz, upper: {upper_freq:.2f} Hz")

            if filter_type == "bandpass":
                filtered_waveform = torchaudio.functional.bandpass_biquad(
                    waveform, sample_rate, center_freq, Q=q_factor
                )
            else:  # bandstop
                filtered_waveform = torchaudio.functional.band_biquad(
                    waveform, sample_rate, center_freq, Q=q_factor
                )
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

        return ({'waveform': filtered_waveform.unsqueeze(0), 'sample_rate': sample_rate},)
    
#https://github.com/paulnasca/paulstretch_python
#https://github.com/paulnasca/paulstretch_python/blob/master/paulstretch_stereo.py
class SignalProcessingPaulStretch:
    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        return {
            "required": {
                "audio": ("AUDIO", {"forceInput": True}),
                "stretch_factor": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 100.0, "step": 0.1}),
                "window_size_seconds": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 10.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    CATEGORY = "Signal Processing"
    FUNCTION = "process"

    def process(
        self,
        audio: Dict[str, torch.Tensor],
        stretch_factor: float,
        window_size_seconds: float
    ) -> Tuple[Dict[str, torch.Tensor]]:
        """
        Apply the PaulStretch algorithm to time-stretch the input audio without altering its pitch.

        Parameters:
            audio (Dict[str, torch.Tensor]): Input audio with 'waveform' and 'sample_rate'.
            stretch_factor (float): Factor by which to stretch the audio (1.0 = no stretch).
            window_size_seconds (float): Window size in seconds.

        Returns:
            Tuple[Dict[str, torch.Tensor]]: PaulStretched audio with original sample rate.
        """
        # Initialize logging

        # Conditional processing: If stretch_factor is 1.0, return original audio
        if stretch_factor == 1.0:
            return ({'waveform': audio['waveform'], 'sample_rate': audio['sample_rate']},)

        # Squeeze the first dimension (e.g., batch dimension) to get shape (C, N) or (N,)
        waveform = audio['waveform'].squeeze(0).cpu().numpy()  # Shape: (C, N) or (N,) for mono
        sample_rate = audio['sample_rate']

        # Handle mono audio by converting to stereo by duplicating the channel
        if waveform.ndim == 1:
            waveform = np.tile(waveform, (2, 1))  # Shape: (2, N)

        nchannels, nsamples = waveform.shape

        # Optimize window size to be divisible by 2, 3, and 5
        window_size = int(window_size_seconds * sample_rate)
        if window_size < 16:
            window_size = 16
        window_size = self.optimize_windowsize(window_size)
        window_size = int(window_size / 2) * 2  # Ensure even window size
        half_window_size = int(window_size / 2)

        # Correct the end of the waveform by applying a fade-out
        end_size = int(sample_rate * 0.05)
        if end_size < 16:
            end_size = 16
        waveform[:, -end_size:] *= np.linspace(1, 0, end_size)

        # Compute displacement inside the input file
        start_pos = 0.0
        displace_pos = (window_size * 0.5) / stretch_factor

        # Create custom window function as in original code
        window = np.power(1.0 - np.power(np.linspace(-1.0, 1.0, window_size), 2.0), 1.25)

        # Initialize old windowed buffer
        old_windowed_buf = np.zeros((nchannels, window_size))

        # Initialize list to store output frames
        output_frames = []

        # Processing loop
        frame_count = 0
        while True:
            # Get the windowed buffer
            istart_pos = int(math.floor(start_pos))
            buf = waveform[:, istart_pos:istart_pos + window_size]
            if buf.shape[1] < window_size:
                buf = np.pad(buf, ((0, 0), (0, window_size - buf.shape[1])), 'constant')
            buf = buf * window

            # Get the amplitudes of the frequency components and discard the phases
            freqs = np.abs(np.fft.rfft(buf, axis=1))

            # Randomize the phases by multiplication with a random complex number with modulus=1
            ph = np.random.uniform(0, 2 * np.pi, (nchannels, freqs.shape[1])) * 1j
            freqs = freqs * np.exp(ph)

            # Do the inverse FFT
            buf_ifft = np.fft.irfft(freqs, n=window_size, axis=1)

            # Window again the output buffer
            buf_ifft *= window

            # Overlap-add the output
            output = buf_ifft[:, 0:half_window_size] + old_windowed_buf[:, half_window_size:window_size]
            old_windowed_buf = buf_ifft

            # Clamp the values to -1..1
            output = np.clip(output, -1.0, 1.0)

            # Append to output_frames
            output_frames.append(output)

            # Increment start_pos
            start_pos += displace_pos
            frame_count += 1

            # Check if we have reached the end of the input
            if start_pos >= nsamples:
                break

            # Optional: Log progress every 10%
            num_frames = max(int(math.ceil((nsamples - window_size) / (window_size * 0.5)) + 1), 1)
            if frame_count % max(int(num_frames / 10), 1) == 0:
                progress = (start_pos / nsamples) * 100

        # Concatenate all output frames horizontally
        output_array = np.hstack(output_frames)

        # Clamp final output to ensure no clipping
        output_array = np.clip(output_array, -1.0, 1.0)

        # Convert back to torch tensor
        output_tensor = torch.from_numpy(output_array).float()

        # Unsqueeze to add the first dimension back (e.g., batch dimension)
        output_tensor = output_tensor.unsqueeze(0)  # Shape: (1, C, N_stretched)


        return ({'waveform': output_tensor, 'sample_rate': sample_rate},)

    @staticmethod
    def optimize_windowsize(n: int) -> int:
        """
        Optimize the window size to be divisible by 2, 3, and 5.

        Parameters:
            n (int): Initial window size.

        Returns:
            int: Optimized window size.
        """
        orig_n = n
        while True:
            n = orig_n
            while (n % 2) == 0:
                n /= 2
            while (n % 3) == 0:
                n /= 3
            while (n % 5) == 0:
                n /= 5

            if n < 2:
                break
            orig_n += 1
        return orig_n


class SignalProcessingPadSynth:
    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        return {
            "required": {
                "samplerate": ("INT", {"default": 44100, "min": 8000, "max": 96000, "step": 1}),
                "fundamental_freq": ("FLOAT", {"default": 261.0, "min": 20.0, "max": 2000.0, "step": 1.0}),
                "bandwidth_cents": ("FLOAT", {"default": 40.0, "min": 10.0, "max": 100.0, "step": 1.0}),
                "number_harmonics": ("INT", {"default": 64, "min": 1, "max": 128, "step": 1}),
                "amplitude_per_harmonic": ("STRING", {"default": "", "multiline": False, "hint": "Provide a JSON array of amplitudes per harmonic, e.g., [0.0, 1.0, 0.5, ...] or leave empty for default."})
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    CATEGORY = "Signal Processing"
    FUNCTION = "process"

    def process(
        self,
        samplerate: int,
        fundamental_freq: float,
        bandwidth_cents: float,
        number_harmonics: int,
        amplitude_per_harmonic: str,
    ) -> Tuple[Dict[str, torch.Tensor]]:
        """
        Apply PADsynth algorithm to generate audio.

        Parameters:
            samplerate (int): Sampling rate in Hz.
            fundamental_freq (float): Fundamental frequency in Hz.
            bandwidth_cents (float): Bandwidth in cents for Gaussian profile.
            number_harmonics (int): Number of harmonics to generate.
            amplitude_per_harmonic (str): JSON-formatted string specifying amplitudes per harmonic. Leave empty for defaults.

        Returns:
            Tuple[Dict[str, torch.Tensor]]: Generated audio with waveform and sample rate.
        """

        import json
        # Initialize logging

        # Define FFT size
        N = 262144  # As per C++ code

        # Initialize amplitude array A
        if not amplitude_per_harmonic.strip():
            # Use default amplitude distribution
            A = torch.zeros(number_harmonics, dtype=torch.double)
            A[0] = 0.0  # A[0] is not used
            for i in range(1, number_harmonics):
                A[i] = 1.0 / i
                if (i % 2) == 0:
                    A[i] *= 2.0
        else:
            # Parse JSON string to list
            try:
                amplitude_list = json.loads(amplitude_per_harmonic)
                if not isinstance(amplitude_list, list):
                    raise ValueError("amplitude_per_harmonic must be a JSON array.")
                if len(amplitude_list) != number_harmonics:
                    raise ValueError("Length of amplitude_per_harmonic must match number_harmonics.")
                # Convert to torch tensor
                A = torch.tensor(amplitude_list, dtype=torch.double)
            except json.JSONDecodeError as e:
                raise ValueError("amplitude_per_harmonic must be a valid JSON array.") from e
            except Exception as e:
                raise

        # Initialize frequency amplitude and phase arrays
        freq_amp = torch.zeros(N // 2, dtype=torch.double)
        freq_phase = torch.rand(N // 2, dtype=torch.double) * 2.0 * math.pi  # Random phases between 0 and 2pi

        # Define Gaussian profile function
        def profile(fi: torch.Tensor, bwi: torch.Tensor) -> torch.Tensor:
            x = fi / bwi
            x_sq = x ** 2
            # Avoid computing exp(-x^2) for x_sq > 14.71280603
            mask = x_sq <= 14.71280603
            result = torch.zeros_like(x_sq)
            result[mask] = torch.exp(-x_sq[mask]) / bwi[mask]
            return result

        # Convert bandwidth from cents to Hz
        # bw_Hz = (2^(bw/1200) -1) * f * nh
        # Convert bandwidth_cents to multiplier
        bw_multiplier = 2.0 ** (bandwidth_cents / 1200.0) - 1.0

        # Populate frequency amplitude array
        for nh in range(1, number_harmonics):
            f_nh = fundamental_freq * nh
            bw_Hz = bw_multiplier * f_nh
            bwi = bw_Hz / (2.0 * samplerate)
            fi = f_nh / samplerate  # Normalized frequency

            # Create tensors for frequency bins
            i = torch.arange(N // 2, dtype=torch.double)
            # Normalized frequency for each bin
            normalized_freq = i / N  # Equivalent to i * (samplerate / N) / samplerate = i / N

            # Compute profile
            fi_tensor = torch.full_like(i, fi)
            bwi_tensor = torch.full_like(i, bwi)
            profile_values = profile(normalized_freq - fi_tensor, bwi_tensor)

            # Update frequency amplitude
            freq_amp += profile_values * A[nh]

        # Construct complex frequency domain tensor
        real = freq_amp * torch.cos(freq_phase)
        imag = freq_amp * torch.sin(freq_phase)
        freq_complex = torch.complex(real, imag)  # Shape: (N//2,)

        # Perform IFFT using torch.fft.irfft
        smp = torch.fft.irfft(freq_complex, n=N)  # Shape: (N,)

        # Normalize the signal to prevent clipping
        max_val = torch.max(torch.abs(smp))
        if max_val < 1e-5:
            max_val = 1e-5  # Prevent division by zero
        smp = smp / (max_val * math.sqrt(2))  # Normalize to 1/sqrt(2) as in C++ code

        # Convert to float32 for saving
        smp = smp.float()

        # Prepare waveform tensor: (C, N)
        waveform_out = smp.unsqueeze(0)  # Mono audio

        # Reshape waveform_out to include batch dimension: (1, C, N)
        waveform_out = waveform_out.unsqueeze(0)  # Shape: (1, C, N)

        # Return the synthesized audio
        return ({'waveform': waveform_out, 'sample_rate': samplerate},)


NODE_CLASS_MAPPINGS = {
    "SignalProcessingLoadAudio": SignalProcessingLoadAudio,
    "SignalProcessingFilter": SignalProcessingFilter,
    "SignalProcessingPaulStretch": SignalProcessingPaulStretch,
    "SignalProcessingPadSynth": SignalProcessingPadSynth
}

print('NODE_CLASS_MAPPINGS',NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {
    "SignalProcessingLoadAudio": "(SP) Load Audio",
    "SignalProcessingFilter": "(SP) Filter",
    "SignalProcessingPaulStretch" : "(SP) PaulStretch",
    "SignalProcessingPadSynth": "(SP) PadSynth"
}

print('NODE_DISPLAY_NAME_MAPPINGS',NODE_DISPLAY_NAME_MAPPINGS)
