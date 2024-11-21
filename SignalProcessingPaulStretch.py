import torch

import math
from typing import Tuple, List, Dict

import numpy as np

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

    RETURN_TYPES = ("AUDIO","INT")
    RETURN_NAMES = ("audio","sample_rate")
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

        print('SignalProcessingPaulStretch.waveform_out',output_tensor.shape)


        return {'waveform': output_tensor, 'sample_rate': sample_rate}, sample_rate

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