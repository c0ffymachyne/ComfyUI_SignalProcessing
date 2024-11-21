import torch

import math, json
from typing import Tuple, List, Dict

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

    RETURN_TYPES = ("AUDIO","INT")
    RETURN_NAMES = ("audio","sample_rate")
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

        #audios = []

        #print('SignalProcessingPadSynth.waveform_out',waveform_out.shape)

        #audios.append({'waveform': waveform_out, 'sample_rate': samplerate})

        # Return the synthesized audio
        return {'waveform': waveform_out, 'sample_rate': samplerate}, samplerate