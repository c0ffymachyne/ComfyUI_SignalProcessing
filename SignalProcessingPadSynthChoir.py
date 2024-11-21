import torch
import math
from typing import Tuple, List, Dict

class SignalProcessingPadSynthChoir:
    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        return {
            "required": {
                "samplerate": ("INT", {"default": 44100, "min": 8000, "max": 96000, "step": 1}),
                "base_freq": ("FLOAT", {"default": 130.81, "min": 20.0, "max": 2000.0, "step": 1.0}),
                "step_size": ("INT", {"default": 4, "min": 1, "max": 24, "step": 1}),
                "num_notes": ("INT", {"default": 7, "min": 1, "max": 24, "step": 1}),
                "bandwidth_cents": ("FLOAT", {"default": 60.0, "min": 10.0, "max": 100.0, "step": 1.0}),
                "number_harmonics": ("INT", {"default": 64, "min": 1, "max": 128, "step": 1})
            }
        }

    RETURN_TYPES = ("AUDIO_LIST", "INT")
    RETURN_NAMES = ("audios", "sample_rate")
    CATEGORY = "Signal Processing"
    FUNCTION = "process"

    def process(
        self,
        samplerate: int,
        base_freq: float,
        step_size: int,
        num_notes: int,
        bandwidth_cents: float,
        number_harmonics: int,
    ) -> Tuple[List[Dict[str, torch.Tensor]]]:
        """
        Apply PADsynth choir algorithm to generate multiple audio files.

        Parameters:
            samplerate (int): Sampling rate in Hz.
            base_freq (float): Base frequency in Hz.
            step_size (int): Step size in semitones between notes.
            num_notes (int): Number of notes to generate.
            bandwidth_cents (float): Bandwidth in cents for Gaussian profile.
            number_harmonics (int): Number of harmonics to generate.

        Returns:
            Tuple[List[Dict[str, torch.Tensor]]]: List of generated audios with waveform and sample rate.
        """

        # Define FFT size
        N = 262144  # As per C++ code

        audios = []

        for note_index in range(num_notes):
            note_semitones = step_size * note_index
            f1 = base_freq * (2.0 ** (note_semitones / 12.0))

            # Compute amplitude_per_harmonic with formants
            A = torch.zeros(number_harmonics, dtype=torch.double)
            A[0] = 0.0  # A[0] is not used

            for i in range(1, number_harmonics):
                # Calculate formants based on the C++ choir implementation
                formants = (
                    math.exp(-((i * f1 - 600.0) / 150.0) ** 2) +
                    math.exp(-((i * f1 - 900.0) / 250.0) ** 2) +
                    math.exp(-((i * f1 - 2200.0) / 200.0) ** 2) +
                    math.exp(-((i * f1 - 2600.0) / 250.0) ** 2) +
                    math.exp(-((i * f1) / 3000.0) ** 2) * 0.1
                )
                A[i] = (1.0 / i) * formants
                # Optionally, you can debug amplitude values
                # logger.debug(f"Harmonic {i}: A[{i}]={A[i]:.4f}")

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
            bw_multiplier = 2.0 ** (bandwidth_cents / 1200.0) - 1.0

            # Create tensors for frequency bins
            i = torch.arange(N // 2, dtype=torch.double)
            normalized_freq = i / N  # Equivalent to i / N

            # Compute and accumulate frequency amplitudes for each harmonic
            for nh in range(1, number_harmonics):
                f_nh = f1 * nh
                bw_Hz = bw_multiplier * f_nh
                bwi = bw_Hz / (2.0 * samplerate)
                fi = f_nh / samplerate  # Normalized frequency

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

            

            print('SignalProcessingPadSynthChoir.waveform_out',waveform_out.shape)
            # Append to audios list
            audios.append({'waveform':waveform_out,'sample_rate': samplerate})

        # Return the list of generated audios

        return audios, samplerate