
import torch, torchaudio
import torch
from typing import Dict

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

    RETURN_TYPES = ("AUDIO","INT")
    RETURN_NAMES = ("audio","sample_rate")
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

        filtered_waveform = filtered_waveform.unsqueeze(0)

        print('SignalProcessingFilter.filtered_waveform',filtered_waveform.shape)

        return {'waveform': filtered_waveform, 'sample_rate': sample_rate}, sample_rate