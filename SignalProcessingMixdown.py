import torch

import os, sys
from typing import Tuple, List, Dict, Union
import torchaudio


class SignalProcessingMixdown:
    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        return {
            "required": {
                "audio_inputs": ("AUDIO_LIST", {"default": []}),  # List of audio inputs
                "output_normalization": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
            },
            "optional": {
                "gain_factors": ("FLOAT_LIST", {"default": [], "min": 0.0, "max": 2.0, "step": 0.1}),
                # If empty, default to [1.0] * num_audios
            }
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("mixed_audio", "sample_rate")
    CATEGORY = "Audio Processing"
    FUNCTION = "process"

    def process(
        self,
        audio_inputs: List[Dict[str, Union[torch.Tensor, int]]],
        output_normalization: float,
        gain_factors: List[float] = [],
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Mix down multiple audio inputs into a single audio output with optional individual volume controls.

        Parameters:
            audio_inputs (List[Dict]): List of audio inputs, each containing 'waveform' and 'sample_rate'.
            output_normalization (float): Normalization factor for the mixed audio (0.0 to 1.0).
            gain_factors (List[float], optional): List of gain factors for each audio input.

        Returns:
            Tuple[Dict[str, torch.Tensor], int]: Mixed audio with waveform and sample rate.
        """

        if not audio_inputs:
            raise ValueError("No audio inputs provided for mixing.")

        num_audios = len(audio_inputs)
        print(f"SignalProcessingMixdown: Number of audio inputs: {num_audios}")

        # Handle gain_factors
        if not gain_factors:
            gain_factors = [1.0] * num_audios
            print("SignalProcessingMixdown: No gain_factors provided. Defaulting to 1.0 for all inputs.")
        elif len(gain_factors) != num_audios:
            raise ValueError(f"Number of gain factors ({len(gain_factors)}) does not match number of audio inputs ({num_audios}).")
        else:
            print(f"SignalProcessingMixdown: Gain factors: {gain_factors}")

        # Extract sample rates and verify consistency
        sample_rates = [audio['sample_rate'] for audio in audio_inputs]
        target_sample_rate = sample_rates[0]

        for idx, sr in enumerate(sample_rates):
            if sr != target_sample_rate:
                print(f"SignalProcessingMixdown: Resampling audio {idx} from {sr} Hz to {target_sample_rate} Hz")
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
                audio_inputs[idx]['waveform'] = resampler(audio_inputs[idx]['waveform'])
                audio_inputs[idx]['sample_rate'] = target_sample_rate

        # Determine the maximum length among all audio inputs
        lengths = [audio['waveform'].shape[-1] for audio in audio_inputs]
        max_length = max(lengths)
        print(f"SignalProcessingMixdown: Maximum waveform length: {max_length} samples")

        # Pad or truncate each audio to match the maximum length and apply gain
        for idx, audio in enumerate(audio_inputs):
            waveform = audio['waveform']
            current_length = waveform.shape[-1]
            gain = gain_factors[idx]
            print(f"SignalProcessingMixdown: Processing audio {idx} with gain factor {gain}")

            if current_length < max_length:
                padding = max_length - current_length
                # Pad with zeros (silence) at the end
                waveform = torch.nn.functional.pad(waveform, (0, padding))
                print(f"SignalProcessingMixdown: Padded audio {idx} with {padding} zeros")
            elif current_length > max_length:
                # Truncate the waveform to max_length
                waveform = waveform[:, :, :max_length]
                print(f"SignalProcessingMixdown: Truncated audio {idx} to {max_length} samples")

            # Apply gain
            waveform = waveform * gain
            print(f"SignalProcessingMixdown: Applied gain to audio {idx}")

            audio['waveform'] = waveform

        # Sum all waveforms to create the mix
        mixed_waveform = torch.zeros_like(audio_inputs[0]['waveform'])
        for idx, audio in enumerate(audio_inputs):
            mixed_waveform += audio['waveform']
            print(f"SignalProcessingMixdown: Summed audio {idx} into mixed waveform")

        # Normalize the mixed audio based on the maximum absolute value
        max_val = torch.max(torch.abs(mixed_waveform))
        if max_val < 1e-5:
            max_val = 1e-5  # Prevent division by zero
            print("SignalProcessingMixdown: Maximum value too low, setting to 1e-5 to prevent division by zero")

        # Apply output normalization as a multiplier
        mixed_waveform = (mixed_waveform / max_val) * output_normalization
        print(f"SignalProcessingMixdown: Applied output normalization factor {output_normalization}")

        # Clamp to [-1.0, 1.0] to ensure no clipping occurs
        mixed_waveform = torch.clamp(mixed_waveform, -1.0, 1.0)
        print("SignalProcessingMixdown: Clamped mixed waveform to [-1.0, 1.0]")

        # Ensure the tensor is of type float32
        mixed_waveform = mixed_waveform.float()

        # Ensure the tensor is on CPU and contiguous
        mixed_waveform = mixed_waveform.to('cpu').contiguous()

        print(f"SignalProcessingMixdown: Mixed waveform shape: {mixed_waveform.shape}")

        # Return the mixed audio and sample rate
        return ({'waveform': mixed_waveform, 'sample_rate': target_sample_rate}, target_sample_rate)