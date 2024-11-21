import torch
import os, sys
from typing import Tuple, List, Dict, Union

class SignalProcessingVolumeControl:
    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        return {
            "required": {
                "audio_input": ("AUDIO",),  # Single audio input
                "gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("AUDIO","INT")
    RETURN_NAMES = ("adjusted_audio","sample_rate")
    CATEGORY = "Audio Processing"
    FUNCTION = "process"

    def process(
        self,
        audio_input: Dict[str, Union[torch.Tensor, int]],
        gain: float,
    ) -> Tuple[Dict[str, torch.Tensor]]:
        """
        Adjust the volume of the input audio by applying a gain factor.

        Parameters:
            audio_input (Dict): Dictionary containing 'waveform' and 'sample_rate'.
            gain (float): Gain factor to adjust the volume. >1.0 amplifies, <1.0 attenuates.

        Returns:
            Tuple[Dict[str, torch.Tensor]]: Dictionary with adjusted 'waveform' and 'sample_rate'.
        """

        # Extract waveform and sample rate
        waveform = audio_input.get('waveform')
        sample_rate = audio_input.get('sample_rate')

        if waveform is None or sample_rate is None:
            raise ValueError("Input audio must contain 'waveform' and 'sample_rate'.")

        # Validate waveform tensor
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("Waveform must be a torch.Tensor.")
        if waveform.ndim != 3:
            raise ValueError("Waveform must be a 3D tensor with shape (batch, channels, samples).")
        if waveform.dtype != torch.float32:
            waveform = waveform.float()
        if not waveform.is_contiguous():
            waveform = waveform.contiguous()

        # Apply gain as a multiplier
        adjusted_waveform = waveform * gain
        print(f"AudioVolumeControl: Applied gain factor {gain}")

        # Clamp to [-1.0, 1.0] to prevent clipping
        adjusted_waveform = torch.clamp(adjusted_waveform, -1.0, 1.0)
        print("AudioVolumeControl: Clamped adjusted waveform to [-1.0, 1.0]")

        # Ensure the tensor is on CPU
        if adjusted_waveform.device != torch.device('cpu'):
            adjusted_waveform = adjusted_waveform.to('cpu')

        # Prepare the output dictionary
        adjusted_audio = {
            'waveform': adjusted_waveform,
            'sample_rate': sample_rate
        }

        return adjusted_audio, sample_rate