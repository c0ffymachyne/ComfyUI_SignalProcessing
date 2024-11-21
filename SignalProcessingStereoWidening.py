import torch

#https://github.com/AudioEmerge/Widener
class SignalProcessingStereoWidening:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_input": ("AUDIO",),
                "gain": ("FLOAT", {"default": 2.0, "min": 0, "max": 5.0, "step": 0.1}),
            },
            "optional": {
                "width": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("widened_audio",)
    CATEGORY = "Audio Processing"
    FUNCTION = "process"

    def process(self, audio_input, gain, width=1.2):
        """
        Widen stereo audio or convert mono audio to wide stereo using the provided widening algorithm.

        Parameters:
            audio_input (Dict): Dictionary containing 'waveform' and 'sample_rate'.
            width (float): Width factor (>1.0).

        Returns:
            Tuple[Dict[str, torch.Tensor]]: Dictionary with widened 'waveform' and 'sample_rate'.
        """


        waveform = audio_input.get('waveform')  # [batch, channels, samples]
        sample_rate = audio_input.get('sample_rate')

        if waveform is None or sample_rate is None:
            raise ValueError("Input audio must contain 'waveform' and 'sample_rate'.")

        if not isinstance(waveform, torch.Tensor):
            raise TypeError("Waveform must be a torch.Tensor.")

        if waveform.ndim != 3:
            raise ValueError("Waveform must be a 3D tensor with shape (batch, channels, samples).")

        batch_size, channels, num_samples = waveform.shape

        if channels not in [1, 2]:
            raise ValueError(f"Unsupported number of channels: {channels}. Only mono and stereo are supported.")

        # Calculate coefficients based on the provided width parameter
        tmp = 1.0 / max(1.0 + width, 2.0)  # Scalar

        coef_M = 1.0 * tmp  # Coefficient for mid
        coef_S = width * tmp  # Coefficient for sides

        if channels == 2:
            # Stereo to Widened Stereo
            L = waveform[:, 0, :]  # Left channel [batch, samples]
            R = waveform[:, 1, :]  # Right channel [batch, samples]

            # Apply the widening algorithm
            mid = (L + R) * coef_M  # Mid signal
            sides = (R - L) * coef_S  # Side signal

            widened_L = mid - sides  # New Left channel
            widened_R = mid + sides  # New Right channel

            # Stack the widened channels back into a stereo waveform
            widened_waveform = torch.stack((widened_L, widened_R), dim=1)  # [batch, 2, samples]

        elif channels == 1:
            # Mono to Wide Stereo
            L = waveform[:, 0, :].clone()  # Duplicate mono channel to Left
            R = waveform[:, 0, :].clone()  # Duplicate mono channel to Right

            # Apply the widening algorithm
            mid = (L + R) * coef_M  # Mid signal
            sides = (R - L) * coef_S  # Side signal

            widened_L = mid - sides  # New Left channel
            widened_R = mid + sides  # New Right channel

            # Stack the widened channels into a stereo waveform
            widened_waveform = torch.stack((widened_L, widened_R), dim=1)  # [batch, 2, samples]

        # Clamp the waveform to prevent clipping

        widened_waveform = widened_waveform * gain
        widened_waveform = torch.clamp(widened_waveform, -1.0, 1.0)

        # Prepare the output dictionary
        widened_audio = {
            'waveform': widened_waveform,
            'sample_rate': sample_rate
        }

        return widened_audio, sample_rate