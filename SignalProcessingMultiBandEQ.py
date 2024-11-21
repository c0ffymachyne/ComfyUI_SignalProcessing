import torch
import torchaudio

def normalize_waveform(waveform):
    """
    Normalize the waveform to ensure its maximum absolute value is 1.0.
    
    Parameters:
        waveform (torch.Tensor): Input waveform tensor with shape [batch, channels, samples].
    
    Returns:
        torch.Tensor: Normalized waveform tensor with the same shape.
    """
    max_val = torch.max(torch.abs(waveform))
    if max_val > 1.0:
        waveform = waveform / max_val
    return waveform

class SignalProcessingMultibandEQ:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_input": ("AUDIO",),
                "method" : (['hann','rfft'],)
            },
            "optional": {
                "sub_bass_gain_db": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "bass_gain_db": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "low_mid_gain_db": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "mid_gain_db": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "upper_mid_gain_db": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "presence_gain_db": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
                "brilliance_gain_db": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("equalized_audio",)
    CATEGORY = "Audio Processing"
    FUNCTION = "process"

    def process(
        self,
        audio_input,
        method,
        sub_bass_gain_db=0.0,
        bass_gain_db=0.0,
        low_mid_gain_db=0.0,
        mid_gain_db=0.0,
        upper_mid_gain_db=0.0,
        presence_gain_db=0.0,
        brilliance_gain_db=0.0
    ):
        if method == 'rfft':
            return self.process_rfft(audio_input,sub_bass_gain_db,bass_gain_db,low_mid_gain_db,mid_gain_db,upper_mid_gain_db,presence_gain_db,brilliance_gain_db)
        if method == 'hann':
            return self.process_hann(audio_input,sub_bass_gain_db,bass_gain_db,low_mid_gain_db,mid_gain_db,upper_mid_gain_db,presence_gain_db,brilliance_gain_db)

    def process_rfft(
        self,
        audio_input,
        sub_bass_gain_db=0.0,
        bass_gain_db=0.0,
        low_mid_gain_db=0.0,
        mid_gain_db=0.0,
        upper_mid_gain_db=0.0,
        presence_gain_db=0.0,
        brilliance_gain_db=0.0
    ):
        """
        Apply multiband equalization to the input audio.

        Parameters:
            audio_input (Dict): Dictionary containing 'waveform' and 'sample_rate'.
            sub_bass_gain_db (float): Gain for the Sub-Bass band in decibels.
            bass_gain_db (float): Gain for the Bass band in decibels.
            low_mid_gain_db (float): Gain for the Low-Mid band in decibels.
            mid_gain_db (float): Gain for the Mid band in decibels.
            upper_mid_gain_db (float): Gain for the Upper-Mid band in decibels.
            presence_gain_db (float): Gain for the Presence band in decibels.
            brilliance_gain_db (float): Gain for the Brilliance band in decibels.

        Returns:
            Tuple[Dict[str, torch.Tensor]]: Dictionary with equalized 'waveform' and 'sample_rate'.
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

        # Convert gains from dB to linear scale
        sub_bass_gain = 10 ** (sub_bass_gain_db / 20)
        bass_gain = 10 ** (bass_gain_db / 20)
        low_mid_gain = 10 ** (low_mid_gain_db / 20)
        mid_gain = 10 ** (mid_gain_db / 20)
        upper_mid_gain = 10 ** (upper_mid_gain_db / 20)
        presence_gain = 10 ** (presence_gain_db / 20)
        brilliance_gain = 10 ** (brilliance_gain_db / 20)

        # Perform FFT
        waveform_fft = torch.fft.rfft(waveform, dim=-1)  # [batch, channels, freq_bins]

        # Frequency bins
        freq_bins = torch.linspace(0, sample_rate / 2, steps=waveform_fft.size(-1), device=waveform.device)

        # Define frequency bands
        bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'upper_mid': (2000, 4000),
            'presence': (4000, 6000),
            'brilliance': (6000, 20000)
        }

        # Initialize gain factors tensor
        gain_factors = torch.ones_like(waveform_fft)  # [batch, channels, freq_bins]

        # Apply gains to each band
        for band, (f_start, f_end) in bands.items():
            if band == 'sub_bass':
                gain = sub_bass_gain
            elif band == 'bass':
                gain = bass_gain
            elif band == 'low_mid':
                gain = low_mid_gain
            elif band == 'mid':
                gain = mid_gain
            elif band == 'upper_mid':
                gain = upper_mid_gain
            elif band == 'presence':
                gain = presence_gain
            elif band == 'brilliance':
                gain = brilliance_gain
            else:
                gain = 1.0  # Default gain

            # Create mask for the current band
            mask = (freq_bins >= f_start) & (freq_bins < f_end)
            gain_factors[:, :, mask] *= gain

        # Apply gain factors
        waveform_fft_eq = waveform_fft * gain_factors  # [batch, channels, freq_bins]

        # Perform inverse FFT
        waveform_eq = torch.fft.irfft(waveform_fft_eq, n=num_samples, dim=-1)  # [batch, channels, samples]

        # Normalize the waveform
        waveform_eq = normalize_waveform(waveform_eq)  # [batch, channels, samples]

        # Prepare the output dictionary
        equalized_audio = {
            'waveform': waveform_eq,
            'sample_rate': sample_rate
        }

        return (equalized_audio, )

    def process_hann(
        self,
        audio_input,
        sub_bass_gain_db=0.0,
        bass_gain_db=0.0,
        low_mid_gain_db=0.0,
        mid_gain_db=0.0,
        upper_mid_gain_db=0.0,
        presence_gain_db=0.0,
        brilliance_gain_db=0.0
    ):
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

        # Limit maximum gain to prevent artifacts
        max_gain_db = 12.0
        gains_db = {
            'sub_bass': min(sub_bass_gain_db, max_gain_db),
            'bass': min(bass_gain_db, max_gain_db),
            'low_mid': min(low_mid_gain_db, max_gain_db),
            'mid': min(mid_gain_db, max_gain_db),
            'upper_mid': min(upper_mid_gain_db, max_gain_db),
            'presence': min(presence_gain_db, max_gain_db),
            'brilliance': min(brilliance_gain_db, max_gain_db)
        }

        gains_linear = {band: 10 ** (gain_db / 20) for band, gain_db in gains_db.items()}

        # FFT parameters
        n_fft = 2048
        hop_length = n_fft // 2
        window = torch.hann_window(n_fft, device=waveform.device, periodic=True)

        # Perform STFT
        waveform_stft = torch.stft(
            waveform.view(-1, num_samples),
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
            center=True,
            pad_mode='reflect'
        )  # [batch*channels, freq_bins, time_frames]

        # Frequency bins
        freq_bins = torch.linspace(0, sample_rate / 2, steps=waveform_stft.size(1), device=waveform.device)

        # Initialize gain factors tensor
        gain_factors = torch.ones_like(waveform_stft, device=waveform.device)

        # Define frequency bands
        bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'upper_mid': (2000, 4000),
            'presence': (4000, 6000),
            'brilliance': (6000, sample_rate / 2)
        }

        # Apply gains to each band with smoother transitions
        for band, (f_start, f_end) in bands.items():
            gain = gains_linear[band]

            # Smooth transition using Gaussian profile
            center_freq = (f_start + f_end) / 2
            bandwidth = (f_end - f_start) / 2
            gain_profile = torch.exp(-0.5 * ((freq_bins - center_freq) / (bandwidth / 2)) ** 2)
            gain_profile = 1 + (gain - 1) * gain_profile
            gain_factors *= gain_profile.unsqueeze(1)

        # Apply gain factors
        waveform_stft_eq = waveform_stft * gain_factors

        # Perform inverse STFT
        waveform_eq = torch.istft(
            waveform_stft_eq,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            length=num_samples,
            center=True
        )

        # Reshape back to original dimensions
        waveform_eq = waveform_eq.view(batch_size, channels, num_samples)

        # Normalize the waveform
        waveform_eq = normalize_waveform(waveform_eq)

        # Prepare the output dictionary
        equalized_audio = {
            'waveform': waveform_eq,
            'sample_rate': sample_rate
        }

        return (equalized_audio,)
