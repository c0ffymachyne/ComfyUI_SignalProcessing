import torch
import torchaudio


def normalize_waveform(waveform):
    max_val = torch.max(torch.abs(waveform))
    if max_val > 1.0:
        waveform = waveform / max_val
    return waveform

class SignalProcessingMultiBandEQExperimental:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_input": ("AUDIO",),
            },
            "optional": {
                "sub_bass_gain_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0, "step": 0.1}),
                "bass_gain_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0, "step": 0.1}),
                "low_mid_gain_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0, "step": 0.1}),
                "mid_gain_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0, "step": 0.1}),
                "upper_mid_gain_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0, "step": 0.1}),
                "presence_gain_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0, "step": 0.1}),
                "brilliance_gain_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("equalized_audio",)
    CATEGORY = "Audio Processing"
    FUNCTION = "process"

    def process(
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

        # Limit gain to prevent artifacts
        max_gain_db = 24.0
        min_gain_db = -24.0
        gains_db = {
            'sub_bass': max(min(sub_bass_gain_db, max_gain_db), min_gain_db),
            'bass': max(min(bass_gain_db, max_gain_db), min_gain_db),
            'low_mid': max(min(low_mid_gain_db, max_gain_db), min_gain_db),
            'mid': max(min(mid_gain_db, max_gain_db), min_gain_db),
            'upper_mid': max(min(upper_mid_gain_db, max_gain_db), min_gain_db),
            'presence': max(min(presence_gain_db, max_gain_db), min_gain_db),
            'brilliance': max(min(brilliance_gain_db, max_gain_db), min_gain_db),
        }

        # Convert gains from dB to linear scale
        gains_linear = {band: 10 ** (gain_db / 20) for band, gain_db in gains_db.items()}

        # STFT parameters
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
        gain_factors = torch.ones_like(waveform_stft)

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
            center_freq = (f_start + f_end) / 2
            bandwidth = (f_end - f_start)
            if bandwidth == 0:
                bandwidth = f_end * 0.01

            gain_profile = 1 + (gain - 1) * torch.exp(-0.5 * ((freq_bins - center_freq) / (bandwidth / 2)) ** 2)
            gain_profile = gain_profile.unsqueeze(0).unsqueeze(2)
            gain_factors = gain_factors * gain_profile

        # Apply gain factors
        waveform_stft_eq = waveform_stft * gain_factors

        # Bass enhancement using compression
        # Define compressor parameters
        threshold = -20.0  # in dB
        ratio = 4.0  # compression ratio
        knee = 5.0  # in dB
        makeup_gain = 5.0  # in dB

        # Convert to linear scale
        threshold_linear = 10 ** (threshold / 20)
        makeup_gain_linear = 10 ** (makeup_gain / 20)

        # Apply compression to sub-bass and bass bands
        bass_mask = torch.zeros_like(freq_bins)
        bass_indices = ((freq_bins >= bands['sub_bass'][0]) & (freq_bins <= bands['bass'][1]))
        bass_mask[bass_indices] = 1.0
        bass_mask = bass_mask.unsqueeze(0).unsqueeze(2)

        # Get magnitude and phase
        magnitude = torch.abs(waveform_stft_eq)
        phase = torch.angle(waveform_stft_eq)

        # Compress magnitude in bass frequencies
        magnitude_compressed = magnitude.clone()

        bass_magnitude = magnitude * bass_mask
        bass_magnitude_db = 20 * torch.log10(bass_magnitude + 1e-8)

        # Apply soft knee compression
        over_threshold = bass_magnitude_db - threshold
        compression_amount = torch.zeros_like(over_threshold)
        compression_mask1 = over_threshold > -knee/2
        compression_mask2 = over_threshold > knee/2
        compression_amount[compression_mask1] = (1 / ratio - 1) * (over_threshold[compression_mask1] + knee/2) ** 2 / (2 * knee)
        compression_amount[compression_mask2] = (1 / ratio - 1) * over_threshold[compression_mask2]
        gain_reduction_db = compression_amount
        gain_reduction_linear = 10 ** (gain_reduction_db / 20)

        magnitude_compressed = magnitude_compressed * (1 + gain_reduction_linear * bass_mask)

        # Apply makeup gain
        magnitude_compressed = magnitude_compressed * makeup_gain_linear

        # Reconstruct the compressed STFT
        waveform_stft_eq = magnitude_compressed * torch.exp(1j * phase)

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

        return (equalized_audio, )