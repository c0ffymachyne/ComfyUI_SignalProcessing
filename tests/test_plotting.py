import pytest
from torchvision.transforms import ToPILImage
from .conftest import TestData
from ..core.plotting import get_spectogram, get_wave, save_image
from ..visuals.SignalProcessingSpectrogram import SignalProcessingSpectrogram

TEST_NAME = "plotting"


@pytest.mark.parametrize(
    "n_fft, n_mels, xlim",
    [
        (4096, 128 * 1, 8192),
        (4096, 128 * 2, 8192),
        (4096, 128 * 4, 8192),
        (4096, 128 * 6, 8192),
    ],
)
def test_plotting_spectogram_general(
    test_data: TestData, n_fft: int, n_mels: int, xlim: int
) -> None:

    waveform = test_data["audio"]["waveform"].squeeze(0)
    sample_rate = test_data["audio"]["sample_rate"]

    print("waveform", waveform.shape)

    spectogram = get_spectogram(
        waveform, sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels, xlim=xlim
    )

    pest_param_str = test_data["pest_param_str"]
    output_filepath = test_data["output_root"] / f"{pest_param_str}.png"

    save_image(output_filepath, spectogram)

    assert output_filepath.exists(), f"Output file {output_filepath} was not created."
    assert output_filepath is not None, "Processed audio output is None."

    print(f"test_plotting_spectogram_general {test_data['output_root']}")


@pytest.mark.parametrize(
    "stretch_factor, window_size_seconds",
    [(8.0, 0.25)],
)
def test_plotting_waveform_general(
    test_data: TestData, stretch_factor: float, window_size_seconds: float
) -> None:

    waveform = test_data["audio"]["waveform"].squeeze(0)
    sample_rate = test_data["audio"]["sample_rate"]

    print("waveform", waveform.shape)

    spectogram = get_wave(waveform, sample_rate=sample_rate, xlim=4096)

    pest_param_str = test_data["pest_param_str"]
    output_filepath = test_data["output_root"] / f"{pest_param_str}.png"

    save_image(output_filepath, spectogram)

    assert output_filepath.exists(), f"Output file {output_filepath} was not created."
    assert output_filepath is not None, "Processed audio output is None."

    print(f"test_plotting_spectogram_general {test_data['output_root']}")


@pytest.mark.parametrize(
    "stretch_factor, window_size_seconds",
    [(8.0, 0.25)],
)
def test_plotting_waveform_node_general(
    test_data: TestData, stretch_factor: float, window_size_seconds: float
) -> None:

    node = SignalProcessingSpectrogram()

    output = node.process(audio_input=test_data["audio"])[0]

    pest_param_str = test_data["pest_param_str"]
    output_filepath = test_data["output_root"] / f"{pest_param_str}.png"

    to_pil = ToPILImage()
    print("rgb_image[0] -----------------------------", output[0].shape)
    rgb_image = output[0][..., :3]
    rgb_image = rgb_image.permute(2, 0, 1)
    print("rgb_image -----------------------------", rgb_image.shape)
    spectogram = to_pil(rgb_image)
    print("spectogram -----------------------------", spectogram)
    save_image(output_filepath, spectogram)

    assert output_filepath.exists(), f"Output file {output_filepath} was not created."
    assert output is not None, "Processed audio output is None."

    print(f"test_pitch_shift_general {test_data['output_root']}")
