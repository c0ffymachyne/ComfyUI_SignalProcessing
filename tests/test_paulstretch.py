import pytest
from .conftest import TestData
from ..core.io import audio_from_comfy_3d_to_disk
from ..effects.SignalProcessingPaulStretch import SignalProcessingPaulStretch

TEST_NAME = "paulstretch"


@pytest.mark.parametrize(
    "stretch_factor, window_size_seconds",
    [
        (9.0, 0.25),
        (6.0, 0.25),
        (3.0, 0.25),
        (1.0, 0.25),
        (2.0, 0.25),
        (2.0, 0.5),
        (2.0, 0.75),
        (2.0, 1.0),
        (2.0, 3.0),
        (2.0, 6.0),
    ],
)
def test_paul_stretch_general(
    test_data: TestData, stretch_factor: float, window_size_seconds: float
) -> None:

    node = SignalProcessingPaulStretch()
    output = node.process(
        audio_input=test_data["audio"],
        stretch_factor=stretch_factor,
        window_size_seconds=window_size_seconds,
    )[0]

    pest_param_str = test_data["pest_param_str"]
    output_filepath = test_data["output_root"] / f"{pest_param_str}.wav"
    audio_from_comfy_3d_to_disk(output, output_filepath)

    assert output_filepath.exists(), f"Output file {output_filepath} was not created."
    assert output is not None, "Processed audio output is None."

    print(f"test_paul_stretch_general {test_data['output_root']}")
