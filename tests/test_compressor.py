import pytest
from ..tests.conftest import TestData
from typing import Tuple
from ..core.io import audio_from_comfy_3d_to_disk
from ..processors.SignalProcessingCompressor import SignalProcessingCompressor

TEST_NAME = "compression"

params: list[Tuple[float, float, float, float]] = [
    (0.9, 0.1, 60.0, 0.3),
    (0.3, 0.1, 60.0, 0.3),
    (0.1, 0.1, 60.0, 0.3),
    (-0.0, 0.1, 60.0, 0.3),
    (-0.1, 0.1, 60.0, 0.3),
    (-0.3, 0.1, 60.0, 0.3),
    (-0.9, 0.1, 60.0, 0.3),
]


@pytest.mark.parametrize(
    "comp, attack, release, filter_param",
    params,
)
def test_compressor_general(
    test_data: TestData, comp: float, attack: float, release: float, filter_param: float
) -> None:
    """
    Test SignalProcessingCompressor with various parameter configurations.
    """
    node = SignalProcessingCompressor()

    # Process input audio
    output = node.process(
        audio_input=test_data["audio"],
        comp=comp,
        attack=attack,
        release=release,
        filter_param=filter_param,
    )[0]

    # Save the output audio
    pest_param_str = test_data["pest_param_str"]
    output_filepath = test_data["output_root"] / f"{pest_param_str}.wav"
    audio_from_comfy_3d_to_disk(output, output_filepath)

    # Assertions
    assert output_filepath.exists(), f"Output file {output_filepath} was not created."
    assert output is not None, "Processed audio output is None."

    print(f"test_compressor_general {test_data['output_root']}")
