import pytest
from ..tests.conftest import TestData
from ..core.io import audio_from_comfy_3d_to_disk
from ..processors.SignalProcessingFilter import (
    SignalProcessingFilter,
)

TEST_NAME = "filter"


@pytest.mark.parametrize(
    "cutoff, q_factor",
    [
        (0.9, 0.707),
        (0.6, 0.707),
        (0.3, 0.707),
        (0.1, 0.707),
        (0.0, 0.707),
    ],
)
def test_filter_general(test_data: TestData, cutoff: float, q_factor: float) -> None:

    node: SignalProcessingFilter = SignalProcessingFilter()
    modes: str = node.INPUT_TYPES()["required"]["filter_type"][0]  # Extract modes

    for mode in modes:
        output = node.process(
            audio_input=test_data["audio"],
            cutoff=cutoff,
            filter_type=mode,
            q_factor=q_factor,
        )[0]

        pest_param_str = test_data["pest_param_str"]
        output_filepath = test_data["output_root"] / f"{mode}-{pest_param_str}.wav"
        audio_from_comfy_3d_to_disk(output, output_filepath)

        assert (
            output_filepath.exists()
        ), f"Output file {output_filepath} was not created."
        assert output is not None, f"Processed audio output is None for mode {mode}."

    print(f"test_filter_general {test_data['output_root']}")
