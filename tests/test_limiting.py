import pytest
from ..tests.conftest import TestData
from typing import Tuple
from ..core.io import audio_from_comfy_3d_to_disk
from ..processors.SignalProcessingLimiter import SignalProcessingLimiter

TEST_NAME = "limiting"

params: list[Tuple[float, float, float]] = [
    (0.0, 100.0, 600.0),
    (10.0, 100.0, 600.0),
    (30.0, 100.0, 600.0),
    (90.0, 100.0, 600.0),
]


@pytest.mark.parametrize(
    "threshold, slope, release_ms",
    params,
)
def test_limiting_general(
    test_data: TestData, threshold: float, slope: float, release_ms: float
) -> None:

    node = SignalProcessingLimiter()
    modes = node.INPUT_TYPES()["required"]["mode"][0]  # Extract modes

    for mode in modes:

        output = node.process(
            audio_input=test_data["audio"],
            mode=mode,
            threshold=threshold,
            slope=slope,
            release_ms=release_ms,
        )[0]

        pest_param_str = test_data["pest_param_str"]
        output_filepath = test_data["output_root"] / f"{mode}-{pest_param_str}.wav"
        audio_from_comfy_3d_to_disk(output, output_filepath)

        assert (
            output_filepath.exists()
        ), f"Output file {output_filepath} was not created."
        assert output is not None, f"Processed audio output is None for mode {mode}."

    print(f"test_limiting_general {test_data['output_root']}")
