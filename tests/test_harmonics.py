import pytest
from ..tests.conftest import TestData
from typing import Tuple
from ..core.io import audio_from_comfy_3d_to_disk
from ..processors.SignalProcessingHarmonicsEnhancer import (
    SignalProcessingHarmonicsEnhancer,
)

TEST_NAME = "harmonics"

params: list[Tuple[str, int, float, float]] = [
    ("2,3", 440, 6.0, 0.707),
    ("2,3", 440, 3.0, 0.707),
    ("2,3", 440, 1.0, 0.707),
    ("2,3", 440, 0.0, 0.707),
]


@pytest.mark.parametrize(
    "harmonics, base_frequency, gain_db, Q",
    params,
)
def test_harmonics_general(
    test_data: TestData, harmonics: str, base_frequency: int, gain_db: int, Q: float
) -> None:

    node = SignalProcessingHarmonicsEnhancer()
    modes: str = node.INPUT_TYPES()["required"]["mode"][0]  # Extract modes

    for mode in modes:
        output = node.process(
            audio_input=test_data["audio"],
            harmonics=harmonics,
            mode=mode,
            base_frequency=base_frequency,
            gain_db=gain_db,
            Q=Q,
        )[0]

        pest_param_str = test_data["pest_param_str"]
        output_filepath = test_data["output_root"] / f"{mode}-{pest_param_str}.wav"
        audio_from_comfy_3d_to_disk(output, output_filepath)

        assert (
            output_filepath.exists()
        ), f"Output file {output_filepath} was not created."
        assert output is not None, f"Processed audio output is None for mode {mode}."

    print(f"test_harmonics_general {test_data['output_root']}")
