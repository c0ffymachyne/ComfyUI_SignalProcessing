import pytest
from .conftest import TestData
from ..core.io import audio_from_comfy_3d_to_disk
from ..effects.SignalProcessingPitchShifter import SignalProcessingPitchShifter

TEST_NAME = "pitchshift"


@pytest.mark.parametrize(
    "pitch_shift_factor",
    [(3), (1), (0), (-1), (-3)],
)
def test_pitch_shift_general(test_data: TestData, pitch_shift_factor: int) -> None:

    node = SignalProcessingPitchShifter()

    output = node.process(
        audio_input=test_data["audio"], pitch_shift_factor=pitch_shift_factor
    )[0]

    pest_param_str = test_data["pest_param_str"]
    output_filepath = test_data["output_root"] / f"{pest_param_str}.wav"
    audio_from_comfy_3d_to_disk(output, output_filepath)

    assert output_filepath.exists(), f"Output file {output_filepath} was not created."
    assert output is not None, "Processed audio output is None."

    print(f"test_pitch_shift_general {test_data['output_root']}")
