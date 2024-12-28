import pytest
from ..tests.conftest import TestData
from ..core.io import audio_from_comfy_3d_to_disk
from ..processors.SignalProcessingSaturation import SignalProcessingSaturation

TEST_NAME = "saturation"


@pytest.mark.parametrize(
    "drive",
    [
        (90.0),
        (60.0),
        (30.0),
        (10.0),
        (0.0),
    ],
)
def test_saturation_general(test_data: TestData, drive: float) -> None:

    node = SignalProcessingSaturation()
    modes = node.INPUT_TYPES()["required"]["mode"][0]  # Extract modes

    for mode in modes:

        output = node.process(audio_input=test_data["audio"], mode=mode, drive=drive)[0]

        pest_param_str = test_data["pest_param_str"]
        output_filepath = test_data["output_root"] / f"{mode}-{pest_param_str}.wav"
        audio_from_comfy_3d_to_disk(output, output_filepath)

        assert (
            output_filepath.exists()
        ), f"Output file {output_filepath} was not created."
        assert output is not None, f"Processed audio output is None for mode {mode}."

    print(f"test_saturation_general {test_data['output_root']}")
