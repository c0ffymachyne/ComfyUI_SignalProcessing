import pytest
from ..tests.conftest import TestData
from ..core.io import audio_from_comfy_3d_to_disk
from ..effects.SignalProcessingStereoWidening import SignalProcessingStereoWidening

TEST_NAME = "widening"


@pytest.mark.parametrize(
    "width",
    [
        (3.0),
        (2.0),
        (1.0),
        (0.5),
        (0.25),
        (0.0),
    ],
)
def test_widening_general(test_data: TestData, width: float) -> None:

    node = SignalProcessingStereoWidening()
    modes = node.INPUT_TYPES()["required"]["mode"][0]

    for mode in modes:

        output = node.process(audio_input=test_data["audio"], mode=mode, width=width)[0]

        pest_param_str = test_data["pest_param_str"]
        output_filepath = test_data["output_root"] / f"{mode}-{pest_param_str}.wav"
        audio_from_comfy_3d_to_disk(output, output_filepath)

        assert (
            output_filepath.exists()
        ), f"Output file {output_filepath} was not created."
        assert output is not None, f"Processed audio output is None for mode {mode}."

    print(f"test_widening_general {test_data['output_root']}")
