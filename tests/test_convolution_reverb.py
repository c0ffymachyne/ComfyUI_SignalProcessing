import pytest
from .conftest import TestData
from ..core.io import audio_from_comfy_3d_to_disk
from ..effects.SignalProcessingConvolutionReverb import (
    SignalProcessingConvolutionReverb,
)

TEST_NAME = "convolution_reverb"


@pytest.mark.parametrize(
    "impulse_response, wet_dry",
    [("ir.wav", 1.0), ("ir.wav", 0.6), ("ir.wav", 0.3), ("ir.wav", 0.0)],
)
def test_convolution_reverb_general(
    test_data: TestData, impulse_response: str, wet_dry: float
) -> None:
    SignalProcessingConvolutionReverb.ir_directory = test_data["inputs_root"] / "ir"
    node = SignalProcessingConvolutionReverb()

    node.INPUT_TYPES()

    output = node.process(
        impulse_response=impulse_response,
        audio_input=test_data["audio"],
        wet_dry=wet_dry,
    )[0]

    pest_param_str = test_data["pest_param_str"]
    output_filepath = test_data["output_root"] / f"{pest_param_str}.wav"
    audio_from_comfy_3d_to_disk(output, output_filepath)

    assert output_filepath.exists(), f"Output file {output_filepath} was not created."
    assert output is not None, "Processed audio output is None."

    print(f"test_convolution_reverb_general {test_data['output_root']}")
