import pytest
from .conftest import TestData
from ..core.io import audio_from_comfy_3d_to_disk
from ..processors.SignalProcessingNormalizer import SignalProcessingNormalizer

TEST_NAME = "normalizer"


@pytest.mark.parametrize(
    "target_rms, target_lufs_db, target_peak, target_auto, target_auto_alpha",
    [
        (-0.0, -14.0, -0.0, -0.0, -0.0),
        (0.3, -14.0, 0.3, 0.3, 0.3),
        (0.6, -14.0, 0.6, 0.6, 0.6),
        (0.9, -14.0, 0.9, 0.9, 0.9),
    ],
)
def test_normalizer_general(
    test_data: TestData,
    target_rms: float,
    target_lufs_db: float,
    target_peak: float,
    target_auto: float,
    target_auto_alpha: float,
) -> None:

    node = SignalProcessingNormalizer()
    modes = node.INPUT_TYPES()["required"]["mode"][0]

    for mode in modes:

        output = node.process(
            audio_input=test_data["audio"],
            mode=mode,
            target_rms=target_rms,
            target_lufs_db=target_lufs_db,
            target_peak=target_peak,
            target_auto=target_auto,
            target_auto_alpha=target_auto_alpha,
        )[0]

        pest_param_str = test_data["pest_param_str"]
        output_filepath = test_data["output_root"] / f"{mode}-{pest_param_str}.wav"
        audio_from_comfy_3d_to_disk(output, output_filepath)

        assert (
            output_filepath.exists()
        ), f"Output file {output_filepath} was not created."
        assert output is not None, f"Processed audio output is None for mode {mode}."

    print(f"test_normalizer_general {test_data['output_root']}")
