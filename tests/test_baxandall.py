import pytest
from ..tests.conftest import TestData
from typing import Tuple
from ..core.io import audio_from_comfy_3d_to_disk
from ..processors.SignalProcessingBaxandallEQ import (
    SignalProcessingBaxandallEQ,
    SignalProcessingBaxandall3BandEQ,
)


TEST_NAME = "baxandall"

params: list[Tuple[float, float]] = [
    (9.0, 0.0),
    (6.0, 0.0),
    (3.0, 0.0),
    (1.0, 0.0),
    (0.0, 9.0),
    (0.0, 6.0),
    (0.0, 3.0),
    (0.0, 1.0),
    (0.0, 0.0),
]


@pytest.mark.parametrize(
    "bass_gain_db, treble_gain_db",
    params,
)
def test_baxandalleq_general(
    test_data: TestData, bass_gain_db: float, treble_gain_db: float
) -> None:

    node = SignalProcessingBaxandallEQ()

    output = node.process(
        audio_input=test_data["audio"],
        bass_gain_db=bass_gain_db,
        treble_gain_db=treble_gain_db,
    )[0]

    pest_param_str = test_data["pest_param_str"]
    output_filepath = test_data["output_root"] / f"{pest_param_str}.wav"
    audio_from_comfy_3d_to_disk(output, output_filepath)

    assert output_filepath.exists(), f"Output file {output_filepath} was not created."
    assert output is not None, "Processed audio output is None."

    print(f"test_baxandalleq_general {test_data['output_root']}")


@pytest.mark.parametrize(
    "bass_gain_db,mid_gain_db,treble_gain_db,low_freq,mid_freq,high_freq,mid_q",
    [
        (9.0, 9.0, 9.0, 100.0, 1000.0, 10000.0, 0.707),
        (6.0, 6.0, 6.0, 100.0, 1000.0, 10000.0, 0.707),
        (3.0, 3.0, 3.0, 100.0, 1000.0, 10000.0, 0.707),
        (1.0, 1.0, 1.0, 100.0, 1000.0, 10000.0, 0.707),
        (0.0, 0.0, 0.0, 100.0, 1000.0, 10000.0, 0.707),
    ],
)
def test_baxandalleq3band_general(
    test_data: TestData,
    bass_gain_db: float,
    mid_gain_db: float,
    treble_gain_db: float,
    low_freq: float,
    mid_freq: float,
    high_freq: float,
    mid_q: float,
) -> None:

    node = SignalProcessingBaxandall3BandEQ()

    output = node.process(
        audio_input=test_data["audio"],
        bass_gain_db=bass_gain_db,
        mid_gain_db=mid_gain_db,
        treble_gain_db=treble_gain_db,
        low_freq=low_freq,
        mid_freq=mid_freq,
        high_freq=high_freq,
        mid_q=mid_q,
    )[0]

    pest_param_str = test_data["pest_param_str"]
    output_filepath = test_data["output_root"] / f"{pest_param_str}.wav"
    audio_from_comfy_3d_to_disk(output, output_filepath)

    assert output_filepath.exists(), f"Output file {output_filepath} was not created."
    assert output is not None, "Processed audio output is None."

    print(f"test_baxandalleq3band_general {test_data['output_root']}")
