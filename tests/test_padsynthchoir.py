from pathlib import Path
from ..tests.conftest import TestData
from ..generators.SignalProcessingPadSynthChoir import SignalProcessingPadSynthChoir

TEST_NAME = "synth_and_mixdown"
OUTPUT_ROOT = Path(f"ComfyUI_SignalProcessing/audio/outputs/{TEST_NAME}")


def test_synth_and_mixdown(test_data: TestData) -> None:

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    samplerate = 44100
    base_freq = 440.0
    step_size = 4
    num_notes = 5
    bandwidth_cents = 60.0
    number_harmonics = 32

    synth_node = SignalProcessingPadSynthChoir()
    synth_output, sample_rate = synth_node.process(
        samplerate=samplerate,
        base_freq=base_freq,
        step_size=step_size,
        num_notes=num_notes,
        bandwidth_cents=bandwidth_cents,
        number_harmonics=number_harmonics,
    )

    print(f"test_synth_and_mixdown {test_data['output_root']}")
