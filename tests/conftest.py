import pytest
from typing import Dict, Union
from pathlib import Path
import torch
from ..core.io import from_disk_as_raw_3d

# Test data roots
DATA_ROOT = Path("ComfyUI_SignalProcessing/audio")
INPUT_FILE = DATA_ROOT / "inputs/002-orig.mp4"
INPUT_FILE = DATA_ROOT / "inputs/pf-01.mp3"
INPUT_IR_FILE = DATA_ROOT / "inputs/ir.wav"
INPUTS_ROOT = DATA_ROOT / "inputs"
INPUT_FILES = {
    file.name: file.resolve() for file in INPUTS_ROOT.rglob("*") if file.is_file()
}

TestData = Dict[str, Union[Dict[str, Union[torch.Tensor, int]], Path]]


@pytest.fixture
def test_data(request) -> TestData:

    test_name = request.node.name  # Automatically get the current test function name
    test_name = str(request.node.function.__name__)

    OUTPUT_ROOT = DATA_ROOT / f"outputs/{test_name}"
    INPUTS_ROOT = DATA_ROOT / "inputs"

    param_values = (
        request.node.callspec.params if hasattr(request.node, "callspec") else {}
    )
    pest_param_str = "_".join(f"{key}-{value}" for key, value in param_values.items())

    audio_slice_begin_seconds: float = 60.0
    audio_slice_duration_seconds: float = 120.0

    # Prepare audio data
    audio, sample_rate = from_disk_as_raw_3d(
        str(INPUT_FILE.absolute()),
        try_gpu=True,
        start_seconds=audio_slice_begin_seconds,
        duration_seconds=audio_slice_duration_seconds,
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists

    audio_to_comfy: Dict[torch.Tensor, int] = {
        "waveform": audio,
        "sample_rate": sample_rate,
    }

    return {
        "audio": audio_to_comfy,
        "output_root": OUTPUT_ROOT,
        "inputs_root": INPUTS_ROOT,
        "test_name": test_name,
        "pest_param_str": pest_param_str,
    }
