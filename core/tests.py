from pathlib import Path

# Reusable constants
DATA_ROOT = Path("ComfyUI_SignalProcessing/audio")


def get_output_file_path(output_root: Path, test_name: str, mode: str) -> Path:
    """Generate output file path for a given test."""
    return output_root / f"{test_name}-{mode}.wav"
