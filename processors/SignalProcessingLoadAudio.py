#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    Audio loading node
"""

import sys
import os
import torch
from typing import Dict, Tuple, Any, Union

from ..core.io import from_disk_as_dict_3d
import folder_paths

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))


class SignalProcessingLoadAudio:
    supported_formats = ["wav", "mp3", "ogg", "m4a", "flac", "mp4"]
    input_dir = os.path.join(folder_paths.get_input_directory(), "samples")

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Any]:
        supported_extensions = tuple(
            f".{fmt.lower()}" for fmt in SignalProcessingLoadAudio.supported_formats
        )

        files, _ = folder_paths.recursive_search(SignalProcessingLoadAudio.input_dir)
        filtered_files = [x for x in files if x.lower().endswith(supported_extensions)]
        files = [
            os.path.join(SignalProcessingLoadAudio.input_dir, x) for x in filtered_files
        ]

        return {
            "required": {
                "audio_file": (sorted(files), {"image_upload": True}),
                "gain": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    CATEGORY = "Signal Processing"
    FUNCTION = "process"

    def process(
        self, audio_file: str, gain: float
    ) -> Tuple[Dict[str, Union[torch.Tensor, int]]]:
        return from_disk_as_dict_3d(audio_file=audio_file, gain=gain)
