#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    Waveform image rendering node
"""

import torch
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from ..core.utilities import comfy_root_to_syspath
comfy_root_to_syspath() # add comfy to sys path for dev

from ..core.io import audio_to_comfy_3d

class SignalProcessingWaveform:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_input": ("AUDIO",),
            },
            "optional": {
                "color": ("STRING", {"default": "black"}),
                "background_color": ("STRING", {"default": "white"}),
                "width": ("INT", {"default": 800, "min": 100, "max": 4000, "step": 100}),
                "height": ("INT", {"default": 200, "min": 50, "max": 1000, "step": 50}),
                "line_width": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("waveform_image",)
    CATEGORY = "Audio Processing"
    FUNCTION = "process"

    def process(self, audio_input, color="white", background_color="black", width=800, height=200, line_width=1.0):
        waveform = audio_input.get('waveform')  # [channels, samples] or [batch, channels, samples]
        sample_rate = audio_input.get('sample_rate')

        # Convert to mono by averaging channels
        if waveform.ndim == 3:
            # [batch, channels, samples]
            waveform = waveform.mean(dim=1, keepdim=True)  # [batch, 1, samples]
            waveform = waveform.squeeze(0)  # [1, samples]
        elif waveform.ndim == 2:
            # [channels, samples]
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)  # [1, samples]
            else:
                waveform = waveform.unsqueeze(0)  # [1, samples]
        elif waveform.ndim == 1:
            # [samples]
            waveform = waveform.unsqueeze(0)  # [1, samples]
        else:
            raise ValueError(f"Unsupported waveform shape: {waveform.shape}")

        # Convert waveform to numpy
        waveform = waveform.to(dtype = torch.float32)
        waveform_np = waveform.squeeze().detach().cpu().numpy()  # [samples]

        # Create a matplotlib figure without axes
        plt.figure(figsize=(width/100, height/100), dpi=96)
        plt.axis('off')
        plt.margins(0,0)
        plt.gca().set_facecolor(background_color)
        plt.gca().set_position([0, 0, 1, 1])

        # Plot the waveform
        plt.plot(waveform_np, color=color, linewidth=line_width)
        plt.ylim(-1.1, 1.1)  # Set y-axis limits to -1 and 1
        plt.tight_layout(pad=0)

        # Save the plot to a buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()

        # Load the image from the buffer
        buf.seek(0)
        waveform_image = Image.open(buf).convert("RGB")

        # Resize if necessary
        waveform_image = waveform_image.resize((width, height), Image.BILINEAR)

        # Convert to numpy array and normalize to [0,1]
        image_np = np.array(waveform_image).astype(np.float32) / 255.0  # [H, W, 3]

        # Convert to torch tensor and add batch dimension
        image = torch.from_numpy(image_np)[None,]

        return (image, )

class SignalProcessingWaveform2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_input": ("AUDIO",),
            },
            "optional": {
                "color": ("STRING", {"default": "black"}),
                "background_color": ("STRING", {"default": "white"}),
                "width": ("INT", {"default": 800, "min": 100, "max": 4000, "step": 100}),
                "height": ("INT", {"default": 200, "min": 50, "max": 1000, "step": 50}),
                "line_width": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("waveform_image",)
    CATEGORY = "Audio Processing"
    FUNCTION = "process"

    def process(self, audio_input, color="black", background_color="white", width=800, height=200, line_width=1.0):
        waveform = audio_input.get('waveform')  # [channels, samples] or [batch, channels, samples]
        sample_rate = audio_input.get('sample_rate')

        # Convert to mono by averaging channels
        if waveform.ndim == 3:
            # [batch, channels, samples]
            waveform = waveform.mean(dim=1, keepdim=True)  # [batch, 1, samples]
            waveform = waveform.squeeze(0)  # [1, samples]
        elif waveform.ndim == 2:
            # [channels, samples]
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)  # [1, samples]
            else:
                waveform = waveform.unsqueeze(0)  # [1, samples]
        elif waveform.ndim == 1:
            # [samples]
            waveform = waveform.unsqueeze(0)  # [1, samples]
        else:
            raise ValueError(f"Unsupported waveform shape: {waveform.shape}")

        # Ensure waveform is in float32
        waveform = waveform.to(dtype=torch.float32)
        waveform_np = waveform.squeeze().detach().cpu().numpy()  # [samples]

        # Create a matplotlib figure without axes
        plt.figure(figsize=(width / 100, height / 100), dpi=96)
        plt.axis('off')
        plt.margins(0, 0)
        ax = plt.gca()
        ax.set_facecolor(background_color)
        ax.set_position([0, 0, 1, 1])

        # Plot the waveform with fixed y-axis limits
        plt.plot(waveform_np, color=color, linewidth=line_width)
        plt.ylim(-1, 1)  # Set y-axis limits to -1 and 1
        plt.tight_layout(pad=0)

        # Save the plot to a buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()

        # Load the image from the buffer
        buf.seek(0)
        waveform_image = Image.open(buf).convert("RGB")

        # Resize if necessary
        waveform_image = waveform_image.resize((width, height), Image.BILINEAR)

        # Convert to numpy array and normalize to [0,1]
        image_np = np.array(waveform_image).astype(np.float32) / 255.0  # [H, W, 3]

        # Convert to torch tensor and add batch dimension
        image = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

        return (image, )

if __name__ == "__main__":

    from pathlib import Path
    from ..core.io import from_disk_as_raw_2d, audio_to_comfy_3d
    
    node = SignalProcessingWaveform()
    samples_path = Path('ComfyUI_SignalProcessing/audio/inputs/song.mp4')

    source_path = samples_path.absolute()
    source_audio, source_audio_sample_rate = from_disk_as_raw_2d(source_path)

    input = audio_to_comfy_3d(source_audio, source_audio_sample_rate)[0]

    result = node.process(input)[0]

    print('result',result)


