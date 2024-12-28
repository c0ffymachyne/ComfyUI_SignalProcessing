#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    various plotting methods for debugging and visualization
"""
import numpy as np
import torch
import torchaudio
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, List
from matplotlib.figure import Figure


def _figure_to_image(figure: Figure, dpi: int = 96) -> Image.Image:
    """Convert a Matplotlib figure to a high-resolution RGB PIL Image."""
    figure.set_dpi(dpi)
    figure.canvas.draw()
    data = np.frombuffer(figure.canvas.tostring_argb(), dtype=np.uint8)
    width, height = figure.canvas.get_width_height()
    image = data.reshape((height, width, 4))  # ARGB format

    # Convert ARGB to RGB
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_image[:, :, 0] = image[:, :, 1]  # Red
    rgb_image[:, :, 1] = image[:, :, 2]  # Green
    rgb_image[:, :, 2] = image[:, :, 3]  # Blue

    return Image.fromarray(rgb_image)


def get_wave(
    waveform: torch.Tensor,
    sample_rate: int,
    title: str = "Waveform",
    xlim: int = 1000,
    ylim: int = 1000,
) -> Image:
    waveform = waveform.cpu().numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)

    if num_channels > 1:
        axes[c].set_ylabel(f"Channel {c+1}")
    if xlim:
        axes[c].set_xlim(xlim)
    if ylim:
        axes[c].set_ylim(ylim)

    figure.suptitle(title)

    waveform_image = _figure_to_image(figure)

    return waveform_image


def get_spectogram(
    waveform: torch.Tensor,
    sample_rate: int,
    n_fft: int = 4096,
    n_mels: int = 512,
    title: str = "Spectrogram",
    xlim: int = 8192,
    dpi: int = 96,  # Set a high DPI for better image resolution
) -> np.ndarray:
    """Generate and plot a high-resolution spectrogram from a waveform."""

    # Parameters for Mel Spectrogram
    win_length = n_fft // 2
    hop_length = n_fft // 4  # Smaller hop for better time resolution

    spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        normalized=True,
        power=2.0,  # Using power spectrogram
        norm="slaney",
        n_mels=n_mels,
        mel_scale="slaney",
    ).to(waveform.device, dtype=waveform.dtype)

    # Compute spectrogram
    mel_spectrogram = spectrogram_transform(waveform).cpu()

    # Convert to decibel scale for better visualization
    spectrogram = torchaudio.transforms.AmplitudeToDB(top_db=80)(
        mel_spectrogram
    ).numpy()

    # Plot the spectrogram
    num_channels, _ = waveform.shape
    figure, axes = plt.subplots(
        num_channels, 1, figsize=(20, 10 * num_channels), squeeze=False, dpi=dpi
    )
    figure.suptitle(title, fontsize=16)

    for i, ax in enumerate(axes[:, 0]):  # Unpack axes
        ax.imshow(
            spectrogram[i],
            origin="lower",
            aspect="auto",
            extent=[0, xlim, 0, sample_rate / 2],
            cmap="magma",
        )
        ax.set_title(f"Channel {i + 1}", fontsize=14)
        ax.set_xlabel("Time (frames)", fontsize=12)
        ax.set_ylabel("Frequency (Hz)", fontsize=12)
        ax.tick_params(axis="both", which="major", labelsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Convert the figure to a high-resolution image
    waveform_image = _figure_to_image(figure, dpi=dpi)
    plt.close(figure)  # Close the figure to free up memory
    return waveform_image


def save_image(filepath: str, image: Image) -> Image:
    image.save(filepath, format="PNG", quality=95)


def save_harmonic_spectrum(
    waveform: torch.Tensor,
    sample_rate: int,
    output_image: str,
    figsize: Tuple[int, int] = (12, 6),
    num_harmonics: int = 10,
) -> None:
    """
    Generate the harmonic spectrum of a waveform and save it as an image.

    Parameters:
        waveform (torch.Tensor): Audio waveform tensor.
        sample_rate (int): Sample rate of the audio.
        output_image (str): Path to save the output image.
        figsize (tuple): Size of the output figure in inches.
        num_harmonics (int): Number of harmonics to calculate.

    Returns:
        None
    """
    # Ensure mono audio (combine channels if necessary)
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Perform FFT to get the frequency domain
    fft = torch.fft.fft(waveform).to(device=waveform.device)
    magnitude = torch.abs(fft[0])  # Magnitude of the FFT
    frequencies = torch.fft.fftfreq(waveform.size(1), d=1 / sample_rate).to(
        device=waveform.device
    )

    # Extract the fundamental frequency
    fundamental_idx = torch.argmax(magnitude[: len(magnitude) // 2])
    fundamental_freq = frequencies[fundamental_idx]

    # Calculate harmonic frequencies
    harmonic_frequencies = [fundamental_freq * (i + 1) for i in range(num_harmonics)]
    harmonic_amplitudes = [
        magnitude[int(harmonic / sample_rate * len(magnitude))]
        for harmonic in harmonic_frequencies
    ]
    harmonic_frequencies = torch.tensor(harmonic_frequencies).cpu().numpy()
    harmonic_amplitudes = torch.tensor(harmonic_amplitudes).cpu().numpy()

    # Plot the harmonic spectrum
    plt.figure(figsize=figsize)
    plt.plot(harmonic_frequencies, harmonic_amplitudes, color="blue", linewidth=2)
    plt.title("Harmonic Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Save the image
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    plt.close()


PREDEFINED_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def plot_multiple_harmonic_spectra(
    audio_data: List[Tuple[torch.Tensor, int, str]],
    output_image: str,
    figsize: Tuple[int, int] = (12, 6),
    num_harmonics: int = 16,
    upper_bound: int = 140000,
    title: str = "Harmonic Spectrum",
) -> None:
    """
    Optimized: Plot high-resolution harmonic spectra for multiple audio waveforms.

    Parameters:
        audio_data (list): List of (waveform, sample_rate, label) tuples.
        output_image (str): Path to save the combined output image.
        figsize (tuple): Size of the figure.
        num_harmonics (int): Number of harmonics to calculate.
        upper_bound (int): Maximum frequency to display.
        title (str): Title of the plot.
    """
    plt.figure(figsize=figsize)

    for idx, (waveform, sample_rate, label) in enumerate(audio_data):
        # Ensure mono audio
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0)

        # Move waveform to GPU for efficient computation
        device = (
            waveform.device
            if waveform.is_cuda
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        waveform = waveform.to(device)

        # High-resolution FFT (Zero-padding for better frequency resolution)
        n_fft = 1 * waveform.size(0)  # Zero-padding factor of 4
        fft = torch.fft.fft(waveform, n=n_fft).to(device=waveform.device)
        magnitude = torch.abs(fft[: n_fft // 2])
        frequencies = torch.fft.fftfreq(n_fft, d=1 / sample_rate)[: n_fft // 2]

        # Find fundamental frequency
        fundamental_idx = torch.argmax(
            magnitude[: len(magnitude) // 4]
        )  # Search in the first quarter
        fundamental_freq = frequencies[fundamental_idx]

        # Precompute harmonic frequencies
        harmonic_freqs = fundamental_freq * torch.arange(
            1, num_harmonics + 1, device=device
        )
        harmonic_indices = (harmonic_freqs / (sample_rate / n_fft)).long()
        harmonic_amplitudes = magnitude[harmonic_indices].cpu().numpy()

        # Limit harmonic frequencies to upper bound
        harmonic_freqs = harmonic_freqs[harmonic_freqs <= upper_bound].cpu().numpy()
        harmonic_amplitudes = harmonic_amplitudes[: len(harmonic_freqs)]

        # Convert data to CPU for plotting
        frequencies_cpu = frequencies.cpu().numpy()
        magnitude_cpu = magnitude.cpu().numpy()

        # Plot the full spectrum
        color = PREDEFINED_COLORS[idx % len(PREDEFINED_COLORS)]
        plt.plot(
            frequencies_cpu,
            magnitude_cpu,
            color=color,
            alpha=0.4,
            label=f"Full Spectrum ({label})",
        )

        # Overlay harmonic peaks
        plt.vlines(
            harmonic_freqs,
            ymin=0,
            ymax=harmonic_amplitudes,
            color=color,
            linewidth=1.5,
            linestyle="--",
            label=f"Harmonics ({label})",
        )

    plt.title(title)
    plt.xscale("log")
    plt.xlim(20, upper_bound)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Save and close the figure
    plt.tight_layout()
    plt.savefig(output_image, dpi=150)
    plt.close()
