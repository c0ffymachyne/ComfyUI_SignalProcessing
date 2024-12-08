#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    Open Source Stereo Widening Plugin
    The copyright of the source code is waived by the CCO-1.0 license putting it in public domain with some nuance, read the license please for more details.

    This is directy copy of the part of the code corresponding paper by Orchisama Das
        https://www.dafx.de/paper-archive/2024/papers/DAFx24_paper_92.pdf

    License : Creative Commons Zero v1.0 Universal
    Link To The License : 
        https://github.com/orchidas/StereoWidener/blob/main/LICENSE

    Repository : https://github.com/orchidas/StereoWidener
"""


import os
import pyfar as pf
import numpy as np
import numpy.typing as npt

from pathlib import Path
from enum import Enum
from typing import Optional, Dict, Union, Tuple
from abc import ABC
from scipy.signal import sosfilt
from scipy.signal import fftconvolve


this_directory = os.path.dirname(os.path.realpath(__file__))
data_directory = os.path.join(os.path.split(this_directory)[0], "data", "widener")

VN_PATH = Path(f"{data_directory}/init_vn_filters.txt")
OPT_VN_PATH = Path(f"{data_directory}/opt_vn_filters.txt")

print(VN_PATH.absolute())
print(os.path.dirname(os.path.realpath(__file__)))


class FilterbankType(Enum):
    AMP_PRESERVE = "amplitude-preserve"
    ENERGY_PRESERVE = "energy-preserve"


class DecorrelationType(Enum):
    ALLPASS = "allpass"
    VELVET = "velvet"
    OPT_VELVET = "opt_velvet"


import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import List
import matplotlib.pyplot as plt

_HIGH_EPS = 1e9


def ms_to_samps(ms: npt.NDArray[float], /, fs: float) -> npt.NDArray[int]:
    """Calculate the nearest integer number of samples corresponding to the given time duration in milliseconds.

    Args:
        ms (NDArray): Duration, in milliseconds
        fs (float): Sample rate, in Hertz.

    Returns:
        An NDArray containing the nearest corresponding number of samples
    """
    return np.round(ms * 1e-3 * fs).astype(int)


def half_hann_fade(length: int, fade_out: bool = False) -> npt.NDArray:
    """Generate a half Hann window for fading out or in a signal.

    Args:
        length (int): The length of the fade.
        fade_out (bool, optional): If True, fade out, if False, fade in. Defaults to False.

    Returns:
        npt.NDArray: The half Hann window signal (one-dimensional)
    """
    n = np.linspace(start=0, stop=1, num=length)
    fade: npt.NDArray = 0.5 - 0.5 * np.cos(np.pi * (n + int(fade_out)))
    return fade


def warp_pole_angle(rho: float, pole_freq: Union[float, np.ndarray]) -> npt.NDArray:
    """
    Warp pole angles according to warping factor rho
    Args:
        rho (float): warping factor, 0 < rho < 1 will zoom in on lower frequencies.
        pole_freq (float, np.ndarray): pole frequencies in radians/sec
    Returns:
        np.ndarray: the pole warped angles
    """
    poles_warped = np.exp(1j * pole_freq)
    lambdam = np.log((rho + poles_warped) / (1 + rho * poles_warped))
    return np.imag(lambdam)


def decorrelate_allpass_filters(
    fs: float, nbiquads: int = 250, max_grp_del_ms: float = 30.0
):
    """
    Return cascaded allpass SOS sections with randomised phase to perform signal decorrelation
    Args:
        fs (float): sample rate in Hz
        nbiquds (int): number of AP biquad sections
        max_grp_del_ms (float): maximum group delay in each frequency band
    Returns:
        np.ndarray: 6 x num_biquads AP filter coefficients
    """

    max_grp_del = (1.0 - max_grp_del_ms * 1e-3) / (1 + max_grp_del_ms * 1e-3)
    # each pole radius should give max group delay of 30ms
    ap_rad = np.random.uniform(high=max_grp_del, low=0.5, size=nbiquads)
    # uniformly distributed pole frequencies
    ap_pole_freq = np.random.uniform(low=0, high=2 * np.pi, size=nbiquads)

    # warp pole angles to ERB filterbank
    warp_factor = 0.7464 * np.sqrt(2.0 / np.pi * np.arctan(0.1418 * fs)) + 0.03237
    ap_pole_freq_warped = warp_pole_angle(warp_factor, ap_pole_freq)

    # allpass filter biquad cascade
    poles = ap_rad * np.exp(1j * ap_pole_freq_warped)
    sos_sec = np.zeros((nbiquads, 6))
    # numerator coefficients
    sos_sec[:, 0] = np.abs(poles) ** 2
    sos_sec[:, 1] = -2 * np.real(poles)
    sos_sec[:, 2] = np.ones(nbiquads)
    # denominator coefficients
    sos_sec[:, 3] = np.ones(nbiquads)
    sos_sec[:, 4] = -2 * np.real(poles)
    sos_sec[:, 5] = np.abs(poles) ** 2

    return sos_sec


def process_allpass(
    input_signal: np.ndarray,
    fs: float,
    num_biquads: int = 200,
    max_grp_del_ms: float = 30.0,
) -> np.ndarray:
    """
    For an input stereo signal, pass both channels through
    cascade of allpass filters, and return the output
    """
    _, num_channels = input_signal.shape
    if num_channels > 2:
        input_signal = input_signal.T
        num_channels = 2
    if num_channels != 2:
        raise RuntimeError("Input signal must be stereo!")

    output_signal = np.zeros_like(input_signal)
    sos_section = np.zeros((num_channels, num_biquads, 6))

    for chan in range(num_channels):
        sos_section[chan, ...] = decorrelate_allpass_filters(
            fs, nbiquads=num_biquads, max_grp_del_ms=max_grp_del_ms
        )
        output_signal[:, chan] = sosfilt(
            sos_section[chan, ...], input_signal[:, chan], zi=None
        )

    return output_signal


class LeakyIntegrator:
    """Leaky integrator for signal envelope detection"""

    def __init__(
        self, fs: float, attack_time_ms: float = 5.0, release_time_ms: float = 50.0
    ):
        self.fs = fs
        self.attack_time_ms = attack_time_ms
        self.release_time_ms = release_time_ms

    def process(self, input_signal: NDArray) -> NDArray:
        """Estimate the signal amplitude envelope with a leaky integrator.

        Leaky integrator = a first-order IIR low-pass filter.

        Args:
            input_signal (npt.NDArray): The impulse response (should be 1-dimensional array or only the 1st column is taken)
            fs (float): Sample rate
            attack_time_ms (float): Integrator attack time in milliseconds, by default 5
            release_time_ms (float): Integrator release time in milliseconds, by default 50

        Returns:
            npt.NDArray: The envelope of the impulse response

        """
        # find envelope with a leaky integrator
        if input_signal.ndim == 2 and input_signal.shape[1] > 1:
            input_signal = input_signal[:, 0]

        tau_a = self.attack_time_ms * self.fs * 1e-3
        tau_r = self.release_time_ms * self.fs * 1e-3
        signal_length = len(input_signal)
        signal_env = np.zeros_like(input_signal)
        for n in range(1, signal_length):
            if input_signal[n] > signal_env[n - 1]:
                signal_env[n] = signal_env[n - 1] + (1 - np.exp(-1 / tau_a)) * (
                    np.abs(input_signal[n]) - signal_env[n - 1]
                )
            else:
                signal_env[n] = signal_env[n - 1] + (1 - np.exp(-1 / tau_r)) * (
                    np.abs(input_signal[n]) - signal_env[n - 1]
                )

        return signal_env


class OnsetDetector:
    """
    Onset detector with a leaky integrator"""

    def __init__(
        self,
        fs: float,
        attack_time_ms: float = 5.0,
        release_time_ms: float = 20.0,
        min_onset_hold_ms: float = 80.0,
        min_onset_sep_ms: float = 50.0,
    ):
        """
        Args:
            fs (float): sampling rate in Hz
            attack_time_ms (float): leaky integrator attack time
            release_time_ms (float): leaky integrator release time
            min_onset_hold_ms (float): minimum time to wait
                                       before an onset becomes an offset
            min_onset_sep_ms (float): minimum separation between two
                                      onsets in ms
        """
        self.fs = fs
        self.leaky = LeakyIntegrator(fs, attack_time_ms, release_time_ms)
        self._onset_flag = []
        self.min_onset_hold_samps = int(ms_to_samps(min_onset_hold_ms, self.fs))
        self.min_onset_sep_samps = int(ms_to_samps(min_onset_sep_ms, self.fs))
        self._threshold = None
        self._signal_env = None

    @property
    def signal_env(self) -> NDArray:
        return self._signal_env

    @property
    def threshold(self) -> NDArray:
        return self._threshold

    @property
    def running_sum_thres(self) -> float:
        return self._running_sum_thres

    @property
    def onset_flag(self) -> List[bool]:
        return self._onset_flag

    @staticmethod
    def check_local_peak(cur_samp: float, prev_samp: float, next_samp: float):
        """
        Given the current, previous and next samples, check if the current sample
        is a local peak
        """
        if cur_samp > prev_samp and cur_samp > next_samp:
            return True
        else:
            return False

    @staticmethod
    def check_direction(
        cur_samp: float, prev_samp: float, next_samp: float, is_rising: bool = True
    ) -> bool:
        """
        Check whether the signal envelope is rising or falling.
        The flag `is_rising` is used to check for rising envelopes
        """
        if is_rising:
            return True if cur_samp > prev_samp and cur_samp < next_samp else False
        else:
            return True if cur_samp < prev_samp and cur_samp > next_samp else False

    def process(self, input_signal: NDArray, to_plot: bool = False):
        """Given an input signal, find the location of onsets"""
        if input_signal.ndim == 2 and input_signal.shape[1] > 1:
            input_signal = input_signal[:, 0]
        num_samp = len(input_signal)
        self._signal_env = self.leaky.process(input_signal)
        # onet flag is a list of bools
        self._onset_flag = [False for k in range(num_samp)]
        # threshold for onset calculation, calculated dynamically
        self._threshold = np.ones(num_samp) * _HIGH_EPS
        # running sum to calculate mean of the signal envelope
        self._running_sum_thres = 0.0
        hold_counter = 0
        inhibit_counter = 0

        for k in range(1, num_samp - 1):
            cur_samp = self._signal_env[k]
            prev_samp = self._signal_env[k - 1]
            next_samp = self._signal_env[k + 1]

            is_local_peak = self.check_local_peak(cur_samp, prev_samp, next_samp)

            # running sum of the signal envelope
            self._running_sum_thres += self.signal_env[k]

            # threshold is 1.4 * mean of envelope if there is a local peak
            self._threshold[k] = (
                2 * (self._running_sum_thres / k)
                if is_local_peak
                else self._threshold[k - 1]
            )

            # if an onset is detected, the flag will be true for a minimum number
            # of frames to prevent false offset detection
            if 0 < hold_counter < self.min_onset_hold_samps:
                hold_counter += 1
                self.onset_flag[k] = True
                continue
            # if an offset is detected, the flag will be false for a minimum number
            # of frames to prevent false onset detection
            elif 0 < inhibit_counter < self.min_onset_sep_samps:
                inhibit_counter += 1
                self.onset_flag[k] = False
                continue
            else:
                hold_counter = 0
                inhibit_counter = 0
                # if the signal is rising and the value is greater than the
                # mean of the thresholds so far
                if self.check_direction(
                    cur_samp, prev_samp, next_samp, is_rising=True
                ) and cur_samp > (self._threshold[k]):
                    self._onset_flag[k] = True
                    hold_counter += 1
                # if the signal is fallng and the value is lesser than the
                # mean of the thresholds so far
                elif self.check_direction(
                    cur_samp, prev_samp, next_samp, is_rising=False
                ) and cur_samp < (self._threshold[k]):
                    self._onset_flag[k] = False
                    inhibit_counter += 1

        if to_plot:
            ax = self.plot(input_signal)

    def plot(self, input_signal: NDArray):
        """Plot the input signal and the detected signal, threshold and onsets"""
        num_samp = len(input_signal)
        time_vector = np.arange(0, num_samp / self.fs, 1.0 / self.fs)
        onset_pos = np.zeros_like(time_vector)
        onset_idx = np.where(self._onset_flag)[0]
        onset_pos[onset_idx] = 1.0

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(time_vector, input_signal, label="input signal")
        ax.plot(time_vector, self._signal_env, label="envelope")
        ax.plot(time_vector, self._threshold, label="threshold")
        ax.plot(time_vector, onset_pos, "k--", label="onsets")
        ax.legend(loc="lower left")
        ax.set_ylim([-1.0, 1.0])
        plt.show()

        return ax


def process_velvet(
    input_signal: np.ndarray, fs: float, vn_seq_path: Optional[Path] = None
) -> np.ndarray:
    """Process a stereo input with two channels of VN sequence"""
    _, num_channels = input_signal.shape
    if num_channels > 2:
        input_signal = input_signal.T
        num_channels = 2
    if num_channels != 2:
        raise RuntimeError("Input signal must be stereo!")

    try:
        vn_seq = np.loadtxt(vn_seq_path, dtype="f", delimiter=" ")
    except:
        raise OSError("Error reading file!")

    output_signal = np.zeros_like(input_signal)
    for chan in range(num_channels):
        output_signal[:, chan] = fftconvolve(input_signal[:, chan], vn_seq[chan, :])[
            : output_signal.shape[0]
        ]

    return output_signal


class StereoWidener(ABC):
    """Parent stereo widener class"""

    def __init__(
        self,
        input_stereo: np.ndarray,
        fs: float,
        decorr_type: DecorrelationType,
        beta: float,
        detect_transient: bool = False,
        onset_detection_params: Optional[Dict] = None,
        xfade_win_len_ms: float = 1.0,
    ):
        """Args:
        input_stereo (ndarray) : input stereo signal
        fs (float) : sampling frequency
        decorr_type (Decorrelation type) : decorrelation type (allpass, velvet or opt_velvet)
        beta (between 0 and pi/2): crossfading factor (initial)
        detect_transient (bool): whether to add a transient detection block
        onset_detection_params (dict, optional): dictionary of parameters for onset detection
        xfade_win_len_ms (float): crossfading window used during transients
        """

        self.input_signal = input_stereo
        self.fs = fs
        self.decorr_type = decorr_type
        self.beta = beta
        self.detect_transient = detect_transient

        _, self.num_channels = input_stereo.shape
        if self.num_channels > 2:
            self.input_signal = self.input_signal.T
            self.num_channels = 2
        if self.num_channels != 2:
            raise RuntimeError("Input signal must be stereo!")

        self.decorrelated_signal = self.decorrelate_input()
        if self.detect_transient:
            if onset_detection_params is not None:
                self.onset_detector = OnsetDetector(
                    self.fs,
                    min_onset_hold_ms=onset_detection_params["min_onset_hold_ms"],
                    min_onset_sep_ms=onset_detection_params["min_onset_sep_ms"],
                )
            else:
                self.onset_detector = OnsetDetector(self.fs)
            self.xfade_win_len_samps = int(ms_to_samps(xfade_win_len_ms, self.fs))
            self.xfade_in_win = half_hann_fade(self.xfade_win_len_samps, fade_out=False)
            self.xfade_out_win = half_hann_fade(self.xfade_win_len_samps, fade_out=True)

    def decorrelate_input(self) -> np.ndarray:
        if self.decorr_type == DecorrelationType.ALLPASS:
            decorrelated_signal = process_allpass(
                self.input_signal, self.fs, num_biquads=200
            )
        elif self.decorr_type == DecorrelationType.VELVET:
            decorrelated_signal = process_velvet(self.input_signal, self.fs, VN_PATH)
        elif self.decorr_type == DecorrelationType.OPT_VELVET:
            decorrelated_signal = process_velvet(
                self.input_signal, self.fs, OPT_VN_PATH
            )
        else:
            raise NotImplementedError("Other decorrelators are not available")
        return decorrelated_signal

    def get_onset_flag(self, input: npt.ArrayLike) -> np.ndarray:
        """Returns the onset locations and the signal envelope"""
        self.onset_detector.process(input)
        return self.onset_detector.onset_flag

    def process(self):
        pass


def filter_in_subbands(
    input_signal: np.ndarray,
    fs: int,
    bands_per_octave: int = 3,
    freq_range=(20, 16000),
    filter_length: int = 4096,
) -> Tuple[npt.NDArray, npt.NDArray]:

    signal = pf.classes.audio.Signal(input_signal, fs)
    signal_subband, centre_frequencies = (
        pf.dsp.filter.reconstructing_fractional_octave_bands(
            signal, bands_per_octave, freq_range, n_samples=filter_length
        )
    )

    return signal_subband.time, centre_frequencies


def calculate_interchannel_coherence(
    x: np.ndarray, y: np.ndarray, time_axis: int
) -> npt.NDArray:
    return np.abs(np.sum(x * y, axis=time_axis)) / np.sqrt(
        np.sum(x**2, axis=time_axis) * np.sum(y**2, axis=time_axis)
    )


def calculate_interchannel_cross_correlation_matrix(
    signals: np.ndarray,
    fs: int,
    num_channels: int,
    time_axis: int = -1,
    channel_axis: int = 0,
    return_single_coeff: bool = False,
    bands_per_octave: int = 3,
    freq_range=(20, 16000),
):
    """Returns a matrix of ICC values for each channel axis in signals"""
    if time_axis != -1:
        signals = np.moveaxis(signals, 0, 1)
        channel_axis = 0
        time_axis = -1

    # passthrough filterbank
    if not return_single_coeff:
        signals_subband, centre_frequencies = filter_in_subbands(
            signals, fs, bands_per_octave=bands_per_octave, freq_range=freq_range
        )
        num_f = len(centre_frequencies)
        # make sure the channel axis is in the beginning
        signals_subband = np.moveaxis(signals_subband, 1, 0)

        icc_matrix = np.ones((num_f, num_channels, num_channels))
    else:
        icc_matrix = np.ones((num_channels, num_channels))

    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                continue
            if return_single_coeff:
                icc_matrix[i, j] = calculate_interchannel_coherence(
                    signals[i, :], signals[j, :], time_axis=time_axis
                )
            else:
                icc_matrix[:, i, j] = calculate_interchannel_coherence(
                    signals_subband[i, :, :],
                    signals_subband[j, :, :],
                    time_axis=time_axis,
                )
    if return_single_coeff:
        return icc_matrix
    else:
        return icc_matrix, centre_frequencies


class StereoWidenerFrequencyBased(StereoWidener):

    def __init__(
        self,
        input_stereo: np.ndarray,
        fs: float,
        filterbank_type: FilterbankType,
        decorr_type: DecorrelationType,
        beta: Tuple[float, float],
        cutoff_freq: float,
    ):
        """Frequency based stereo widener
        Args:
            input_stereo (ndarray): input stereo signal
            fs (float): sampling rate
            filterbank_type (Filterbank type): amplitude or energy preserving
            decorr_type (Decorrelation type): allpass, velvet or opt-velvet
            beta (Tuple(float, float)): cross-fading gain for low and high frequencies
            cutoff_freq (float): cutoff frequency of filterbank (Hz)
        """

        super().__init__(input_stereo, fs, decorr_type, beta)
        self.filterbank_type = filterbank_type
        self.cutoff_freq = cutoff_freq
        self.get_filter_coefficients()

    def get_filter_coefficients(self):
        if self.filterbank_type == FilterbankType.AMP_PRESERVE:
            # Linkwitz Riley crossover filterbank
            filters = pf.dsp.filter.crossover(
                signal=None, N=4, frequency=self.cutoff_freq, sampling_rate=self.fs
            )
            self.lowpass_filter_coeffs = pf.classes.filter.FilterSOS(
                filters.coefficients[0, ...], self.fs
            )
            self.highpass_filter_coeffs = pf.classes.filter.FilterSOS(
                filters.coefficients[1, ...], self.fs
            )

        elif self.filterbank_type == FilterbankType.ENERGY_PRESERVE:

            self.lowpass_filter_coeffs = pf.dsp.filter.butterworth(
                signal=None,
                N=16,
                frequency=self.cutoff_freq,
                btype="lowpass",
                sampling_rate=self.fs,
            )

            self.highpass_filter_coeffs = pf.dsp.filter.butterworth(
                signal=None,
                N=16,
                frequency=self.cutoff_freq,
                btype="highpass",
                sampling_rate=self.fs,
            )

        else:
            raise NotImplementedError(
                "Only Butterworth and LR crossover filters are available"
            )

    def filter_in_subbands(self, signal: np.ndarray) -> np.ndarray:
        """Filter signal into two frequency bands"""
        pf_signal = pf.classes.audio.Signal(signal, self.fs)
        lowpass_signal = self.lowpass_filter_coeffs.process(pf_signal).time
        highpass_signal = self.highpass_filter_coeffs.process(pf_signal).time
        return np.vstack((lowpass_signal, highpass_signal))

    def update_beta(self, new_beta: Tuple[float, float]):
        self.beta = new_beta

    def update_cutoff_frequency(self, new_cutoff_freq: float):
        self.cutoff_freq = new_cutoff_freq
        self.get_filter_coefficients()

    def process(self):
        stereo_output = np.zeros_like(self.input_signal)
        filtered_input = np.zeros((2, self.input_signal.shape[0], self.num_channels))
        filtered_decorr = np.zeros_like(filtered_input)

        for chan in range(self.num_channels):
            filtered_input[..., chan] = self.filter_in_subbands(
                self.input_signal[:, chan]
            )
            filtered_decorr[..., chan] = self.filter_in_subbands(
                self.decorrelated_signal[:, chan]
            )

            for k in range(self.num_channels):
                stereo_output[:, chan] += np.cos(self.beta[k]) * np.squeeze(
                    filtered_input[k, :, chan]
                ) + np.sin(self.beta[k]) * np.squeeze(filtered_decorr[k, :, chan])

        return stereo_output

    def calculate_interchannel_coherence(self, output_signal: np.ndarray):
        icc_matrix, icc_freqs = calculate_interchannel_cross_correlation_matrix(
            output_signal,
            fs=self.fs,
            num_channels=self.num_channels,
            time_axis=0,
            channel_axis=-1,
            bands_per_octave=3,
            freq_range=(20, self.fs / 2.0),
        )
        icc_vector = np.squeeze(icc_matrix[..., 0, 1])
        return icc_vector, icc_freqs
