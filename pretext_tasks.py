"""

    Contains all the pretext tasks from the research works:
    1. Self-supervised learning for ECG based emotion recognition --> Code: https://code.engineering.queensu.ca/17ps21/SSL-ECG
    2. Subject-aware contrastive learning for bio-signals --> Code: https://github.com/zacharycbrown/ssl_baselines_for_biosignal_feature_extraction
    3. Self-supervised representation learning from ECG signal --> Code:https://github.com/ogrisel/eegssl
    4. Sense and learn: Self-supervision for omnipresent sensors
    5. Uncovering the structure of clinical EEG signals with self-supervised learning --> 

"""

import numpy as np
import math
# import cv2
import sys

sys.path.append("../../Google Drive/Python Scripts/")

import utils as utl

PRETEXT_TASKS = [
    'NOISE',
    'NOISE_SNR',
    'SCALED',
    'VERTICAL_FLIP',
    'HORIZONTAL_FLIP',
    'PERMUTATION'
]


def create_pretext_dataset(x, pretext_task, batch_size, one_hot=True):
    """
    :param x: numpy array
    :param pretext_task: string name of pretext task
    :param batch_size: (int)
    :param one_hot: Bool, default True
    :return: features and labels (X, Y)
    """
    assert pretext_task in PRETEXT_TASKS

    if pretext_task == 'NOISE':
        x_ = add_noise(x, 0.5)

    elif pretext_task == 'NOISE_SNR':
        x_ = add_noise_with_snr(x, 0.5)

    elif pretext_task == 'SCALED':
        x_ = scaled(x, 2)

    elif pretext_task == 'VERTICAL_FLIP':
        x_ = negate(x)

    elif pretext_task == 'HORIZONTAL_FLIP':
        x_ = hor_flip(x)

    elif pretext_task == 'PERMUTATION':
        x_ = permute(x, 3)

    else:
        raise ValueError("Invalid pretext task %s", pretext_task)

    x_p = np.concatenate([x, x_])
    y_p = np.concatenate([np.zeros(x.shape[0], dtype=int),
                          np.ones(x_.shape[0], dtype=int)])

    if one_hot:
        y_p = utl.get_hot_labels(y_p)

    return utl.create_tf_dataset(x_p, y_p, batch_size=batch_size)


# First the pre-text tasks from 1.
def add_noise(signal, noise_amount):
    """
        Add Gaussian noise to the signal
    """
    noise = np.random.normal(1, noise_amount, np.shape(signal))
    noised_signal = signal + noise
    return noised_signal


def add_noise_with_snr(signal, noise_amount):
    """
        Add Gaussian noise for the specified noise amount (db)
        created using: https://stackoverflow.com/a/53688043/10700812
    """

    target_snr_db = noise_amount  # 20
    x_watts = signal ** 2  # Calculate signal power and convert to dB
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)  # Calculate noise then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts),
                                   len(x_watts))  # Generate an sample of white noise
    noised_signal = signal + noise_volts  # noise added signal

    return noised_signal


def scaled(signal, factor):
    """"
        Scale the signal by the given factor: signal * factor
    """
    scaled_signal = signal * factor
    return scaled_signal


def negate(signal):
    """
        Negate the signal i.e., vertical flip
    """
    negated_signal = signal * (-1)
    return negated_signal


def hor_flip(signal):
    """
        Flip the signal along the time axis i.e., horizontal flip
    """
    hor_flipped = np.flip(signal)
    return hor_flipped


def permute(signal, pieces):
    """
        Randomly permute the temporal locations of the signal.
        signal: numpy array (batch x window)
        pieces: number of segments along time
    """
    pieces = int(np.ceil(np.shape(signal)[0] / (np.shape(signal)[0] // pieces)).tolist())
    piece_length = int(np.shape(signal)[0] // pieces)

    sequence = list(range(0, pieces))
    np.random.shuffle(sequence)

    permuted_signal = np.reshape(signal[:(np.shape(signal)[0] // pieces * pieces)], (pieces, piece_length)).tolist() + [
        signal[(np.shape(signal)[0] // pieces * pieces):]]
    permuted_signal = np.asarray(permuted_signal)[sequence]
    permuted_signal = np.hstack(permuted_signal)

    return permuted_signal


# def time_warp(signal, sampling_freq, pieces, stretch_factor, squeeze_factor):
#     """
#         Randomly stretch or squeeze the signal
#         signal: numpy array (batch x window)
#         sampling freq
#         pieces: number of segments along time
#         stretch factor
#         squeeze factor
#     """
#
#     total_time = np.shape(signal)[0] // sampling_freq
#     segment_time = total_time / pieces
#     sequence = list(range(0, pieces))
#     stretch = np.random.choice(sequence, math.ceil(len(sequence) / 2), replace=False)
#     squeeze = list(set(sequence).difference(set(stretch)))
#     initialize = True
#     for i in sequence:
#         orig_signal = signal[int(i * np.floor(segment_time * sampling_freq)):int(
#             (i + 1) * np.floor(segment_time * sampling_freq))]
#         orig_signal = orig_signal.reshape(np.shape(orig_signal)[0], 1)
#         if i in stretch:
#             output_shape = int(np.ceil(np.shape(orig_signal)[0] * stretch_factor))
#             new_signal = cv2.resize(orig_signal, (1, output_shape), interpolation=cv2.INTER_LINEAR)
#             if initialize == True:
#                 time_warped = new_signal
#                 initialize = False
#             else:
#                 time_warped = np.vstack((time_warped, new_signal))
#         elif i in squeeze:
#             output_shape = int(np.ceil(np.shape(orig_signal)[0] * squeeze_factor))
#             new_signal = cv2.resize(orig_signal, (1, output_shape), interpolation=cv2.INTER_LINEAR)
#             if initialize == True:
#                 time_warped = new_signal
#                 initialize = False
#             else:
#                 time_warped = np.vstack((time_warped, new_signal))
#     return time_warped
