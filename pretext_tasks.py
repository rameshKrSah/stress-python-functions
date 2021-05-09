"""

    Contains all the pretext tasks from the research works:
    1. Self-supervised learning for ECG based emotion recognition --> Code: https://code.engineering.queensu.ca/17ps21/SSL-ECG
    2. Subject-aware contrastive learning for bio-signals --> Code: https://github.com/zacharycbrown/ssl_baselines_for_biosignal_feature_extraction
    3. Self-supervised representation learning from ECG signal --> Code:https://github.com/ogrisel/eegssl
    4. Sense and learn: Self-supervision for omnipresent sensors
    5. Uncovering the structure of clinical EEG signals with self-supervised learning --> 

"""


"""
    There are different categories of pretext task. First we have the transformation pretext
    tasks. 
    
        1. Scaling
        2. Vertical flip
        3. Horizontal flip
        4. Permutation
        5. Adding noise
        6. Rotation
        7. Time warping i.e., compressing and squeezing the signal
        8. Channel shuffling
"""

import numpy as np
import math
# import cv2
import sys
from scipy.stats import kurtosis, skew
from scipy.signal import find_peaks

sys.path.append("../Python Scripts/")

import utils as utl

PRETEXT_TASKS = [
    'NOISE',
    'NOISE_SNR',
    'SCALED',
    'VERTICAL_FLIP',
    'HORIZONTAL_FLIP',
    'PERMUTATION'
]

def create_pretext_dataset_multiclass(x):
    data_len = x.shape[0]

    x_p = np.copy(x)
    y_p = np.zeros(data_len, dtype=int)

    # now add the data for all pre-text tasks
    x_p = np.concatenate([x_p, add_noise(x, 0.5)])
    y_p = np.concatenate([y_p, np.ones(data_len, dtype=int) * 1])
    
    x_p = np.concatenate([x_p, scaled(x, 2)])
    y_p = np.concatenate([y_p, np.ones(data_len, dtype=int) * 2])

    x_p = np.concatenate([x_p, negate(x)])
    y_p = np.concatenate([y_p, np.ones(data_len, dtype=int) * 3])

    x_p = np.concatenate([x_p, hor_flip(x)])
    y_p = np.concatenate([y_p, np.ones(data_len, dtype=int) * 4])

    x_p = np.concatenate([x_p, permute(x, 3)])
    y_p = np.concatenate([y_p, np.ones(data_len, dtype=int) * 5])

    return utl.split_into_train_val_test(x_p, y_p, test_split=0.25)


def create_pretext_dataset(x, pretext_task):
    """
    :param x: numpy array
    :param pretext_task: string name of pretext task
    :param batch_size: (int)
    :param one_hot: Bool, default True
    :param tf_dataset: Bool, If True return TF dataset else numpy arrays. Default False
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

    elif pretext_task == 'ALL':
        x_ = add_noise(x, 0.5)
        x_ = np.concatenate([x_, scaled(x, 2)])
        x_ = np.concatenate([x_, negate(x)])
        x_ = np.concatenate([x_, hor_flip(x)])
        x_ = np.concatenate([x_, permute(x, 3)])
    else:
        raise ValueError("Invalid pretext task %s", pretext_task)

    x_p = np.concatenate([x, x_])
    y_p = np.concatenate([np.zeros(x.shape[0], dtype=int),
                          np.ones(x_.shape[0], dtype=int)])

    print(f"x-vanilla {x.shape[0]}")
    print(f"x-pretext {x_.shape[0]}")
    print(f"total {x_p.shape[0]}")

    return utl.split_into_train_val_test(x_p, y_p, test_split=0.25)

"""
    Paper: https://arxiv.org/abs/1910.07497
    This paper has most of the transformation pretext tasks.

    - Add Gaussian noise
    - Scale the signal by a constant
    - Vertical flip
    - Horizontal flip
    - Pemutate the sub-segments of a signal window
    - Time warping
"""

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
    # determine the length of each sub segment
    piece_length = int(signal.shape[1] // pieces)

    # also number of sub-segments
    pieces = int(np.ceil(signal.shape[1] / piece_length).tolist())

    # generate a random sequence for positioning sub-segments
    sequence = list(range(0, pieces))
    np.random.seed(0)
    np.random.shuffle(sequence)

    # for each sub-segments, get the data indices
    indices = []
    for sp in sequence:
        indices.extend(list(range(sp * 80, sp * 80 + 80)))
    print(sequence)

    # return the permuted signal
    permuted_signal = signal[:, indices, ]
    return permuted_signal


def time_warp(signal, sampling_freq, pieces, stretch_factor, squeeze_factor):
    """
        Randomly stretch or squeeze the signal
        signal: numpy array (batch x window)
        sampling freq
        pieces: number of segments along time
        stretch factor
        squeeze factor
    """

    # length of the signal in terms of time
    total_time = signal.shape[1] // sampling_freq

    # the time length for the sub-segment
    segment_time = total_time / pieces
    sequence = list(range(0, pieces))

    # stretch = np.random.choice(sequence, math.ceil(len(sequence) / 2), replace=False)
    # squeeze = list(set(sequence).difference(set(stretch)))
    # initialize = True
    # for i in sequence:
    #     orig_signal = signal[int(i * np.floor(segment_time * sampling_freq)):int(
    #         (i + 1) * np.floor(segment_time * sampling_freq))]
    #     orig_signal = orig_signal.reshape(np.shape(orig_signal)[0], 1)
    #     if i in stretch:
    #         output_shape = int(np.ceil(np.shape(orig_signal)[0] * stretch_factor))
    #         new_signal = cv2.resize(orig_signal, (1, output_shape), interpolation=cv2.INTER_LINEAR)
    #         if initialize == True:
    #             time_warped = new_signal
    #             initialize = False
    #         else:
    #             time_warped = np.vstack((time_warped, new_signal))
    #     elif i in squeeze:
    #         output_shape = int(np.ceil(np.shape(orig_signal)[0] * squeeze_factor))
    #         new_signal = cv2.resize(orig_signal, (1, output_shape), interpolation=cv2.INTER_LINEAR)
    #         if initialize == True:
    #             time_warped = new_signal
    #             initialize = False
    #         else:
    #             time_warped = np.vstack((time_warped, new_signal))
    # return time_warped

"""
    Paper: https://arxiv.org/pdf/2009.13233.pdf

    1. Blend or mix one signal modality with another or one sample with another from the same modality.
    2. Predict the blend coefficient using a regression method. Binary cross-entropy with a logistic function in the 
        output layer resulted in better generalization. 
    3. Feature prediction from masked window. Approximate 8 summary statistics of a masked temporal segment within a signal.
        Train the multi-head network with Huber loss to predict statistics of the missing sequence. 
    4. Temporal shift prediction: Estimate the number of steps by which the samples are circularly-shifted in their temporal 
        dimension.
    5. Modality denosing: If the network is taksed to reconstruct the original input from corroupted or mixed modality signals 
        then it forces the network to identify core signal characterstics while learning usable representation in the process.
        For this train an encoder-decoder network end-to-end to minimum the MSE loss between ground truth (original signal) 
        and corrupted signals.
    6. Odd segment prediction: Identify the unrelated subsegment that does not belong to the input under consideration, where
        the rest of the sequences are in the correct order. 

"""

def odd_segment(signal):
    n_segments, segment_len, channels = signal.shape

    # # split the segments into length of 60
    # segments_position = [0, 1, 2, 3, 4]
    # segments_indices = [np.arange(0, 60 * i) for i in segments_position]

    np.random.seed(33)
    # get the indices of the sub-segments that will be swapped
    swap_position = np.random.randint(low=0, high=3, size=n_segments)

    # get the indices of the segments in signal from which sub-segments to be swapped is extracted
    swap_segment_index = np.random.randint(low=0, high=n_segments, size=n_segments)

    swapped_signal = np.copy(signal)
    for i, j, sp in zip(swap_segment_index, swap_position, swapped_signal):
        sp[:, j * 60:(j+1) * 60, :] = signal[i, j * 60:(j+1) * 60, :]
    
    x = np.concatenate([signal, swapped_signal])
    y = np.concatenate([np.zeros(n_segments, dtype=int), 
                        np.ones(n_segments, dtype=int)])
    return x, y

def temporal_shift(signal, type_='classification'):
    n_segments, segment_len, channels = signal.shape

    # we know that segment length is 240. 
    shift_ranges = [(0, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 240)]
    
    shifted_x = np.zeros(shape=(1, segment_len, 1))
    y = np.zeros(shape=(1))
    
    for i, range_ in enumerate(shift_ranges):
        # randomly select a shifting factor for the current range
        shift_factor = np.random.randint(low=range_[0], high=range_[1])[0]

        # now circularly-shift the signal segments by the shift factor
        shifted_x = np.concatenate([shifted_x, np.roll(signal, shift=shift_factor, axis = 1)])

        # now the labels
        if type_ == 'classification':
            y = np.concatenate([y, np.ones(n_segments) * i])
        elif type_ == 'regression':
            y = np.concatenate([y, np.ones(n_segments) * shift_factor])

    return shifted_x[1:], y[1:]


def _calc_summary(segment):
    # calculate summary statistics of the segment
    stats = []

    stats.append(np.mean(segment))
    stats.append(np.std(segment))
    stats.append(np.max(segment))
    stats.append(np.min(segment))
    stats.append(np.median(segment))
    stats.append(kurtosis(segment))
    stats.append(skew(segment))
    peaks, _ = find_peaks(segment)
    stats.append(np.len(peaks))

    return stats

def feature_from_masked_window(signal):
    """ 
        Randomly sample the segment length and the starting point. From the selected subsequence, extract 
        8 basic features: mean, standard deviation, maximum, minimum, median, kurtosis, skewness, and 
        number of peaks. Mask the segment with zeros.
    """

    n_segments, segment_len, channel = signal.shape
    np.random.seed(0)

    # get the random length for the subsequence: it cannot be equal to segment length
    sub_length = np.random.randint(low=0, high=segment_len // 2)[0]

    # get the random starting position for the cutout window
    start_index = np.random.randint(low=0, high=segment_len-sub_length, size=n_segments)

    # the end index for the cutout window
    end_index = [s + sub_length for s in start_index]

    # calculate the summary statistics and mask the subsequence
    summary_stats = []
    masked_segments = np.copy(signal)
    i = 0
    for pp in masked_segments:
        summary_stats.append(_calc_summary(pp[start_index[i]:end_index[i]]))
        pp[start_index[i]:end_index[i]] = 0
        i += 1
    
    return masked_segments, summary_stats





"""
    Paper: https://arxiv.org/pdf/2007.16104.pdf

    Relative positioning: Segments are displaced along the time scale and labels created based on whether 
    two time segments are close together or far apart. 

    Temporal shuffling: Create three segments and label is determined based on their relative ordering.

"""

# def relative_positioning(signal):


"""

    Paper: https://arxiv.org/pdf/2007.04871v1.pdf

    1. Temporal cutout: a random contiguous section of the time-series signal (cutout window) is replaced with zeros (22)
    2. Temporal delays: the time-series data is randomly delayed in time
    3. Bandstop filtering: the signal content at a randomly selected frequency band is filtered out using a bandstop filter
    4. Signal mixing: another time instance or subject data is added to the signal to simulate correlated noise

    Temporal cutout was the most effective transformation followed by temporal delay and signal mixing. 

    The effect of temporal transformations was the promotion of temporal consistency where neighboring time points should be
    close in the embedding space and more distant time points should be farther. 
"""


def signal_mixing(signal):
    n_segments, segment_len, channel = signal.shape
    np.random.seed(55)

    # each segment is mixed with another random one. 
    # we don't check whether we are mixing a segment with itself or not. 
    indices = np.random.randint(low=0, high=n_segments, size=n_segments)

    # mixed or blended segments. We can also use a coefficient for mixing the signals. 
    mixed_segments = np.add(signal, signal[indices])

    # we may need to normalize the mixed segments since addition may result 
    # in out of bounds values compared to original signal
    return mixed_segments


def temporal_cut(signal):
    n_segments, segment_len, channel = signal.shape
    np.random.seed(33)

    # get the random length of the cutout window: cannot be more than half the signal length
    cutout_length = np.random.randint(low=0, high=segment_len//2)[0]

    # get the random starting position for the cutout window
    start_index = np.random.randint(low=0, high=segment_len-cutout_length, size=n_segments)

    # the end index for the cutout window
    # end_index = [s + c if s + c <= segment_len else segment_len for s, c in zip(start_index, cutout_length)]
    end_index = cutout_length + start_index

    # now mask the values in the range [start_index:end_index] for all sensor segments
    masked_segments = np.copy(signal)
    i = 0
    for pp in masked_segments:
        pp[start_index[i]:end_index[i]] = 0
        i += 1
    
    print(f"Temporal cut \nStart index {start_index[0]}, end index {end_index[0]}, and cutout length {cutout_length[0]}")
    return masked_segments



if __name__ == '__main__':
    signal = np.random.rand(100, 240, 1)
    print(f"Orignal signal shape {signal.shape}")
    print(f"Original min {np.min(signal)} max {np.max(signal)}")

    processed_signal = temporal_cut(signal)
    print(f"Temporal cut signal shape{processed_signal.shape}")
    print(f"{np.where(processed_signal[0] == 0)[0]}")

    processed_signal = signal_mixing(signal)
    print(f"Signal mixing shape {processed_signal.shape}")
    print(f"Signal mixing min {np.min(processed_signal)} max {np.max(processed_signal)}")




