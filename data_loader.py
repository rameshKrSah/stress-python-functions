
import numpy as np
import sys

sys.path.append("../Python Scripts/")

import utils as utl

def create_dataset(stress, not_stress, path=True, reshape=True, train_test=True, balance_classes=False):
    """Create dataset given stress and not-stress data or path. 
    
    
    stress -- path or data of stress class
    not_stress -- path or data of not-stress class
    path -- whether values of first and seconds arguments are path or not (default True)
    reshape -- whether to reshape the data as (n_samples, window_length, n_channels) (default True)
    train_test -- whether to split the data into train and test sets (default True)
    balance_classes -- whether to balance the data between the classes (default True)

    """
    # if path load the datasets
    if path:
        stress_segments = utl.read_data(stress)
        not_stress_segments = utl.read_data(not_stress)
    else:
        stress_segments = stress
        not_stress_segments = not_stress

    # select equal number of not-stress and stress segments
    if balance_classes == True:
        not_stress_segments = utl.select_random_samples(not_stress_segments, stress_segments.shape[0])

    # concatenate the stress and not-stress data
    x = np.concatenate([stress_segments, not_stress_segments], axis=0)
    y = np.concatenate([
        np.ones(len(stress_segments), dtype=int),
        np.zeros(len(not_stress_segments), dtype=int)
    ], axis=0)

    # reshape is instructed
    if reshape:
        x = x.reshape(-1, x.shape[1], 1)

    # split into train, test, and val if instructed
    if train_test:
        x_tr, x_val, x_ts, y_tr, y_val, y_ts = utl.split_into_train_val_test(x, y, test_split=0.25)
        return x_tr, x_ts, y_tr, y_ts, utl.get_hot_labels(y_tr), utl.get_hot_labels(y_ts)
    else:
        return x, y