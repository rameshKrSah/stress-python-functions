
import numpy as np
import sys

sys.path.append("../Python Scripts/")

import utils as utl

def load_dataset(stress_path, not_stress_path, reshape = True, train_test = True, balance_classes = False):
    """
        Load the EDA data from the given paths.

    :param stress_path: path to the stress eda data
    :param not_stress_path: path to not stress eda data
    :param reshape: Boolean to reshape. Default True
    :param train_test: Boolean for train-test split. Default True
    :param balance_classes: Boolean to whether balance the stress and not-stress classes samples. 
    :return: x,y pair(s)
    """
    # load the data
    stress_segments = utl.read_data(stress_path)
    not_stress_segments = utl.read_data(not_stress_path)

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
        return x_tr, x_val, x_ts, y_tr, y_val, y_ts
    else:
        return x, y

