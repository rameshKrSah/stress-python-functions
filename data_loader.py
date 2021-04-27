
import numpy as np
import sys

sys.path.append("../Python Scripts/")

import utils as utl

def load_dataset(stress_path, not_stress_path, reshape = True, train_test = True):
    """
        Load the EDA data from the given paths.

    :param stress_path: path to the stress eda data
    :param not_stress_path: path to not stress eda data
    :param reshape: Boolean to reshape. Default True
    :param train_test: Boolean for train-test split. Default True
    :return: x,y pair(s)
    """
    stress_segments = utl.read_data(stress_path)
    not_stress_segments = utl.read_data(not_stress_path)

    x = np.concatenate([stress_segments, not_stress_segments], axis=0)
    y = np.concatenate([
        np.ones(len(stress_segments), dtype=int),
        np.zeros(len(not_stress_segments), dtype=int)
    ], axis=0)

    if reshape:
        x = x.reshape(-1, x.shape[1], 1)

    if train_test:
        x_tr, x_val, x_ts, y_tr, y_val, y_ts = utl.split_into_train_val_test(x, y, test_split=0.25)
        return x_tr, x_val, x_ts, y_tr, y_val, y_ts
    else:
        return x, y

