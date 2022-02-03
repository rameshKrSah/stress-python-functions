from scipy.signal import butter, filtfilt, lfilter
  

# Low Pass Filter removes noise from the EDA data  https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
def eda_lpf(order = 4, fs = 4, cutoff = 1):
    """
    Low pass filter for EDA sampled at 4hz. Cutoff frequency defined at 1hz.
    """
    nyq = 0.5 * fs
    low = cutoff / nyq
    b, a = butter(order, low, btype='lowpass', analog=True)
    return b, a

def butter_lowpass_filter_eda(data):
    """
        Butterworth low pass filter for 1d EDA data.
    """
    b, a = eda_lpf()
    y = lfilter(b, a, data)
    return y

# High Pass Filter is used to separate the SCL and SCR components from the EDA signal
def eda_hpf(order = 1, fs = 4, cutoff = 0.05):
    """
        High pass filter for EDA data sampled at 4hz. Cutoff frequency is set at 0.05hz.
     """
    nyq = 0.5 * fs
    high = cutoff / nyq
    b, a = butter(order, high, btype='highpass')
    return b, a

def butter_highpass_filter_eda(data):
    """ High pass filter for 1d EDA data.
    """
    b, a = eda_hpf()
    y = lfilter(b, a, data)
    return y
