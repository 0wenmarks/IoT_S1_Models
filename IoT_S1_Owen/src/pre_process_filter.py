import numpy as np
from scipy.signal import butter, filtfilt, medfilt

FS = 50

def lowpass(x, cutoff, fs=FS, order=3):
    b, a = butter(order, cutoff / (fs / 2), btype="low")
    return filtfilt(b, a, x, axis=0)

def clean_acc(acc):
    filtered = medfilt(acc, kernel_size=(3, 1))
    filtered = lowpass(filtered, cutoff=20)
    gravity = lowpass(filtered, cutoff=0.3)
    body = filtered - gravity
    return filtered, gravity, body