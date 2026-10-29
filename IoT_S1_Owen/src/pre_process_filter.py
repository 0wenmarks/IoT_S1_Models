import numpy as np
import pandas as pd
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

ACTIVITIES = {
    1: "walking",
    4: "sitting",
    5: "standing",
    7: "stand_to_sit",
    8: "sit_to_stand",
}

WINDOW = 128
STEP = 64

CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

FEATURE_NAMES = []
for stat in ["mean", "std", "min", "max"]:
    for channel in CHANNELS:
        FEATURE_NAMES.append(stat + "_" + channel)


def clean(signal):
    """Remove noise from a signal: median filter, then 20 Hz low-pass."""
    signal = medfilt(signal, kernel_size=(3, 1))
    return lowpass(signal, cutoff=20)

def window_features(window):
    """Turn one window (128 rows x 6 columns) into 24 numbers."""
    return np.concatenate([
        window.mean(axis=0),
        window.std(axis=0),
        window.min(axis=0),
        window.max(axis=0),
    ])

def build_table(raw_folder):
    """Turn all the HAPT raw files into one table: one row per window."""
    labels = np.loadtxt(f"{raw_folder}/labels.txt", dtype=int)
    rows = []

    for exp in np.unique(labels[:, 0]):
        # All the label rows for this session, and who recorded it
        session = labels[labels[:, 0] == exp]
        user = session[0, 1]

        # Load and clean the whole session, then put acc and gyro side by side
        acc = clean(np.loadtxt(f"{raw_folder}/acc_exp{exp:02d}_user{user:02d}.txt"))
        gyro = clean(np.loadtxt(f"{raw_folder}/gyro_exp{exp:02d}_user{user:02d}.txt"))
        signals = np.hstack([acc, gyro])

        for _, _, activity, start, end in session:
            if activity not in ACTIVITIES:
                continue  # skip stairs, lying and the lie transitions

            segment = signals[start - 1 : end]

            for i in range(0, len(segment) - WINDOW + 1, STEP):
                window = segment[i : i + WINDOW]
                features = window_features(window)
                rows.append([user, ACTIVITIES[activity]] + list(features))

    return pd.DataFrame(rows, columns=["user", "activity"] + FEATURE_NAMES)