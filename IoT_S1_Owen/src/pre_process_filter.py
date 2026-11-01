"""Turn the raw HAPT recordings into a table of features, ready for model training."""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, medfilt

FS = 50  # sampling rate of the phone sensors, in Hz
CUTOFF = 20  # low-pass cutoff in Hz: human movement sits below this, most noise above it

# Each window is 128 samples (2.56 s at 50 Hz). Moving along by 64 samples
# means neighbouring windows overlap by 50%.
WINDOW = 128
STEP = 64

# HAPT activity IDs we keep, and their names. Any ID not listed here is skipped.
ACTIVITIES = {
    1: "walking",
    4: "sitting",
    5: "standing",
    7: "stand_to_sit",
    8: "sit_to_stand",
}

CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

# Column names in the same order window_features() returns its values:
# mean_acc_x, mean_acc_y, ... max_gyro_z
FEATURE_NAMES = [
    f"{stat}_{channel}"
    for stat in ["mean", "std", "min", "max"]
    for channel in CHANNELS
]


def clean(signal):
    """Remove noise from a signal (rows = samples, columns = x, y, z)."""
    # Median filter: removes one-sample spikes. (3, 1) means 3 samples along
    # time and 1 across columns, so each axis is filtered on its own.
    signal = medfilt(signal, kernel_size=(3, 1))

    # 3rd-order Butterworth low-pass. butter() takes the cutoff as a fraction
    # of the Nyquist frequency (FS / 2). filtfilt() runs the filter forwards
    # and backwards, so the signal isn't shifted in time.
    b, a = butter(3, CUTOFF / (FS / 2), btype="low")
    return filtfilt(b, a, signal, axis=0)


def window_features(window):
    """Turn one window (128 rows x 6 columns) into 24 numbers: 4 stats per channel."""
    return np.concatenate([
        window.mean(axis=0),
        window.std(axis=0),
        window.min(axis=0),
        window.max(axis=0),
    ])


def build_table(raw_folder):
    """Turn all the HAPT raw files into one table: one row per window."""
    # Each row of labels.txt is one labelled segment:
    # experiment, user, activity, start sample, end sample
    labels = np.loadtxt(f"{raw_folder}/labels.txt", dtype=int)
    rows = []

    for exp in np.unique(labels[:, 0]):
        # All the label rows for this session, and who recorded it
        session = labels[labels[:, 0] == exp]
        user = session[0, 1]

        # Clean the whole session before cutting it up, so the filters don't
        # create artefacts at the edge of each segment. Then put acc and gyro
        # side by side, giving 6 columns.
        acc = clean(np.loadtxt(f"{raw_folder}/acc_exp{exp:02d}_user{user:02d}.txt"))
        gyro = clean(np.loadtxt(f"{raw_folder}/gyro_exp{exp:02d}_user{user:02d}.txt"))
        signals = np.hstack([acc, gyro])

        for _, _, activity, start, end in session:
            if activity not in ACTIVITIES:
                continue  # skip stairs, lying and the lie transitions

            # Labels count from 1 and include the end sample, hence start - 1
            segment = signals[start - 1 : end]

            # Slide the window along the segment. Any leftover samples at the
            # end, or a segment shorter than one window, produce no row.
            for i in range(0, len(segment) - WINDOW + 1, STEP):
                features = window_features(segment[i : i + WINDOW])
                rows.append([user, ACTIVITIES[activity], *features])

    return pd.DataFrame(rows, columns=["user", "activity"] + FEATURE_NAMES)
