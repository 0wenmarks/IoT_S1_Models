"""Turn the raw HAPT recordings into a table of features, ready for model training."""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, medfilt

FS = 50  # sampling rate of the phone sensors, in Hz

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
FEATURE_NAMES = []
for stat in ["mean", "std", "min", "max"]:
    for channel in CHANNELS:
        name = stat + "_" + channel
        FEATURE_NAMES.append(name)
FEATURE_NAMES.append("change_acc_x")
FEATURE_NAMES.append("change_acc_y")
FEATURE_NAMES.append("change_acc_z")



def lowpass(signal, cutoff):
    """Butterworth low-pass filter: keeps changes slower than `cutoff` Hz."""
    nyquist = FS / 2
    b, a = butter(3, cutoff / nyquist, btype="low")
    filtered = filtfilt(b, a, signal, axis=0)
    return filtered

def clean(signal):
    """Remove noise: median filter (removes spikes), then a 20 Hz low-pass filter."""
    no_spikes = medfilt(signal, kernel_size=(3, 1))
    smooth = lowpass(no_spikes, cutoff=20)
    return smooth

def window_features(window):
    """Turn one window (128 rows x 6 columns) into 24 numbers: 4 stats per channel. (+3 for the change in acc for all 3 axis)"""
    means = window.mean(axis=0)   # 6 numbers: the tilt
    stds = window.std(axis=0)     # 6 numbers: how much movement
    mins = window.min(axis=0)     # 6 numbers: lowest values
    maxs = window.max(axis=0)     # 6 numbers: highest values
 
    # How the acceleration changes from the first half of the window to the second half
    # (shows the DIRECTION of a movement, e.g. sitting down vs standing up)
    half = len(window) // 2
    first_half = window[:half, 0:3]
    second_half = window[half:, 0:3]
    change = second_half.mean(axis=0) - first_half.mean(axis=0)   # 3 numbers
 
    features = np.concatenate([means, stds, mins, maxs, change])
    return features

def make_row(user, activity, window):
    """One row of the table: who, what, then the 27 features."""
    features = window_features(window)
    row = [user, activity] + list(features)
    return row

def to_table(rows):
    """Turn a list of rows into a table with named columns."""
    columns = ["user", "activity"] + FEATURE_NAMES
    table = pd.DataFrame(rows, columns=columns)
    return table

def build_table(raw_folder):
    """Turn all the HAPT raw files into one table: one row per window."""
    labels = np.loadtxt(f"{raw_folder}/labels.txt", dtype=int)
    rows = []

    experiments = np.unique(labels[:, 0])
    for exp in experiments:
        # The label rows for this session, and who recorded it
        is_this_session = labels[:, 0] == exp
        session = labels[is_this_session]
        user = session[0, 1]

        # Load and clean the whole session
        acc_file = f"{raw_folder}/acc_exp{exp:02d}_user{user:02d}.txt"
        gyro_file = f"{raw_folder}/gyro_exp{exp:02d}_user{user:02d}.txt"
        acc = clean(np.loadtxt(acc_file))
        gyro = clean(np.loadtxt(gyro_file))
        signals = np.hstack([acc, gyro])

        # Cut out each labelled segment we want, then cut it into windows
        for label_row in session:
            activity = label_row[2]
            start = label_row[3]
            end = label_row[4]

            if activity not in ACTIVITIES:
                continue   # skip activities we don't use

            segment = signals[start - 1 : end]   # labels count from 1, Python from 0
            name = ACTIVITIES[activity]

            last_start = len(segment) - WINDOW
            for window_start in range(0, last_start + 1, STEP):
                window_end = window_start + WINDOW
                window = segment[window_start:window_end]
                rows.append(make_row(user, name, window))

    return to_table(rows)

def load_sensor_logger(folder, trim_seconds=10):
    """Load one Sensor Logger recording as 6 columns at exactly 50 Hz, matching HAPT:
    acc x, y, z in g (including gravity), then gyro x, y, z in rad/s."""
    acc = pd.read_csv(f"{folder}/Accelerometer.csv")
    grav = pd.read_csv(f"{folder}/Gravity.csv")
    gyro = pd.read_csv(f"{folder}/Gyroscope.csv")
    meta = pd.read_csv(f"{folder}/Metadata.csv")
 
    # Find where the recording ends: at the first gap of over 1 second, if there is one
    acc_time = acc["seconds_elapsed"].values
    time_between = np.diff(acc_time)
    gap_positions = np.where(time_between > 1)[0]
    if len(gap_positions) > 0:
        end_time = acc_time[gap_positions[0]]
    else:
        end_time = acc_time[-1]
 
    # A list of times exactly 1/50 of a second apart
    start_time = acc_time[0]
    grid = np.arange(start_time, end_time, 1 / FS)
 
    # Estimate every sensor's value at each grid time (each sensor has its own timestamps)
    acc_x = np.interp(grid, acc["seconds_elapsed"], acc["x"])
    acc_y = np.interp(grid, acc["seconds_elapsed"], acc["y"])
    acc_z = np.interp(grid, acc["seconds_elapsed"], acc["z"])
    grav_x = np.interp(grid, grav["seconds_elapsed"], grav["x"])
    grav_y = np.interp(grid, grav["seconds_elapsed"], grav["y"])
    grav_z = np.interp(grid, grav["seconds_elapsed"], grav["z"])
    gyro_x = np.interp(grid, gyro["seconds_elapsed"], gyro["x"])
    gyro_y = np.interp(grid, gyro["seconds_elapsed"], gyro["y"])
    gyro_z = np.interp(grid, gyro["seconds_elapsed"], gyro["z"])
 
    # Total acceleration = movement + gravity, converted from m/s2 to g (like HAPT)
    total_x = (acc_x + grav_x) / 9.80665
    total_y = (acc_y + grav_y) / 9.80665
    total_z = (acc_z + grav_z) / 9.80665
 
    # iPhones recorded without "standardisation" give acceleration the opposite sign to Android
    platform = meta["platform"][0]
    standardised = meta["standardisation"][0]
    if platform == "ios" and standardised == False:
        total_x = -total_x
        total_y = -total_y
        total_z = -total_z

    # Put the 6 columns side by side, in the same order as HAPT
    signals = np.column_stack([total_x, total_y, total_z, gyro_x, gyro_y, gyro_z])
 
    # Cut off the start and end, where the phone is being handled
    trim = int(trim_seconds * FS)
    signals = signals[trim : len(signals) - trim]
    return signals
 
 
def activity_table(folder, user, activity):
    """A recording of one activity (walking, sitting or standing) -> one row per window."""
    raw = load_sensor_logger(folder)
    signals = clean(raw)
    rows = []
 
    last_start = len(signals) - WINDOW
    for window_start in range(0, last_start + 1, STEP):
        window_end = window_start + WINDOW
        window = signals[window_start:window_end]
        rows.append(make_row(user, activity, window))
 
    return to_table(rows)
 
 
def transitions_table(folder, user):
    """The sit-down/stand-up recording -> one window centred on each transition.
    In a front pocket the phone's long side (y-axis) runs along the thigh, so
    gravity is mostly along y when standing, and not when sitting."""
    raw = load_sensor_logger(folder, trim_seconds=5)
    signals = clean(raw)
 
    # Work out sitting or standing at every moment, from which way gravity points
    acceleration = signals[:, 0:3]
    gravity = lowpass(acceleration, cutoff=0.3)
    gravity_y = gravity[:, 1]
    is_sitting = np.abs(gravity_y) < 0.5
 
    rows = []
    half = WINDOW // 2
    for i in range(1, len(is_sitting)):
        sitting_now = is_sitting[i]
        sitting_before = is_sitting[i - 1]
        if sitting_now == sitting_before:
            continue   # posture didn't change here
 
        # Take one window centred on the moment the posture changed
        window_start = i - half
        window_end = window_start + WINDOW
        if window_start < 0 or window_end > len(signals):
            continue   # too close to the start or end of the recording
        window = signals[window_start:window_end]
 
        if sitting_now:
            activity = "stand_to_sit"
        else:
            activity = "sit_to_stand"
        rows.append(make_row(user, activity, window))
 
    return to_table(rows)



def person_table(user, folder):
    """All four recordings for one person (in Walking, Sitting, Standing, Transitions folders)."""
    walking = activity_table(f"{folder}/Walking", user, "walking")
    sitting = activity_table(f"{folder}/Sitting", user, "sitting")
    standing = activity_table(f"{folder}/Standing", user, "standing")
    transitions = transitions_table(f"{folder}/Transitions", user)
 
    table = pd.concat([walking, sitting, standing, transitions], ignore_index=True)
    return table