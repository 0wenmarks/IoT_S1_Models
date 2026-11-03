#I acknowledge the use of Claude https://claude.ai for code debugging. All output was critically evaluated and modified.





"""Turn the raw HAPT recordings into a table of features, ready for model training."""

# Overview of the pipeline. Two data sources end up in the same table format:
#
#   HAPT dataset:   build_table()
#   Own recordings: person_table() -> activity_table() / transitions_table()
#                   (load_sensor_logger() first converts them to match HAPT)
#
# Both then follow the same steps:
#   1. clean()           remove noise from the 6 sensor channels
#   2. sliding windows   cut the signal into short, fixed-length chunks
#   3. window_features() summarise each chunk as 27 numbers
#   4. to_table()        collect the rows into a DataFrame, one row per window

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, medfilt

# HAPT was recorded at 50 Hz, so our own recordings are resampled to the same rate.
# Without this, the same number of samples would cover a different length of time.
FS = 50  # sampling rate of the phone sensors, in Hz

# Each window is 128 samples (2.56 s at 50 Hz). Moving along by 64 samples
# means neighbouring windows overlap by 50%, so a movement that falls across
# the edge of one window is still seen whole in the next.
WINDOW = 128
STEP = 64

# HAPT activity IDs we keep, and their names. Any ID not listed here is skipped.
# The skipped IDs are the stairs (2, 3), laying (6) and the lying-down transitions (9-12).
ACTIVITIES = {
    1: "walking",
    4: "sitting",
    5: "standing",
    7: "stand_to_sit",
    8: "sit_to_stand",
}

# The 6 sensor channels. This order must match the columns of the signal arrays
# used below: acceleration is columns 0-2 and gyroscope is columns 3-5.
CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

# Column names in the same order window_features() returns its values:
# mean_acc_x, mean_acc_y, ... max_gyro_z
# That is 4 stats x 6 channels = 24 names, plus the 3 "change" names = 27 in total.
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
    # Nyquist is the highest frequency a signal sampled at FS can represent (25 Hz here)
    nyquist = FS / 2
    # Design a 3rd-order filter. butter() takes the cutoff as a fraction of Nyquist,
    # and returns the filter's coefficients (b, a) rather than any filtered data.
    b, a = butter(3, cutoff / nyquist, btype="low")
    # filtfilt runs the filter forwards then backwards, so the output isn't shifted
    # in time relative to the input. axis=0 filters each column along time separately.
    filtered = filtfilt(b, a, signal, axis=0)
    return filtered

def clean(signal):
    """Remove noise: median filter (removes spikes), then a 20 Hz low-pass filter."""
    # Replace each sample with the median of itself and its 2 neighbours in time.
    # A kernel of (3, 1) means 3 rows (time) by 1 column, so channels are not mixed.
    # medfilt pads the edges with zeros, so the very first and last samples may be
    # pulled towards zero.
    no_spikes = medfilt(signal, kernel_size=(3, 1))
    # Remove what's left of the fast sensor noise. Gravity and body movement are much
    # slower than 20 Hz, so they pass through.
    smooth = lowpass(no_spikes, cutoff=20)
    return smooth

def window_features(window):
    """Turn one window (128 rows x 6 columns) into 24 numbers: 4 stats per channel. (+3 for the change in acc for all 3 axis)"""
    # axis=0 works down each column, giving one number per channel
    means = window.mean(axis=0)   # 6 numbers: the tilt
    stds = window.std(axis=0)     # 6 numbers: how much movement
    mins = window.min(axis=0)     # 6 numbers: lowest values
    maxs = window.max(axis=0)     # 6 numbers: highest values
 
    # How the acceleration changes from the first half of the window to the second half
    # (shows the DIRECTION of a movement, e.g. sitting down vs standing up)
    half = len(window) // 2
    # 0:3 selects columns 0-2, the accelerometer only
    first_half = window[:half, 0:3]
    second_half = window[half:, 0:3]
    change = second_half.mean(axis=0) - first_half.mean(axis=0)   # 3 numbers
 
    # Join into one flat array of 27 numbers, in the same order as FEATURE_NAMES
    features = np.concatenate([means, stds, mins, maxs, change])
    return features

def make_row(user, activity, window):
    """One row of the table: who, what, then the 27 features."""
    features = window_features(window)
    # list() is needed so that + joins two lists, rather than NumPy adding element by element
    row = [user, activity] + list(features)
    return row

def to_table(rows):
    """Turn a list of rows into a table with named columns."""
    # Same order as the rows built by make_row()
    columns = ["user", "activity"] + FEATURE_NAMES
    table = pd.DataFrame(rows, columns=columns)
    return table

def build_table(raw_folder):
    """Turn all the HAPT raw files into one table: one row per window."""
    # Each row of labels.txt describes one labelled segment, with these 5 columns:
    # experiment, user, activity ID, start sample, end sample
    labels = np.loadtxt(f"{raw_folder}/labels.txt", dtype=int)
    rows = []

    # One experiment is one recording session, with one pair of acc/gyro files
    experiments = np.unique(labels[:, 0])
    for exp in experiments:
        # The label rows for this session, and who recorded it
        is_this_session = labels[:, 0] == exp
        session = labels[is_this_session]
        user = session[0, 1]   # the same in every row of a session, so take the first

        # Load and clean the whole session
        acc_file = f"{raw_folder}/acc_exp{exp:02d}_user{user:02d}.txt"
        gyro_file = f"{raw_folder}/gyro_exp{exp:02d}_user{user:02d}.txt"
        # Filtering before cutting up the session means the filters' edge effects only
        # affect the ends of the session, not the ends of every segment.
        acc = clean(np.loadtxt(acc_file))
        gyro = clean(np.loadtxt(gyro_file))
        signals = np.hstack([acc, gyro])   # 6 columns, in the order of CHANNELS

        # Cut out each labelled segment we want, then cut it into windows
        for label_row in session:
            activity = label_row[2]
            start = label_row[3]
            end = label_row[4]

            if activity not in ACTIVITIES:
                continue   # skip activities we don't use

            segment = signals[start - 1 : end]   # labels count from 1, Python from 0
            name = ACTIVITIES[activity]

            # Slide along the segment. The +1 is because range() stops before its end value.
            # Leftover samples too few to fill a window are dropped, and a segment shorter
            # than one window makes last_start negative, so it gives no windows at all.
            last_start = len(segment) - WINDOW
            for window_start in range(0, last_start + 1, STEP):
                window_end = window_start + WINDOW
                window = segment[window_start:window_end]
                rows.append(make_row(user, name, window))

    return to_table(rows)

def load_sensor_logger(folder, trim_seconds=10):
    """Load one Sensor Logger recording as 6 columns at exactly 50 Hz, matching HAPT:
    acc x, y, z in g (including gravity), then gyro x, y, z in rad/s."""
    # Accelerometer.csv holds movement only, with gravity kept separately in Gravity.csv.
    # HAPT's accelerometer includes gravity, so the two are added back together below.
    acc = pd.read_csv(f"{folder}/Accelerometer.csv")
    grav = pd.read_csv(f"{folder}/Gravity.csv")
    gyro = pd.read_csv(f"{folder}/Gyroscope.csv")
    meta = pd.read_csv(f"{folder}/Metadata.csv")
 
    # Find where the recording ends: at the first gap of over 1 second, if there is one
    acc_time = acc["seconds_elapsed"].values
    time_between = np.diff(acc_time)
    # Anything after a gap is ignored, because interpolating across it would invent data.
    # gap_positions[0] is the index of the last sample before the first gap.
    gap_positions = np.where(time_between > 1)[0]
    if len(gap_positions) > 0:
        end_time = acc_time[gap_positions[0]]
    else:
        end_time = acc_time[-1]
 
    # A list of times exactly 1/50 of a second apart
    start_time = acc_time[0]
    grid = np.arange(start_time, end_time, 1 / FS)
 
    # Estimate every sensor's value at each grid time (each sensor has its own timestamps)
    # np.interp draws a straight line between each sensor's two nearest readings. Using
    # the same grid for all of them lines the sensors up and gives a steady 50 Hz.
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
    # 9.80665 m/s2 is one standard g
    total_x = (acc_x + grav_x) / 9.80665
    total_y = (acc_y + grav_y) / 9.80665
    total_z = (acc_z + grav_z) / 9.80665
 
    # iPhones recorded without "standardisation" give acceleration the opposite sign to Android
    # Only the acceleration is flipped here, not the gyroscope.
    platform = meta["platform"][0]
    standardised = meta["standardisation"][0]
    if platform == "ios" and standardised == False:
        total_x = -total_x
        total_y = -total_y
        total_z = -total_z

    # Put the 6 columns side by side, in the same order as HAPT
    signals = np.column_stack([total_x, total_y, total_z, gyro_x, gyro_y, gyro_z])
 
    # Cut off the start and end, where the phone is being handled
    # trim is in samples. The recording must be longer than 2 x trim_seconds, or nothing is left.
    trim = int(trim_seconds * FS)
    signals = signals[trim : len(signals) - trim]
    return signals
 
 
def activity_table(folder, user, activity):
    """A recording of one activity (walking, sitting or standing) -> one row per window."""
    # The whole recording is one activity, so every window gets the same label
    raw = load_sensor_logger(folder)
    signals = clean(raw)
    rows = []
 
    # Same sliding window as build_table()
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
    # A shorter trim than the default, which keeps more of the recording
    raw = load_sensor_logger(folder, trim_seconds=5)
    signals = clean(raw)
 
    # Work out sitting or standing at every moment, from which way gravity points
    acceleration = signals[:, 0:3]
    # Below 0.3 Hz, almost all that's left is gravity, since body movement is faster
    gravity = lowpass(acceleration, cutoff=0.3)
    gravity_y = gravity[:, 1]
    # A value of 0.5 g on y means the thigh is 60 degrees from vertical (cos 60 = 0.5).
    # abs() makes this work whichever way up the phone is in the pocket.
    is_sitting = np.abs(gravity_y) < 0.5
 
    rows = []
    half = WINDOW // 2
    # Look for every sample where the posture differs from the one before, which is why
    # this starts at 1. Each change gives a window, so if gravity_y wobbles around the
    # 0.5 threshold, one real transition can produce several windows.
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
 
        # The posture just changed, so sitting now means the person was standing before
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
 
    # ignore_index=True renumbers the rows from 0, instead of repeating each table's numbering
    table = pd.concat([walking, sitting, standing, transitions], ignore_index=True)
    return table