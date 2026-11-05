# this code Model was created with code adapted from claude
import os #imports the operating system tools for finding files and folders
import re #imports regex, used to find sit and stand words in file names
import zipfile #lets the code open zip files without unzipping them first
import numpy as np #math tools for arrays and the window calculations
import pandas as pd #data commands, reads the csv and txt tables
import matplotlib.pyplot as plt #draws the confusion matrix plot
from scipy.signal import medfilt, butter, filtfilt#uses these for filtering data the same way it is done in UCI HAPT
from sklearn.model_selection import LeaveOneGroupOut #splits the data so each group gets tested once
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay #tools that measure how good the model is
from sklearn.tree import DecisionTreeClassifier, export_text #the decision tree model and the tool that prints its rules


# ---------------- Settings ----------------
CLASS_NAMES = ["sitting", "standing"]#Class names for the two classes we are trying to predict
FS = 50          # sampling rate in Hz 
WINDOW_SIZE = 128 # window size for samples matching UCI HAPT (2.56 seconds at 50 Hz)
STEP = 64 # step size for sliding sample windows when creating the mean std min and max based on the data

CHANNELS = ["body_acc_x", "body_acc_y", "body_acc_z",#acclerometer data
            "gyro_x", "gyro_y", "gyro_z",#gyroscope data
            "grav_x", "grav_y", "grav_z"]#uses gravity from accerlerometer to calculate tilt of the phone similar to how it was done in th UCI HAPT
FEATURE_NAMES = [f"{ch}_{stat}" for stat in ["mean", "std", "min", "max"] for ch in CHANNELS]#builds the 36 feature names (4 stats x 9 channels) used for the tree printout


# ---------------- Finding and reading files ----------------
def is_sensor_name(filename):#checks if a file is an accelerometer or gyroscope csv
    name = filename.lower()#lowercase so capital letters don't matter
    return name.endswith(".csv") and ("accelerometer" in name or "gyroscope" in name)#true only for csv files with accelerometer or gyroscope in the name


def label_from_name(text):
    """0 = sitting, 1 = standing, None = the name doesn't clearly say one or the other."""
    name = text.lower()#lowercase the text so capital letters don't matter
    has_sit = re.search(r"(?<![a-z])sit(ting)?(?![a-z])", name) is not None#true if sit or sitting appears as its own word
    has_stand = re.search(r"(?<![a-z])stand(ing)?(?![a-z])", name) is not None#true if stand or standing appears as its own word
    if has_sit and not has_stand:#only sitting words were found
        return 0#sitting label
    if has_stand and not has_sit:#only standing words were found
        return 1#standing label
    return None#both or neither were found so the file gets skipped


def find_recordings(data_dir):
    """Yield each recording as a .zip path, or as a folder that holds loose sensor CSVs."""
    for root, _, files in os.walk(data_dir):#goes through every folder inside data_dir
        for f in files:#checks each file in this folder
            if f.lower().endswith(".zip"):#a zip file is a recording on its own
                yield os.path.join(root, f)#hands back the full path to the zip
        if any(is_sensor_name(f) for f in files):#checks if this folder has loose accelerometer or gyroscope csvs
            yield root#hands back the folder as one recording


def iter_sensor_tables(source):
    """Yield (filename, DataFrame) for each accelerometer/gyroscope CSV in a zip or folder."""
    if os.path.isdir(source):#checks if the recording is a folder
        for f in sorted(os.listdir(source)):#goes through its files in alphabetical order
            if is_sensor_name(f):#only keeps the accelerometer and gyroscope csvs
                yield f.lower(), pd.read_csv(os.path.join(source, f))#reads the csv and hands back its name and table
    elif source.lower().endswith(".zip"):#otherwise checks if it is a zip
        with zipfile.ZipFile(source) as z:#opens the zip without unzipping it to disk
            for member in z.namelist():#goes through every file inside the zip
                base = os.path.basename(member)#gets just the file name, without the folder path inside the zip
                if is_sensor_name(base):#only keeps the accelerometer and gyroscope csvs
                    with z.open(member) as f:#opens that csv inside the zip
                        yield base.lower(), pd.read_csv(f)#reads it and hands back its name and table


# ---------------- Cleaning the signals ----------------
def lowpass(signal, cutoff_hz, order=3):
    """Zero-phase Butterworth low-pass filter applied along the time axis."""
    b, a = butter(order, cutoff_hz / (FS / 2), btype="low")#designs the filter, dividing the cutoff by half the sampling rate converts Hz into the form the filter needs
    return filtfilt(b, a, signal, axis=0)#runs the filter forward and backward along time so the signal doesn't shift


def denoise(values):
    """Median filter removes spikes, then a 20 Hz low-pass removes remaining jitter."""
    spikes_removed = np.column_stack(#starts a new table with the spikes removed, one column at a time
        [medfilt(values[:, i], kernel_size=3) for i in range(values.shape[1])]#runs a median filter on each column, kernel size 3 looks at 3 readings at a time
    )#closes the table that was built column by column
    return lowpass(spikes_removed, 20.0)#removes jitter above 20 Hz


def load_recording(source):#reads one recording and returns its cleaned 9 channel table, or None if a sensor file is missing
    tables = {}#empty dictionary to hold the accelerometer and gyroscope tables
    for name, df in iter_sensor_tables(source):#goes through each sensor csv in this recording
        if "accelerometer" in name:#finds the accelerometer file by its name
            tables["acc"] = df#saves it under the acc key
        elif "gyroscope" in name:#finds the gyroscope file by its name
            tables["gyro"] = df#saves it under the gyro key
    if "acc" not in tables or "gyro" not in tables:#checks both sensor files were found
        return None#skips this recording if one is missing

    acc = tables["acc"].rename(columns=str.lower)#makes all the accelerometer column names lowercase
    gyro = tables["gyro"].rename(columns=str.lower)#makes all the gyroscope column names lowercase
    acc = acc[["time", "x", "y", "z"]].rename(columns={"x": "acc_x", "y": "acc_y", "z": "acc_z"})#keeps time and xyz and renames them with acc_ in front
    gyro = gyro[["time", "x", "y", "z"]].rename(columns={"x": "gyro_x", "y": "gyro_y", "z": "gyro_z"})#keeps time and xyz and renames them with gyro_ in front

    merged = pd.merge_asof(acc.sort_values("time"), gyro.sort_values("time"),#lines up each gyroscope reading with the closest accelerometer time
                           on="time", direction="nearest").dropna().reset_index(drop=True)#matches on time, drops empty rows and resets the row numbers

    # Clean the accelerometer and gyroscope signals (same steps as the HAPT researchers)
    acc_clean = denoise(merged[["acc_x", "acc_y", "acc_z"]].values)#cleans the accelerometer xyz signals
    gyro_clean = denoise(merged[["gyro_x", "gyro_y", "gyro_z"]].values)#cleans the gyroscope xyz signals

    # Gravity is the slow part of the accelerometer (below 0.3 Hz); it shows the phone's tilt
    gravity = lowpass(acc_clean, 0.3)#pulls out the slow gravity part of the accelerometer
    body_acc = acc_clean - gravity#subtracts gravity to leave just the body movement

    values = np.hstack([body_acc, gyro_clean, gravity])#joins body acc, gyro and gravity side by side into 9 columns
    return pd.DataFrame(values, columns=CHANNELS)#turns the array back into a table with the channel names


# ---------------- Cutting windows and building features ----------------
def make_windows(rec, label):
    values = rec[CHANNELS].values#pulls out the 9 channels as an array
    X = []#empty list to hold one feature row per window
    for start in range(0, len(values) - WINDOW_SIZE + 1, STEP):#moves the window start forward by STEP each time, stopping before the recording ends
        win = values[start:start + WINDOW_SIZE]#cuts out one window of 128 readings
        X.append(np.concatenate([win.mean(axis=0), win.std(axis=0),#calculates the mean and std for each channel
                                 win.min(axis=0), win.max(axis=0)]))#calculates the min and max for each channel and adds them as one row
    X = np.array(X).reshape(-1, len(FEATURE_NAMES))#turns the list into a 2D table, reshape keeps the shape right even if no windows were made
    return X, np.full(len(X), label)#returns the feature rows and a label for every window


# ---------------- Evaluation and plotting ----------------
def evaluate_and_plot(X, y, groups, feature_names, title, group_names=None, group_label="recording"):#Plots the results of the decision tree model and prints out the accuracy and classification report
    #
    y_pred = np.empty_like(y)#creates an empty array with the same shape and type as y
    per_group = []#creeates a empty list to store summaries of the results for each group

    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):#loops through each set, leaving one out for testing and using the rest for training
        model = DecisionTreeClassifier(max_depth=4, random_state=42)#4 questions and random state is the seed like a minecraft seed its possible to reporoduce the same tree with the same data and seed
        model.fit(X[train_idx], y[train_idx])#selects feature rows for all data except the left out one
        preds = model.predict(X[test_idx])#uses the trained data to predict the left out data
        y_pred[test_idx] = preds#saves the predictions made

        g = groups[test_idx][0]#takes held out data and gets its group number to see if it was sitting or standing
        name = group_names[g] if group_names is not None else f"{group_label} {g}"#picks from group naems will check the names and will create them if there isnt ay
        true_name = CLASS_NAMES[y[test_idx][0]]#turns number 1 or 0 into sitting or standing
        acc = (preds == y[test_idx]).mean()#measures how often model was correct for this group
        per_group.append((true_name, acc, len(test_idx), name))#adds this group's summary to the list
#------Prints out the results of the model, the confusion matrix and the decision tree rules below --------------
    print(f"\n===== {title} =====")#prints a title so you know which model the results belong to
    print("Accuracy (all windows):", round(accuracy_score(y, y_pred), 3))#prints the overall accuracy across every window
    print(classification_report(y, y_pred, labels=[0, 1],#prints precision, recall and f1 for sitting and standing
                                target_names=CLASS_NAMES, zero_division=0))#gives the labels their names and avoids errors when a class has no predictions

    print(f"Per-{group_label} results:")#heading for the per group list
    for true_name, acc, n, name in per_group:#goes through each group summary one at a time
        print(f"  {true_name:9s} | {acc:6.1%} correct | {n} windows | {name}")#prints the true label, percent correct, window count and name for each group

    cm = confusion_matrix(y, y_pred, labels=[0, 1])#builds the confusion matrix from the true and predicted labels
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES).plot(cmap="Blues")#turns the matrix into a picture with sitting and standing labels
    plt.title(f"Confusion Matrix: {title}")#adds a title to the plot
    plt.show()#opens the plot window

    # Tree trained on all data, shown for the rules printout 
    final_model = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X, y)#trains one tree on every window, this is only used for the rules printout
    print("Decision tree rules (trained on all windows):")#heading for the tree rules
    print(export_text(final_model, feature_names=feature_names))#prints the rules the tree learned, written as if/else questions


# ---------------- 1. UCI HAPT dataset ---------------- #Training and testing the decision tree model based on the UCI HAPT dataset
def run_hapt(data_dir, sitting_id=4, standing_id=5):#runs the HAPT data through the model, sitting is id 4 and standing is id 5
    X_all = pd.read_csv(os.path.join(data_dir, "X_train.txt"), sep=r"\s+", header=None)#loads the feature table, the values are separated by spaces and there are no headers
    X_all.columns = [f"feature_{i+1}" for i in range(X_all.shape[1])]#names the 561 columns feature_1 to feature_561 since the file has no headers
    y_all = pd.read_csv(os.path.join(data_dir, "y_train.txt"), header=None, names=["activity_id"])#loads the activity label for every window, 1 to 12
    subject_all = pd.read_csv(os.path.join(data_dir, "subject_id_train.txt"),#loads which person each window came from
                              header=None, names=["subject"])#names the column subject and has no headers in the file

    df = pd.concat([X_all, subject_all, y_all], axis=1)#joins the features, subject and labels side by side into one table
    df = df[df["activity_id"].isin([sitting_id, standing_id])].reset_index(drop=True)#keeps only sitting and standing rows and resets the row numbers

    feature_cols = [c for c in df.columns if c.startswith("feature_")]#makes a list of the feature column names
    X = df[feature_cols].values#the feature values the model learns from
    y = df["activity_id"].map({sitting_id: 0, standing_id: 1}).values#converts the activity ids into 0 for sitting and 1 for standing
    groups = df["subject"].values#the person each window came from, used to split the data

    # Groups are subject IDs here, so pass them through with their real names
    unique_subjects = sorted(np.unique(groups))#sorted list of each unique person id
    subject_index = {s: i for i, s in enumerate(unique_subjects)}#maps each person id to a position number starting at 0
    group_idx = np.array([subject_index[s] for s in groups])#replaces every window's person id with its position number
    group_names = [f"subject {s}" for s in unique_subjects]#readable names for each person, used in the printout

    evaluate_and_plot(X, y, group_idx, feature_cols, "UCI HAPT Dataset",#runs the leave one subject out test and shows the results
                      group_names=group_names, group_label="subject")#passes the subject names and label so the printout says subject


# ---------------- 2. Sensor Logger data ---------------- 
def run_sensor_logger(data_dir):
    all_X, all_y, all_groups = [], [], []#empty lists to collect windows, labels and group numbers from every recording
    summary = []#empty list to hold one summary line per recording used
    group_id = 0#counter that gives each recording its own number

    for source in find_recordings(data_dir):#goes through every recording found
        label_text = os.path.relpath(source, data_dir)   # folder + file names, used for labeling
        label = label_from_name(label_text)#works out sitting (0) or standing (1) from the name
        if label is None:#no clear sitting or standing word in the name
            print(f"SKIPPED (name doesn't say sitting or standing): {label_text}")#tells you which file was skipped and why
            continue#moves on to the next recording

        rec = load_recording(source)#loads and cleans the recording
        if rec is None:#the recording was missing a sensor file
            print(f"SKIPPED (no accelerometer + gyroscope CSV): {label_text}")#tells you which file is missing a sensor
            continue#moves on to the next recording

        Xi, yi = make_windows(rec, label)#cuts the recording into windows and gets their features and labels
        if len(Xi) == 0:#the recording is too short to make even one window
            print(f"SKIPPED (too short for one window): {label_text}")#tells you the recording was too short
            continue#moves on to the next recording

        all_X.append(Xi)#adds this recording's features to the full list
        all_y.append(yi)#adds this recording's labels to the full list
        all_groups.append(np.full(len(yi), group_id))#gives every window in this recording the same group number
        summary.append(("sitting" if label == 0 else "standing", label_text, len(rec), len(Xi)))#saves the label, name, row count and window count for the printout
        group_id += 1#moves the counter up for the next recording

    print("\nRecordings used:")#heading for the list of recordings
    for label_name, label_text, rows, windows in summary:#goes through each recording summary
        print(f"  {label_name:9s} | {label_text} | {rows} rows -> {windows} windows")#prints the label, file name, row count and window count

    X = np.vstack(all_X)#stacks every recording's windows into one big table
    y = np.concatenate(all_y)#joins all the labels into one list
    groups = np.concatenate(all_groups)#joins all the group numbers into one list
    print("\nSensor Logger windows:", X.shape,#prints the size of the feature table
          "| sitting:", int((y == 0).sum()),#counts how many windows are sitting
          "| standing:", int((y == 1).sum()),#counts how many windows are standing
          "| recordings:", group_id)#prints how many recordings were used

    evaluate_and_plot(X, y, groups, FEATURE_NAMES, "Sensor Logger Data",#runs the leave one recording out test and shows the results
                      group_names=[s[1] for s in summary], group_label="recording")#passes the recording names so the printout shows them


# ---------------- Run both ----------------
# %%
if __name__ == "__main__":#only runs the code below when this file is run directly, not when it is imported
    PROJECT_DIR = r"C:\Users\zicob\Documents\GitHub\IoT_S1_Models\IoT_S1_Zico\Basic_Decision_tree_Model22"

    run_hapt(data_dir=os.path.join(PROJECT_DIR, "Train"))#trains and tests the model on the HAPT data
    run_sensor_logger(data_dir=os.path.join(PROJECT_DIR, "Data"))#trains and tests the model on sensor logger recordings