"""Train and run a bidirectional-LSTM activity classifier.

Training CSVs require: timestamp, ax, ay, az, gx, gy, gz, label
Prediction CSVs require: timestamp, ax, ay, az, gx, gy, gz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

SENSORS = ["ax", "ay", "az", "gx", "gy", "gz"]


def read_csv(path, labelled):
    frame = pd.read_csv(path)
    needed = {"timestamp", *SENSORS}
    if labelled:
        needed.add("label")
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing: {', '.join(sorted(missing))}")
    columns = ["timestamp", *SENSORS] + (["label"] if "label" in frame else [])
    frame = frame[columns].copy()
    for column in ["timestamp", *SENSORS]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["timestamp", *SENSORS])
    if "label" in frame:
        frame["label"] = frame["label"].astype(str).str.strip().str.lower()
        frame = frame[frame["label"] != ""]
    return frame.sort_values("timestamp").reset_index(drop=True)


def resample(frame, sample_hz):
    """Interpolate irregular sensor callbacks to a fixed sample rate."""
    time = frame["timestamp"].to_numpy(dtype=float).copy()
    if np.median(np.abs(time)) > 1e10:
        time /= 1e9 if np.median(np.abs(time)) > 1e14 else 1e3
    time -= time[0]
    if len(time) < 2 or time[-1] <= 0:
        return frame.iloc[0:0]
    target = np.arange(0, time[-1], 1 / sample_hz)
    result = {"timestamp": target}
    for column in SENSORS:
        result[column] = np.interp(target, time, frame[column])
    if "label" in frame:
        after = np.searchsorted(time, target).clip(1, len(time) - 1)
        before = after - 1
        closest = np.where(target - time[before] < time[after] - target, before, after)
        result["label"] = frame["label"].to_numpy()[closest]
    return pd.DataFrame(result)


def windows(frame, sample_hz, seconds, overlap, label_ids=None):
    size = round(sample_hz * seconds)
    stride = max(1, round(size * (1 - overlap)))
    values = frame[SENSORS].to_numpy(dtype=np.float32)
    samples, targets = [], []
    for start in range(0, len(frame) - size + 1, stride):
        end = start + size
        if label_ids is not None:
            labels = frame["label"].iloc[start:end]
            label = labels.mode().iat[0]
            if label not in label_ids or (labels == label).mean() < 0.8:
                continue
            targets.append(label_ids[label])
        samples.append(values[start:end])
    x = np.asarray(samples, dtype=np.float32)
    if not len(x):
        x = np.empty((0, size, len(SENSORS)), dtype=np.float32)
    return x, np.asarray(targets, dtype=np.int32)


def csv_files(source):
    source = Path(source)
    return [source] if source.is_file() else sorted(source.rglob("*.csv"))


def load_training_data(source, sample_hz, seconds, overlap):
    files = csv_files(source)
    names = set()
    for path in files:
        names.update(read_csv(path, labelled=True)["label"].unique())
    labels = sorted(names)
    if len(labels) < 2:
        raise ValueError("Training needs at least two activity labels.")
    label_ids = {label: index for index, label in enumerate(labels)}
    all_x, all_y = [], []
    for path in files:
        frame = resample(read_csv(path, labelled=True), sample_hz)
        x, y = windows(frame, sample_hz, seconds, overlap, label_ids)
        if len(x):
            all_x.append(x)
            all_y.append(y)
    if not all_x:
        raise ValueError("No labelled windows were found.")
    return np.concatenate(all_x), np.concatenate(all_y), labels


def load_uci_data(source, split):
    """Load UCI's raw IMU windows for the three activities used by this model."""
    root = Path(source) / split
    signals = root / "Inertial Signals"
    axes = [np.loadtxt(signals / f"{kind}_{axis}_{split}.txt", dtype=np.float32)
            for kind in ("total_acc", "body_gyro") for axis in "xyz"]
    x = np.stack(axes, axis=2)
    x[:, :, :3] *= 9.80665  # UCI acceleration is in g; Sensor Logger uses m/s².
    raw_y = np.loadtxt(root / f"y_{split}.txt", dtype=np.int32)
    names = {1: "walking", 4: "sitting", 5: "standing"}
    keep = np.isin(raw_y, list(names))
    labels = ["walking", "sitting", "standing"]
    y = np.asarray([labels.index(names[value]) for value in raw_y[keep]], dtype=np.int32)
    return x[keep], y, labels


def metrics(y_true, y_pred, labels):
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        matrix[actual, predicted] += 1
    rows, supports = [], matrix.sum(axis=1)
    for index, label in enumerate(labels):
        true_positive = matrix[index, index]
        precision = true_positive / max(1, matrix[:, index].sum())
        recall = true_positive / max(1, supports[index])
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        rows.append({"label": label, "precision": precision, "recall": recall, "f1": f1,
                     "support": int(supports[index])})
    weights = supports / max(1, supports.sum())
    return {
        "accuracy": float(np.trace(matrix) / max(1, matrix.sum())),
        "macro_f1": float(np.mean([row["f1"] for row in rows])),
        "weighted_f1": float(np.dot(weights, [row["f1"] for row in rows])),
        "per_class": rows,
        "confusion_matrix": matrix.tolist(),
    }


def save_plots(history, results, labels, weights):
    """Save an epoch-history graph and row-normalised confusion-matrix heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history.history["accuracy"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history.history["accuracy"], label="Training")
    axes[0].plot(epochs, history.history["val_accuracy"], label="Validation")
    axes[0].set(title="Accuracy by epoch", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.05))
    axes[1].plot(epochs, history.history["loss"], label="Training")
    axes[1].plot(epochs, history.history["val_loss"], label="Validation")
    axes[1].set(title="Loss by epoch", xlabel="Epoch", ylabel="Loss")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(weights.with_suffix(".epochs.png"), dpi=180)
    plt.close(figure)

    counts = np.asarray(results["confusion_matrix"])
    normalised = counts / np.maximum(1, counts.sum(axis=1, keepdims=True))
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    figure.colorbar(image, ax=axis, label="Proportion of actual activity")
    axis.set(title="Confusion matrix", xlabel="Predicted activity", ylabel="Actual activity",
             xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels)
    for row in range(len(labels)):
        for column in range(len(labels)):
            colour = "white" if normalised[row, column] > 0.5 else "black"
            axis.text(column, row, f"{normalised[row, column]:.0%}\n({counts[row, column]})",
                      ha="center", va="center", color=colour)
    figure.tight_layout()
    figure.savefig(weights.with_suffix(".confusion.png"), dpi=180)
    plt.close(figure)


def save_model(model, path, labels, sample_hz, seconds, overlap, mean, std, results, history):
    weights = Path(path)
    if not str(weights).endswith(".weights.h5"):
        weights = weights.with_suffix(".weights.h5")
    weights.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(weights)
    metadata = {"labels": labels, "sample_hz": sample_hz, "window_seconds": seconds,
                "overlap": overlap, "mean": mean.tolist(), "std": std.tolist()}
    weights.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    weights.with_suffix(".metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    save_plots(history, results, labels, weights)
    return weights


def make_model(samples, classes):
    """Bidirectional LSTM that learns from IMU sequences directly."""
    inputs = tf.keras.Input(shape=(samples, len(SENSORS)))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train(args):
    x, y, labels = load_training_data(args.data, args.sample_hz, args.window_seconds, args.overlap)
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(x))
    split = max(1, round(len(x) * args.validation_fraction))
    validation, training = indices[:split], indices[split:]
    if len(training) == 0:
        raise ValueError("Add more labelled windows before training.")
    mean, std = x[training].mean(axis=(0, 1)), x[training].std(axis=(0, 1))
    std = np.maximum(std, 1e-6)
    x = (x - mean) / std
    tf.keras.utils.set_random_seed(args.seed)
    model = make_model(x.shape[1], len(labels))
    history = model.fit(x[training], y[training], validation_data=(x[validation], y[validation]),
                        epochs=args.epochs, batch_size=args.batch_size, verbose=2)
    predicted = model.predict(x[validation], verbose=0).argmax(axis=1)
    results = metrics(y[validation], predicted, labels)
    weights = save_model(model, args.model_out, labels, args.sample_hz, args.window_seconds,
                         args.overlap, mean, std, results, history)
    print(json.dumps(results, indent=2))
    print(f"Saved {weights} and {weights.with_suffix('.metrics.json')}")


def train_uci(args):
    train_x, train_y, labels = load_uci_data(args.data, "train")
    test_x, test_y, _ = load_uci_data(args.data, "test")
    mean, std = train_x.mean(axis=(0, 1)), train_x.std(axis=(0, 1))
    std = np.maximum(std, 1e-6)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std
    tf.keras.utils.set_random_seed(args.seed)
    model = make_model(train_x.shape[1], len(labels))
    history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=args.epochs,
                        batch_size=args.batch_size, verbose=2)
    predicted = model.predict(test_x, verbose=0).argmax(axis=1)
    results = metrics(test_y, predicted, labels)
    weights = save_model(model, args.model_out, labels, 50, 2.56, 0.5, mean, std, results, history)
    print(json.dumps(results, indent=2))
    print(f"Saved {weights} and {weights.with_suffix('.metrics.json')}")


def predict(args):
    weights = Path(args.model)
    if not str(weights).endswith(".weights.h5"):
        weights = weights.with_suffix(".weights.h5")
    metadata = json.loads(weights.with_suffix(".json").read_text(encoding="utf-8"))
    frame = resample(read_csv(args.data, labelled=False), metadata["sample_hz"])
    x, _ = windows(frame, metadata["sample_hz"], metadata["window_seconds"], metadata["overlap"])
    if not len(x):
        raise ValueError("Recording is shorter than one analysis window.")
    x = (x - np.asarray(metadata["mean"], dtype=np.float32)) / np.asarray(metadata["std"], dtype=np.float32)
    model = make_model(x.shape[1], len(metadata["labels"]))
    model.load_weights(weights)
    activities = np.asarray(metadata["labels"])[model.predict(x, verbose=0).argmax(axis=1)]
    report = pd.DataFrame({"window": np.arange(1, len(activities) + 1), "activity": activities})
    if args.output:
        report.to_csv(args.output, index=False)
        print(f"Saved {args.output}")
    else:
        print(report.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--sample-hz", type=int, default=50)
    common.add_argument("--window-seconds", type=float, default=2.56)
    common.add_argument("--overlap", type=float, default=0.5)
    training = commands.add_parser("train", parents=[common])
    training.add_argument("--data", required=True, help="Labelled CSV file or directory")
    training.add_argument("--model-out", default="models/activity.weights.h5")
    training.add_argument("--epochs", type=int, default=30)
    training.add_argument("--batch-size", type=int, default=64)
    training.add_argument("--validation-fraction", type=float, default=0.2)
    training.add_argument("--seed", type=int, default=42)
    training.set_defaults(run=train)
    uci = commands.add_parser("train-uci", help="Train and evaluate on UCI's labelled raw IMU data")
    uci.add_argument("--data", default="data/uci_har_raw/UCI HAR Dataset")
    uci.add_argument("--model-out", default="models/activity.weights.h5")
    uci.add_argument("--epochs", type=int, default=30)
    uci.add_argument("--batch-size", type=int, default=64)
    uci.add_argument("--seed", type=int, default=42)
    uci.set_defaults(run=train_uci)
    prediction = commands.add_parser("predict")
    prediction.add_argument("--data", required=True, help="Unlabelled IMU CSV")
    prediction.add_argument("--model", required=True)
    prediction.add_argument("--output", help="Optional CSV for predicted activities")
    prediction.set_defaults(run=predict)
    default_run = ["train", "--data", "data/training", "--overlap", "0", "--epochs", "8",
                   "--batch-size", "32", "--model-out", "models/phone_activity_4class.weights.h5"]
    args = parser.parse_args(sys.argv[1:] or default_run)
    args.run(args)


if __name__ == "__main__":
    main()
