"""Held-out-run motor-imagery robustness experiment on PhysioNet EEGMMIDB."""

import argparse
import json
from pathlib import Path

import mne
import numpy as np
from mne.datasets import eegbci
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 2027


def load_features(subject, run, path):
    files = eegbci.load_data(subject, [run], path=str(path), update_path=False, verbose=False)
    raw = mne.io.read_raw_edf(files[0], preload=True, verbose=False)
    eegbci.standardize(raw)
    raw.filter(8, 30, picks="eeg", verbose=False)
    events, event_id = mne.events_from_annotations(raw, event_id={"T1": 1, "T2": 2}, verbose=False)
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0.5, tmax=3.5,
                        baseline=None, preload=True, picks="eeg", verbose=False)
    signal = epochs.get_data(copy=True)
    x = np.log(np.var(signal, axis=2) + 1e-15)
    y = (epochs.events[:, 2] == 2).astype(int)
    return x, y


def fit(x, y):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000, random_state=SEED))
    model.fit(x, y)
    return model


def mask(x, median, rng, fraction=0.25):
    result = x.copy()
    n_channels = max(1, round(x.shape[1] * fraction))
    missing = rng.choice(x.shape[1], n_channels, replace=False)
    result[:, missing] = median[missing]
    return result


def metrics(y, predicted):
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
            "confusion": confusion_matrix(y, predicted, labels=[0, 1]).tolist()}


def subject_result(subject, path):
    train_parts = [load_features(subject, run, path) for run in (4, 8)]
    x_train = np.concatenate([part[0] for part in train_parts])
    y_train = np.concatenate([part[1] for part in train_parts])
    x_test, y_test = load_features(subject, 12, path)
    median = np.median(x_train, axis=0)
    ordinary = fit(x_train, y_train)
    rng = np.random.default_rng(SEED + subject)
    robust_x = np.concatenate([x_train] + [mask(x_train, median, rng) for _ in range(5)])
    robust_y = np.tile(y_train, 6)
    robust = fit(robust_x, robust_y)
    scores = {"subject": subject, "train_n": len(y_train), "test_n": len(y_test),
              "ordinary_clean": metrics(y_test, ordinary.predict(x_test)),
              "robust_clean": metrics(y_test, robust.predict(x_test))}
    for name, model in (("ordinary", ordinary), ("robust", robust)):
        masked_scores = []
        for repeat in range(5):
            test_rng = np.random.default_rng(SEED + subject * 100 + repeat)
            masked_scores.append(metrics(y_test, model.predict(mask(x_test, median, test_rng))))
        scores[name + "_masked"] = masked_scores
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results/pilot.json"))
    args = parser.parse_args()
    args.data.mkdir(parents=True, exist_ok=True)
    all_scores = []
    for subject in args.subjects:
        print(f"Subject {subject}", flush=True)
        all_scores.append(subject_result(subject, args.data))
    summary = {}
    for key in ("ordinary_clean", "robust_clean", "ordinary_masked", "robust_masked"):
        values = []
        for row in all_scores:
            scores = row[key] if isinstance(row[key], list) else [row[key]]
            values.append(np.mean([score["balanced_accuracy"] for score in scores]))
        summary[key] = {"mean_balanced_accuracy": float(np.mean(values)),
                        "subject_values": [float(v) for v in values]}
    for condition in ("clean", "masked"):
        ordinary = np.asarray(summary[f"ordinary_{condition}"]["subject_values"])
        robust = np.asarray(summary[f"robust_{condition}"]["subject_values"])
        differences = robust - ordinary
        summary[f"robust_minus_ordinary_{condition}"] = {
            "mean_balanced_accuracy_difference": float(np.mean(differences)),
            "subject_differences": differences.tolist(),
        }
    report = {"seed": SEED,
              "dataset": "PhysioNet EEG Motor Movement/Imagery v1.0.0",
              "dataset_doi": "10.13026/C28G6P",
              "train_runs": [4, 8], "test_run": 12,
              "masked_channel_fraction": 0.25, "mask_unit": "whole recording", "mask_repeats": 5,
              "task": "left versus right fist motor imagery", "subjects": all_scores, "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
