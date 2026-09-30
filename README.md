# Can motor-imagery decoding survive missing EEG electrodes?

An applied BCI reliability study using real 64-channel EEG, rather than a workshop demo. We compare a standard left-vs-right imagined fist classifier with a model trained on simulated electrode dropouts. The question is whether the robust model preserves performance when 25% of channels go missing at inference time, and what it costs on clean signals.

## Data and protocol

Source: [PhysioNet EEG Motor Movement/Imagery Dataset v1.0.0](https://physionet.org/content/eegmmidb/1.0.0/), by Schalk and colleagues (DOI: [10.13026/C28G6P](https://doi.org/10.13026/C28G6P), Open Data Commons Attribution License v1.0). It contains 64-channel EEG from 109 subjects. This pilot uses subjects 1-5, imagery runs 4 and 8 for training and run 12 for a held-out evaluation, avoiding random epoch splits across a continuous recording. [MNE's run table](https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html) confirms these are the left/right motor-imagery runs. We classify event T1 (left fist imagery) versus T2 (right fist imagery). Rest periods and executed movements are excluded.

The pipeline filters 8-30 Hz, crops each cue to 0.5-3.5 seconds, calculates per-channel log variance, and fits an L2 logistic regression. Preprocessing parameters and channel medians are estimated from training runs only. For the robust model, training examples are copied with random channel features replaced by training medians. Each mask removes the same 25% of electrodes for a whole recording, mimicking persistent contact loss. At test time we compare clean data and five seeded dropout masks. All reported scores are per-subject and macro-averaged; there is **no** claim of cross-subject generalization.

## Reproduce

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python experiment.py --subjects 1 2 3 4 5
.venv/bin/python -m unittest -v test_experiment.py
```

EDF downloads are cached under `data/` and are not committed. Run `--help` for options. The output JSON includes sample counts, per-subject accuracy, balanced accuracy, macro F1, and confusion matrices. The script is deterministic given the declared random seed. This is an offline analysis, not a medical or assistive device.

## Pilot result

Each of the five subjects contributed 30 training and 15 held-out trials. The values below are mean **subject-level balanced accuracy** (0.50 is chance for two balanced classes), with the exact per-subject and confusion-matrix results in [results/pilot.json](results/pilot.json).

| Classifier | Clean run | 25% electrodes missing |
| --- | ---: | ---: |
| Ordinary training | 0.521 | 0.519 |
| Electrode-dropout training | 0.509 | 0.506 |

Dropout augmentation was **1.25 percentage points worse** in both conditions on this pilot. The hypothesis that this simple augmentation improves missing-electrode robustness is **not supported** here. Neither classifier decodes reliably on the held-out run; these numbers should not be presented as a working BCI. Subject-level differences are heterogeneous, and 15 test trials per subject make each estimate noisy.

## Interpretation and next steps

The held-out run was not used for model selection. This is a deliberately simple baseline and a *negative* pilot result, not a novel clinical method. The sample of five subjects is too small for a population claim. A stronger follow-up would pre-register a subject-level analysis, expand toward all 109 subjects, compare spatial-filter and channel-selection methods using training-run validation only, and report subject-level confidence intervals on a new held-out cohort. Missing electrodes are simulated by feature replacement, which is only a proxy for real contact loss and artifacts.
