# Implementation Overview

**Author:** Mobin Kheibary  
**Supervisor:** Dr. Shiva Kamkar

This document explains the revised implementation in practical terms. It
describes the code that is currently used, not the archived legacy pipeline.

## 1. Data loading and provenance

`src/data_contract.py` validates every local video NPZ before it enters an
experiment. It checks the filename, embedded subject/date/type fields, required
keys, the two-column gaze shape, and the existence of stored picture segments.
It keeps the subject ID, session ID, task ID, picture ID, original path, and
missing-data fraction with each segment.

Some people have more than one local session. The default configuration selects
the earliest valid session per person. This is deterministic, explicit, and
written to `cohort_manifest.json`; it is not an implicit overwrite.

## 2. Feature creation

Each stored picture segment is a sequence of `(x, y)` gaze coordinates. The
feature extractor first records the missing fraction. If a coordinate stream
has at least two valid points, missing values are linearly interpolated. A
segment with fewer than two valid coordinate values cannot be processed and is
recorded as excluded.

The extractor then creates these feature families:

| Family | Examples | Purpose |
|---|---|---|
| Position | mean, median, standard deviation, range, IQR, skewness | where gaze samples occur |
| Velocity and acceleration | mean/percentile/max speed, acceleration magnitude | how quickly gaze changes |
| Fixation and saccade summaries | counts, rates, durations, amplitude, peak speed | stable periods and rapid transitions |
| Scanpath | total path length and length per second | overall movement amount |
| Entropy and dispersion | two-dimensional and one-dimensional count entropy | how spread or concentrated gaze is |

Speed is the Euclidean distance between adjacent samples multiplied by the
configured sampling rate. A sample is marked as a saccade when its speed is
above the configured threshold; consecutive non-saccade samples meeting the
minimum duration form a fixation. These are simple, reproducible operational
definitions, not a claim that the thresholds are universally optimal.

Entropy is calculated from normalized histogram counts. The current revision
does not include grid/AOI features because their interpretation would require a
verified screen-coordinate system and fixed screen boundaries.

## 3. Task-disjoint experiment

The configuration names three lists of task IDs: training, validation, and
test. `src/protocol.py` rejects the run if any task appears in more than one
list. It also retains only people represented in all three partitions.

Pictures from training tasks train the model. Pictures from validation tasks
choose among a small set of candidate models. Test-task pictures are untouched
until the chosen model is refit on training plus validation rows and evaluated
once. This prevents test results from deciding which model configuration to
report.

## 4. Model overview

The baseline intentionally uses small, interpretable models:

- **Nearest centroid:** standardizes each feature and assigns a picture to the
  identity whose average training feature vector is closest.
- **Multinomial logistic regression:** standardizes features, then learns one
  set of weights per known identity. The model outputs a score for every
  identity; the largest score is the predicted identity.

Before either model fits, features with zero variance in the training rows are
removed. This exact training-derived filter is applied unchanged to validation
and test rows. Standard scaling is also learned only from the fitting rows.

## 5. Evaluation outputs

The runner saves the exact prediction for every held-out picture. It reports
top-1 accuracy, top-k accuracy, and mean reciprocal rank. Top-k means the true
identity appears among the k highest-scoring predicted identities. Mean
reciprocal rank rewards a correct identity more when it is ranked nearer the
top.

Uncertainty is summarized with a bootstrap that resamples participants rather
than individual pictures. A label-permutation test randomly reassigns identity
labels at the participant level to form a chance comparison. Both results are
stored with the run configuration, seed, package versions, and source hashes.

## 6. Files to read

1. `configs/baseline.json` - all default experimental choices.
2. `scripts/inspect_data.py` - cohort inspection command.
3. `scripts/run_experiment.py` - full experiment orchestration.
4. `src/revised_features.py` - exact feature calculations.
5. `src/revised_models.py` and `src/revised_metrics.py` - models and metrics.
6. `tests/` - executable examples of contract, feature, split, and metric
   behaviour.
