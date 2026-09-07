# Task-Disjoint Gaze Identity Identification

**Author:** Mobin Kheibary  
**Supervisor:** Dr. Shiva Kamkar

## Question and data

Can eye-gaze behavior identify a known participant when the model is evaluated
on video tasks not used for training? We analysed the local EasyCog video NPZ
files using only eye-tracking arrays. The source paper describes EasyCog as a
multimodal cognitive-assessment dataset; this project does not make a clinical
prediction.

The local files use `task1` through `task9` and stored `taskN_pic` picture
segments. Because the standalone data description uses a different numbering
convention, this report names tasks by their local keys rather than assigning
unverified cognitive-load labels. The cohort/session policy and every source
file hash are saved in `artifacts/task_disjoint_baseline/cohort_manifest.json`.

## Method

The earliest valid local session was selected for each subject. Each stored
picture segment was converted into 40 gaze features covering position, velocity,
acceleration, fixation/saccade summaries, scanpath length, and count-normalized
entropy. AOI-grid features were deliberately excluded because screen bounds and
coordinate semantics have not yet been independently verified.

The protocol was task-disjoint: `task1` and `task2` supplied training rows,
`task3` selected the model, and `task7` and `task8` were held out for the final
test. The code rejects any overlap between train, validation, and test task
sets. It compared nearest centroid and two regularized logistic-regression
models. Feature variance filtering and scaling were fit from training rows.

## Result of the revised baseline

The saved run retained 42 common subjects after segment-level checks. It used
813 training rows, 409 validation rows, and 829 test rows. Validation selected
logistic regression with `C=0.1`. On the held-out tasks, top-1 accuracy was
26.66% (closed-set chance: 2.38%); top-5 accuracy was 59.35%; mean reciprocal
rank was 0.417. A participant-bootstrap 95% interval for top-1 accuracy was
19.37% to 34.53%. The recorded subject-label permutation p-value was 0.000999
using 1,000 permutations.

These are descriptive results for this exact local cohort and configuration.
They do not show a causal cognitive-load effect, a clinical result, or a
performance guarantee for other participants, sessions, devices, or tasks.

## Limitations and next steps

The NPZ task numbering must be reconciled with authoritative task labels before
making task-domain claims. Repeat sessions should support a separate temporal
stability analysis. Coordinate calibration and invalid-sample conventions must
be verified before using screen AOIs or interpreting pixel velocity
physiologically. Finally, the baseline should be repeated with predeclared
alternative disjoint task splits and, ideally, an independent cohort.

## Reference

Hu et al. (2026). *The EasyCog Dataset: Towards Easier Cognitive Assessment
with Passive Video Watching*. Proceedings of the ACM on Interactive, Mobile,
Wearable and Ubiquitous Technologies, 10(1), Article 4.
https://doi.org/10.1145/3789682
