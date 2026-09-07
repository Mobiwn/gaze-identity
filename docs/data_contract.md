# Local data contract

The revised code accepts local video files named
`<numeric-id>_patient-<date>-video.npz`. It requires the NPZ keys `subject`,
`date`, `type`, and `task_et`; it rejects a file when the filename subject/date
disagree with the embedded values or when `type` is not `video`.

For each locally observed one-based task key `task1` through `task9`, the code
requires both a two-column task array and a non-empty `taskN_pic` sequence of
two-column picture arrays. The picture arrays are the analysis unit. The full
task array is validated but is not re-segmented by the revised pipeline.

The task-number to cognitive-domain/load mapping is intentionally not encoded
as a scientific fact in the revised code. The standalone local data description
uses a different, zero-based convention. A future report may attach task labels
only after documenting an authoritative mapping from the source paper/data.

## Cohort policy

`configs/baseline.json` selects the earliest locally valid session per subject.
The run manifest records selected files, SHA-256 hashes, picture counts, and
all load/selection exclusions. Picture-level quality filtering occurs after
cohort selection and is recorded separately in `segment_exclusions.json`.
