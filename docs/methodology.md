# Revised baseline methodology

The baseline is closed-set identification: the same set of known participants
appears in training, validation, and test data, while whole task IDs remain
disjoint across those partitions. The default configuration trains on `task1`
and `task2`, selects a model on `task3`, and evaluates once on `task7` and
`task8`. This is a task-transfer protocol, not a subject-generalization test.

Each retained stored picture produces one feature row. Features summarize gaze
position, velocity, acceleration, fixation/saccade runs, scanpath length, and
count-normalized spatial entropy. Missing coordinates are linearly interpolated
only when at least two finite values exist. AOI-grid features are excluded until
the coordinate system and screen bounds are verified.

The candidate set is deliberately small: nearest centroid and two regularized
multinomial logistic-regression configurations. Variance filtering and scaling
are fit on training rows only. Model choice is based on validation top-1
accuracy; test tasks are not used for selection. The final selected model is
then refit on train plus validation rows before one test evaluation.

Reported metrics are top-1 accuracy, top-k accuracy, and mean reciprocal rank.
The result artifact also records a participant-bootstrap interval and a
subject-label permutation p-value. Neither replaces replication on an
independent cohort or establishes causality.
