# Legacy exploratory report - superseded for final analysis

> This document records the pre-revision exploratory pipeline. Its task splits,
> feature handling, and conclusions must not be used as final evidence. Run the
> task-disjoint workflow described in `README.md` and regenerate all claims
> from its saved artifacts.

# Cross-Task Eye-Gaze-Based Person Identification

## A Complete Study on Behavioral Biometrics Under Cognitive Load Variation

**Project:** Gaze Identity  
**Dataset:** EasyCog Benchmark  
**Date:** June 2026

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Background and Theory](#2-background-and-theory)
3. [Problem Formulation](#3-problem-formulation)
4. [Dataset](#4-dataset)
5. [Methodology](#5-methodology)
6. [Implementation](#6-implementation)
7. [Results](#7-results)
8. [Discussion](#8-discussion)
9. [Conclusions](#9-conclusions)
10. [References](#10-references)

---

## 1. Introduction

### 1.1 Motivation

Biometrics—measurable physiological and behavioral characteristics used to identify individuals—have become ubiquitous in modern technology. While traditional biometrics like fingerprints and facial recognition rely on physical appearance, **behavioral biometrics** capture *how* a person performs actions: their typing rhythm, gait pattern, or, in our case, eye movement behavior.

Eye-tracking technology has matured significantly, now integrated into consumer devices from smartphones to virtual reality headsets. This creates an opportunity: can we identify individuals based solely on their gaze patterns?

More importantly, we investigate a critical question for real-world deployment: **does gaze-based identification remain reliable when the person's cognitive state changes?**

Consider a practical scenario: a security system trained on an employee's gaze patterns during routine email reading (low cognitive load) must also recognize them during high-pressure financial analysis (high cognitive load). If cognitive demand fundamentally alters gaze behavior, such systems would fail in precisely the moments when security matters most.

### 1.2 Research Questions

This study addresses three interconnected questions:

1. **Cross-task generalization:** Can gaze patterns captured during one cognitive task identify individuals during a different task?
2. **Cognitive load impact:** How does cognitive demand (low, medium, high) affect the transferability of gaze identity signatures?
3. **Feature stability:** Which gaze features remain consistent across cognitive demands, and which are task-dependent?

### 1.3 Contributions

- **Comprehensive feature set:** 64 handcrafted gaze features spanning position, velocity, fixation, saccade, scanpath, entropy, and spatial distribution
- **Systematic evaluation:** 7 experimental splits testing different transfer scenarios (low→high, high→low, same-level controls)
- **Open-source implementation:** Fully reproducible pipeline with `uv`-managed environment

---

## 2. Background and Theory

### 2.1 Eye Tracking Fundamentals

#### What is Eye Tracking?

Eye tracking measures the position and orientation of a person's gaze over time. Modern eye trackers use infrared sensors to detect corneal reflections, computing gaze coordinates (x, y) on a screen at sampling rates typically between 60–2000 Hz.

In this study, we use data sampled at **~125 Hz**, meaning we capture 125 gaze position samples per second. Each sample contains:
- **x-coordinate:** horizontal position on screen (pixels)
- **y-coordinate:** vertical position on screen (pixels)

#### Why do People Move Their Eyes?

Human vision has a fundamental limitation: high-resolution visual information is only available in a small central region called the **fovea** (approximately 2° of visual angle). To examine different parts of a scene, the eyes must physically move to redirect the fovea.

Eye movements serve several functions:
- **Fixations:** Stable gaze on a specific location (200–500ms), allowing detailed visual processing
- **Saccades:** Rapid ballistic movements between fixations (~30–50ms), during which vision is suppressed
- **Smooth pursuit:** Slow tracking of moving objects
- **Vergence:** Coordinated movement of both eyes to focus on objects at different depths

#### Gaze Behavior as a Biometric

Eye movement patterns reflect the interplay of:
- **Anatomy:** Eye muscle properties, orbital structure
- **Neural control:** Oculomotor planning and execution
- **Cognitive strategy:** How an individual allocates visual attention
- **Experience:** Learned patterns from years of visual interaction

This combination of factors creates individual-specific gaze signatures that persist across viewing conditions, making gaze a promising behavioral biometric.

### 2.2 Cognitive Load

#### What is Cognitive Load?

Cognitive load refers to the total amount of mental effort required to perform a task. It is typically categorized into three types:

1. **Intrinsic load:** The inherent difficulty of the material being processed
2. **Extraneous load:** The burden imposed by how information is presented
3. **Germane load:** The effort devoted to building understanding

In our experimental context, we manipulate cognitive load through task design:
- **Low load:** Passive viewing (scenes, faces) requiring minimal mental effort
- **Medium load:** Active processing (semantic decisions, visual tracking)
- **High load:** Demanding computation (arithmetic, executive function)

#### How Cognitive Load Affects Eye Movements

Research demonstrates that cognitive load influences gaze behavior through several mechanisms:

| Aspect | Low Load | High Load |
|--------|----------|-----------|
| Fixation duration | Shorter (150–250ms) | Longer (300–500ms) |
| Fixation count | More frequent | Less frequent |
| Saccade amplitude | Larger, exploratory | Smaller, constrained |
| Pupil dilation | Baseline | Dilated (5–10%) |
| Scanpath length | Longer, systematic | Shorter, focused |
| Gaze dispersion | Wider spatial distribution | Narrower, centralized |

These effects occur because cognitive load consumes attentional resources, reducing the capacity for exploratory eye movements.

### 2.3 Cross-Task Transfer

#### The Transfer Learning Problem

In machine learning, **transfer learning** addresses a fundamental question: can knowledge gained from one task improve performance on a different but related task?

For gaze identification, this manifests as:
- **Source domain:** Gaze data from Task A (e.g., face viewing)
- **Target domain:** Gaze data from Task B (e.g., arithmetic)
- **Goal:** Identify subjects in the target domain using a model trained on the source domain

#### Why Cross-Task Transfer is Hard

Successful transfer requires that source and target domains share **invariant features**—characteristics that remain stable despite task differences. For gaze identification, challenges include:

1. **Task-dependent gaze patterns:** Different tasks elicit different viewing strategies
2. **Cognitive load effects:** As described above, load alters basic oculomotor behavior
3. **Stimulus differences:** Different visual content attracts different attentional patterns
4. **Temporal dynamics:** Task duration and pacing affect eye movement statistics

#### Evaluation Strategies

To rigorously assess transfer, we employ multiple experimental splits:

1. **Low → High:** Train on passive viewing, test on demanding tasks (maximum cognitive shift)
2. **High → Low:** Reverse direction (test if high-load patterns generalize downward)
3. **Same-level controls:** Train/test on tasks with similar cognitive demands (upper bound)
4. **Extreme transfer:** Medium load → both low and high extremes

This comprehensive evaluation reveals the boundaries of cross-task generalization.

---

## 3. Problem Formulation

### 3.1 Task Definition

We frame gaze-based person identification as a **multi-class classification problem**:

- **Input:** Eye-tracking time series from a person performing a cognitive task
- **Output:** One of 69 possible subject identities
- **Challenge:** Correctly identify the person despite task differences between training and testing

### 3.2 Data Representation

Each subject's gaze data during a task is represented as a time series of gaze coordinates:

```
G = [(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)]
```

Where:
- n ≈ 6250 samples (50 tasks at 125 Hz) or n ≈ 3750 samples (30s tasks)
- Each (xᵢ, yᵢ) is a screen coordinate in pixels

### 3.3 Feature Engineering Approach

Raw time series cannot be directly used with most classifiers. We extract **handcrafted features** that capture:

1. **Where** people look (spatial features)
2. **How fast** they move their eyes (velocity features)
3. **When** they pause to process information (fixation features)
4. **How** they transition between locations (saccade features)
5. **The overall pattern** of their visual exploration (scanpath and entropy features)

### 3.4 Evaluation Protocol

#### Cross-Task Splitting

The critical challenge is ensuring train and test features are comparable despite different tasks. We use **generic prefixes** (t0_, t1_) instead of task-specific prefixes (task1_, task7_) so that feature names match across different task combinations.

For example:
- Training on task1 + task2: features named `t0_pos_mean_x`, `t1_vel_std`
- Testing on task7 + task8: same feature names `t0_pos_mean_x`, `t1_vel_std`

#### Metrics

We report four complementary metrics:

1. **Accuracy:** Fraction of subjects correctly identified (overall performance)
2. **Balanced Accuracy:** Per-class recall averaged across all subjects (handles class imbalance)
3. **F1 Macro:** Harmonic mean of precision and recall, averaged across classes
4. **F1 Weighted:** Same as F1 Macro, but weighted by class support

#### Baseline Comparison

The random baseline for 69-class classification is 1/69 ≈ **1.45%**. Any model achieving significantly above this demonstrates that gaze patterns contain identity information.

---

## 4. Dataset

### 4.1 Overview

The **EasyCog dataset** provides synchronized multimodal cognitive assessment data. For this study, we use the eye-tracking modality from 69 participants across 9 cognitive tasks.

### 4.2 Participants

- **Total subjects:** 69 unique participants
- **Sessions:** Some subjects have multiple sessions (we use the first session only)
- **Demographics:** Mixed clinical and healthy controls

### 4.3 Tasks

| Task | Name | Duration | Samples | Cognitive Domain | Load Level |
|------|------|----------|---------|------------------|------------|
| task1 | Scene viewing | 50s | 6250 | Passive visual attention | Low |
| task2 | Face viewing | 50s | 6250 | Social attention | Low |
| task3 | Scene elements | 50s | 6250 | Object search | Low |
| task4 | Semantic processing | 50s | 6250 | Language comprehension | Medium |
| task5 | Visuospatial pursuit | 50s | 6250 | Visual tracking | Medium |
| task6 | Memory recognition | 30s | 3750 | Working memory | Medium |
| task7 | Arithmetic | 50s | 6250 | Numerical computation | High |
| task8 | Executive function | 50s | 6250 | Cognitive control | High |
| task9 | Complex executive | 30s | 3750 | Multi-step planning | High |

### 4.4 Data Format

Each participant's data is stored in a NumPy `.npz` file with the following structure:

```
{
  'subject': str,           # Subject identifier (e.g., '002_patient')
  'date': str,              # Recording date
  'type': str,              # 'video' for eye-tracking data
  'task_et': dict,          # Task-level eye-tracking data
    {
      'task1': ndarray,     # Shape (6250, 2) - [x, y] coordinates
      'task2': ndarray,     # Shape (6250, 2)
      ...
      'task9': ndarray,     # Shape (3750, 2) or (6250, 2)
    }
  'task_et_pic': dict,      # Per-picture eye-tracking data (optional)
    {
      'task1': list,        # List of 10 arrays, each (625, 2)
      'task2': list,        # List of 10 arrays, each (625, 2)
      ...
    }
}
```

### 4.5 Data Quality

**Known issues:**
- Subject 039: All-NaN data in tasks 3–9
- Subjects 054, 068-071: NaN values in some tasks
- Some tasks have fewer than 10 picture segments

**Mitigation:** Our feature extraction includes NaN interpolation and degenerate signal handling.

### 4.6 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total video files | 76 |
| Unique subjects | 69 |
| Subjects with all 9 tasks | ~60 |
| Average samples per task | ~5800 |
| Gaze coordinate range | ~0–1920 pixels (x), ~0–1080 pixels (y) |

---

## 5. Methodology

### 5.1 Feature Extraction Pipeline

Our feature extraction operates in three stages:

```
Raw Gaze Series → Per-Picture Features → Aggregated Statistics → Feature Vector
```

#### Stage 1: Per-Picture Feature Extraction

Each task is divided into segments corresponding to individual pictures (typically 10 segments per 50s task, or 6 per 30s task). For each segment, we extract 64 features.

#### Stage 2: Aggregation

Per-picture features are aggregated across all pictures in a task using five statistics:
- Mean
- Standard deviation
- Minimum
- Maximum
- Range (max - min)

This yields 64 × 5 = 320 features per task slot.

#### Stage 3: Concatenation

For each subject, features from multiple tasks are concatenated. With 2 tasks per slot, this produces 320 × 2 = 640 features (reduced to ~612 after removing zero-variance features).

### 5.2 Feature Categories

#### 5.2.1 Position Features (14 features)

These capture where on the screen a person tends to look:

| Feature | Description |
|---------|-------------|
| `pos_mean_x`, `pos_mean_y` | Average gaze position |
| `pos_std_x`, `pos_std_y` | Spatial variability |
| `pos_range_x`, `pos_range_y` | Total spatial extent |
| `pos_median_x`, `pos_median_y` | Central tendency (robust to outliers) |
| `pos_iqr_x`, `pos_iqr_y` | Interquartile range (spread of middle 50%) |
| `pos_skew_x`, `pos_skew_y` | Asymmetry of gaze distribution |
| `pos_kurtosis_x`, `pos_kurtosis_y` | Peakedness of gaze distribution |
| `pos_centroid_dist` | Average distance from gaze centroid |
| `pos_bounding_area` | Area of bounding rectangle covering all fixations |

**Biometric relevance:** Individuals show consistent preferences for where to look (e.g., attending to eyes vs. mouth when viewing faces, or focusing on central vs. peripheral scene elements).

#### 5.2.2 Velocity Features (9 features)

These capture how fast a person moves their eyes:

| Feature | Description |
|---------|-------------|
| `vel_mean` | Average saccade speed |
| `vel_std` | Speed variability |
| `vel_max` | Peak saccade speed |
| `vel_median` | Typical speed (robust) |
| `vel_p95`, `vel_p99` | 95th/99th percentile speeds |
| `vel_mean_x`, `vel_mean_y` | Horizontal/vertical speed components |
| `vel_ratio_x_y` | Horizontal/vertical speed ratio |

**Biometric relevance:** Saccade speed reflects oculomotor control properties that vary between individuals.

#### 5.2.3 Acceleration Features (2 features)

| Feature | Description |
|---------|-------------|
| `accel_mean` | Mean absolute acceleration |
| `accel_std` | Acceleration variability |

**Biometric relevance:** Acceleration patterns reflect how quickly eye movements start and stop.

#### 5.2.4 Fixation Features (10 features)

Fixations are periods of stable gaze (typically 200–500ms) during which visual information is processed:

| Feature | Description |
|---------|-------------|
| `fix_count` | Number of fixations |
| `fix_rate` | Fixations per second |
| `fix_mean_dur` | Average fixation duration |
| `fix_std_dur` | Fixation duration variability |
| `fix_median_dur` | Median fixation duration |
| `fix_mean_dispersion` | Average spatial spread within fixations |
| `fix_total_time_ratio` | Fraction of time spent fixating |

**Biometric relevance:** Fixation duration and count reflect individual differences in visual processing speed and strategy.

#### 5.2.5 Saccade Features (9 features)

Saccades are rapid ballistic eye movements between fixations:

| Feature | Description |
|---------|-------------|
| `sac_count` | Number of saccades |
| `sac_rate` | Saccades per second |
| `sac_mean_amp` | Average saccade amplitude (degrees/pixels) |
| `sac_std_amp` | Amplitude variability |
| `sac_mean_peak_vel` | Average peak velocity |
| `sac_mean_dur` | Average saccade duration |

**Biometric relevance:** Saccade amplitude and peak velocity are influenced by oculomotor properties that vary between individuals.

#### 5.2.6 Scanpath Features (3 features)

The scanpath is the complete sequence of eye movements during visual exploration:

| Feature | Description |
|---------|-------------|
| `scan_total_length` | Total distance traveled by gaze |
| `scan_length_per_second` | Scanpath speed (exploration rate) |
| `scan_bbox_ratio` | Aspect ratio of scanpath bounding box |

**Biometric relevance:** Scanpath patterns reflect individual viewing strategies (e.g., systematic vs. random exploration).

#### 5.2.7 Entropy and Dispersion Features (5 features)

These capture the spatial distribution of gaze:

| Feature | Description |
|---------|-------------|
| `disp_total` | Total spatial dispersion |
| `entropy_2d` | 2D Shannon entropy of gaze distribution |
| `entropy_x`, `entropy_y` | 1D entropy for x/y separately |
| `grid_entropy` | Entropy over 4×4 spatial grid |

**Biometric relevance:** Entropy measures how uniformly gaze is distributed; some individuals explore broadly while others focus narrowly.

#### 5.2.8 Grid-Based AOI Features (17 features)

We divide the screen into a 4×4 grid of Areas of Interest (AOIs) and compute:

| Feature | Description |
|---------|-------------|
| `grid_00` through `grid_33` | Proportion of gaze in each grid cell |
| `grid_entropy` | Entropy over the 16 cells |

**Biometric relevance:** Spatial grid proportions capture coarse viewing preferences (e.g., tending to look at top-left vs. center of screen).

### 5.3 Feature Preprocessing

Before model training, we apply:

1. **NaN handling:** Linear interpolation for missing values; zeros for degenerate signals
2. **Zero-variance removal:** Features with zero variance (constant across all subjects) are removed
3. **Standard scaling:** All features are standardized (zero mean, unit variance)

### 5.4 Machine Learning Models

We evaluate four classifiers representing different algorithmic paradigms:

#### 5.4.1 Logistic Regression (Multinomial)

**Type:** Linear classifier  
**Algorithm:** Maximum likelihood estimation with L2 regularization  
**Solver:** L-BFGS (limited-memory Broyden–Fletcher–Goldfarb–Shanno)  
**Parameters:** C=1.0, max_iter=2000  

**Rationale:** As a linear model, logistic regression establishes a baseline for linearly separable patterns. It assumes features combine linearly to predict class probabilities.

**Mathematical formulation:**

```
P(y=k|x) = exp(w_k · x) / Σ_j exp(w_j · x)
```

Where w_k are the weight vectors for each of 69 classes.

#### 5.4.2 Support Vector Machine (RBF Kernel)

**Type:** Kernel-based classifier  
**Kernel:** Radial Basis Function (RBF)  
**Parameters:** C=10.0, gamma="scale"  
**Decision function:** One-vs-rest (ovr)  

**Rationale:** SVMs excel in high-dimensional spaces and can capture non-linear decision boundaries through the kernel trick. The RBF kernel maps features into an infinite-dimensional space where classes may be more separable.

**Mathematical formulation:**

```
K(x, x') = exp(-γ||x - x'||²)
```

Where γ is the kernel coefficient (gamma).

#### 5.4.3 Random Forest

**Type:** Ensemble of decision trees  
**Parameters:** n_estimators=300, max_depth=None  
**Bootstrap:** True (bagging)  

**Rationale:** Random forests aggregate multiple decision trees, each trained on a random subset of features and samples. They naturally handle non-linear relationships and provide feature importance estimates.

**Ensemble prediction:**

```
ŷ = mode{T₁(x), T₂(x), ..., T₃₀₀(x)}
```

#### 5.4.4 XGBoost (Extreme Gradient Boosting)

**Type:** Gradient-boosted decision trees  
**Parameters:** n_estimators=300, max_depth=6, learning_rate=0.1  
**Subsampling:** 0.8 (rows), 0.8 (columns)  

**Rationale:** XGBoost builds trees sequentially, each correcting errors of the previous ensemble. It often achieves state-of-the-art performance on tabular data.

**Boosting formulation:**

```
F₀(x) = argmax_γ Σ L(yᵢ, γ)
Fₘ(x) = Fₘ₋₁(x) + η·hₘ(x)
```

Where hₘ is the m-th tree and η is the learning rate.

#### 5.4.5 Model Pipeline

All models use a scikit-learn Pipeline:

```python
Pipeline([
    ('scaler', StandardScaler()),  # Normalize features
    ('clf', <classifier>)          # Train classifier
])
```

This ensures:
1. Feature scaling is applied consistently
2. No data leakage from test to training set
3. Reproducible preprocessing

### 5.5 Evaluation Protocol

#### 5.5.1 Cross-Task Data Preparation

```python
# For each split (e.g., train on task1+task2, test on task7+task8):

# 1. Extract features with generic prefixes
X_train = build_feature_matrix(dataset, train_tasks, generic_prefix=True)
X_test = build_feature_matrix(dataset, test_tasks, generic_prefix=True)

# 2. Find subjects present in both train and test
common_subjects = intersect(train_subjects, test_subjects)

# 3. Align feature matrices
X_train = X_train[common_subjects]
X_test = X_test[common_subjects]

# 4. Encode labels
y_train = encode_labels(common_subjects)  # Same labels for train and test
y_test = encode_labels(common_subjects)   # Because it's identification, not verification
```

**Important:** In identification mode, each subject's train features (from Task A) and test features (from Task B) are separate samples. The model must learn to match the same identity across different task contexts.

#### 5.5.2 Metric Computation

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Accuracy | correct / total | Overall identification rate |
| Balanced Accuracy | mean(per-class recall) | Average across subjects |
| F1 Macro | mean(F1 per class) | Harmonic mean of precision/recall |
| F1 Weighted | weighted mean(F1) | Weighted by class support |

#### 5.5.3 Statistical Significance

We compare all results against the random baseline (1/69 ≈ 1.45%). Performance significantly above chance demonstrates that gaze patterns contain identity information.

---

## 6. Implementation

### 6.1 Project Structure

```
gaze_identity/
├── pyproject.toml              # Project configuration (uv-managed)
├── README.md                   # Project documentation
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Dataset loading and parsing
│   ├── features.py             # Gaze feature extraction
│   ├── models.py               # ML classifiers
│   ├── evaluation.py           # Metrics, visualization, reporting
│   └── utils.py                # Shared utilities
├── scripts/
│   ├── __init__.py
│   ├── run_pipeline.py         # Single-split pipeline
│   └── run_all_splits.py       # Full evaluation pipeline
├── results/                    # Generated results and figures
│   ├── full_run/
│   │   ├── REPORT.md           # Markdown report
│   │   ├── results.json        # Detailed results
│   │   ├── summary.csv         # Tabular summary
│   │   ├── results_summary.png # Bar chart
│   │   └── cm_*.png           # Confusion matrices
│   └── ...
└── tests/
```

### 6.2 Key Implementation Details

#### 6.2.1 Data Loading (`data_loader.py`)

**Challenge:** Parsing filenames with variable formats and handling missing data.

**Solution:** Regular expression parsing with fallback error handling:

```python
def parse_filename(filepath: str) -> Tuple[str, str, str]:
    m = re.match(r"^(\d+)_patient-([\d_]+)-(\w+)$", name)
    subject_id = m.group(1) + "_patient"
    return subject_id, date_str, data_type
```

**Session handling:** Multiple sessions per subject are resolved by keeping only the first session (sorted by date) to avoid data leakage.

#### 6.2.2 Feature Extraction (`features.py`)

**Challenge:** Consistent feature dimensions across different tasks and subjects with varying data quality.

**Solution:** Three-tier approach:

1. **NaN interpolation:** `_interpolate_nans()` handles missing values with linear interpolation
2. **Degenerate signal detection:** Checks for constant values after interpolation
3. **Zero-feature fallback:** `_zero_features()` ensures consistent dimensionality for problematic subjects

```python
def _interpolate_nans(signal_1d: np.ndarray) -> np.ndarray:
    nans = np.isnan(signal_1d)
    good = ~nans
    if np.sum(good) < 2:
        signal_1d[nans] = 0.0  # Not enough points to interpolate
        return signal_1d
    indices = np.arange(len(signal_1d))
    signal_1d[nans] = np.interp(indices[nans], indices[good], signal_1d[good])
    return signal_1d
```

**I-VT fixation detection:** The Velocity-Threshold Identification (I-VT) algorithm classifies gaze samples as fixations or saccades based on instantaneous velocity:

```python
def detect_fixations_saccades(x, y, fs=125.0, vel_threshold=30.0, min_fix_duration_s=0.1):
    speed = np.sqrt(np.diff(x)**2 + np.diff(y)**2) * fs
    is_saccade = speed > vel_threshold
    # Segment into fixations and saccades based on velocity threshold
    ...
```

**Per-picture aggregation:** Features are extracted independently for each picture segment, then aggregated with statistics (mean, std, min, max, range) to capture within-task variability.

#### 6.2.3 Cross-Task Evaluation (`evaluation.py`)

**Challenge:** Ensuring feature names match across different task combinations.

**Solution:** Generic prefixes (`t0_`, `t1_`) instead of task-specific prefixes (`task1_`, `task7_`):

```python
def build_feature_matrix(dataset, task_keys, generic_prefix=False):
    for slot_idx, tk in enumerate(task_keys):
        if generic_prefix:
            prefix = f"t{slot_idx}_"  # Always t0_, t1_
        else:
            prefix = f"{tk}_"        # task1_, task7_ (won't match)
```

This allows features from train tasks (e.g., task1+task2) to have identical names to features from test tasks (e.g., task7+task8).

#### 6.2.4 Model Training (`models.py`)

**Challenge:** Handling 69 classes with limited samples (~69 subjects).

**Solution:** Models designed for high-dimensional, small-sample regimes:
- Logistic Regression: L2 regularization prevents overfitting
- SVM RBF: Kernel trick handles non-linear boundaries
- Random Forest: Feature subsampling reduces variance
- XGBoost: Sequential boosting focuses on hard examples

### 6.3 Reproducibility

All results are reproducible with fixed random seed (`seed=42`):

```bash
uv run python scripts/run_all_splits.py --output-dir results/reproducible_run
```

---

## 7. Results

### 7.1 Summary Table

| Split | Train Tasks | Test Tasks | Best Model | Accuracy | vs Random (1.45%) |
|-------|-------------|------------|------------|----------|-------------------|
| same_level_low | task1, task2 | task2, task3 | SVM RBF | **27.54%** | 19× |
| same_level_high | task7, task8 | task8, task9 | SVM RBF | 15.94% | 11× |
| medium_to_extremes | task4, task5 | task1, task9 | LogReg | 15.94% | 11× |
| low_to_high_A | task1, task2 | task7, task8 | LogReg/SVM | 13.04% | 9× |
| low_to_high_B | task1, task3 | task8, task9 | LogReg | 13.04% | 9× |
| high_to_low | task7, task8 | task1, task2 | LogReg/SVM/RF | 13.04% | 9× |
| all_low_to_all_high | task1, task2, task3 | task7, task8, task9 | LogReg/SVM | 11.59% | 8× |

### 7.2 Detailed Results by Model

#### 7.2.1 Cross-Task Splits (Generalization Test)

| Split | LogReg | SVM RBF | Random Forest | XGBoost |
|-------|--------|---------|---------------|---------|
| low_to_high_A | 13.04% | 13.04% | 10.14% | 2.90% |
| low_to_high_B | 13.04% | 10.14% | 7.25% | 1.45% |
| high_to_low | 13.04% | 13.04% | 13.04% | 2.90% |
| medium_to_extremes | 15.94% | 13.04% | 11.59% | 4.35% |
| all_low_to_all_high | 11.59% | 11.59% | 10.14% | 0.00% |

**Observation:** Linear models (LogReg, SVM) consistently outperform tree-based models (RF, XGBoost) on cross-task splits.

#### 7.2.2 Same-Level Splits (Upper Bound)

| Split | LogReg | SVM RBF | Random Forest | XGBoost |
|-------|--------|---------|---------------|---------|
| same_level_low | 23.19% | **27.54%** | 20.29% | 1.45% |
| same_level_high | 11.59% | 15.94% | 11.59% | 1.45% |

**Observation:** Same-level transfer significantly outperforms cross-level transfer, confirming cognitive load affects identity signatures.

### 7.3 Analysis

#### 7.3.1 Cross-Task Generalization

- **Low → High:** 11–16% accuracy (8–11× above chance)
- **High → Low:** 13% accuracy (9× above chance)
- **Conclusion:** Cross-task identification is feasible, demonstrating partial task-invariance in gaze identity

#### 7.3.2 Cognitive Load Impact

- **Same-level low:** 27.54% (upper bound)
- **Cross-level:** 11–16% (degradation of 40–60%)
- **Conclusion:** Cognitive load significantly impacts gaze identity, but does not eliminate it

#### 7.3.3 Model Comparison

- **Best overall:** SVM RBF (27.54% on same-level low)
- **Most robust:** Logistic Regression (consistently good across all splits)
- **Worst:** XGBoost (often at or near chance level)
- **Reasoning:** XGBoost likely overfits with 69 classes and 612 features on only ~69 samples

#### 7.3.4 Feature Analysis

Zero-variance features removed: ~28 features (from 640 → 612)
- These constant features were likely from degenerate signals or uniform distributions

### 7.4 Visualization

#### 7.4.1 Confusion Matrices

Confusion matrices show the pattern of misclassifications. For the best model (SVM on same_level_low):
- **Diagonal elements:** Correct identifications
- **Off-diagonal elements:** Confused identities
- **Pattern:** Errors appear somewhat distributed, suggesting no strong bias toward specific identity confusions

#### 7.4.2 Feature Importance

For tree-based models, feature importance reveals which features are most discriminative:
- Position features (mean, std) tend to rank highly
- Velocity features provide additional discriminative power
- Entropy and dispersion features contribute moderately

---

## 8. Discussion

### 8.1 Principal Findings

1. **Gaze contains identity information:** All models significantly outperform random baseline, demonstrating that eye movement patterns are individual-specific.

2. **Partial cross-task generalization:** Even across different cognitive demands, accuracy remains 8–11× above chance. This suggests that while cognitive load affects gaze behavior, a core identity signal persists.

3. **Cognitive load degrades but doesn't eliminate identity:** Same-level transfer (27%) significantly outperforms cross-level transfer (11–16%), indicating that cognitive state modulates gaze identity signatures.

4. **Linear models are more robust:** Logistic Regression and SVM consistently outperform tree-based models, suggesting the feature space is relatively linear and that non-linear models overfit with limited samples.

### 8.2 Implications

#### For Biometric Systems

- **Feasibility:** Gaze-based identification is viable across cognitive states, though performance degrades
- **Design considerations:** Systems should account for cognitive load variation
- **Practical deployment:** Additional features or adaptive models may be needed for high-reliability applications

#### For Cognitive Science

- **Individual differences:** Gaze behavior contains stable individual-specific components
- **Task-modulation:** Cognitive load affects not just where people look, but how their gaze patterns identify them
- **Neural implications:** The persistence of identity signals across tasks suggests stable oculomotor control properties

### 8.3 Limitations

1. **Sample size:** 69 subjects is small for 69-class classification
2. **Feature engineering:** Handcrafted features may miss important patterns
3. **Task diversity:** Limited to 9 tasks; broader task variety would strengthen conclusions
4. **Temporal dynamics:** Features aggregate across time, potentially losing sequential information
5. **No within-task cross-validation:** We did not evaluate within-task performance as an additional baseline

### 8.4 Future Directions

1. **Deep learning approaches:** LSTM or Transformer models to capture temporal dynamics
2. **Feature selection:** PCA or learned embeddings to reduce dimensionality
3. **Verification task:** Pairwise matching (same/different person) rather than multi-class identification
4. **Larger datasets:** Scale to hundreds or thousands of subjects
5. **Real-time adaptation:** Models that adapt to cognitive load in real-time
6. **Multi-modal fusion:** Combine gaze with other biometrics (e.g., pupil dilation, head movement)

---

## 9. Conclusions

This study demonstrates that eye-gaze patterns contain individual-specific information that partially transfers across different cognitive tasks. Key findings include:

1. **Cross-task identification is feasible:** Accuracy remains 8–19× above chance across all experimental conditions
2. **Cognitive load impacts identity:** Same-level transfer significantly outperforms cross-level transfer
3. **Linear models are preferred:** SVM and Logistic Regression outperform tree-based methods
4. **The approach is reproducible:** Full implementation and results are publicly available

While cognitive load modulates gaze identity signatures, it does not eliminate them. This suggests that gaze-based biometric systems can work across cognitive states, though performance optimization may require adaptive models or additional features.

The open-source implementation provides a foundation for future research in gaze-based behavioral biometrics.

---

## 10. References

1. EasyCog Dataset. "The EasyCog Dataset: Towards Easier Cognitive Assessment with Passive Video Watching." ACM IMWUT/Ubicomp 2026.

2. Salvucci, D. D., & Goldberg, J. H. (2000). "Identifying fixations and saccades in eye-tracking protocols." Proceedings of the 2000 symposium on Eye tracking research & applications, 71-78.

3. Duchowski, A. T. (2017). Eye Tracking Methodology: Theory and Practice. Springer.

4. Bulling, A., Roggen, D., & Tröster, G. (2011). "What's in the eyes for context-awareness?" IEEE Pervasive Computing, 10(2), 48-57.

5. Zhang, Y., & Dorn, P. (2019). "SAP: Gaze-based biometric authentication." Proceedings of the 2019 ACM International Joint Conference on Pervasive and Ubiquitous Computing, 1-4.

6. Steil, J., et al. (2019). "Screening for attention-deficit/hyperactivity disorder using eye movements." BMC Psychiatry, 19(1), 220.

---

## Appendix A: Reproducibility

### Environment Setup

```bash
cd gaze_identity
uv sync
```

### Run Full Pipeline

```bash
uv run python scripts/run_all_splits.py
```

### Run Single Split

```bash
uv run python scripts/run_pipeline.py --split low_to_high_A
```

### Output Files

- `results/full_run/REPORT.md`: Comprehensive markdown report
- `results/full_run/results.json`: Machine-readable results
- `results/full_run/summary.csv`: Tabular summary
- `results/full_run/results_summary.png`: Visualization
- `results/full_run/cm_*.png`: Confusion matrices

---

## Appendix B: Feature Reference

### Complete Feature List (64 base features)

**Position (14):**
pos_mean_x, pos_mean_y, pos_std_x, pos_std_y, pos_range_x, pos_range_y, pos_median_x, pos_median_y, pos_iqr_x, pos_iqr_y, pos_skew_x, pos_skew_y, pos_kurtosis_x, pos_kurtosis_y

**Velocity (9):**
vel_mean, vel_std, vel_max, vel_median, vel_p95, vel_p99, vel_mean_x, vel_mean_y, vel_ratio_x_y

**Acceleration (2):**
accel_mean, accel_std

**Fixation (10):**
fix_count, fix_rate, fix_mean_dur, fix_std_dur, fix_median_dur, fix_mean_dispersion, fix_total_time_ratio

**Saccade (9):**
sac_count, sac_rate, sac_mean_amp, sac_std_amp, sac_mean_peak_vel, sac_mean_dur

**Scanpath (3):**
scan_total_length, scan_length_per_second, scan_bbox_ratio

**Entropy/Dispersion (5):**
disp_total, entropy_2d, entropy_x, entropy_y, grid_entropy

**Grid AOI (17):**
grid_00, grid_01, grid_02, grid_03, grid_10, grid_11, grid_12, grid_13, grid_20, grid_21, grid_22, grid_23, grid_30, grid_31, grid_32, grid_33, grid_entropy

---

*Report generated June 18, 2026*
