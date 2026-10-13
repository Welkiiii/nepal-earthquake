# Nepal Earthquake Damage Prediction

## Project Overview

This project uses the DrivenData **Richter's Predictor: Modeling Earthquake Damage** dataset to predict the post-earthquake damage grade of buildings in Nepal.

The target variable is `damage_grade`:

| Grade | Interpretation |
|---|---|
| 1 | Low damage |
| 2 | Medium damage |
| 3 | Severe damage or near-complete destruction |

The project goes beyond a simple leaderboard exercise. In addition to predictive performance, it investigates how much of the signal comes from geographic location, how much comes from structural characteristics, whether engineered features add measurable value, how the model behaves on unseen regions, and whether predicted severe-damage probabilities can support post-earthquake inspection prioritization.

Official competition page:

https://www.drivendata.org/competitions/57/nepal-earthquake/

GitHub repository:

https://github.com/Welkiiii/nepal-earthquake

## Main Research Questions

1. How accurately can building damage severity be predicted from the available pre-earthquake features?
2. Which models perform best on the fixed validation set?
3. Which structural, material, age, usage, and geographic variables are most predictive?
4. Do the supplementary engineered features improve predictive performance?
5. How much predictive power comes from geography compared with building structure?
6. Does the model generalize to completely unseen geographic regions?
7. Can the model rank buildings effectively for post-earthquake inspection when inspection capacity is limited?

## Repository Structure

```text
group_project/
│
├── Modeling/
│   ├── 03_Model_Training_Evaluation.ipynb
│   ├── requirements_modeling.txt
│   ├── README_Modeling.md
│   ├── Nepal_Earthquake_Modeling_Report_EN_v2.docx
│   │
│   ├── results/
│   │   ├── model_comparison.csv
│   │   ├── best_confusion_matrix.csv
│   │   ├── feature_engineering_ablation.csv
│   │   ├── geography_vs_structure.csv
│   │   ├── random_vs_geographic_holdout.csv
│   │   ├── compact_model_shap_importance.csv
│   │   ├── inspection_prioritization.csv
│   │   └── submission_lightgbm_ohe.csv
│   │
│   ├── figures/
│   │   ├── 01_model_comparison.png
│   │   ├── 02_confusion_matrix.png
│   │   ├── 03_feature_engineering_ablation.png
│   │   ├── 04_geography_vs_structure.png
│   │   ├── 05_random_vs_geo_holdout.png
│   │   ├── 06_shap_importance.png
│   │   └── 07_inspection_prioritization.png
│   │
│   └── leaderboard/
│       ├── leaderboard_experiments.csv
│       ├── submission_A_lgb400.csv
│       ├── submission_B_lgb400_logit20.csv
│       └── submission_C_blend_adjusted.csv
│
├── Preprocess & EDA/
│   ├── raw_data/
│   │   ├── train_values.csv
│   │   ├── train_labels.csv
│   │   ├── test_values.csv
│   │   └── submission_format.csv
│   │
│   ├── processed_data/
│   │   ├── train_features_engineered.csv
│   │   ├── test_features_engineered.csv
│   │   ├── X_train.csv
│   │   ├── X_validation.csv
│   │   ├── y_train.csv
│   │   ├── y_validation.csv
│   │   ├── train_validation_split.csv
│   │   └── feature_metadata.json
│   │
│   ├── supplementary_processed_data/
│   │   ├── train_features_complete.csv
│   │   ├── test_features_complete.csv
│   │   ├── train_features_complete.pkl
│   │   ├── test_features_complete.pkl
│   │   ├── train_labels.csv
│   │   ├── submission_format.csv
│   │   ├── train_validation_split_complete.csv
│   │   └── feature_schema.json
│   │
│   ├── supplementary_outputs/
│   │   ├── figures/
│   │   └── tables/
│   │
│   ├── figures/
│   ├── 01_Preprocessing_EDA.ipynb
│   ├── 02_Supplementary_Preprocessing_Geography_EDA.ipynb
│   ├── DATA_PROCESSING_LOG.md
│   ├── Nepal_Earthquake_Preprocessing_EDA_Complete_Report.docx
│   └── Supplementary_Preprocessing_Handoff_Note.docx
│
├── presentation/
├── report/
└── README.md
```

Some output folders are generated when the notebooks are executed, so a fresh clone may initially contain fewer derived files.

## End-to-End Workflow

```text
Official raw competition data
            ↓
01_Preprocessing_EDA.ipynb
            ↓
Core data validation, EDA, feature engineering,
fixed 80/20 stratified train-validation split
            ↓
processed_data/
            ↓
02_Supplementary_Preprocessing_Geography_EDA.ipynb
            ↓
Additional engineered features,
geographic damage EDA,
complete modeling dataset
            ↓
supplementary_processed_data/
            ↓
03_Model_Training_Evaluation.ipynb
            ↓
Model benchmarks, ablation studies,
geographic holdout, SHAP,
inspection prioritization,
final submission
```

The notebooks should be run in the order **01 → 02 → 03**.

## Dataset Summary

The raw competition files used in this project are:

| File | Rows | Columns | Purpose |
|---|---:|---:|---|
| `train_values.csv` | 260,601 | 39 | Training building features including `building_id` |
| `train_labels.csv` | 260,601 | 2 | `building_id` and `damage_grade` |
| `test_values.csv` | 86,868 | 39 | Official competition test features |
| `submission_format.csv` | 86,868 | 2 | Required submission structure |

The preprocessing audit confirmed:

* No missing cells in the four official files.
* No exact duplicate rows.
* `building_id` is unique in each file.
* Training feature IDs align exactly with training labels.
* Test feature IDs align exactly with the submission template.
* Train and test feature schemas are consistent.
* Binary superstructure and secondary-use indicators contain valid 0/1 values.

`building_id` is retained throughout data processing for traceability and submission alignment. It is removed from the model feature matrix before training.

## Target Distribution

The full labeled dataset has the following class distribution:

| Damage Grade | Proportion |
|---|---:|
| Grade 1 | 9.64% |
| Grade 2 | 56.89% |
| Grade 3 | 33.47% |

The target is therefore imbalanced, especially for Grade 1.

The fixed 80/20 stratified split preserves this distribution almost exactly:

| Split | Grade 1 | Grade 2 | Grade 3 |
|---|---:|---:|---:|
| Train | 0.096407 | 0.568913 | 0.334680 |
| Validation | 0.096410 | 0.568907 | 0.334683 |

The split contains:

* 208,480 training observations
* 52,121 validation observations

`random_state = 42` is used throughout the main pipeline.

# Notebook 01: Core Preprocessing and EDA

File:

```text
Preprocess & EDA/01_Preprocessing_EDA.ipynb
```

Notebook 01 performs the main raw-data validation, exploratory analysis, feature engineering, and fixed train-validation split.

## Core Preprocessing Decisions

### Geographic IDs

The following variables are treated as categorical identifiers:

```text
geo_level_1_id
geo_level_2_id
geo_level_3_id
```

These values represent geographic categories. Their numeric values do not imply continuous magnitude or distance.

### Building Age

The original `age` variable is retained.

A special value of `age = 995` appears in 1,390 training observations, approximately 0.53% of the training set. Rather than deleting these observations, Notebook 01 creates:

```text
age_is_995
age_capped_100
```

Definitions:

```text
age_is_995 = 1 if age == 995 else 0
age_capped_100 = min(age, 100)
```

This preserves the unusual-value information while also creating a more robust version of age for modeling.

## Engineered Features from Notebook 01

Notebook 01 creates six explicit engineered features:

| Feature | Definition | Motivation |
|---|---|---|
| `age_is_995` | Indicator for `age == 995` | Explicitly preserve the unusual age code |
| `age_capped_100` | `min(age, 100)` | Reduce influence of the extreme age code |
| `material_count` | Sum of all available `has_superstructure_*` indicators | Approximate structural material complexity |
| `height_per_floor` | `height_percentage / count_floors_pre_eq` | Relative building height after controlling for floor count |
| `families_per_floor` | `count_families / count_floors_pre_eq` | Simple occupancy-density proxy |
| `has_reinforced_concrete` | 1 if either RC indicator is present | Identify RC-related construction |

Notebook 01 also saves the fixed train-validation assignment and the engineered train/test tables.

## Main EDA Findings from Notebook 01

The EDA indicates several strong descriptive associations:

* Grade 2 is the majority class.
* Building age has a nonlinear relationship with damage severity.
* Foundation type has a large association with damage composition.
* Roof type is strongly associated with damage grade.
* Secondary use is associated with different observed damage distributions.
* Floor count has a nonlinear relationship with severe damage.
* Material type matters more than the simple number of materials.
* Reinforced concrete, especially engineered RC, is associated with much lower observed Grade-3 rates than traditional masonry categories.
* Numerical features generally have weak individual linear correlation with the target, suggesting that nonlinear models and feature interactions may be useful.

These are descriptive associations. They are not causal estimates.

# Notebook 02: Supplementary Preprocessing and Geography EDA

File:

```text
Preprocess & EDA/02_Supplementary_Preprocessing_Geography_EDA.ipynb
```

Notebook 02 starts from the engineered feature tables generated by Notebook 01. It does not recreate the entire preprocessing pipeline.

## Additional Features from Notebook 02

### `secondary_use_count`

The ten detailed secondary-use indicators are summed:

```text
secondary_use_count = sum(has_secondary_use_*)
```

Observed training counts:

| Count | Buildings |
|---:|---:|
| 0 | 231,445 |
| 1 | 28,599 |
| 2 | 557 |

### `area_per_family`

Defined as:

```text
area_percentage / count_families
```

For buildings where `count_families = 0`, the feature is set to 0 while the original family-count variable remains available to the model.

Approximately 8.01% of the training buildings have `count_families = 0`.

### `rc_category`

The original binary `has_reinforced_concrete` feature is retained, and Notebook 02 adds a more detailed category:

```text
no_rc
non_engineered_rc
engineered_rc
mixed_rc
```

Training counts:

| RC Category | Buildings |
|---|---:|
| `no_rc` | 245,464 |
| `non_engineered_rc` | 11,004 |
| `engineered_rc` | 4,038 |
| `mixed_rc` | 95 |

### `age_group`

Age is grouped into:

```text
0-10
11-20
21-40
41-60
61-100
100+
995_special
```

The `995_special` category prevents the unusual `age = 995` code from contaminating the normal `100+` age group.

## Geographic Damage EDA

Notebook 02 produces damage summaries for:

```text
geo_level_1_id
geo_level_2_id
geo_level_3_id
```

For each geographic category, it computes:

* Number of buildings
* Mean damage grade
* Grade-1 rate
* Grade-2 rate
* Grade-3 rate

A key finding is the large regional heterogeneity at Geo Level 1:

```text
Minimum observed Grade-3 rate ≈ 8.55%
Maximum observed Grade-3 rate ≈ 80.75%
```

This result motivated the later geography-versus-structure ablation and geographic holdout experiment.

The geographic target summaries are used only for EDA and presentation. They are not merged back into the training feature matrix, so they do not create target leakage.

## Final Modeling Dataset

After Notebook 02, the final feature table has:

```text
260,601 training rows
86,868 test rows
49 total columns including building_id
48 usable modeling features after removing building_id
```

The preferred modeling inputs are:

```text
train_features_complete.pkl
test_features_complete.pkl
```

The pickle format is preferred because it preserves categorical dtypes.

# Pre-Modeling Validation

Before modeling, the pipeline was checked for:

* Required files
* Expected train/test shapes
* Matching train/test feature columns
* Unique and aligned `building_id`
* Presence of all four Notebook-02 features
* Preservation of all Notebook-01 features
* Absence of `damage_grade` from the feature matrix
* Zero missing values
* Zero infinite numeric values
* Correct categorical dtypes
* Consistency of `feature_schema.json`
* Correct split sizes
* Exact agreement between the Notebook-01 and Notebook-02 split assignments

The final validation output was:

```text
ALL PREPROCESSING CHECKS PASSED
DATA IS READY FOR 03 MODEL TRAINING
```

# Notebook 03: Modeling, Evaluation and Experimental Analysis

File:

```text
Modeling/03_Model_Training_Evaluation.ipynb
```

Notebook 03 contains the complete modeling stage.

## Evaluation Metrics

The project uses multiple evaluation metrics because the target is both imbalanced and ordinal.

### Micro F1

The official DrivenData competition metric.

For this single-label multiclass problem:

```text
Micro F1 = Accuracy
```

### Macro F1

Averages class-level F1 equally and is more sensitive to weak performance on minority classes.

### Grade-3 Recall

Measures how many truly severely damaged buildings are identified.

### Ordinal MAE

Uses the natural ordering of Grades 1, 2, and 3.

A Grade 1 → Grade 3 error is more serious than a Grade 1 → Grade 2 error.

### Quadratic Weighted Kappa

Measures ordinal agreement while penalizing large disagreements more heavily.

### Catastrophic Error Rate

Defined as:

```text
CER = P(|true_grade - predicted_grade| = 2)
```

This captures direct Grade-1 ↔ Grade-3 errors.

# Model Benchmark

The following models were evaluated on the same fixed validation split:

| Model | Micro F1 | Macro F1 | Grade-3 Recall | QWK | CER |
|---|---:|---:|---:|---:|---:|
| LightGBM + One-Hot | **0.7474** | **0.6937** | **0.6222** | **0.6142** | 0.0031 |
| LightGBM, compact encoding | 0.7387 | 0.6816 | 0.6079 | 0.5993 | 0.0026 |
| Linear Logistic Baseline | 0.7359 | 0.6835 | 0.5488 | 0.5922 | 0.0051 |
| XGBoost + One-Hot | 0.7331 | 0.6700 | 0.5792 | 0.5838 | 0.0023 |
| Random Forest | 0.7172 | 0.6446 | 0.5340 | 0.5495 | 0.0020 |
| Decision Tree | 0.7087 | 0.6529 | 0.6069 | 0.5609 | 0.0062 |
| Majority Baseline | 0.5689 | 0.2417 | 0.0000 | 0.0000 | 0.0000 |

## Best Model

The strongest tested validation model is:

```text
LightGBM + One-Hot Encoding
```

Validation performance:

```text
Micro F1          0.7474
Macro F1          0.6937
Grade-3 Recall    0.6222
QWK               0.6142
Ordinal MAE       ≈ 0.2557
CER               ≈ 0.0031
```

The linear baseline performs surprisingly well, which suggests that a large amount of signal is additive. LightGBM still improves both overall performance and Grade-3 recall, showing that nonlinearities and interactions add measurable value.

# Best-Model Error Analysis

The confusion matrix shows that:

* Grade 2 is the easiest class.
* Grade 1 is the hardest class.
* Grade 3 has useful recall while maintaining relatively high precision.
* Extreme Grade-1 ↔ Grade-3 errors are rare.

Approximately 161 of 52,121 validation observations are catastrophic two-grade errors, corresponding to a CER of roughly 0.31%.

# Supplementary Feature Ablation

A controlled LightGBM comparison was used to test whether Notebook 02 improves predictive performance.

| Feature Set | Micro F1 |
|---|---:|
| Notebook 01 features | 0.7398 |
| Notebook 01 + 02 features | 0.7387 |

The difference is very small and slightly negative.

The evidence therefore does not support claiming that the four supplementary features improve LightGBM validation accuracy.

They remain useful for richer representation, interpretation, and targeted analysis.

# Geography versus Structure

The project compares three feature groups using the same model family.

| Feature Set | Micro F1 | Macro F1 | Grade-3 Recall |
|---|---:|---:|---:|
| Geography only | 0.7166 | ≈ 0.616 | ≈ 0.589 |
| Structure only | 0.6201 | ≈ 0.531 | ≈ 0.308 |
| Full compact model | 0.7387 | 0.6816 | 0.6079 |

The main conclusion is:

**Geography is the dominant source of predictive information in the random validation setting, while structural characteristics provide meaningful incremental information.**

The full model improves Micro F1 by about 2.2 percentage points over the geography-only model.

This is a predictive ablation. It is not a causal decomposition.

# Geographic Holdout

A second validation design holds out entire `geo_level_1_id` regions.

This experiment asks whether the model can generalize to broad geographic regions it never sees during training.

Approximate result:

```text
Random stratified validation Micro F1    0.7387
Unseen Geo-Level-1 holdout Micro F1      0.5109
```

QWK also falls substantially.

This demonstrates a major transferability gap.

The geographic holdout also changes the target distribution. Grade 3 represents roughly 28% of the geographic-holdout training subset and roughly 45% of the held-out regions.

The performance drop therefore reflects both unseen geography and target-distribution shift.

The project does not interpret the entire difference as pure geographic memorization.

# Explainability

The compact LightGBM model is used for SHAP because it preserves interpretable original feature names.

The strongest global predictors include:

1. `geo_level_1_id`
2. `geo_level_2_id`
3. `age`
4. `foundation_type`
5. `has_superstructure_mud_mortar_stone`
6. `other_floor_type`
7. `ground_floor_type`
8. `geo_level_3_id`
9. `roof_type`
10. `count_floors_pre_eq`
11. `has_superstructure_cement_mortar_brick`
12. `area_percentage`

The SHAP ranking is consistent with the earlier EDA.

Geographic variables dominate, followed by building age, foundation, structural material, floor type, roof type, and floor count.

SHAP values represent predictive association, not causal effect.

# Post-Earthquake Inspection Prioritization

The best model's predicted probability of Grade 3 is used as a risk score.

The validation buildings are ranked from highest to lowest severe-damage probability.

Selected results:

| Inspection Budget | Grade-3 Precision | Grade-3 Recall |
|---|---:|---:|
| Top 5% | ≈ 96% | ≈ 14% |
| Top 10% | ≈ 93.9% | ≈ 28.1% |
| Top 20% | ≈ 84.1% | ≈ 50.2% |
| Top 30% | ≈ 74% | ≈ 66.7% |

This suggests that the model can concentrate severely damaged buildings into high-priority inspection groups.

For example, inspecting only the highest-risk 20% of validation buildings captures about half of all Grade-3 buildings.

These results are measured on the random validation distribution. Deployment to new regions would require additional validation and likely recalibration.

# Official DrivenData Submission

The first official submission achieved:

```text
Public Micro F1 = 0.7440
```

The best internal validation score was:

```text
Validation Micro F1 = 0.7474
```

The difference is only:

```text
0.0034
```

This close agreement suggests that the fixed random validation set provides a good estimate of competition test performance under a similar distribution.

At the time the leaderboard was checked, the observed leading public score was approximately:

```text
0.7564
```

The gap between the first project submission and the observed leader was approximately 0.0124.

# Optional Leaderboard Optimization

The academic project is complete before this section.

Leaderboard optimization is treated as an optional extension.

Three additional submission candidates are generated.

## Candidate A

```text
submission_A_lgb400.csv
```

Method:

```text
LightGBM + One-Hot
400 trees
```

This matches the strongest internal validation configuration more closely than the first submitted full-data model.

Recommended interpretation:

```text
Safest next submission
```

## Candidate B

```text
submission_B_lgb400_logit20.csv
```

Method:

```text
80% LightGBM probability
20% Linear Logistic probability
```

Validation result:

```text
Micro F1 ≈ 0.7490
Macro F1 ≈ 0.6957
```

The logistic model is weaker individually, but the ensemble can improve performance because its error pattern differs from LightGBM.

This is the recommended conservative ensemble experiment.

## Candidate C

```text
submission_C_blend_adjusted.csv
```

Method:

```text
Candidate B probability blend
Grade-1 score × 0.95
Grade-3 score × 1.07
```

Validation result:

```text
Micro F1 ≈ 0.7496
```

This is the strongest fixed-validation leaderboard candidate so far.

It also has the highest risk of validation overfitting because the class-score multipliers were selected on the same validation set.

## Recommended Submission Order

1. Candidate A
2. Candidate B
3. Candidate C

Each upload should be treated as a controlled experiment and recorded before trying the next candidate.

# Installation

Recommended Python version:

```text
Python 3.10+
```

Install modeling dependencies:

```bash
pip install -r Modeling/requirements_modeling.txt
```

If working directly inside a Jupyter notebook and a package is missing:

```python
%pip install lightgbm xgboost shap
```

Core packages include:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
lightgbm
xgboost
shap
jupyter
```

# Running the Project

## Step 1: Prepare the official raw data

Place the four competition files in:

```text
Preprocess & EDA/raw_data/
```

Required files:

```text
train_values.csv
train_labels.csv
test_values.csv
submission_format.csv
```

## Step 2: Run Notebook 01

Open:

```text
Preprocess & EDA/01_Preprocessing_EDA.ipynb
```

Run all cells.

Confirm that `processed_data/` contains:

```text
train_features_engineered.csv
test_features_engineered.csv
X_train.csv
X_validation.csv
y_train.csv
y_validation.csv
train_validation_split.csv
feature_metadata.json
```

## Step 3: Run Notebook 02

Open:

```text
Preprocess & EDA/02_Supplementary_Preprocessing_Geography_EDA.ipynb
```

Run all cells.

Confirm that `supplementary_processed_data/` contains:

```text
train_features_complete.csv
test_features_complete.csv
train_features_complete.pkl
test_features_complete.pkl
train_validation_split_complete.csv
feature_schema.json
```

The expected final train/test dimensions are:

```text
Train: 260,601 × 49
Test:   86,868 × 49
```

After removing `building_id`, there should be:

```text
48 modeling features
```

## Step 4: Run the Preflight Validation Cell

The preprocessing pipeline should report:

```text
ALL PREPROCESSING CHECKS PASSED
DATA IS READY FOR 03 MODEL TRAINING
```

Do not proceed to modeling if any assertion fails.

## Step 5: Run Notebook 03

Open:

```text
Modeling/03_Model_Training_Evaluation.ipynb
```

Run the main project sections in order.

The notebook covers:

* Majority baseline
* Linear logistic baseline
* Decision Tree
* Random Forest
* Compact LightGBM
* XGBoost with one-hot encoding
* LightGBM with one-hot encoding
* Unified evaluation
* Confusion-matrix analysis
* Notebook-01 versus Notebook-02 feature ablation
* Geography-only versus structure-only versus full-model ablation
* Geographic holdout
* SHAP analysis
* Inspection prioritization
* Final competition submission
* Optional leaderboard optimization

# Reproducibility and Leakage Safeguards

The project follows several safeguards.

## Fixed split

All major random-holdout comparisons use the same fixed 80/20 split.

## Training-only preprocessing

When a model requires learned preprocessing, category mappings, one-hot encoders, scalers, or similar transformations are fitted on training data and then applied to validation data.

## Test isolation

The official test set is used only for final competition prediction.

It is not used for model selection.

## Geography target summaries

Observed regional Grade-3 rates are used only for EDA.

They are never inserted into the training matrix.

Any future target encoding would need an out-of-fold implementation.

## `building_id`

`building_id` is used for traceability, label alignment, split alignment, and submission.

It is removed before fitting predictive models.

# Important Interpretation Caveats

## Predictive association is not causality

EDA, feature importance, SHAP, and ablation measure association and predictive value.

They do not prove that modifying a foundation, roof, material, age, or geographic feature would causally change earthquake damage.

## Geographic holdout is a domain-shift test

The geographic holdout changes both geographic composition and damage-grade distribution.

Its lower performance should be interpreted as a broader transferability challenge.

## Leaderboard optimization is separate from the main research story

The main course-project evidence is:

```text
EDA
→ model comparison
→ feature ablation
→ geography vs structure
→ geographic holdout
→ SHAP
→ inspection prioritization
```

Leaderboard tuning is an optional final extension.

# Key Project Conclusions

1. **LightGBM with one-hot encoded categorical variables is the strongest tested random-validation model.**

2. **The best internal Micro F1 is approximately 0.7474.**

3. **The first official DrivenData submission achieved 0.7440, closely matching the internal validation estimate.**

4. **Geography is the dominant source of predictive signal in the random validation setting.**

5. **Structural characteristics still add measurable incremental predictive value beyond geography.**

6. **The four additional Notebook-02 engineered features do not materially improve LightGBM validation accuracy.**

7. **Model performance falls substantially when evaluated on completely unseen broad geographic regions.**

8. **SHAP identifies geography, age, foundation, construction material, floor type, roof type, and floor count as important predictors.**

9. **Predicted Grade-3 probabilities can prioritize inspections efficiently within the random-validation distribution.**

10. **The project supports a broader conclusion that strong leaderboard performance under a familiar geographic distribution does not guarantee equally strong transfer to new regions.**

# Recommended Presentation Story

For the final 20-minute group presentation, the modeling story can be summarized as:

```text
Problem and dataset
        ↓
EDA and preprocessing
        ↓
Strong geographic heterogeneity
        ↓
Model benchmark
        ↓
Best model: LightGBM + One-Hot
        ↓
Feature engineering ablation
        ↓
Geography vs structure
        ↓
Geographic generalization challenge
        ↓
SHAP interpretation
        ↓
Inspection prioritization
        ↓
Official DrivenData score
        ↓
Limitations and future work
```

The leaderboard section should remain brief unless the course specifically rewards competition ranking.

# Main Deliverables

## Preprocessing and EDA

```text
Preprocess & EDA/01_Preprocessing_EDA.ipynb
Preprocess & EDA/02_Supplementary_Preprocessing_Geography_EDA.ipynb
Nepal_Earthquake_Integrated_Preprocessing_EDA_Report_EN.docx
```

## Modeling

```text
Modeling/03_Model_Training_Evaluation.ipynb
Modeling/Nepal_Earthquake_Modeling_Report_EN_v2.docx
Modeling/results/
Modeling/figures/
Modeling/leaderboard/
```

## Submission

```text
Modeling/results/submission_lightgbm_ohe.csv
```

Optional leaderboard submissions:

```text
Modeling/leaderboard/submission_A_lgb400.csv
Modeling/leaderboard/submission_B_lgb400_logit20.csv
Modeling/leaderboard/submission_C_blend_adjusted.csv
```

# Team Workflow

The coding work is concentrated in two members.

One coding member owns the core preprocessing and EDA in Notebook 01.

The second coding member owns the supplementary feature/geography pipeline in Notebook 02 and the full modeling and experimental analysis in Notebook 03.

The remaining group members can use the generated English reports, figures, result tables, and this README to prepare the final presentation and written report without needing to rerun or modify the modeling code.

# Future Extensions

Possible extensions include:

* Native categorical CatBoost on the full feature set.
* More systematic LightGBM hyperparameter tuning.
* Probability calibration.
* Ordinal-specific classification methods.
* More robust geographic cross-validation.
* External seismic-intensity or geospatial information, subject to competition rules.
* Model calibration and decision thresholds optimized for inspection budgets.
* Ensemble stacking using out-of-fold predictions.
* Uncertainty estimation for model metrics and risk rankings.

# Status

Current project status:

```text
Data validation                 Complete
EDA                             Complete
Core preprocessing              Complete
Supplementary preprocessing     Complete
Fixed split validation          Complete
Model benchmark                 Complete
Feature ablation                Complete
Geography vs structure          Complete
Geographic holdout              Complete
SHAP interpretation             Complete
Inspection prioritization       Complete
Official submission             Complete
First public score              0.7440
Leaderboard optimization        In progress
Presentation                    In progress
Final written report            In progress
```

