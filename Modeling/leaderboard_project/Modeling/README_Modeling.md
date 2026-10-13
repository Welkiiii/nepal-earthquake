# Nepal Earthquake Modeling

This folder contains the modeling stage of the project.

## Main file
`03_Model_Training_Evaluation.ipynb`

## Expected input
The notebook expects the repository structure:

```text
group_project/
├── Modeling/
│   └── 03_Model_Training_Evaluation.ipynb
└── Preprocess & EDA/
    ├── raw_data/
    ├── processed_data/
    └── supplementary_processed_data/
```

Run Notebooks 01 and 02 first.

## Verified best validation model
LightGBM + one-hot encoded categorical features:
- Micro F1: 0.7474
- Macro F1: 0.6937
- Grade-3 Recall: 0.6222
- QWK: 0.6142

## Key project finding
Geographic variables carry very strong predictive signal. Performance deteriorates sharply when complete Geo-Level-1 regions are held out, so random-split performance should not be interpreted as evidence of equally strong transfer to new regions.


## Optional leaderboard optimization

The first official DrivenData submission scored **0.7440**.

Three additional submission candidates are in `leaderboard/`:

1. `submission_A_lgb400.csv`
   - exact 400-tree LightGBM configuration used by the strongest validation model
   - safest next submission

2. `submission_B_lgb400_logit20.csv`
   - 80% LightGBM + 20% linear-logistic probability blend
   - fixed-validation Micro F1: approximately **0.7490**
   - recommended conservative ensemble experiment

3. `submission_C_blend_adjusted.csv`
   - Candidate B plus mild class-score adjustment
   - fixed-validation Micro F1: approximately **0.7496**
   - more exposed to validation overfitting

Leaderboard tuning is intentionally separated from the main academic analysis.
