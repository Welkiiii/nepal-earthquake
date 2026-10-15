# Nepal Earthquake Damage Prediction

NTU2026-FINTECH-MH6810 Course project for predicting building damage severity in the DrivenData **Richter's Predictor: Modeling Earthquake Damage** competition.

DrivenData: Richter's Predictor — Modeling Earthquake Damage

https://www.drivendata.org/competitions/57/nepal-earthquake/

## Project Goal

Predict `damage_grade` for buildings affected by the 2015 Nepal earthquake and analyze which factors drive model performance.

Target classes:

- **1** — low damage
- **2** — medium damage
- **3** — severe damage

## Project Structure

```text
group_project/
├── Preprocess & EDA/
│   ├── 01_Preprocessing_EDA.ipynb
│   ├── 02_Supplementary_Preprocessing_Geography_EDA.ipynb
│   ├── raw_data/
│   ├── processed_data/
│   └── supplementary_processed_data/
│   ├── figures/
│
├── Modeling/
│   ├── 03_Model_Training_Evaluation.ipynb
│   ├── 04_CatBoost_LightGBM_Tuning_Ensemble.ipynb
│   ├── figures/
│   ├── figures_04_tuning/
│   ├── results/
│   └── results_04_tuning/
│
└── README.md
```

## Workflow

```text
Raw Data
   ↓
01 Preprocessing & EDA
   ↓
02 Supplementary Feature Engineering + Geography EDA
   ↓
03 Model Training, Evaluation and Analysis
   ↓
04 Leaderboard Optimization
```

## Main Methods

The project includes:

- data quality checks and EDA
- feature engineering
- fixed stratified train / validation split
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- probability ensembling
- feature ablation
- geography vs structure analysis
- geographic holdout validation
- SHAP interpretation
- post-earthquake inspection prioritization

## Key Results

Best standard validation model:

- **LightGBM + One-Hot**
- Validation Micro F1: **0.7474**

Best leaderboard model:

- **Tuned CatBoost + Tuned LightGBM probability ensemble**
- Public Micro F1: **0.7515**
- Public rank: **252 / 2,761**
- Approximately **Top 9.1%**

Leaderboard progression:

| Version | Public Micro F1 |
|---|---:|
| Original | 0.7440 |
| LightGBM 400 trees | 0.7451 |
| LGBM + Logistic | 0.7456 |
| Adjusted blend | 0.7460 |
| CatBoost | 0.7486 |
| Multi-model ensemble | 0.7494 |
| **Tuned CatBoost + Tuned LightGBM** | **0.7515** |

## Main Findings

- Geography is the strongest source of predictive signal.
- Structural variables still add useful incremental information.
- Model performance drops substantially on completely unseen geographic regions.
- Foundation type, roof type, age, construction material, and floor-related features are important predictors.
- The final ensemble performs substantially better than the original baseline submission.

## Running the Project

Install the required packages:

```bash
pip install -r Modeling/requirements_modeling.txt
```

Run the notebooks in order:

```text
01_Preprocessing_EDA.ipynb
02_Supplementary_Preprocessing_Geography_EDA.ipynb
03_Model_Training_Evaluation.ipynb
04_CatBoost_LightGBM_Tuning_Ensemble.ipynb
```

## Detailed Documentation

For the full methodology, EDA findings, modeling experiments, ablation studies, interpretability analysis, and leaderboard results, refer to the project report in the `report/` folder.

