# STAT 440 — Project P1

## Project files

- `data/`: pipe network and historical leak CSV files.
- `nds/01_baseline.ipynb`: exploratory analysis and baseline scheduling notebook.
- `results/simple_baseline/`: saved plots, schedule, and candidate CSV.
- `requirements.txt`: Python dependencies.

## Leak prediction notebooks

Run these in order, from top to bottom:

1. **`nds/05_feature_engineering_selection.ipynb`** owns cleaning, EDA, extraction, reference-tree training/selection, encoding and imputation. Its four candidates are adjusted age, log length, material + surface as one group, and the 75 m network leak rate. It compares all six pairs and keeps two groups. Material/surface indicators are included only when that group is selected.
2. **`nds/06_model_training_comparison.ipynb`** reads the numerical CSVs and compares tuned random forest with tuned XGBoost on **2025 and 2026**. It chooses the higher mean annual ROC-AUC, refits that winning family on all labelled 2020–2026 data and saves the model. It does not recreate/select features, encode categories or fit imputers.

All code for this workflow is inside the two notebooks. They do not import project Python modules. The original team notebooks and `model.py` remain available as earlier work.

The main data handoff is:

- **`data/model_training_data.csv`**: 250,776 labelled 2020–2026 rows, with ID/year/target and the two selected full-data groups. The current full-data pair is adjusted age + network leak rate, so there are two numeric predictor columns.
- `data/model_folds/2025/` and `2026/`: prepared inner-training/validation and full-training/test CSVs. These preserve each year's training-only selection and imputation. The chosen groups can differ between years, but both model families receive identical inputs within each year.
- `data/model_folds/full_training/`: prepared 2020–2025/2026 tuning pair used before refitting the winning family on all labelled rows.

Generated results are separate from data:

- `results/feature_selection/`: selected feature names, candidate-combination scores and decision-tree reference scores.
- `results/model_comparison/`: the two-model leaderboard, annual scores, tuning results, historical evaluation probabilities, `best_model.joblib` and `best_model_details.json`.

The original `data/pipes.csv` and `data/train.csv` are inputs and are never overwritten by these notebooks. After changing source data or feature code, run Notebook 05 and then Notebook 06 again.

The evaluation splits are 2020–2024 → 2025 and 2020–2025 → 2026. Selection and tuning use an earlier inner split within each training period. Two candidate groups are retained; a material/surface group expands into multiple indicator columns if selected. These are reused development results, with less evidence of stability than a longer evaluation history. They are not an untouched test set or an official cost score. No future-year prediction dataset or forecast is generated.

To use the environment already installed on this computer:

```sh
source .venv/bin/activate
jupyter lab
```

Choose **Python (STAT 440 .venv)** as the notebook kernel. Both notebooks use the project's main `.venv` environment. For a new environment, install `requirements.txt` using the setup below. On macOS, XGBoost also needs the OpenMP runtime (already installed on this computer). Notebook 1's network-distance calculation and Notebook 2's random-forest tuning can each take several minutes. Both notebooks contain saved outputs for reading without rerunning.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd nds
jupyter lab
```
