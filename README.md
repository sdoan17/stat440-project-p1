# STAT 440 — Project P1

## Project files

- `data/`: pipe network and historical leak CSV files.
- `nds/01_baseline.ipynb`: exploratory analysis and baseline scheduling notebook.
- `results/simple_baseline/`: saved plots, schedule, and candidate CSV.
- `requirements.txt`: Python dependencies.

## Leak prediction notebooks

Run these in order, from top to bottom:

1. **`nds/03_feature_engineering_selection.ipynb`** owns cleaning, EDA, extraction, reference-tree training/selection, encoding and imputation. Its four candidates are adjusted age, log length, material + surface as one group, and the 75 m network leak rate. It compares all six pairs and keeps two groups. Material/surface indicators are included only when that group is selected.
2. **`nds/04_model_training_comparison.ipynb`** reads the numerical CSVs and compares tuned random forest with tuned XGBoost on **2025 and 2026**. It chooses the higher mean annual ROC-AUC, refits that winning family on all labelled 2020–2026 data and saves the model. It does not recreate/select features, encode categories or fit imputers.
3. **`nds/05_cost_optimization.ipynb`** searches circle groupings and work orderings against the complete 2027–2051 construction-plus-repair objective. It loads the saved model, projects three annual first-leak scenarios with survival, compares individual and paired circles, enforces the $10M annual allocation with unused-budget carryover, and checks a frozen selected plan with paired stochastic replays. It exports one headerless `"(x, y)",r` file at `output/XGBoost_model_plan.csv`. All detailed costs, schedules, assumptions and search results are kept separately in `results/cost_optimization/`. These are model-based planning estimates, not an official score or a proven global optimum.

Notebook 05 documents requirements from `StarKingdom.pdf` in Markdown cells. Its code follows a data → prediction → candidate comparison → evaluation → export workflow, with short cells and no inline comments. The PDF specifies the construction formula and rates, incidental copper/brass replacement, and zero polyurethane leak risk. Future risk projections, mixed-surface aggregation and annual replay timing are documented separately as modeling choices. The 25-year window beginning in 2027 ends in 2051; this is also before the briefing's end-of-2052 date.

All code for this workflow is inside these notebooks. They do not import project Python modules. The original team notebooks and `model.py` remain available as earlier work.

The main data handoff is:

- **`data/model_training_data.csv`**: 250,776 labelled 2020–2026 rows, with ID/year/target and the two selected full-data groups. The current full-data pair is adjusted age + network leak rate, so there are two numeric predictor columns.
- `data/model_folds/2025/` and `2026/`: prepared inner-training/validation and full-training/test CSVs. These preserve each year's training-only selection and imputation. The chosen groups can differ between years, but both model families receive identical inputs within each year.
- `data/model_folds/full_training/`: prepared 2020–2025/2026 tuning pair used before refitting the winning family on all labelled rows.

Generated results are separate from data:

- `results/feature_selection/`: selected feature names, candidate-combination scores and decision-tree reference scores.
- `results/model_comparison/`: the two-model leaderboard, annual scores, tuning results, historical evaluation probabilities, `best_model.joblib` and `best_model_details.json`.

The original `data/pipes.csv` and `data/train.csv` are inputs and are never overwritten by these notebooks. After changing source data or feature code, run Notebooks 03, 04 and 05 again.

The evaluation splits are 2020–2024 → 2025 and 2020–2025 → 2026. Selection and tuning use an earlier inner split within each training period. Two candidate groups are retained; a material/surface group expands into multiple indicator columns if selected. These are reused development results, with less evidence of stability than a longer evaluation history. They are not an untouched test set or an official cost score. Notebook 04 stops at model selection and refitting. Notebook 05 separately optimizes a 25-year work plan using explicit risk scenarios and budget carryover.

To use the environment already installed on this computer:

```sh
source .venv/bin/activate
jupyter lab
```

Choose **Python (STAT 440 .venv)** as the notebook kernel. These notebooks use the project's main `.venv` environment. For a new environment, install `requirements.txt` using the setup below. On macOS, XGBoost also needs the OpenMP runtime (already installed on this computer). Notebook 1's network-distance calculation and Notebook 2's random-forest tuning can each take several minutes. The notebooks contain saved outputs for reading without rerunning.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd nds
jupyter lab
```
