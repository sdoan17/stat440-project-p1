# STAT 440 — Star Kingdom Pipe Replacement

Plan the replacement of iron gas pipes over 25 years with a $10 million annual construction budget. Use historical leak data to decide where to dig and which pipes to replace first, minimizing construction and repair costs.

## What to run

These are different team approaches—you do not need to run everything.

| File | Use it to… |
|---|---|
| [01_baseline.ipynb](nds/01_baseline.ipynb) | Explore the network and build a simple oldest-first replacement plan. |
| [02_model_clean.ipynb](nds/02_model_clean.ipynb) | Build an age-based leak model and compare replacement strategies. Runs independently. |
| [03_feature_engineering_selection.ipynb](nds/03_feature_engineering_selection.ipynb) | Prepare the data and select two of four candidate feature groups. |
| [04_model_training_comparison.ipynb](nds/04_model_training_comparison.ipynb) | Compare random forest and XGBoost using the data from Notebook 03. |
| [05_cost_optimization.ipynb](nds/05_cost_optimization.ipynb) | Use the saved model from Notebook 04 to generate a 25-year replacement plan. |
| [model.py](model.py) | Run a separate gradient-boosting and age-based planning approach. |

For the random forest/XGBoost workflow, run **03 → 04 → 05**. The final plan is saved to **`output/XGBoost_model_plan.csv`**.

## Setup

From the project folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd nds
jupyter lab
```

Run each notebook from top to bottom using the `.venv` Python kernel. For `model.py`, activate the same environment and run `python ../model.py` from `data/`; it saves `pipe_risk.csv` and `submission.csv` there.

## Folders

- `data/`: pipe information, historical leaks, and prepared training data.
- `nds/`: notebooks.
- `results/`: evaluation results and planning diagnostics.
- `output/`: notebook submission plans.

Submission files contain one circular work unit per line, with no header: `"(x, y)",radius`. Coordinates and radii are in metres; rows give the work order.
