# STAT 440 — Project P1

Course project exploring a baseline pipe replacement schedule using pipe network data and historical leak records.

## Project files

- `data/`: pipe network and historical leak CSV files.
- `nds/01_baseline.ipynb`: exploratory analysis and baseline scheduling notebook.
- `results/simple_baseline/`: saved plots, schedule, and candidate CSV.
- `requirements.txt`: Python dependencies.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd nds
jupyter lab
```

Open `01_baseline.ipynb`. The notebook expects its working directory to be `nds/` so that it can locate the data and results folders.

The notebook and saved results are work in progress.
