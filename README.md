# Bangalore Traffic Prediction

## Overview
This repository contains the code and notebooks for analyzing Bengaluru traffic data and training baseline and Random Forest models on the data.

Project layout:
- `datasets/` – raw and processed CSV inputs referenced by the notebooks.
- `src/notebooks/` – analysis and modeling notebooks to reproduce the results.
- `src/` – source utilities supporting data preparation.
- `pyproject.toml` – dependency and tool configuration managed via `uv` or standard `pip` workflows.

## Environment Setup

### Prerequisites
- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`

### Using `uv`
```bash
git clone https://github.com/pranav738/fods_assignment
cd fods_assignment
uv sync
```
`uv sync` creates a `.venv` directory and installs all dependencies from `pyproject.toml`.

Run commands inside the environment with `uv run`:
- Start Jupyter: `uv run jupyter lab`
- Execute tests: `uv run pytest`
- Add new dependencies: `uv add <package-name>`

### Using `pip`
```bash
git clone https://github.com/pranav738/fods_assignment
cd fods_assignment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install .
```
This installs the project in editable mode using the dependencies declared in `pyproject.toml`. Launch Jupyter or other tooling as usual after activation (`python -m notebook`, `pytest`, etc.).

## Data
The notebooks expect the CSV files already present under `datasets/`:
- `datasets/raw/enriched_sample.csv`
- `datasets/processed/enriched_sample_encoded.csv`
- `datasets/processed/traffic_features_encoded_filtered.csv`

Ensure these files remain in place (or update the notebook paths if you relocate them).

## Notebook Workflow
Run the notebooks in `src/notebooks/` in the following order to replicate the analyses and metrics:

1. `in-depth-analysis-of-bangalore-traffic.ipynb`
   - **Purpose:** Complete exploratory data analysis (EDA).
   - **Input:** `datasets/raw/enriched_sample.csv` and `datasets/processed/enriched_sample_encoded.csv`.
2. `baseline_model.ipynb`
   - **Purpose:** Train and evaluate the baseline Decision Tree model.
   - **Output:** Final test RMSE for the baseline run.
   - **Input:** `datasets/processed/enriched_sample_encoded.csv`.
3. `random_forest.ipynb`
   - **Purpose:** Train and evaluate the initial Random Forest model (v1).
   - **Input:** `datasets/processed/enriched_sample_encoded.csv`.
4. `random_forest_v2.ipynb`
   - **Purpose:** Train and evaluate the improved Random Forest model (v2) used in the paper.
   - **Output:** Final test RMSE for the reported results.
   - **Input:** `datasets/processed/traffic_features_encoded_filtered.csv`.

Open the notebooks with your preferred interface (`uv run jupyter lab`, `jupyter notebook`, or VS Code) and execute each notebook top-to-bottom to reproduce the figures and metrics.

## Reproducing Results
1. Set up the environment using either `uv` or `pip` as described above.
2. Verify the datasets directory structure is intact.
3. Launch Jupyter and run the notebooks sequentially.
4. Optionally, run the automated tests with `uv run pytest` (or `pytest` inside your virtual environment) to ensure the project utilities behave as expected.

