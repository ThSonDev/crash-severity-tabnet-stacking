# Crash Severity — TabNet + Stacking

Adapts the pipeline from *Pedestrian Crash Severity Analysis using TabNet and Stacking Ensemble Models* (Rafe et al., 2024) to the Kaggle `saurabhshahane/road-traffic-accidents` dataset (3-class `Accident_severity`: Slight / Serious / Fatal).

Five models are trained on a shared split / encoding / SMOTE / Optuna / threshold-tuning contract so they can be compared head-to-head: **TabNet**, **Stacking** (LR + XGBoost + ExtraTrees), **MLP** (PyTorch with entity embeddings), **Logistic Regression**, and **LightGBM**.

## Python version

Pinned to **Python 3.12.12**
## Local install (uv)

[uv](https://docs.astral.sh/uv/) is the package manager. Install it first:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Or

```bash
pip install uv
```

Then from the repo root:

```bash
uv sync --group dev          # install runtime + jupyter dev tools
```

Runtime-only install (no Jupyter, e.g. on a server):

```bash
uv sync
```

## Dataset

The CSV is not committed. Download `RTA Dataset.csv` from Kaggle (`saurabhshahane/road-traffic-accidents`) and place it at:

```
data/RTA Dataset.csv
```

Or run this cell (Google Colab):
```
import kagglehub
path = kagglehub.dataset_download("saurabhshahane/road-traffic-accidents")

!mkdir -p /content/data

!rm -rf /content/data/*

!mv "$path"/* /content/data/

!ls -l /content/data
```

Then run `uv run python clean-feature-engineer.py` to produce `data/data_cleaned.csv`.

## Google Colab

The notebooks run on Colab with the free T4 GPU. In a fresh runtime:

```python
!pip install -q scikit-learn xgboost lightgbm pandas imbalanced-learn shap \
    matplotlib numpy torch pytorch-tabnet optuna joblib

# Optional
from google.colab import files
files.upload()
```

Or mount Drive and point the notebook paths at your Drive copy.
Or use the kagglehub download dataset cell above.

## Layout

| Path | Purpose |
|---|---|
| `data/` | Project datasets |
| `clean-feature-engineer.py` | Raw CSV → `data/data_cleaned.csv` |
| `EDA_raw.ipynb` | Quality checks on the raw data |
| `EDA_cleaned.ipynb` | Feature analysis on cleaned data |
| `tabnet.ipynb` | TabNet training + Optuna search |
| `stacking-ensemble.ipynb` | LR + XGB + ExtraTrees stacking |
| `mlp.ipynb` | PyTorch MLP with entity embeddings + Optuna search |
| `lr.ipynb` | Logistic Regression baseline + Optuna search |
| `lightgbm.ipynb` | LightGBM + Optuna search |
| `original/` | Read-only reference (Rafe et al. notebooks) |
| `outputs/` | Per-model metrics JSON + figures (confusion, SHAP, threshold sweep, sensitivity, training curves) |
| `models/` | Fitted artifacts: `{model}_best.{joblib,zip,pt}` |

## Results

Test-set metrics on the held-out 15% split (`n_test = 1,848`). Macro-F1 is the Optuna objective, so it is the primary comparison metric. Numbers shown are **threshold-tuned**; argmax variants are in each notebook's `outputs/{model}_metrics.json`.

| Model | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC (ovr) | At ceiling? |
|---|---:|---:|---:|---:|---|
| **Stacking** (XGB + ExtraTrees + LR → LR meta) | 0.851 | **0.468** | 0.813 | 0.697 | Near ceiling — diverse base learners already average out individual biases; further gains would need more data, not more tuning. |
| **LightGBM** | 0.847 | 0.460 | 0.813 | **0.728** | Near ceiling — Optuna picked params where threshold tuning offered no further win. Highest AUC means it ranks Fatal best; the remaining gap is the 158-sample Fatal floor. |
| **MLP** (entity embeddings) | 0.792 | 0.430 | 0.782 | 0.651 | Some headroom — small dataset (8.6k train rows) under-feeds the embeddings. Bigger Optuna budget or wider embeddings could add a little; fundamentally data-bound. |
| **TabNet** | 0.791 | 0.405 | 0.771 | 0.651 | Not at potential — TabNet's sparse-attention masks are designed for 100k+ row tabular data (Rafe et al. trained on a much larger crash dataset). With 8.6k training rows the architecture under-fits. |
| **Logistic Regression** | 0.517 | 0.351 | 0.586 | 0.591 | At ceiling for the family — argmax macro-F1 is only 0.253, so threshold tuning is doing all the lifting. ROC-AUC ≈ 0.59 says the features are barely linearly separable; a non-linear model is required. |

**Shared ceiling**: the Fatal class has only 158 samples (1.3% of the data), leaving each model ~24 Fatal rows in the test fold. Recall on Fatal is information-bounded; no architecture change will push macro-F1 dramatically higher without more Fatal examples (or a cost-sensitive loss tuned for that one class).

## Reproducing results

The numbers in `outputs/` and `models/` were produced on this hardware:

- CPU: Intel i5-14400F
- RAM: 32 GB
- GPU: NVIDIA RTX 5060 (8 GB VRAM)
- OS: Ubuntu 24.04 LTS
- Python: 3.12.12

GPU is used by TabNet, MLP (both PyTorch), and XGBoost (inside stacking). LightGBM, Logistic Regression, and the ExtraTrees / LR components of stacking are CPU-only. Expect different wall-clock times on other hardware; metrics should match within Optuna seed variance.

## License

