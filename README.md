# 9417 xRFM Project

## Team: One Night Miracle

| Name (sorted alphabetically)       | zID      |
| ---------------------------------- | -------- |
| Alan Wang                          | z5696241 |
| Chalat Phumphiraratthaya (Charlie) | z5623543 |
| Chih-Yueh Liu (Jeff)               | z5631242 |
| Enjia Wu (Eta)                     | z5456801 |
| Isaac Dwadattusyah Haikal Azziz    | z5640942 |

---

## Quick Start

### Initiate project & Install libraries (using pip)

```bash
make init
```

---

## How to reproduce the results

**1. Clone the repo and install dependencies**

```bash
git clone <repo-url>
cd 9417-xrfm-project
make init
```

**2. Preprocess datasets**

Run each preprocessing notebook to process raw datasets under `data/raw/`.

```
notebooks/01_preprocess/01_abalone.py
notebooks/01_preprocess/01_diamonds.py
notebooks/01_preprocess/01_superconductor.py
notebooks/01_preprocess/01_obesity_level.py
notebooks/01_preprocess/01_sleep_health.py
```

Each notebook generates train/test splits and metadata under `data/processed/`:

**3. Run benchmarking**

Each model has one notebook per dataset under `notebooks/02_benchmarks/` (except for TabTFN). Run the notebooks for each model you want to evaluate:

```
notebooks/02_benchmarks/xrfm/
notebooks/02_benchmarks/random_forest/
notebooks/02_benchmarks/xgboost/
notebooks/02_benchmarks/tabpfn/
```

All results (JSON file and figures) are saved to `results/02_benchmarks/<dataset>/`.

**4. Run interpretability analysis**

```
notebooks/03_interpretability/03_interpretability.py
```

All results (JSON file and figures) are saved to `results/03_interpretability/`.

**5. Run learning curve experiments**

Subsamples the training set at several sizes and plots test performance and training time versus n for all models:

```
notebooks/04_learning_curve/04_learning_curve.py
```

All results (figures and tables) are saved to `results/04_learning_curve/`.

---

## Folder Structure

```
9417-xrfm-project/
├── data/
│   ├── DATA.md                          # data dictionary
│   ├── raw/                             # unprocessed datasets (original)
│   │   ├── abalone/
│   │   ├── diamonds/
│   │   ├── superconductor/
│   │   ├── obesity_level/
│   │   └── sleep_health/
│   └── processed/                       # processed datasets
│       ├── abalone/
│       ├── diamonds/
│       ├── superconductor/
│       ├── obesity_level/
│       └── sleep_health/
│
├── notebooks/
│   ├── 01_preprocess/                   # data preprocessing (one script per dataset)
│   │   ├── 01_abalone.py
│   │   ├── 01_diamonds.py
│   │   ├── 01_superconductor.py
│   │   ├── 01_obesity_level.py
│   │   └── 01_sleep_health.py
│   ├── 02_benchmarks/                   # model benchmarking (one notebook per model x dataset)
│   │   ├── xrfm/
│   │   ├── random_forest/
│   │   ├── xgboost/
│   │   └── tabpfn/
│   ├── 03_interpretability/
│   │   └── 03_interpretability.py       # interpretability analysis
│   └── 04_learning_curve/
│       ├── 04_learning_curve.py         # learning curve experiments
│       └── subsample_plot.ipynb
│
├── results/
│   ├── 02_benchmarks/                   # benchmark outputs per dataset
│   │   ├── abalone/
│   │   ├── diamonds/
│   │   ├── superconductor/
│   │   ├── obesity_level/
│   │   └── sleep_health/
│   ├── 03_interpretability/             # interpretability figures and tables
│   └── 04_learning_curve/               # learning curve figures and JSON
│
├── src/
│   └── benchmark/                       # local library named `benchmark`
│       ├── models/
│       │   ├── search_spaces.py         # Optuna hyperparameter search spaces
│       │   ├── random_forest_model.py   # Random Forest model wrapper
│       │   ├── xgboost_model.py         # XGBoost model wrapper
│       │   ├── xrfm_model.py            # xRFM model wrapper
│       │   └── tabpfn_model.py          # TabPFN model wrapper
│       ├── config.py                    # shared configs and paths
│       └── metrics.py                   # evaluation metrics
│
├── makefile                             # project commands
├── pyproject.toml                       # package metadata and dependencies
└── README.md
```

---

## Resources

| Resource | Description | Link |
| -------- | ----------- | ---- |
| Data Dictionary | Dataset descriptions and feature definitions for all 5 datasets | [DATA.md](data/DATA.md) |
| Saved Models | Pre-trained model artifacts (not committed; place under `models/<dataset>/`) | [OneDrive](https://unsw-my.sharepoint.com/:f:/g/personal/z5640942_ad_unsw_edu_au/IgAaWkN4UptgRrku5dDH6pYSAYOFHNRWxHAppZo7_y_fg_g?e=2sSK8u&xsdata=MDV8MDJ8fGQ2NGJjNGZmNDg4NzRhZmU1YzMzMDhkZWEwYzRkMmYxfDNmZjZjZmE0ZTcxNTQ4ZGJiOGUxMDg2N2I5ZjlmYmEzfDB8MHw2MzkxMjQ5NjQzNDYyNzY1Nzl8VW5rbm93bnxWR1ZoYlhOVFpXTjFjbWwwZVZObGNuWnBZMlY4ZXlKRFFTSTZJbFJsWVcxelgwRlVVRk5sY25acFkyVmZVMUJQVEU5R0lpd2lWaUk2SWpBdU1DNHdNREF3SWl3aVVDSTZJbGRwYmpNeUlpd2lRVTRpT2lKUGRHaGxjaUlzSWxkVUlqb3hNWDA5fDF8TDJOb1lYUnpMekU1T21VMk9ERTJPR1JsWXpNek5UUXlOakpoWkRreU16SXpZVEUwTVdRMVlUTmxRSFJvY21WaFpDNTJNaTl0WlhOellXZGxjeTh4TnpjMk9EazVOak15TkRBeXwxYzBiZDYxNzFkY2Q0YzE4ZWNlNjA4ZGVhMGM0ZDJmMHxmZWMxZTk1ZGM3MDU0NGE5OTU5NTA0N2ExMmU3YzA1Mw%3D%3D&sdata=V3oxL00xUzJWU3lqdmEzUlc1c3NTTUlhM1VSOHY5eFMraVRhVWJGNEtGWT0%3D&ovuser=3ff6cfa4-e715-48db-b8e1-0867b9f9fba3%2Cz5623543%40ad.unsw.edu.au) |
