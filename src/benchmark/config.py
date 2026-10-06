from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

RESULTS = ROOT / "results"
RESULTS_BENCHMARK = RESULTS / "02_benchmarks"

RANDOM_STATE = 9417
TEST_SIZE = 0.2
NUM_FOLDS = 5
N_TRIALS = 100

DATASETS = {
    "abalone": {
        "task": "regression",
        "target_col": "rings",
        "cat_cols": ["sex"],
    },
    "diamonds": {
        "task": "regression",
        "target_col": "price",
        "cat_cols": ["cut", "color", "clarity"],
    },
    "obesity_level": {
        "task": "classification",
        "target_col": "NObeyesdad",
        "num_cols": ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"],
        "cat_cols": ["CAEC", "CALC", "MTRANS"],
        "bin_cols": [
            "Gender",
            "family_history_with_overweight",
            "FAVC",
            "SMOKE",
            "SCC",
        ],
    },
    "sleep_health": {
        "task": "classification",
        "target_col": "sleep_disorder_risk",
        "cat_cols": [
            "gender",
            "occupation",
            "country",
            "chronotype",
            "mental_health_condition",
            "season",
            "day_type",
        ],
        "bin_cols": ["sleep_aid_used", "shift_work"],
    },
    "superconductor": {
        "task": "regression",
        "target_col": "critical_temp",
    },
}
