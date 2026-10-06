import optuna
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

optuna.trial.Trial.suggest_distribution = lambda self, name, dist: self._suggest(
    name, dist
)

SEARCH_SPACES: dict = {
    "random_forest": {
        "n_estimators": IntDistribution(30, 120),
        "max_depth": IntDistribution(5, 12),
        "min_samples_split": IntDistribution(5, 12),
        "min_samples_leaf": IntDistribution(2, 6),
        "max_features": CategoricalDistribution(["sqrt", "log2"]),
        "bootstrap": CategoricalDistribution([True]),
    },
    "xgboost": {
        "n_estimators": IntDistribution(100, 800),
        "learning_rate": FloatDistribution(0.01, 0.2, log=True),
        "max_depth": IntDistribution(3, 10),
        "min_child_weight": IntDistribution(1, 7),
        "subsample": FloatDistribution(0.6, 1.0),
        "colsample_bytree": FloatDistribution(0.6, 1.0),
        "gamma": FloatDistribution(1e-8, 1.0, log=True),
        "reg_alpha": FloatDistribution(1e-8, 1.0, log=True),
        "reg_lambda": FloatDistribution(0.1, 10.0, log=True),
    },
    "xrfm": {
        "bandwidth": FloatDistribution(1.0, 200.0, log=True),
        "exponent": FloatDistribution(0.7, 1.4),
        "diag": CategoricalDistribution([False, True]),
        "kernel_type": CategoricalDistribution(["kpq", "k2q"]),
        "norm_p": lambda e: FloatDistribution(e, e + 0.8 * (2 - e)),
        "reg": FloatDistribution(1e-6, 1.0, log=True),
    },
}
