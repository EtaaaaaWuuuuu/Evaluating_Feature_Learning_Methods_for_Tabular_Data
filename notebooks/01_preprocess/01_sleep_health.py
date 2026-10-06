# %%
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from benchmark.config import DATA_PROCESSED, DATA_RAW, DATASETS, RANDOM_STATE, TEST_SIZE

cfg = DATASETS["sleep_health"]
raw = DATA_RAW / "sleep_health" / "sleep_health.csv"
processed = DATA_PROCESSED / "sleep_health"

TARGET_COLUMN = cfg["target_col"]
N_SAMPLE = 20000

# %%
df = pd.read_csv(raw)
df.drop(columns=["person_id"], inplace=True)

df = df.sample(n=N_SAMPLE, ignore_index=True, random_state=RANDOM_STATE)

df.drop(columns=["cognitive_performance_score", "felt_rested"], inplace=True)
df.head()

# %%
df.info()


# %%
df.describe()


# %%
df["rem_hrs"] = df["sleep_duration_hrs"] * df["rem_percentage"] / 100
df["deep_sleep_hrs"] = df["sleep_duration_hrs"] * df["deep_sleep_percentage"] / 100

df["light_sleep_hrs"] = df["sleep_duration_hrs"] - (
    df["rem_hrs"] + df["deep_sleep_hrs"]
)

df["wake_rate"] = df["wake_episodes_per_night"] / df["sleep_duration_hrs"]

# %%
df.head()


# %%
df.describe()


# %%
print("target_col")
print(TARGET_COLUMN)

cat_cols = cfg.get("cat_cols", [])

print("\ncat_cols")
print(cat_cols)

bin_cols = cfg.get("bin_cols", [])

print("\nbin_cols")
print(bin_cols)

num_cols = (
    df.drop(columns=[TARGET_COLUMN] + cat_cols + bin_cols)
    .select_dtypes(include=["int64", "float64"])
    .columns.to_list()
)

print("\nnum_cols")
print(num_cols)

# %%
df[cat_cols].head()

# %%
df["shift_work"].value_counts()


# %%
train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df[TARGET_COLUMN],
)

print(len(train_df))
print(len(test_df))


# %%
metadata = {
    "num_cols": num_cols,
    "cat_cols": cat_cols,
    "bin_cols": bin_cols,
    "target_col": TARGET_COLUMN,
}

with open(processed / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

train_df.to_csv(processed / "train.csv", index=False)
test_df.to_csv(processed / "test.csv", index=False)
