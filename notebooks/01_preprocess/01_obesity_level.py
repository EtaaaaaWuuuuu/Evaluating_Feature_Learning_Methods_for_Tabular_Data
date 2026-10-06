# %%
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from benchmark.config import DATA_PROCESSED, DATA_RAW, DATASETS, RANDOM_STATE, TEST_SIZE

cfg = DATASETS["obesity_level"]
raw = DATA_RAW / "obesity_level" / "obesity_level.csv"
processed = DATA_PROCESSED / "obesity_level"

TARGET_COLUMN = cfg["target_col"]

# %%
df = pd.read_csv(raw)
df.head()

# %%
df.info()

# %%
df.describe()

# %%
# tidy column that have comma coz SMOTE
# snap them to the nearest integer

df["FCVC"] = np.floor(df["FCVC"] + 0.5).astype(int)
df["CH2O"] = np.floor(df["CH2O"] + 0.5).astype(int)
df["FAF"] = np.floor(df["FAF"] + 0.5).astype(int)
df["TUE"] = np.floor(df["TUE"] + 0.5).astype(int)

df["Age"] = np.floor(df["Age"] + 0.5).astype(int)
df["NCP"] = np.floor(df["NCP"] + 0.5).astype(int)

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

num_cols = cfg.get("num_cols", [])
print("\nnum_cols")
print(num_cols)

# %%
print("Categorical Values")
for col in cat_cols:
    print(f"{col}: {np.unique(df[col])}")

print("\nBinary Values")
for col in bin_cols:
    print(f"{col}: {np.unique(df[col])}")

# %%
df[cat_cols].head()

# %%
df[bin_cols].head()

# %%
df["NObeyesdad"].value_counts()

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
# %%
