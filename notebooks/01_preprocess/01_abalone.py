# %%
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from benchmark.config import DATA_PROCESSED, DATA_RAW, DATASETS, RANDOM_STATE, TEST_SIZE

cfg = DATASETS["abalone"]
raw = DATA_RAW / "abalone" / "abalone.csv"
processed = DATA_PROCESSED / "abalone"

TARGET_COLUMN = cfg["target_col"]

# %%
df = pd.read_csv(raw)
df.head()

# %%
df.info()

# %%
df.describe()

# %%
print("target_col")
print(TARGET_COLUMN)

cat_cols = cfg.get("cat_cols", [])
print("\ncat_cols")
print(cat_cols)

num_cols = (
    df.drop(columns=[TARGET_COLUMN] + cat_cols)
    .select_dtypes(include=["int64", "float64"])
    .columns.to_list()
)

print("\nnum_cols")
print(num_cols)

# %%
print(df["sex"].nunique())
print(df["sex"].unique())


# %%
train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

print(len(train_df))
print(len(test_df))

# %%
metadata = {
    "num_cols": num_cols,
    "cat_cols": cat_cols,
    "target_col": TARGET_COLUMN,
}

with open(processed / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

train_df.to_csv(processed / "train.csv", index=False)
test_df.to_csv(processed / "test.csv", index=False)
# %%
