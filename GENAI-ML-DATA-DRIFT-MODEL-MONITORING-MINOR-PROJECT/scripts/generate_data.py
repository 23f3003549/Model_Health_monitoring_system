import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# path

INPUT_FILE = "data/cleaned.csv"

REFERENCE_FILE = "data/reference.csv"
PRODUCTION_FILE = "data/production.csv"

# settings 

RANDOM_STATE = 42

REFERENCE_SIZE = 0.80

# load cleaned data

print("\n[1] Loading cleaned dataset...")
df= pd.read_csv(INPUT_FILE)
print(f"Cleaned dataset shape:{df.shape}")

# Identify target

TARGET_COLUMN = "default_payment_next_month"

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' not found."
    )

print(f"Target column: {TARGET_COLUMN}")

# split data 

print("\n[2] Splitting dataset...")

reference_df , production_df = train_test_split(
    df,
    test_size=1-REFERENCE_SIZE,
    random_state=RANDOM_STATE,
    stratify= df[TARGET_COLUMN]

)

# reset index
reference_df = reference_df.reset_index(drop=True)
production_df=production_df.reset_index(drop =True)

print(f"Reference dataset:  {reference_df.shape}")
print(f"Production dataset: {production_df.shape}")

# select features for drift 

print("\n[3] Selecting features for controlled drift...")

DRIFT_CONFIG = {
    "age": {
        "type": "shift",
        "value": 5
    },

    "limit_bal": {
        "type": "multiply",
        "value": 1.25
    },

    "bill_amt1": {
        "type": "multiply",
        "value": 1.20
    },

    "pay_amt1": {
        "type": "multiply",
        "value": 1.30
    }
}

# check features 

for feature in DRIFT_CONFIG:
    if feature not in production_df.columns:
         print(
            f"WARNING: Feature '{feature}' "
            f"not found in dataset."
        )
    else:
        print(f"Drift will be introduced into: {feature}")

# show original statistics 

print("\n[4] Statistics BEFORE drift")

for feature in DRIFT_CONFIG:
     if feature in production_df.columns:

        reference_mean = reference_df[feature].mean()
        production_mean = production_df[feature].mean()

        print(
            f"{feature:15s} | "
            f"Reference mean: {reference_mean:12.2f} | "
            f"Production mean: {production_mean:12.2f}"
        )

# introduce control drift 

print("\n[5] Introducing controlled distribution drift...")

for feature , config in DRIFT_CONFIG.items():
    if feature not in production_df.columns:
        continue

    if config["type"]=="shift":
        production_df[feature]=(
            production_df[feature]+config["value"]
        )
    elif config["type"]=="multiply":
        production_df[feature] = (
            production_df[feature]
            * config["value"]
        )


# keep value realistic 

# Age should remain realistic
if "age" in production_df.columns:

    production_df["age"] = production_df["age"].clip(
        lower=18,
        upper=80
    )

# Credit limit should not become negative
if "limit_bal" in production_df.columns:

    production_df["limit_bal"] = production_df[
        "limit_bal"
    ].clip(lower=0)


# Bill amounts should not become negative
for column in [
    "bill_amt1",
    "bill_amt2",
    "bill_amt3",
    "bill_amt4",
    "bill_amt5",
    "bill_amt6"
]:
    if column in production_df.columns:
        production_df[column]=production_df[column].clip(lower=0)

# Payment amounts should not become negative
for column in [
    "pay_amt1",
    "pay_amt2",
    "pay_amt3",
    "pay_amt4",
    "pay_amt5",
    "pay_amt6"
]:
     if column in production_df.columns:

        production_df[column] = production_df[
            column
        ].clip(lower=0)

# target column not modified 

print("\n[7] Protecting target column...")

print(
    f"Target distribution in reference:\n"
    f"{reference_df[TARGET_COLUMN].value_counts()}"
)

print(
    f"Target distribution in production:\n"
    f"{production_df[TARGET_COLUMN].value_counts()}"
)

# final statistics 

print("\n[8] Statistics AFTER drift")

for feature in DRIFT_CONFIG:

    if feature in production_df.columns:

        reference_mean = reference_df[feature].mean()
        production_mean = production_df[feature].mean()

        print(
            f"{feature:15s} | "
            f"Reference mean: {reference_mean:12.2f} | "
            f"Production mean: {production_mean:12.2f}"
        )

# save datasets

print("\n[9] Saving datasets...")

os.makedirs("data", exist_ok=True)

reference_df.to_csv(
    REFERENCE_FILE,
    index=False
)

production_df.to_csv(
    PRODUCTION_FILE,
    index=False
)

print(f"Reference dataset saved to: {REFERENCE_FILE}")
print(f"Production dataset saved to: {PRODUCTION_FILE}")

# summary 
print("\nDataset sizes:")
print(f"Reference:  {len(reference_df)} rows")
print(f"Production: {len(production_df)} rows")