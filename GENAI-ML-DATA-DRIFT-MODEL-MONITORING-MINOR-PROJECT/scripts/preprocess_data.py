import os
import pandas as pd 
import numpy as np
 
# PATH
RAW_FILE= "data/raw/default of credit card clients.xls"
OUTPUT_FILE="data/cleaned.csv"

# LOAD DATA

print("\n [1] loading raw datasets....")

df= pd.read_excel(RAW_FILE,header=1)

print (f"Raw dataset shape:{df.shape}")

# CLEAN COLUMN NAMES 

print("\n[2] Cleaning column names...")

df.columns=( df.columns.astype(str).str.strip().str.lower().str.replace(" ","_").str.replace("/","_").str.replace("-","_"))

print("Columns:")
print(df.columns.tolist())

# REMOVE UNNAME AND EMPTYCOLUMNS

print("\n[3] Removing unnecessary columns...")

unnammed_columns = [ col for col in df.columns
                    if "unnamed"in col.lower()]
if unnammed_columns:
    print("Removing:", unnammed_columns)
    df=df.drop(columns=unnammed_columns)

# REMOVE DUPLICATES

print("\n[4] Checking duplicate records...")

duplicate_count=df.duplicated().sum()

print(f" duplicated rows found:{duplicate_count}")

if duplicate_count>0:
    df= df.drop_duplicates()
    print(f"Removed{duplicate_count} duplicate rows.")

#     IDENTIFY TARGET COLUMN

print("\n[5] Identifying target column...")

target_candidates = [
    "default_payment_next_month",
    "default_payment_next_month_",
    "default_payment_next_month__"
]

target_column = None

for column in target_candidates:
    if column in df.columns:
        target_column = column
        break

if target_column is None:
    # Fallback: search for a column containing "default"
    for column in df.columns:
        if "default" in column.lower():
            target_column=column
            break


if target_column is None:
    raise ValueError(
        "Target column could not be identified. "
        "Check the printed column names."
    )

print(f"Target column: {target_column}")   

# REMOVE ID COLUMN
print("\n[6] Handling ID column...")

if "id" in df.columns:
    print("Removing ID column...")
    df= df.drop(columns=["id"])

# HANDLE MISSING VALUES
    
print("\n[7] Checking missing values...")

missing_values =df.isnull().sum()
total_missing=missing_values.sum()
print(f"Total missing values: {total_missing}")

if total_missing>0:
    print("\n Missing values by column:")
    print(missing_values[missing_values>0])

    # Numerical columns → median
    numerical_columns =df.select_dtypes(include=[np.number]).columns
    for column in numerical_columns:
        if df[column].isnull().sum()>0:
            df[column]=df[column].fillna(df[column].median())


    # categorical columns to mode 

    categorical_columns = df.select_dtypes(
        exclude=[np.number]
    ).columns  
    for column in categorical_columns:

        if df[column].isnull().sum() > 0:
            df[column] = df[column].fillna(
                df[column].mode()[0]
            ) 
    print ("Missing values handled.")
else:
    print("No missing value found.")    

# coonvert numeric columns 

print("\n[8] Converting numeric columns...")

for column in df.columns:
    if column!=target_column:
        df[column]=pd.to_numeric(df[column], errors= "coerce")

# convert target to numeric 
df[target_column]=pd.to_numeric(
    df[target_column],
    errors="coerce"
)

#  HANDLE INVALID VALUES

print("\n[9] Checking invalid values...")

# Remove rows where target is missing
before = len(df)

df = df.dropna(
    subset=[target_column]
)

after = len(df)

if before != after:
    print(f"Removed {before - after} rows with missing target.")

# HANDLE OUTLIERS

print("\n[10] Checking extreme values...")

numerical_columns = df.select_dtypes(
    include=[np.number]
).columns

for column in numerical_columns:
    q1 = df[column].quantile(0.25)
    q3 =  df[column].quantile(0.75)
    iqr = q3-q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = (
        (df[column] < lower) |
        (df[column] > upper)
    ).sum()
    if outliers > 0:
        print(
            f"{column}: {outliers} potential outliers"
        )

#  RESET INDEX

df = df.reset_index(drop=True)


#  save clean dataset

print("\n[11] Saving cleaned dataset...")

os.makedirs(
    "data",
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"Saved to: {OUTPUT_FILE}")

#  FINAL DATASET INFORMATION

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)

print(f"\nFinal shape: {df.shape}")

print("\nFinal columns:")
for column in df.columns:
    print(f"  - {column}")

print("\nTarget:")
print(f"  {target_column}")

print("\nTarget distribution:")
print(df[target_column].value_counts())

print("\nMissing values after preprocessing:")
print(df.isnull().sum().sum())

print("\nData types:")
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head())

print("\n✓ Clean dataset created successfully.")