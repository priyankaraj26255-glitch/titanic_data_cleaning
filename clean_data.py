"""End-to-end cleaning pipeline for the Kaggle Titanic dataset.

Usage:  python clean_data.py [--raw data/raw/titanic.csv] [--out data/cleaned/titanic_cleaned.csv]
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def iqr_fences(s: pd.Series, k: float = 1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()

    # --- Step 2: missing values -------------------------------------------
    df["Has_Cabin"] = df["Cabin"].notna().astype(int)          # keep the signal, drop the sparse column
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    rare = df["Title"].value_counts()[lambda s: s < 10].index
    df["Title"] = df["Title"].replace(dict.fromkeys(rare, "Rare"))
    df["Age"] = df["Age"].fillna(df.groupby(["Title", "Pclass"])["Age"].transform("median"))
    df["Age"] = df["Age"].fillna(df["Age"].median())            # safety net for empty groups
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df = df.drop(columns=["Cabin"])

    # --- Steps 3-4: outliers (cap at IQR fences) ----------------------------
    for col in ["Age", "Fare"]:
        lo, hi = iqr_fences(df[col])
        df[col] = df[col].clip(lower=lo, upper=hi)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/titanic.csv")
    ap.add_argument("--out", default="data/cleaned/titanic_cleaned.csv")
    a = ap.parse_args()
    out = clean(pd.read_csv(a.raw))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"Saved {a.out}  shape={out.shape}  nulls={int(out.isna().sum().sum())}")
