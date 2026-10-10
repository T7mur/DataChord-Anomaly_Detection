# -*- coding: utf-8 -*-
"""ETL 6.2 - Extract Year/Month/Day/Hour/Minute/Second (RECREATED; function copied from the report screenshot)."""
import re
import sys
import pandas as pd


def find_timestamp_columns(df):
    cols = []
    for col in df.columns:
        sample_values = df[col].dropna().astype(str).head(100)
        if any(re.search(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', val) or re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{4}', val)
               or any(char in val for char in ":") for val in sample_values):
            cols.append(col)
    return cols


# Function to extract date components and save them in the existing file
def extract_date_components(df, timestamp_columns):
    for col in timestamp_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')  # Convert to datetime

        df[f"{col}_Year"] = df[col].dt.year
        df[f"{col}_Month"] = df[col].dt.month
        df[f"{col}_Day"] = df[col].dt.day
        df[f"{col}_Hour"] = df[col].dt.hour
        df[f"{col}_Minute"] = df[col].dt.minute
        df[f"{col}_Second"] = df[col].dt.second
    return df


def main(files):
    for path in files:
        df = pd.read_csv(path)
        cols = find_timestamp_columns(df)
        df = extract_date_components(df, cols)
        df.to_csv(path, index=False)
        print(f"{path}: added components for {cols}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["firewall_logs.csv", "ids_logs.csv", "user_behavior_analytics.csv"])
