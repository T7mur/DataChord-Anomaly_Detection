# -*- coding: utf-8 -*-
"""ETL 6.3 - Weekday / Weekend classification (RECREATED; functions copied from the report screenshot)."""
import re
import sys
import pandas as pd


# Define function to standardize timestamps without modifying the format
def detect_and_classify_timestamps(df):
    timestamp_columns = []

    for col in df.columns:
        sample_values = df[col].dropna().astype(str).head(100)

        # Identify timestamp columns based on content patterns
        if any(re.search(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', val) or re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{4}', val) or any(char in val for char in ":") for val in sample_values):
            timestamp_columns.append(col)

    if timestamp_columns:
        print(f"  -> Detected timestamp columns: {timestamp_columns}")

    return timestamp_columns


# Function to add weekday/weekend classification while preserving format
def add_weekday_weekend(df, timestamp_columns):
    for col in timestamp_columns:
        try:
            # Convert to datetime while handling errors
            df[col] = pd.to_datetime(df[col], errors='coerce')

            # Preserve original format
            original_format = df[col].dt.strftime('%Y-%m-%d %H:%M:%S') if df[col].dtype == 'datetime64[ns]' else df[col]

            # Add new classification column
            df[f"{col}_Weekday_Weekend"] = df[col].dt.dayofweek.apply(lambda x: "Weekend" if x >= 5 else "Weekday")

            # Restore original format
            df[col] = original_format
        except Exception as e:
            print(f"  -> Error processing {col}: {e}")

    return df


def main(files):
    for path in files:
        df = pd.read_csv(path)
        cols = detect_and_classify_timestamps(df)
        df = add_weekday_weekend(df, cols)
        df.to_csv(path, index=False)


if __name__ == "__main__":
    main(sys.argv[1:] or ["firewall_logs.csv", "ids_logs.csv", "user_behavior_analytics.csv"])
