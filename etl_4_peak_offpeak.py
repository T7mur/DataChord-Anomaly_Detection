# -*- coding: utf-8 -*-
"""ETL 6.4 - Peak / Off-Peak classification (RECREATED; functions copied from the report screenshot).
Peak = Mon-Fri, 07:00-19:00."""
import re
import sys
import pandas as pd


# Function to detect timestamp columns dynamically
def detect_timestamp_columns(df):
    timestamp_columns = []

    for col in df.columns:
        sample_values = df[col].dropna().astype(str).head(100)

        # Identify timestamp columns based on content patterns
        if any(re.search(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', val) or re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{4}', val) or any(char in val for char in ":") for val in sample_values):
            timestamp_columns.append(col)

    if timestamp_columns:
        print(f"  -> Detected timestamp columns: {timestamp_columns}")

    return timestamp_columns


# Function to classify Peak vs. Off-Peak hours without modifying timestamps
def add_peak_offpeak(df, timestamp_columns, peak_hours=(7, 19), peak_days=[0, 1, 2, 3, 4]):
    for col in timestamp_columns:
        try:
            # Convert to datetime while handling errors
            df[col] = pd.to_datetime(df[col], errors='coerce')

            # Preserve original format
            original_format = df[col].dt.strftime('%Y-%m-%d %H:%M:%S') if df[col].dtype == 'datetime64[ns]' else df[col]

            # Add Peak/Off-Peak classification
            df[f"{col}_Peak_OffPeak"] = df[col].apply(
                lambda x: "Peak" if (pd.notna(x) and x.dayofweek in peak_days and peak_hours[0] <= x.hour < peak_hours[1]) else "Off-Peak"
            )

            # Restore original format
            df[col] = original_format
        except Exception as e:
            print(f"  -> Error processing {col}: {e}")

    return df


def main(files):
    for path in files:
        df = pd.read_csv(path)
        cols = detect_timestamp_columns(df)
        df = add_peak_offpeak(df, cols)
        df.to_csv(path, index=False)


if __name__ == "__main__":
    main(sys.argv[1:] or ["firewall_logs.csv", "ids_logs.csv", "user_behavior_analytics.csv"])
