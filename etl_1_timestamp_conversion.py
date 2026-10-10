# -*- coding: utf-8 -*-
"""ETL 6.1 - Timestamp conversion (RECREATED; function bodies copied from the report screenshot).
Standardises raw timestamp strings / epoch values into datetimes (YYYY-MM-DD HH:MM:SS)."""
import sys
import re
import pandas as pd
from dateutil import parser


# Define function to standardize timestamps
def standardize_timestamp(value):
    try:
        # Convert epoch timestamps
        if isinstance(value, (int, float)) and len(str(int(value))) >= 10:
            return pd.to_datetime(value, unit='s')
        # Convert using dateutil.parser
        return parser.parse(str(value))
    except (ValueError, TypeError):
        return None  # Return None if conversion fails


# Function to detect timestamp variations
def detect_timestamp_variations(values):
    detected_formats = set()
    for value in values:
        try:
            parsed_date = parser.parse(value, fuzzy=True)
            detected_formats.add(parsed_date.strftime('%Y-%m-%d %H:%M:%S'))
        except:
            continue
    return detected_formats


def find_timestamp_columns(df):
    cols = []
    for col in df.columns:
        sample_values = df[col].dropna().astype(str).head(100)
        if any(re.search(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', val) or re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{4}', val)
               or any(char in val for char in ":") for val in sample_values):
            cols.append(col)
    return cols


def main(files):
    for path in files:
        df = pd.read_csv(path)
        for col in find_timestamp_columns(df):
            df[col] = df[col].apply(standardize_timestamp)
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
            print(f"{path}: standardised column '{col}'")
        df.to_csv(path, index=False)          # saved back into the existing file


if __name__ == "__main__":
    main(sys.argv[1:] or ["firewall_logs.csv", "ids_logs.csv", "user_behavior_analytics.csv"])
