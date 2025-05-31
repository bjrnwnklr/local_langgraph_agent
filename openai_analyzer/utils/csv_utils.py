"""
csv_utils.py - Functions for reading from and writing to CSV files, with validation.
"""

import pandas as pd

REQUIRED_COLUMNS = ["user location", "user query", "answer", "csat"]


def load_csv(file_path):
    """
    Read the CSV file into a pandas DataFrame and validate required columns.
    :param file_path: Path to the CSV file.
    :return: pandas DataFrame with the CSV contents.
    :raises: ValueError if required columns are missing.
    """
    # Read CSV (assume default encoding and comma delimiter; adjust if needed)
    df = pd.read_csv(file_path)
    # Validate that all required columns are present (case-sensitive match)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input CSV is missing required columns: {missing_cols}")
    return df


def save_csv(df, file_path):
    """
    Save the DataFrame to a CSV file.
    :param df: pandas DataFrame to save.
    :param file_path: Destination file path.
    """
    df.to_csv(file_path, index=False)
