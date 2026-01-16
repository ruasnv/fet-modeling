# src/data/loader.py

import pandas as pd
import os


class DataLoader:
    def __init__(self, raw_data_path):
        """
        Initializes the DataLoader with the path to the raw data.
        """
        self.raw_data_path = raw_data_path

    def _clean_column_names(self, df):
        """
        Cleans column names by removing leading/trailing whitespace.
        """
        # Strip whitespace from all column names
        df.columns = df.columns.str.strip()
        return df

    def load_raw_data(self):
        """
        Loads raw data from the specified CSV path.
        Assumes 'vg', 'vd', 'id', 'vb', 'w', 'l', 'temp' columns exist.

        Returns:
            pandas.DataFrame: The loaded raw DataFrame, or None if an error occurs.
        """
        print(f"Loading raw data from {self.raw_data_path}...")
        if not os.path.exists(self.raw_data_path):
            print(f"Error: Raw data file not found at {self.raw_data_path}.")
            return None

        try:
            df = pd.read_csv(self.raw_data_path)

            # --- Ensure: Clean column names immediately after loading ---
            df = self._clean_column_names(df)

            print(f"Raw data loaded successfully. {len(df)} rows found.")
            return df
        except Exception as e:
            print(f"Error loading raw data: {e}")
            return None
