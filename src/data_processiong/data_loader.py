# src/data_processing/data_loader.py

import pandas as pd
import os


class DataLoader:
    def __init__(self, raw_data_path):
        """
        Initializes the DataLoader with the path to the raw data.
        """
        self.raw_data_path = raw_data_path

    def load_raw_data(self):
        """
        Loads raw data from the specified CSV path.
        Assumes 'vg', 'vd', 'id', 'vb', 'w', 'l', 'temp' columns exist.
        Adds 'vgs' and 'vds' columns, assuming Vs=0 (source grounded).

        Returns:
            pandas.DataFrame: The loaded raw DataFrame, or None if an error occurs.
        """
        print(f"Loading raw data from {self.raw_data_path}...")
        if not os.path.exists(self.raw_data_path):
            print(f"Error: Raw data file not found at {self.raw_data_path}.")
            return None

        try:
            df = pd.read_csv(self.raw_data_path)
            # Assuming Vs = 0 (Source terminal is grounded)
            df['vgs'] = df['vg']
            df['vds'] = df['vd']
            print(f"Raw data loaded successfully. {len(df)} rows found.")
            return df
        except Exception as e:
            print(f"Error loading raw data: {e}")
            return None