# src/eda/analyzer.py - Core EDA functionality
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from src.config import settings
from src.utils.helpers import classify_region, calculate_vth


class EDAAnalyzer:
    """
    A class to perform a comprehensive Exploratory Data Analysis (EDA)
    on the MOSFET dataset.

    This class encapsulates all the plotting and analysis logic,
    making the main script clean and focused on orchestration.
    """

    def __init__(self, df):
        """
        Initializes the EDAAnalyzer with the raw dataset and configurations.

        Args:
            df (pd.DataFrame): The input pandas DataFrame (expected to be raw, unfiltered).
        """
        self.df = df.copy()  # Work on a copy to avoid modifying the original DataFrame
        self.output_dir = settings.get('paths.eda_output_dir')

        os.makedirs(self.output_dir, exist_ok=True)

        if settings.get("global_settings.ignore_warnings", True):
            warnings.filterwarnings('ignore')

        self.eda_temp_filter = settings.get("filtering.temperature_filter")

        # Derived quantities for EDA's internal use
        if 'vs' in self.df.columns and 'vg' in self.df.columns:
            self.df['vgs'] = self.df['vg'] - self.df['vs']
        if 'vs' in self.df.columns and 'vd' in self.df.columns:
            self.df['vds'] = self.df['vd'] - self.df['vs']
        if 'vs' in self.df.columns and 'vb' in self.df.columns:
            self.df['vbs'] = self.df['vb'] - self.df['vs']
        if 'vs' in self.df.columns and 'vb' in self.df.columns:
            self.df['vsb'] = self.df['vs'] - self.df['vb']  # For Vth calculation

        # Calculate wOverL for EDA's correlation analysis
        if 'w' in self.df.columns and 'l' in self.df.columns:
            self.df['wOverL'] = self.df['w'] / self.df['l']

        # Calculate log_Id for EDA's correlation analysis
        if 'id' in self.df.columns and settings.get('normalization.log_transform_id'):
            min_current_threshold = 1e-12
            self.df['id_clipped'] = np.clip(self.df['id'], a_min=min_current_threshold, a_max=None)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                self.df['log_Id'] = np.log10(self.df['id_clipped'])

    def run_all_eda(self):
        """
        Executes the full suite of EDA methods in a logical sequence.
        """
        print("\n--- DATASET OVERVIEW ---")
        self._data_overview()

        print("\n--- MISSING VALUES ---")
        self._missing_values()

        print("\n--- BASIC STATISTICS ---")
        self._basic_stats()

        print(f"\n--- Temperature Distribution ---")
        self._plot_temperature_distribution()  # This will show the distribution of ALL temperatures

        print("\n--- Device Size Analysis ---")
        self._analyze_device_sizes()

        print("\n--- Operating Region Distribution ---")
        self._analyze_operating_regions()  # This will use the EDA-specific temp filter

        print("\n--- Feature Correlation Matrix ---")
        self._analyze_correlations()  # This will use the EDA-specific temp filter

    def _data_overview(self):
        """Prints a high-level overview of the dataset."""
        print(f"  Rows, Columns: {self.df.shape}")
        print(f"  Memory usage: {self.df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
        print(f"  Available columns: {list(self.df.columns)}")

    def _missing_values(self):
        """Checks for and reports any missing values."""
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("  No missing values found.")

    def _basic_stats(self):
        """Prints basic descriptive statistics and the first few rows."""
        print(self.df.describe())
        print("\nFirst 5 rows:")
        print(self.df.head())

    def _plot_temperature_distribution(self):
        """Generates and saves a bar chart of temperature distribution."""
        temp_counts = self.df['temp'].value_counts().sort_index()
        print("  Counts of unique temperatures:")
        print(temp_counts.to_string())

        plt.figure()
        temp_counts.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title('Temperature Distribution (All Data)', fontsize=plt.rcParams['axes.titlesize'])
        plt.xlabel('Temperature (°C)', fontsize=plt.rcParams['axes.labelsize'])
        plt.ylabel('Count', fontsize=plt.rcParams['axes.labelsize'])
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(self.output_dir) / 'temperature_distribution.png')
        plt.close()

    def _analyze_device_sizes(self):
        """Analyzes and visualizes the distribution of device sizes."""
        device_sizes = self.df[['w', 'l']].drop_duplicates().sort_values(['l', 'w'])
        print(f"  Number of unique device sizes: {len(device_sizes)}")
        print(f"  Width range: {self.df['w'].min():.2e} to {self.df['w'].max():.2e} meters")
        print(f"  Length range: {self.df['l'].min():.2e} to {self.df['l'].max():.2e} meters")

        print(f"\n  Unique Device Sizes (W, L) in micrometers:")
        for i, (_, row) in enumerate(device_sizes.iterrows()):
            w_um = row['w'] * 1e6
            l_um = row['l'] * 1e6
            print(f"  [{i + 1:2d}] W={w_um:.2f} µm, L={l_um:.2f} µm")

        temp_counts_by_size = self.df.groupby(['w', 'l'])['temp'].nunique().reset_index()
        temp_counts_by_size.rename(columns={'temp': 'temp_count'}, inplace=True)

        device_data_counts_by_size = self.df.groupby(['w', 'l']).size().reset_index()
        device_data_counts_by_size.rename(columns={0: 'data_point_count'}, inplace=True) # Default column name for .size().reset_index() is 0

        temp_pivot = temp_counts_by_size.pivot(index='l', columns='w', values='temp_count').fillna(0)
        data_count_pivot = device_data_counts_by_size.pivot(index='l', columns='w', values='data_point_count').fillna(0)

        temp_pivot.columns = [f"{col * 1e6:.2f} µm" for col in temp_pivot.columns]
        temp_pivot.index = [f"{idx * 1e6:.2f} µm" for idx in temp_pivot.index]
        data_count_pivot.columns = data_count_pivot.columns.astype(float)
        data_count_pivot.index = data_count_pivot.index.astype(float)
        data_count_pivot.columns = [f"{col * 1e6:.2f} µm" for col in data_count_pivot.columns]
        data_count_pivot.index = [f"{idx * 1e6:.2f} µm" for idx in data_count_pivot.index]


        plt.figure(figsize=(10, 7))
        sns.heatmap(temp_pivot, annot=True, cmap='viridis', fmt='.0f', linewidths=.5)
        plt.title('Unique Temperature Measurements by Device Size (W, L)', fontsize=plt.rcParams['axes.titlesize'])
        plt.xlabel('Width (W)', fontsize=plt.rcParams['axes.labelsize'])
        plt.ylabel('Length (L)', fontsize=plt.rcParams['axes.labelsize'])
        plt.tight_layout()
        plt.savefig(Path(self.output_dir) / 'temp_counts_by_device_size.png')
        plt.close()

        plt.figure(figsize=(10, 7))
        sns.heatmap(data_count_pivot, annot=True, cmap='cividis', fmt='.0f', linewidths=.5)
        plt.title('Total Data Points by Device Size (W, L)', fontsize=plt.rcParams['axes.titlesize'])
        plt.xlabel('Width (W)', fontsize=plt.rcParams['axes.labelsize'])
        plt.ylabel('Length (L)', fontsize=plt.rcParams['axes.labelsize'])
        plt.tight_layout()
        plt.savefig(Path(self.output_dir) / 'data_counts_by_device_size.png')
        plt.close()

    def _analyze_operating_regions(self):
        """Analyzes and plots the distribution of operating regions for a specific temperature."""
        # Filter for the specific temperature for this plot
        filtered_df_for_regions = self.df[
            (self.df['id'] > 0) &
            (self.df['vds'] > 0) &  # Use vds here, as it's calculated in __init__
            (np.isclose(self.df['temp'], self.eda_temp_filter))
            ].copy()

        if filtered_df_for_regions.empty:
            print(f"  No data found for region classification at {self.eda_temp_filter}°C after filtering.")
            return

        # Ensure vth is calculated for this filtered subset if not already present
        if 'vth' not in filtered_df_for_regions.columns:
            # Re-calculate vsb if it was not already in the df or if df was filtered
            if 'vsb' not in filtered_df_for_regions.columns:
                filtered_df_for_regions['vsb'] = filtered_df_for_regions['vs'] - filtered_df_for_regions['vb']

            vth_params = settings.get('vth_params')
            filtered_df_for_regions['vth'] = filtered_df_for_regions['vsb'].apply(
                lambda x: calculate_vth(x,
                                        vth0=vth_params['vth0'],
                                        gamma=vth_params['gamma'],
                                        phi_f=vth_params['phi_f']))
            print(f"  - Calculated 'vth' dynamically for region analysis at {self.eda_temp_filter}°C.")

        print(f"  (at {self.eda_temp_filter}°C, using dynamically calculated Vth)")

        filtered_df_for_regions['operating_region'] = filtered_df_for_regions.apply(
            lambda row: classify_region(row, vth_approx_val=row['vth']), axis=1
        )

        region_counts = filtered_df_for_regions['operating_region'].value_counts().reindex(
            ['Cut-off', 'Linear', 'Saturation']
        ).fillna(0)

        print("  Distribution of samples per operating region:")
        print(region_counts.to_string())

        plt.figure(figsize=(8, 6))
        sns.barplot(x=region_counts.index, y=region_counts.values, palette='viridis')
        plt.title(f"Distribution of Operating Regions (at {self.eda_temp_filter}°C, Id>0, Vds>0)",
                  fontsize=plt.rcParams['axes.titlesize'])
        plt.xlabel("Operating Region", fontsize=plt.rcParams['axes.labelsize'])
        plt.ylabel("Number of Samples", fontsize=plt.rcParams['axes.labelsize'])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(self.output_dir) / 'operating_region_distribution.png')
        plt.close()

    def _analyze_correlations(self):
        """Generates and saves a heatmap of feature correlations for a specific temperature."""
        # Filter the data for a specific temperature for a cleaner correlation analysis
        filtered_df_for_corr = self.df[
            np.isclose(self.df['temp'], self.eda_temp_filter)
        ].copy()

        # Ensure engineered features exist for correlation analysis (already done in __init__)
        # but defensive re-creation if the filtered_df_for_corr somehow loses them.
        if 'wOverL' not in filtered_df_for_corr.columns:
            filtered_df_for_corr['wOverL'] = filtered_df_for_corr['w'] / filtered_df_for_corr['l']

        if 'log_Id' not in filtered_df_for_corr.columns:
            min_current_threshold = 1e-12
            filtered_df_for_corr['id_clipped'] = np.clip(filtered_df_for_corr['id'], a_min=min_current_threshold,
                                                         a_max=None)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                filtered_df_for_corr['log_Id'] = np.log10(filtered_df_for_corr['id_clipped'])

        correlation_cols = ['vg', 'vd', 'vb', 'w', 'l', 'wOverL', 'log_Id']
        # Filter to only include columns that exist in the dataframe
        correlation_cols_exist = [col for col in correlation_cols if col in filtered_df_for_corr.columns]

        if not filtered_df_for_corr.empty and len(correlation_cols_exist) > 1:
            correlation_matrix = filtered_df_for_corr[correlation_cols_exist].corr()
            print("\nCorrelation Matrix:")
            print(correlation_matrix)

            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
            plt.title(f"Feature Correlation Matrix (at {self.eda_temp_filter}°C)",
                      fontsize=plt.rcParams['axes.titlesize'])
            plt.savefig(Path(self.output_dir) / 'correlation_matrix.png')
            plt.close()
        else:
            print("  Not enough data or relevant columns to compute correlation matrix.")
