# src/eda/augmented_analyzer.py

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from src.config import settings
from src.utils.helpers import classify_region, calculate_vth


class AugmentedEDAAnalyzer:
    """
    A class to perform EDA on the GAN-augmented MOSFET dataset.

    This class is specifically designed to handle the characteristics of the
    combined original and synthetic data, which contains a single temperature.
    """

    def __init__(self, df):
        """
        Initializes the AugmentedEDAAnalyzer with the augmented dataset.

        Args:
            df (pd.DataFrame): The input augmented pandas DataFrame.
        """
        self.df = df.copy()  # Work on a copy to avoid modifying the original DataFrame
        self.output_dir = Path(settings.get('paths.eda_output_dir'))

        os.makedirs(self.output_dir, exist_ok=True)

        if settings.get("global_settings.ignore_warnings", True):
            warnings.filterwarnings('ignore')

        # Derived quantities for EDA's internal use
        if 'vs' in self.df.columns and 'vg' in self.df.columns:
            self.df['vgs'] = self.df['vg'] - self.df['vs']
        if 'vs' in self.df.columns and 'vd' in self.df.columns:
            self.df['vds'] = self.df['vd'] - self.df['vs']
        if 'vs' in self.df.columns and 'vb' in self.df.columns:
            self.df['vbs'] = self.df['vb'] - self.df['vs']
        if 'vs' in self.df.columns and 'vb' in self.df.columns:
            self.df['vsb'] = self.df['vs'] - self.df['vb']

        # Calculate wOverL and log_Id for correlation analysis
        if 'w' in self.df.columns and 'l' in self.df.columns:
            self.df['wOverL'] = self.df['w'] / self.df['l']

        if 'id' in self.df.columns and settings.get('normalization.log_transform_id'):
            min_current_threshold = 1e-12
            self.df['id_clipped'] = np.clip(self.df['id'], a_min=min_current_threshold, a_max=None)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                self.df['log_Id'] = np.log10(self.df['id_clipped'])

    def run_all_eda(self):
        """
        Executes the full suite of EDA methods.
        """
        print("\n--- DATASET OVERVIEW ---")
        self._data_overview()

        print("\n--- MISSING VALUES ---")
        self._missing_values()

        print("\n--- BASIC STATISTICS ---")
        self._basic_stats()

        print("\n--- Device Size Analysis ---")
        self._analyze_device_sizes()
        self._plot_device_size_distribution()

        print("\n--- Operating Region Distribution ---")
        self._analyze_operating_regions()
        self._analyze_operating_regions_by_reassessing_regions()

        print("\n--- Feature Correlation Matrix ---")
        self._analyze_correlations()

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

    def _plot_device_size_distribution(self):
        """
        Generates and saves a scatter plot of unique device sizes (W vs L).
        """
        print("  - Generating device size dot plot...")
        # Get unique device sizes from the dataframe
        unique_sizes_df = self.df[['w', 'l']].drop_duplicates().copy()

        # Convert to micrometers for better visualization
        unique_sizes_df['l_um'] = unique_sizes_df['l'] * 1e6
        unique_sizes_df['w_um'] = unique_sizes_df['w'] * 1e6

        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            data=unique_sizes_df,
            x='l_um',
            y='w_um',
            alpha=0.6,
            edgecolor='w',
            linewidth=0.5,
            s=50
        )
        plt.title('Distribution of Unique Device Sizes', fontsize=plt.rcParams['axes.titlesize'])
        plt.xlabel(r'Length ($L$) in $\mu m$', fontsize=plt.rcParams['axes.labelsize'])
        plt.ylabel(r'Width ($W$) in $\mu m$', fontsize=plt.rcParams['axes.labelsize'])
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'device_size_dot_plot.png')
        plt.close()
        print("  - Device size dot plot saved.")

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
        device_data_counts_by_size.rename(columns={0: 'data_point_count'}, inplace=True)

    def _analyze_operating_regions(self):
        """
        Analyzes and plots the distribution of operating regions.
        """
        dataframe = self.df

        if dataframe.empty:
            print(f"  No data found, dataset is empty")
            return

        # Check for and handle missing 'operating_region' labels
        if 'operating_region' not in dataframe.columns or dataframe[
            'operating_region'].isnull().any():
            print(
                "  'operating_region' column not found or contains missing values. Calculating for missing rows...")

            # Identify rows with missing 'operating_region'
            missing_mask = dataframe['operating_region'].isnull()

            # Re-calculate vth for the missing rows if needed
            if 'vth' not in dataframe.columns:
                print("  'vth' column not found. Calculating it for missing rows...")
                dataframe['vsb'] = dataframe['vs'] - dataframe['vb']
                vth_params = settings.get('vth_params')
                dataframe['vth'] = dataframe['vsb'].apply(
                    lambda x: calculate_vth(x,
                                            vth0=vth_params['vth0'],
                                            gamma=vth_params['gamma'],
                                            phi_f=vth_params['phi_f']))
            else:
                print("  Using existing 'vth' column.")

            # Recalculate region only for missing samples using .loc
            dataframe.loc[missing_mask, 'operating_region'] = dataframe[
                missing_mask].apply(
                lambda row: classify_region(row, vth_approx_val=row['vth']), axis=1
            )
            print(f"  Successfully calculated and filled {missing_mask.sum()} missing 'operating_region' labels.")

        else:
            print("  Using existing 'operating_region' column for analysis.")

        region_counts = dataframe['operating_region'].value_counts().reindex(
            ['Cut-off', 'Linear', 'Saturation']
        ).fillna(0)

        print("  Distribution of samples per operating region:")
        print(region_counts.to_string())

        plt.figure(figsize=(8, 6))
        sns.barplot(x=region_counts.index, y=region_counts.values, palette='viridis')
        plt.title("Distribution of Operating Regions",
                  fontsize=plt.rcParams['axes.titlesize'])
        plt.xlabel("Operating Region", fontsize=plt.rcParams['axes.labelsize'])
        plt.ylabel("Number of Samples", fontsize=plt.rcParams['axes.labelsize'])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(self.output_dir) / 'operating_region_distribution_augmented.png')
        plt.close()

    def _analyze_operating_regions_by_reassessing_regions(self):
        """
        Re-calculates and plots the distribution of operating regions
        based on dynamic Vth calculation.
        """
        # Create a copy to work with, preserving the original dataframe
        reassessed_df = self.df.copy()

        if reassessed_df.empty:
            print("  No data found, dataset is empty")
            return

        print("  - Re-calculating Vth and operating regions for the entire dataset...")

        # Calculate vsb if it's not present
        if 'vsb' not in reassessed_df.columns:
            reassessed_df['vsb'] = reassessed_df['vs'] - reassessed_df['vb']

        # Recalculate vth for all rows
        vth_params = settings.get('vth_params')
        reassessed_df['vth'] = reassessed_df['vsb'].apply(
            lambda x: calculate_vth(x,
                                    vth0=vth_params['vth0'],
                                    gamma=vth_params['gamma'],
                                    phi_f=vth_params['phi_f']))

        # Reclassify region for all rows
        reassessed_df['operating_region_reassessed'] = reassessed_df.apply(
            lambda row: classify_region(row, vth_approx_val=row['vth']), axis=1
        )

        region_counts = reassessed_df['operating_region_reassessed'].value_counts().reindex(
            ['Cut-off', 'Linear', 'Saturation']
        ).fillna(0)

        print("  Distribution of samples per re-assessed operating region:")
        print(region_counts.to_string())

        plt.figure(figsize=(8, 6))
        sns.barplot(x=region_counts.index, y=region_counts.values, palette='viridis')
        plt.title("Distribution of Re-assessed Operating Regions", fontsize=plt.rcParams['axes.titlesize'])
        plt.xlabel("Operating Region", fontsize=plt.rcParams['axes.labelsize'])
        plt.ylabel("Number of Samples", fontsize=plt.rcParams['axes.labelsize'])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(self.output_dir) / 'operating_region_distribution_reassessed.png')
        plt.close()

    def _analyze_correlations(self):
        """Generates and saves a heatmap of feature correlations."""
        filtered_df_for_corr = self.df

        # Ensure engineered features exist
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
        correlation_cols_exist = [col for col in correlation_cols if col in filtered_df_for_corr.columns]

        if not filtered_df_for_corr.empty and len(correlation_cols_exist) > 1:
            correlation_matrix = filtered_df_for_corr[correlation_cols_exist].corr()
            print("  Correlation Matrix:")
            print(correlation_matrix)

            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
            plt.title("Feature Correlation Matrix", fontsize=plt.rcParams['axes.titlesize'])
            plt.savefig(Path(self.output_dir) / 'correlation_matrix_augmented.png')
            plt.close()
        else:
            print("  Not enough data or relevant columns to compute correlation matrix.")
