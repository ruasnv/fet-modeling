# scripts/eda.py - Exploratory Data Analysis

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting (e.g., on servers)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import sys  # For stdout redirection
from datetime import datetime

# Import custom helpers and data preprocessor
from src.utils.helpers import load_config, setup_environment, calculate_vth, classify_region


# Potentially from data preprocessor if specific filtering logic needed, but for EDA,
# direct pandas operations are often fine.

def run_eda():
    """
    Executes the Exploratory Data Analysis (EDA) process for the MOSFET dataset.
    This script loads raw data, performs initial data inspection, analyzes
    temperature and device size distributions, classifies operating regions,
    and visualizes feature correlations. All outputs (log and plots) are saved
    to a configured directory.
    """
    print("--- Starting EDA Script ---")

    # --- Load Configuration ---
    # Load main configuration for paths and global settings
    main_config_path = 'config/main_config.yaml'
    data_config_path = 'config/data_config.yaml'

    main_config = load_config(main_config_path)
    data_config = load_config(data_config_path)

    # Validate config loading
    if not main_config or not data_config:
        print("Failed to load configurations. Exiting EDA.")
        return

    # Extract paths and settings from configs
    RAW_DATA_PATH = main_config['paths']['raw_data_path']
    EDA_OUTPUT_DIR = main_config['paths']['eda_output_dir']

    # Ensure raw data path is correctly relative if needed, or absolute
    # Assuming RAW_DATA_PATH is correctly defined relative to project root or absolute
    # if RAW_DATA_PATH does not exist, check if '../data/nmoshv.csv' exists relative to this script
    if not os.path.exists(RAW_DATA_PATH):
        # Fallback to a common relative path for scripts
        alternative_raw_data_path = os.path.join(os.path.dirname(__file__), '..', RAW_DATA_PATH)
        if os.path.exists(alternative_raw_data_path):
            RAW_DATA_PATH = alternative_raw_data_path
        else:
            print(f"Error: Raw data file not found at expected path: {RAW_DATA_PATH}")
            print(f"Also tried: {alternative_raw_data_path}")
            print("Please ensure 'nmoshv.csv' is in your 'data/' directory.")
            return

    # Get specific EDA filter settings from data_config (or an 'eda_config.yaml' if you prefer)
    TEMP_FILTER_EDA = data_config['data_filtering']['temperature_filter']
    # If using a vth_eda, it could also come from a config or be determined dynamically later
    VTH_APPROX_EDA = data_config['vth_calculation']['vth0_approx']  # Using vth0 as an approximation for EDA

    # --- Setup Environment (Matplotlib style, output directories, warnings) ---
    setup_environment(main_config_path)  # Pass main config path for global settings
    # The `setup_environment` function already creates `EDA_OUTPUT_DIR`

    # --- Capture terminal output to a log file ---
    # Create a unique log file name with timestamp
    log_file_name = f"eda_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_file_path = os.path.join(EDA_OUTPUT_DIR, log_file_name)

    original_stdout = sys.stdout  # Save original stdout
    try:
        with open(log_file_path, 'w') as f:
            sys.stdout = f  # Redirect stdout to file

            print(f"--- EDA Report ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
            print(f"Raw data loaded from: {RAW_DATA_PATH}")
            print(f"EDA outputs saved to: {EDA_OUTPUT_DIR}")
            print(f"Temperature filter applied for region classification: {TEMP_FILTER_EDA}°C")
            print(f"Approximate Vth used for region classification: {VTH_APPROX_EDA}V")

            # --- Data Loading ---
            print("\n--- LOADING DATA ---")
            df = pd.read_csv(RAW_DATA_PATH)
            print(f"Successfully loaded {len(df)} rows from {RAW_DATA_PATH}")

            # Create Vgs and Vds columns - essential for region classification
            df['vgs'] = df['vg'] - df['vs']
            df['vds'] = df['vd'] - df['vs']
            print("  Created 'vgs' (Vg-Vs) and 'vds' (Vd-Vs) columns.")

            print("\n--- DATASET OVERVIEW ---")
            print(f"  Rows, Columns: {df.shape}")
            print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
            print(f"  Available columns: {list(df.columns)}")

            print("\n--- MISSING VALUES ---")
            missing = df.isnull().sum()
            if missing.sum() > 0:
                print(missing[missing > 0])
            else:
                print("  No missing values found.")

            print("\n--- BASIC STATISTICS ---")
            print(df.describe())
            print("\nFirst 5 rows:")
            print(df.head())

            # --- Temperature Analysis ---
            print(f"\n--- Temperature Distribution ---")
            temp_counts = df['temp'].value_counts().sort_index()
            print("  Counts of unique temperatures:")
            print(temp_counts.to_string())  # Use to_string for full display

            plt.figure(figsize=(8, 6))
            temp_counts.plot(kind='bar', color='skyblue', edgecolor='black')
            plt.title('Temperature Distribution', fontsize=plt.rcParams['axes.titlesize'])
            plt.xlabel('Temperature (°C)', fontsize=plt.rcParams['axes.labelsize'])
            plt.ylabel('Count', fontsize=plt.rcParams['axes.labelsize'])
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(EDA_OUTPUT_DIR, 'temperature_distribution.png'))
            plt.close()

            # --- Device Size Analysis (Width-Length) ---
            print("\n--- Device Size Analysis ---")
            device_sizes = df[['w', 'l']].drop_duplicates().sort_values(['l', 'w'])
            print(f"  Number of unique device sizes: {len(device_sizes)}")
            print(f"  Width range: {df['w'].min():.2e} to {df['w'].max():.2e} meters")
            print(f"  Length range: {df['l'].min():.2e} to {df['l'].max():.2e} meters")

            print(f"\n  Unique Device Sizes (W, L) in micrometers:")
            for i, (_, row) in enumerate(device_sizes.iterrows()):
                w_um = row['w'] * 1e6
                l_um = row['l'] * 1e6
                print(f"  [{i + 1:2d}] W={w_um:.2f} µm, L={l_um:.2f} µm")

            # Validate against paper if applicable (e.g., "Paper reported 27 device sizes...")
            # print(f"\n  Paper reported 27 device sizes, we found: {len(device_sizes)}")

            # Number of temperature measurements for each device size
            temp_counts_by_size = df.groupby(['w', 'l'])['temp'].nunique().reset_index(name='temp_count')
            # Total data points for each device size
            device_data_counts_by_size = df.groupby(['w', 'l']).size().reset_index(name='data_point_count')

            # Pivot tables for heatmap visualization
            temp_pivot = temp_counts_by_size.pivot(index='l', columns='w', values='temp_count').fillna(0)
            data_count_pivot = device_data_counts_by_size.pivot(index='l', columns='w',
                                                                values='data_point_count').fillna(0)

            # Map temp_count for visualization if needed (e.g., 0, 1, many)
            # This 'map_temp_count' function seems specific to replicating a particular visual.
            # If not strictly needed, simply plotting temp_pivot is fine.
            # For general EDA, showing exact unique temp counts is often better.
            # def map_temp_count(count):
            #     if count == 0: return 0
            #     elif count == 1: return 1
            #     else: return 4
            # temp_pivot_mapped = temp_pivot.applymap(map_temp_count)

            # Convert column/index headers to micrometers for plotting labels
            temp_pivot.columns = [f"{col * 1e6:.2f} µm" for col in temp_pivot.columns]
            temp_pivot.index = [f"{idx * 1e6:.2f} µm" for idx in temp_pivot.index]
            data_count_pivot.columns = [f"{col * 1e6:.2f} µm" for col in data_count_pivot.columns]
            data_count_pivot.index = [f"{idx * 1e6:.2f} µm" for idx in data_count_pivot.index]

            # Heatmap for Temperature Measurement Counts
            plt.figure(figsize=(10, 7))
            sns.heatmap(temp_pivot, annot=True, cmap='viridis', fmt='.0f', linewidths=.5)
            plt.title('Unique Temperature Measurements by Device Size (W, L)', fontsize=plt.rcParams['axes.titlesize'])
            plt.xlabel('Width (W)', fontsize=plt.rcParams['axes.labelsize'])
            plt.ylabel('Length (L)', fontsize=plt.rcParams['axes.labelsize'])
            plt.tight_layout()
            plt.savefig(os.path.join(EDA_OUTPUT_DIR, 'temp_counts_by_device_size.png'))
            plt.close()

            # Heatmap for Total Data Points Counts
            plt.figure(figsize=(10, 7))
            sns.heatmap(data_count_pivot, annot=True, cmap='cividis', fmt='.0f', linewidths=.5)
            plt.title('Total Data Points by Device Size (W, L)', fontsize=plt.rcParams['axes.titlesize'])
            plt.xlabel('Width (W)', fontsize=plt.rcParams['axes.labelsize'])
            plt.ylabel('Length (L)', fontsize=plt.rcParams['axes.labelsize'])
            plt.tight_layout()
            plt.savefig(os.path.join(EDA_OUTPUT_DIR, 'data_counts_by_device_size.png'))
            plt.close()

            # --- Operating Region Classification ---
            print(f"\n--- Operating Region Distribution (at {TEMP_FILTER_EDA}°C, using Vth={VTH_APPROX_EDA}) ---")

            # Filter data for region classification: Id > 0, Vd > 0, and specific temperature
            filtered_df_for_regions = df[
                (df['id'] > 0) &
                (df['vd'] > 0) &  # Assuming Vd > 0 for standard MOSFET operation in these regions
                (df['temp'] == TEMP_FILTER_EDA)
                ].copy()  # Ensure working on a copy to avoid SettingWithCopyWarning

            if filtered_df_for_regions.empty:
                print(
                    f"  No data found for region classification after filtering for temp={TEMP_FILTER_EDA}°C, Id>0, Vd>0.")
            else:
                # Calculate Vth dynamically based on available Vb (or assume constant for EDA if no Vb variation)
                # For EDA, we might just use the simple Vth_approx_val.
                # If 'vb' column is available and varies, you might calculate Vth per row.
                # For simplicity in EDA, we use a single VTH_APPROX_EDA value.
                # If you want to use calculate_vth helper, you'd need to pass vsb
                # vth_series = calculate_vth(-filtered_df_for_regions['vb'], vth0=VTH_APPROX_EDA)
                # For EDA, a static approx Vth is often sufficient for initial distribution

                filtered_df_for_regions['operating_region'] = filtered_df_for_regions.apply(
                    lambda row: classify_region(row, VTH_APPROX_EDA), axis=1
                )

                region_counts = filtered_df_for_regions['operating_region'].value_counts().reindex(
                    ['Cut-off', 'Linear', 'Saturation']
                ).fillna(0)  # Reindex to ensure consistent order, fill 0 for missing

                print("  Distribution of samples per operating region:")
                print(region_counts.to_string())

                plt.figure(figsize=(8, 6))
                sns.barplot(x=region_counts.index, y=region_counts.values, palette='viridis')
                plt.title(f"Distribution of Operating Regions (at {TEMP_FILTER_EDA}°C, Id>0, Vd>0)",
                          fontsize=plt.rcParams['axes.titlesize'])
                plt.xlabel("Operating Region", fontsize=plt.rcParams['axes.labelsize'])
                plt.ylabel("Number of Samples", fontsize=plt.rcParams['axes.labelsize'])
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(EDA_OUTPUT_DIR, 'operating_region_distribution.png'))
                plt.close()

            # --- Feature Engineering for Correlation ---
            print("\n--- Feature Correlation Matrix (on filtered data for 27C) ---")
            # Create wOverL only if it's relevant and not already there, for correlation
            if 'wOverL' not in filtered_df_for_regions.columns:
                filtered_df_for_regions['wOverL'] = filtered_df_for_regions['w'] / filtered_df_for_regions['l']

            # Calculate log_Id, handling potential warnings for log(0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # Ignore log of zero warning
                # Add a small epsilon to avoid log(0) if Id can truly be zero
                filtered_df_for_regions['log_Id'] = np.log10(filtered_df_for_regions['id'].clip(lower=1e-12))

            # Select relevant columns for correlation
            correlation_cols = ['vg', 'vd', 'vb', 'w', 'l', 'wOverL', 'log_Id',
                                'temp']  # Include temp if it's in the filtered data
            # Filter columns that actually exist in the dataframe
            correlation_cols_exist = [col for col in correlation_cols if col in filtered_df_for_regions.columns]

            if not filtered_df_for_regions.empty and len(correlation_cols_exist) > 1:
                correlation_matrix = filtered_df_for_regions[correlation_cols_exist].corr()
                print("\nCorrelation Matrix:")
                print(correlation_matrix)

                plt.figure(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
                plt.title("Feature Correlation Matrix", fontsize=plt.rcParams['axes.titlesize'])
                plt.savefig(os.path.join(EDA_OUTPUT_DIR, 'correlation_matrix.png'))
                plt.close()
            else:
                print("  Not enough data or relevant columns to compute correlation matrix after filtering.")

            print(f"\nEDA complete. Detailed report saved to: {log_file_path}")
            print(f"All plots saved to: {EDA_OUTPUT_DIR}")

    except Exception as e:
        print(f"An error occurred during EDA: {e}", file=sys.stderr)
    finally:
        sys.stdout = original_stdout  # Restore original stdout even if an error occurs

    print(f"\nEDA script finished. Check '{EDA_OUTPUT_DIR}' for results and plots.")


if __name__ == "__main__":
    run_eda()