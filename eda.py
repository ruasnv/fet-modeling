# eda.py - Exploratory Data Analysis
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime
from src import helpers
from config import TEMP_FILTER

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 8

# --- Configuration for EDA Output ---
EDA_OUTPUT_DIR = 'reports/eda'
os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)

print("Starting EDA...")

# --- Data Loading ---
RAW_DATA_PATH = 'data/nmoshv.csv'
if not os.path.exists(RAW_DATA_PATH):
    print(f"Error: Raw data file not found at {RAW_DATA_PATH}. Please ensure it's in the 'data' directory.")
    exit()

df = pd.read_csv(RAW_DATA_PATH)

# Create Vgs and Vds columns for region classification later
df['vgs'] = df['vg'] - df['vs']
df['vds'] = df['vd'] - df['vs']

# --- Capture terminal output to a log file ---
log_file_path = os.path.join(EDA_OUTPUT_DIR, f"eda_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
with open(log_file_path, 'w') as f:
    # Redirect stdout to file
    import sys

    original_stdout = sys.stdout
    sys.stdout = f

    print("--- DATASET OVERVIEW ---")
    print(f"  Rows, Columns: {df.shape}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
    print(f"  Available columns: {list(df.columns)}")

    print("\n--- MISSING VALUES ---")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("  No missing values found")

    print("\n--- BASIC STATISTICS ---")
    print(df.describe())
    print("\nFirst 5 rows:")
    print(df.head())

    # --- Temperature Analysis ---
    print(f"\n--- Temperature Distribution ---")
    temp_counts = df['temp'].value_counts().sort_index()
    print(temp_counts)

    plt.figure(figsize=(6, 4))
    temp_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Temperature Distribution')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, 'temperature_distribution.png'))
    plt.close()  # Close plot to free memory and prevent display

    # --- Device Size Analysis (Width-Length) - Replicating Fig-1 ---
    print("\n*** Device Size Analysis ***")
    device_sizes = df[['w', 'l']].drop_duplicates().sort_values(['l', 'w'])
    print(f"  Number of unique device sizes: {len(device_sizes)}")
    print(f"  Width range: {df['w'].min():.2e} to {df['w'].max():.2e} meters")
    print(f"  Length range: {df['l'].min():.2e} to {df['l'].max():.2e} meters")

    print(f"\n  Device sizes (W, L) in micrometers:")
    for i, (_, row) in enumerate(device_sizes.iterrows()):
        w_um = row['w'] * 1e6  # Convert meters to micrometers
        l_um = row['l'] * 1e6  # Convert meters to micrometers
        print(f"  [{i + 1:2d}] W={w_um:.2f} µm, L={l_um:.2f} µm")

    print(f"\n  Paper reported 27 device sizes, we found: {len(device_sizes)}")

    # Number of temperature measurements for each device size
    temp_counts_by_size = df.groupby(['w', 'l'])['temp'].nunique().reset_index(name='temp_count')
    device_count_by_size = df.groupby(['w', 'l']).size().reset_index(name='device_count')

    temp_pivot = temp_counts_by_size.pivot(index='l', columns='w', values='temp_count').fillna(0)
    size_pivot = device_count_by_size.pivot(index='l', columns='w', values='device_count').fillna(0)


    def map_temp_count(count):
        if count == 0:
            return 0
        elif count == 1:
            return 1
        else:
            return 4


    temp_pivot_mapped = temp_pivot.applymap(map_temp_count)

    # Convert column/index headers to micrometers for plotting
    temp_pivot_mapped.columns = [f"{col * 1e6:.2f} µm" for col in temp_pivot_mapped.columns]
    temp_pivot_mapped.index = [f"{idx * 1e6:.2f} µm" for idx in temp_pivot_mapped.index]
    size_pivot.columns = [f"{col * 1e6:.2f} µm" for col in size_pivot.columns]
    size_pivot.index = [f"{idx * 1e6:.2f} µm" for idx in size_pivot.index]

    plt.figure(figsize=(8, 5))
    sns.heatmap(temp_pivot_mapped, annot=True, cmap='viridis', fmt='.0f')
    plt.title('Temperature Measurement Counts by Device Size (W, L)')
    plt.xlabel('Width (W)')
    plt.ylabel('Length (L)')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, 'temp_counts_by_device_size.png'))
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.heatmap(size_pivot, annot=True, cmap='viridis', fmt='.0f')
    plt.title('Counts by Device Size')
    plt.xlabel('Width (W)')
    plt.ylabel('Length (L)')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, 'data_counts_by_device_size.png'))
    plt.close()

#TODO Replace with the Helper function for it. Lets see how the dynamic approach changes the distribution.
    # --- NMOS Classification based on regions ---
    print("\n--- Operating Region Distribution (based on Vth=0.7) ---")
    Vth_eda = 0.7  # Using a local Vth for EDA


    def classify_region_eda(a_row, Vth_val):
        Vgs_val, Vds_val = a_row['vgs'], a_row['vds']
        if Vgs_val < Vth_val:
            return 'Cut-off'
        elif Vds_val < (Vgs_val - Vth_val):
            return 'Linear'
        else:
            return 'Saturation'


    temp_filter_eda = TEMP_FILTER
    filtered_df_eda = df[
        (df['id'] > 0) &
        (df['vd'] > 0) &
        (df['temp'] == temp_filter_eda)
        ].copy()
    filtered_df_eda['operating_region'] = filtered_df_eda.apply(classify_region_eda, axis=1, Vth_val=Vth_eda)

    print(filtered_df_eda['operating_region'].value_counts())

    plt.figure(figsize=(6, 4))
    sns.countplot(x='operating_region', data=filtered_df_eda, order=['Cut-off', 'Linear', 'Saturation'])
    plt.title("Distribution of Operating Regions (at 27C, Id>0, Vd>0)")
    plt.xlabel("Operating Region")
    plt.ylabel("Count")
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, 'operating_region_distribution.png'))
    plt.close()

    # --- Feature Correlation Measurement (on filtered data) ---
    print("\n--- Feature Correlation Matrix (on filtered 27C data) ---")
    filtered_df_eda['wOverL'] = filtered_df_eda['w'] / filtered_df_eda['l']
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        filtered_df_eda['log_Id'] = np.log10(filtered_df_eda['id'])

    correlation_matrix = filtered_df_eda[['vg', 'vd', 'vb', 'w', 'l', 'wOverL', 'log_Id']].corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, 'correlation_matrix.png'))
    plt.close()

    print(f"\nEDA complete. Detailed report saved to {log_file_path}")
    print(f"All plots saved to {EDA_OUTPUT_DIR}")

    # Restore stdout
    sys.stdout = original_stdout

print(f"EDA script finished. Check '{EDA_OUTPUT_DIR}' for reports and plots.")