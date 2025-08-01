# scripts/eda.py - Orchestrates the EDA process
import os
import sys
from datetime import datetime
import pandas as pd
import matplotlib  # Ensure matplotlib is imported for backend setting

# Append the parent directory to the system path to allow module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Use 'Agg' backend for non-interactive plotting
matplotlib.use('Agg')

# Import the new EDA package and helper functions
from src.eda.analyzer import EDAAnalyzer
from src.utils.helpers import setup_environment
from src.config import settings


def run_eda():
    """
    Orchestrates the Exploratory Data Analysis (EDA) process.
    It loads the configurations and raw data, sets up the environment,
    performs initial feature creation (vgs, vds), and then delegates
    all the analysis and plotting to the EDAAnalyzer class.
    All terminal output is redirected to a log file within the EDA output directory.
    """
    print("--- Starting EDA Script ---")
    # Extract paths from configurations
    raw_data_path = settings.get("raw_data_path")
    eda_output_dir = settings.get("eda_output_dir")

    # --- Setup Environment (Matplotlib style, output directories, warnings) ---
    setup_environment()

    # --- Data Loading ---
    # Robust path checking as per your original script
    if not os.path.exists(raw_data_path):
        alternative_raw_data_path = os.path.join(os.path.dirname(__file__), '..', raw_data_path)
        if os.path.exists(alternative_raw_data_path):
            raw_data_path = alternative_raw_data_path
        else:
            print(f"Error: Raw data file not found at expected path: {raw_data_path}")
            print(f"Also tried: {alternative_raw_data_path}")
            print("Please ensure 'nmoshv.csv' is in your 'data/' directory.")
            return

    print(f"Loading raw data from: {raw_data_path}")
    df = pd.read_csv(raw_data_path)

    # Create Vgs and Vds columns (essential for operating region classification in EDA)
    df['vgs'] = df['vg'] - df['vs']
    df['vds'] = df['vd'] - df['vs']
    print("  Created 'vgs' (Vg-Vs) and 'vds' (Vd-Vs) columns.")

    # --- Capture terminal output to a log file ---
    log_file_name = f"eda_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_file_path = os.path.join(eda_output_dir, log_file_name)

    original_stdout = sys.stdout
    try:
        with open(log_file_path, 'w') as f:
            sys.stdout = f

            print(f"--- EDA Report ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
            print(f"Raw data loaded from: {raw_data_path}")
            print(f"EDA outputs saved to: {eda_output_dir}")
            print(
                f"Temperature filter for region classification: {settings.get('temperature_filter')}°C")

            # Initialize and run the EDAAnalyzer
            analyzer = EDAAnalyzer(df)
            analyzer.run_all_eda()

    except Exception as e:
        print(f"An error occurred during EDA: {e}", file=sys.stderr)
    finally:
        sys.stdout = original_stdout  # Restore original stdout

    print(f"\nEDA script finished. Check '{eda_output_dir}' for the detailed report and plots.")


if __name__ == "__main__":
    run_eda()
