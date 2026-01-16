# scripts/run_eda.py - Orchestrates the EDA process
import os
import sys
from datetime import datetime
from pathlib import Path
import matplotlib

# Append the parent directory to the system path to allow module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Use 'Agg' backend for non-interactive plotting
matplotlib.use('Agg')

from src.eda.augmentedAnalyzer import AugmentedEDAAnalyzer
from src.data_processing.data_loader import DataLoader
from src.utils.helpers import setup_environment
from src.config import settings


def run_eda():
    """
    Orchestrates the Exploratory Data Analysis (EDA) process.
    It loads the raw data, sets up the environment, and then delegates
    all the analysis and plotting to the EDAAnalyzer class.
    All terminal output is redirected to a log file within the EDA output directory.
    """
    print("   Starting EDA Script   ")

    # ---Setup Environment---
    setup_environment()  # ensures all necessary directories are created

    # Extract paths from configurations using correct dotted paths
    raw_data_path = settings.get("paths.raw_data_path")
    eda_output_dir = settings.get("paths.eda_output_dir")

    # --- Data Loading ---
    # Use the DataLoader to load the raw data
    data_loader = DataLoader(raw_data_path)
    df = data_loader.load_raw_data()

    if df is None:
        print("Error: Failed to load raw data. Cannot proceed with EDA.")
        return

    print(f"Raw data loaded successfully for EDA. {len(df)} rows found.")

    # Capture terminal output to a log file to keep as a report
    log_file_name = f"eda_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_file_path = Path(eda_output_dir) / log_file_name

    original_stdout = sys.stdout
    try:
        with open(log_file_path, 'w') as f:
            sys.stdout = f

            print(f"   EDA Report ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})    ")
            print(f"Raw data loaded from: {raw_data_path}")
            print(f"EDA outputs saved to: {eda_output_dir}")
            print(f"Temperature filter for region classification: {settings.get('filtering.temperature_filter')}°C")

            # Initialize and run the EDAAnalyzer with the raw DataFrame
            analyzer = AugmentedEDAAnalyzer(df)
            analyzer.run_all_eda()

    except Exception as e:
        # Print to original stderr if an error occurs during redirection
        print(f"An error occurred during EDA: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
    finally:
        sys.stdout = original_stdout

    print(f"\nEDA script finished. Check '{eda_output_dir}' for the detailed report and plots.")

if __name__ == "__main__":
    run_eda()
